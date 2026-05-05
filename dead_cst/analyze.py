from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import libcst as cst
import networkx as nx
from libcst.helpers.module import ModuleNameAndPackage
from libcst.metadata import CodeRange, MetadataWrapper

from ._edges import resolve_edges
from ._fqn import FixedFullyQualifiedNameProvider
from ._visitor import SymbolVisitor
from .branches import (
    DefaultUnreachableRegionDetector,
    UnreachableRegionDetector,
)
from .cache import GraphCache, compute_fingerprint
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode, SymbolTrie, VisitorPayload
from .plugins import (
    EdgePlugin,
    ObserveContext,
    PluginContext,
    apply_ops,
)
from .plugins._core import (
    SYNTHETIC_PATH_PREFIXES,
    make_payload,
)
from .resolvers import (
    ImportResolver,
    PathMap,
    PathResolver,
    default_resolve_import,
    distribution_lookup,
    editable_distribution_roots,
    exported_roots,
    safe_resolve_module,
)

logger = logging.getLogger(__name__)


def order_paths(paths: PathMap) -> list[Path]:
    """Topologically sort ``paths.keys()`` so dependencies are processed first.

    ``paths`` maps each base directory to the list of other paths that
    are added to ``sys.path`` while that base is processed. Only keys
    of ``paths`` are returned -- search-path entries that aren't
    themselves keys (e.g. a venv ``site-packages`` directory) are
    treated as lookup-only by the resolver and are never walked.
    Edges between two keys (``dep -> consumer``) constrain the topo
    order so cross-base import resolution sees deps' symbols before
    the consumer's stitch step runs.
    """
    path_order: nx.DiGraph = nx.DiGraph()
    keys = set(paths)
    for base, search_paths in paths.items():
        path_order.add_node(base)
        for sp in search_paths:
            if sp in keys:
                path_order.add_edge(sp, base)
    return list(nx.topological_sort(path_order))


def _chain_resolvers(resolvers: Sequence[PathResolver]) -> ImportResolver:
    """Compose ``resolvers`` into one ``name -> path`` callable.

    Each resolver's :meth:`PathResolver.resolve_import` is tried in
    order; the first non-``None`` answer wins. With no resolvers,
    falls back to :func:`default_resolve_import` so the analyzer keeps
    working when callers don't pass any (the common public-API case).
    A single-resolver chain skips the closure -- the typical CLI
    invocation passes one resolver, and the chain would just call its
    method directly.
    """
    if not resolvers:
        return default_resolve_import
    if len(resolvers) == 1:
        return resolvers[0].resolve_import

    def _resolve(name: str, search_paths: list[Path]) -> str | Path | None:
        for resolver in resolvers:
            result = resolver.resolve_import(name, search_paths)
            if result is not None:
                return result
        return None

    return _resolve


def build_symbol_graph(
    paths: PathMap,
    *,
    plugins: Sequence[EdgePlugin] = (),
    resolvers: Sequence[PathResolver] = (),
    project_root: Path | None = None,
    cache: GraphCache | None = None,
    unreachable_detector: UnreachableRegionDetector | None = None,
    workers: int | None = None,
) -> nx.MultiDiGraph:
    """Build a directed reachability graph of every top-level symbol under ``paths``.

    Thin wrapper over :class:`Analysis` for callers that want the whole
    graph in one shot. Equivalent to::

        Analysis(
            paths,
            plugins=plugins,
            resolvers=resolvers,
            project_root=project_root,
            cache=cache,
            unreachable_detector=unreachable_detector,
            workers=workers,
        ).materialize_all()

    For incremental queries (single package, lazy materialization,
    scoped cache refresh), construct an :class:`Analysis` directly.

    Each ``.py`` file under each base is parsed with LibCST; modules,
    classes, functions, top-level variables, and module-level imports
    become :class:`SymbolNode` graph nodes. Edges encode "keeps alive"
    relationships:

    * a reference points at its referent,
    * a declaration points at its containing module, and
    * a submodule points at its parent package.

    References made from inside a statically-dead suite are still emitted
    but tagged with :data:`EdgeFlags.DEAD_BRANCH`. Default
    :func:`find_reachable` does not filter on the flag, so those refs
    still propagate liveness through the enclosing decl. The opt-in
    :func:`find_kept_alive_by_dead_branches` returns the set of symbols
    that would become unreachable if every dead suite were removed.

    Third-party imports are surfaced as synthetic ``[external dist] <name>``
    / ``[external file] <name>`` nodes so callers can audit the
    project's dependency surface (see the ``dependencies`` CLI command).

    Plugins run once per base in topological order, after that base's import
    edges have been resolved. Each plugin invocation gets a per-base
    :class:`PluginContext`; its :meth:`PluginContext.parse` lazily reads +
    parses files on first request and memoizes the result for the rest
    of the analysis.

    See :class:`Analysis` for the full parameter documentation; the
    arguments here are passed through verbatim.

    Returns
    -------
    networkx.MultiDiGraph
        Nodes are :class:`SymbolNode` instances. Edges carry a ``flags``
        attribute (:class:`EdgeFlags`); ``DEAD_BRANCH``-flagged edges
        originated inside a statically-dead suite. Entrypoint seeds
        carry ``graph.nodes[node]["entrypoint"] = True``. Per-file
        dead-suite positions are exposed via ``graph.graph["dead_suites"]``,
        a ``{path: tuple[CodeRange]}`` mapping.
    """
    return Analysis(
        paths,
        plugins=plugins,
        resolvers=resolvers,
        project_root=project_root,
        cache=cache,
        unreachable_detector=unreachable_detector,
        workers=workers,
    ).materialize_all()


def _run_observe(
    plugins: Sequence[EdgePlugin],
    path: Path,
    module: cst.Module,
    base_payload: VisitorPayload,
    base: Path,
    project_root: Path,
) -> VisitorPayload:
    """Invoke each plugin's :meth:`EdgePlugin.observe` and collect contributions.

    Returns a single :class:`VisitorPayload` that merges every plugin's
    additions for this file. Plugins that return ``None`` contribute
    nothing. The result is concatenated with the visitor's payload by
    :func:`_merge_payloads` and cached together so that warm runs skip
    both the visitor and the observe pass.
    """
    if not plugins:
        return make_payload()
    ctx = ObserveContext(
        path=path,
        module=module,
        payload=base_payload,
        base=base,
        project_root=project_root,
    )
    payloads: list[VisitorPayload] = []
    for plugin in plugins:
        if not isinstance(plugin, EdgePlugin):
            raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
        contribution = plugin.observe(ctx)
        if contribution is not None:
            payloads.append(contribution)
    return _merge_payloads(*payloads) if payloads else make_payload()


def _process_one_file(
    file: Path,
    *,
    fqn_entry: ModuleNameAndPackage,
    search_paths: list[Path],
    import_resolver: ImportResolver,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    base: Path,
    project_root: Path,
) -> VisitorPayload:
    """Run the visitor + observe pass for a single file and return its payload.

    The caller owns the precomputed FQN entry (built once per base by
    :func:`_collect_base_specs`) so we can construct
    :class:`MetadataWrapper` directly with ``cache=`` injected,
    skipping :class:`FullRepoManager`'s per-instance ``gen_cache``
    rebuild. Same shape ``FullRepoManager.get_metadata_wrapper_for_path``
    builds, just without re-walking the file list every time.
    """
    module = cst.parse_module(file.read_text())
    wrapper = MetadataWrapper(
        module,
        unsafe_skip_copy=True,
        cache={FixedFullyQualifiedNameProvider: fqn_entry},
    )
    visitor = SymbolVisitor(
        file,
        search_paths,
        import_resolver,
        unreachable_detector=detector,
        wrapper=wrapper,
    )
    wrapper.visit(visitor)
    base_payload = visitor.to_payload()
    plugin_payload = _run_observe(plugins, file, wrapper.module, base_payload, base, project_root)
    return _merge_payloads(base_payload, plugin_payload)


@dataclass(frozen=True, slots=True)
class _BaseSpec:
    """Per-base file list partitioned into cache hits and miss files.

    Built once up front in :func:`_collect_base_specs` so the parallel
    path can flatten every base's miss files into a single sorted task
    list and the apply phase can iterate every base in deterministic
    order without re-reading the cache. ``fqn_cache`` covers only the
    miss files -- hit files never go through the visitor, so they
    don't need an FQN entry. ``fingerprint`` is this base's per-base
    cache fingerprint (see :func:`compute_fingerprint`); the recorder
    uses it when writing payloads back, and :func:`_build_base_spec`
    uses it when looking them up.
    """

    base: Path
    search_paths: tuple[Path, ...]
    files: tuple[Path, ...]
    hits: dict[Path, VisitorPayload]
    miss_files: tuple[Path, ...]
    fqn_cache: Mapping[str, ModuleNameAndPackage]
    fingerprint: str


def _build_base_spec(
    base: Path,
    paths: PathMap,
    cache: GraphCache | None,
    fingerprint: str,
) -> _BaseSpec:
    """Build a single :class:`_BaseSpec`: enumerate ``base``'s files and
    partition into cache hits / misses against ``fingerprint``.

    Per-base helper so :class:`Analysis` can refresh one base at a
    time without paying for sibling bases' file walks. Each cache
    lookup uses this base's per-base fingerprint, so a sibling base's
    config change leaves these rows valid. The FQN cache covers only
    the miss files because hit files never go through the visitor.
    """
    search_paths = (base,) + tuple(paths.get(base, []))
    files = tuple(sorted(base.rglob("*.py")))
    hits: dict[Path, VisitorPayload] = {}
    miss_files: list[Path] = []
    for file in files:
        payload = cache.get(file, fingerprint) if cache is not None else None
        if payload is None:
            miss_files.append(file)
        else:
            hits[file] = payload
    fqn_cache: Mapping[str, ModuleNameAndPackage] = (
        FixedFullyQualifiedNameProvider.gen_cache(base, [str(f) for f in miss_files], timeout=5)
        if miss_files
        else {}
    )
    return _BaseSpec(
        base=base,
        search_paths=search_paths,
        files=files,
        hits=hits,
        miss_files=tuple(miss_files),
        fqn_cache=fqn_cache,
        fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _Task:
    """Per-file unit of work for the visitor + observe pass.

    ``fqn_entry`` is this file's slice of the per-base FQN cache built
    once in the parent (see :func:`_collect_base_specs`); the runner
    injects it into a :class:`MetadataWrapper` rather than rebuilding
    a :class:`FullRepoManager`. ``search_paths`` doubles as the
    transition key for ``sys.path`` rebinding + resolver-cache
    invalidation.
    """

    file: Path
    base: Path
    search_paths: tuple[Path, ...]
    fqn_entry: ModuleNameAndPackage
    project_root: Path


@dataclass(slots=True)
class _RunnerState:
    """Mutable state for the per-task runner, shared by serial and worker paths.

    ``sys_path_baseline`` is the original ``sys.path`` captured at
    runner startup; transitions rebuild ``sys.path`` from
    ``search_paths + baseline`` and clear the resolver LRUs. The
    serial caller restores ``sys.path`` from the baseline when the run
    finishes; workers don't bother since the process is about to exit.
    """

    detector: UnreachableRegionDetector
    plugins: tuple[EdgePlugin, ...]
    import_resolver: ImportResolver
    sys_path_baseline: list[str]
    last_search_paths: tuple[Path, ...] | None = field(default=None)


def _rebind_sys_path(search_paths: tuple[Path, ...], baseline: list[str]) -> None:
    """Set ``sys.path`` to ``search_paths + baseline``, deduplicated."""
    new_path: list[str] = []
    seen: set[str] = set()
    for p in search_paths:
        s = str(p)
        if s not in seen:
            new_path.append(s)
            seen.add(s)
    for s in baseline:
        if s not in seen:
            new_path.append(s)
            seen.add(s)
    sys.path[:] = new_path


def _clear_resolver_caches() -> None:
    """Drop the ``sys.path``-derived resolver caches.

    :func:`safe_resolve_module` keys on fullname and
    :func:`distribution_lookup` / :func:`editable_distribution_roots`
    key on ``()`` -- all three read live ``sys.path`` (or
    :mod:`importlib.metadata` against it) and would otherwise return
    stale results across bases (different first-party prefix;
    uv-workspace members shipping their own ``.venv``).
    """
    safe_resolve_module.cache_clear()
    distribution_lookup.cache_clear()
    editable_distribution_roots.cache_clear()


def _process_task(state: _RunnerState, task: _Task) -> tuple[Path, Path, VisitorPayload]:
    """Run one task, transitioning ``sys.path`` + caches if the base changed.

    Used by both the in-process serial loop and the
    :class:`ProcessPoolExecutor` workers; the only difference between
    the two is who owns ``state`` (a local in serial, a per-process
    global in workers).
    """
    if state.last_search_paths != task.search_paths:
        _rebind_sys_path(task.search_paths, state.sys_path_baseline)
        _clear_resolver_caches()
        state.last_search_paths = task.search_paths
    payload = _process_one_file(
        task.file,
        fqn_entry=task.fqn_entry,
        search_paths=list(task.search_paths),
        import_resolver=state.import_resolver,
        detector=state.detector,
        plugins=state.plugins,
        base=task.base,
        project_root=task.project_root,
    )
    return task.base, task.file, payload


_worker_state: _RunnerState | None = None


def _init_worker(
    detector: UnreachableRegionDetector,
    plugins: tuple[EdgePlugin, ...],
    resolvers: tuple[PathResolver, ...],
) -> None:
    """Pool initializer: build the worker's :class:`_RunnerState`.

    Resolver chain is rebuilt locally because :func:`_chain_resolvers`
    can return a closure (not picklable); the resolver instances
    themselves travel as picklable dataclasses.
    """
    global _worker_state
    _worker_state = _RunnerState(
        detector=detector,
        plugins=plugins,
        import_resolver=_chain_resolvers(resolvers),
        sys_path_baseline=list(sys.path),
    )


def _worker_process_task(task: _Task) -> tuple[Path, Path, VisitorPayload]:
    """Pool task: delegate to :func:`_process_task` against the worker's state."""
    assert _worker_state is not None, "_init_worker must run before _worker_process_task"
    return _process_task(_worker_state, task)


def _build_sorted_tasks(base_specs: dict[Path, _BaseSpec], project_root: Path) -> list[_Task]:
    """Flatten every base's miss files into one task list, sorted by search_paths.

    Sorting puts same-base tasks adjacent so a runner sees at most one
    ``sys.path`` transition per base it touches, regardless of total
    file count. Sorting on ``Path`` tuples directly is fine -- it's
    just attribute access per comparison.
    """
    tasks: list[_Task] = [
        _Task(
            file=file,
            base=base,
            search_paths=spec.search_paths,
            fqn_entry=spec.fqn_cache[str(file)],
            project_root=project_root,
        )
        for base, spec in base_specs.items()
        for file in spec.miss_files
    ]
    tasks.sort(key=lambda t: (t.search_paths, t.file))
    return tasks


def _compute_all_miss_payloads(
    *,
    base_specs: dict[Path, _BaseSpec],
    project_root: Path,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    resolvers: Sequence[PathResolver],
    import_resolver: ImportResolver,
    cache: GraphCache | None,
    workers: int | None,
) -> dict[Path, dict[Path, VisitorPayload]]:
    """Run visitor + observe for every cache-miss file across every base.

    Both branches use :func:`_process_task` for the per-task work; they
    differ only in whether the runner state lives on the main process
    or in :class:`ProcessPoolExecutor` workers. The pool is opt-in
    (``workers >= 2`` and at least two miss files); below that, the
    in-process path avoids pool startup cost.

    Cache writes happen on the main process as each payload arrives,
    so a partial run still warms the cache for files that completed.
    Returns ``{base: {file: payload}}`` even for bases with no misses,
    so the caller can index without an existence check.
    """
    out: dict[Path, dict[Path, VisitorPayload]] = {b: {} for b in base_specs}
    total_misses = sum(len(s.miss_files) for s in base_specs.values())
    if total_misses == 0:
        return out

    tasks = _build_sorted_tasks(base_specs, project_root)
    use_pool = workers is not None and workers >= 2 and total_misses >= 2

    def _record(base: Path, file: Path, payload: VisitorPayload) -> None:
        out[base][file] = payload
        if cache is not None:
            cache.put(file, payload, base_specs[base].fingerprint)

    if use_pool:
        assert workers is not None
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_init_worker,
            initargs=(detector, tuple(plugins), tuple(resolvers)),
        ) as pool:
            for base, file, payload in pool.map(_worker_process_task, tasks):
                _record(base, file, payload)
        return out

    # Serial: run every task in-process. Restore ``sys.path`` on the
    # way out so callers (tests, library users) don't see lingering
    # mutations from the runner's per-base rebinds.
    baseline = list(sys.path)
    state = _RunnerState(
        detector=detector,
        plugins=tuple(plugins),
        import_resolver=import_resolver,
        sys_path_baseline=baseline,
    )
    try:
        for task in tasks:
            base, file, payload = _process_task(state, task)
            _record(base, file, payload)
    finally:
        sys.path[:] = baseline
    return out


def _merge_payloads(*payloads: VisitorPayload) -> VisitorPayload:
    """Concatenate the ``nodes``/``edges``/``imports``/``dead_suites`` of every payload."""
    nodes: list[SymbolNode] = []
    edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []
    imports: list[tuple[SymbolNode, Import, CodeRange]] = []
    dead_suites: list[CodeRange] = []
    for p in payloads:
        nodes.extend(p.nodes)
        edges.extend(p.edges)
        imports.extend(p.imports)
        dead_suites.extend(p.dead_suites)
    return VisitorPayload(
        nodes=tuple(nodes),
        edges=tuple(edges),
        imports=tuple(imports),
        dead_suites=tuple(dead_suites),
    )


def _contains(suite: CodeRange, access: CodeRange) -> bool:
    """``True`` iff ``access`` is fully nested inside ``suite``.

    Compares ``(line, column)`` lexicographically at both ends. Suites
    are line-aligned in practice (libcst positions an ``IndentedBlock``
    at its first statement) so the line check usually decides; the
    column tiebreak handles one-line ``if False: x = 1`` suites.
    """
    s_start = (suite.start.line, suite.start.column)
    s_end = (suite.end.line, suite.end.column)
    a_start = (access.start.line, access.start.column)
    a_end = (access.end.line, access.end.column)
    return s_start <= a_start and a_end <= s_end


def _apply_payload(
    payload: VisitorPayload,
    *,
    current_trie: SymbolTrie,
    export_trie: SymbolTrie,
    file_exported: bool,
    symbol_graph: nx.MultiDiGraph,
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
) -> None:
    """Emit ``payload`` into the in-progress per-base structures.

    Drives all node routing off ``SymbolNode.flags`` and ``type``:

    * ``type == "module"`` goes into the graph and the trie; no parent
      edge (modules are themselves the parent target).
    * ``type == "synthetic"`` (plugin-emitted markers) goes into the
      graph only -- no parent edge, no trie entry. Synthetic fqnames
      don't fit the dotted module hierarchy and aren't lookup targets
      for cross-module imports.
    * Other decls go into the graph with a parent-module edge.
      ``NodeFlags.SHADOWED`` excludes them from the trie -- the graph
      keeps the parent edge so the decl stays well-formed, but
      cross-module imports never resolve to it.
    * ``NodeFlags.ENTRYPOINT`` (typically on plugin synthetics)
      seeds reachability: ``graph.nodes[node]["entrypoint"] = True``
      so :func:`find_reachable` starts its BFS from this node.

    Edge flag derivation: each ``(src, dst, access_pos)`` entry has
    its access position tested against ``payload.dead_suites`` for
    containment. If matched, the resulting graph edge gets
    :data:`EdgeFlags.DEAD_BRANCH`. Plugin-emitted edges use
    ``SYNTHETIC_POSITION`` (line 0), which never falls inside a real
    dead suite, so they always land with ``EdgeFlags.NONE``.
    Unresolved cross-file imports accumulate into ``import_edges``
    along with the derived flag and are fed to :func:`resolve_edges`
    once the per-base trie is fully built; resolution preserves the
    flag through every emission.

    Per-file dead-suite positions are stashed on the graph as
    ``graph.graph["dead_suites"][module.path]`` for downstream
    reporting (e.g. "this file has unreachable code at line X").
    """
    module = next(n for n in payload.nodes if n.type == "module")

    def flag_for(pos: CodeRange) -> EdgeFlags:
        return (
            EdgeFlags.DEAD_BRANCH
            if any(_contains(s, pos) for s in payload.dead_suites)
            else EdgeFlags.NONE
        )

    for n in payload.nodes:
        symbol_graph.add_node(n)
        if n.flags & NodeFlags.ENTRYPOINT:
            symbol_graph.nodes[n]["entrypoint"] = True
        if n.type == "synthetic":
            continue
        if n.type != "module":
            symbol_graph.add_edge(n, module, flags=EdgeFlags.NONE)
        if not (n.flags & NodeFlags.SHADOWED):
            current_trie.add_declaration(n)
            if file_exported:
                export_trie.add_declaration(n)

    for src, dst, pos in payload.edges:
        symbol_graph.add_edge(src, dst, flags=flag_for(pos))

    for src, imp, pos in payload.imports:
        import_edges.add((src, imp, flag_for(pos)))

    if payload.dead_suites:
        symbol_graph.graph["dead_suites"][module.path] = payload.dead_suites


@dataclass(slots=True)
class _BaseContribution:
    """One base's pre-stitched contribution to the symbol graph.

    Built once per base by :func:`_build_contribution` from that base's
    payloads. Composed into a target graph (full or closure-scoped) by
    :func:`_compose_contribution`, which then runs cross-base edge
    stitching and plugin :meth:`EdgePlugin.finalize` against the
    composed graph.

    Splitting "intra-base apply" from "cross-base compose" is what
    lets :class:`Analysis` materialize different scopes (full graph
    vs. a single base's closure) without redoing per-file apply work.
    Each contribution is also self-contained: ``import_edges`` is the
    raw cross-file references to feed to :func:`resolve_edges`, and
    ``current_trie`` / ``export_trie`` are this base's lookup tables
    (everything visible internally, vs. what consumers see).
    """

    base: Path
    current_trie: SymbolTrie
    export_trie: SymbolTrie
    base_graph: nx.MultiDiGraph
    import_edges: frozenset[tuple[SymbolNode, Import, EdgeFlags]]
    dead_suites_per_file: dict[Path, tuple[CodeRange, ...]]


def _build_contribution(
    base: Path,
    spec: _BaseSpec,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> _BaseContribution:
    """Apply ``spec``'s per-file payloads into a base-local graph slice.

    Identical per-file work to the per-base loop in
    :func:`build_symbol_graph` pre-refactor: the only change is that
    ``_apply_payload`` mutates a fresh per-base ``nx.MultiDiGraph``
    rather than the shared full graph, so the resulting contribution
    can be composed into different target graphs (full vs. a single
    base's closure) without redoing this work.
    """
    current_trie = SymbolTrie()
    export_trie = SymbolTrie()
    export_roots = exported_roots(base)
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    base_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    base_graph.graph["dead_suites"] = {}
    for file in spec.files:
        payload = spec.hits.get(file)
        if payload is None:
            payload = miss_payloads[file]
        # See ``_apply_payload``: a file's decls go into ``export_trie``
        # only when the file lives under one of the base's exported dirs
        # (or when the base has no export restriction).
        file_exported = export_roots is None or _under_any(file, export_roots)
        _apply_payload(
            payload,
            current_trie=current_trie,
            export_trie=export_trie,
            file_exported=file_exported,
            symbol_graph=base_graph,
            import_edges=import_edges,
        )
    current_trie.add_module_hierarchy_edges(base_graph)
    return _BaseContribution(
        base=base,
        current_trie=current_trie,
        export_trie=export_trie,
        base_graph=base_graph,
        import_edges=frozenset(import_edges),
        dead_suites_per_file=dict(base_graph.graph["dead_suites"]),
    )


def _compose_contribution(
    contrib: _BaseContribution,
    *,
    target_graph: nx.MultiDiGraph,
    symbol_lookup: SymbolTrie,
    plugins: Sequence[EdgePlugin],
    project_root: Path,
) -> None:
    """Merge ``contrib.base_graph`` into ``target_graph``, stitch
    cross-base imports against ``symbol_lookup``, and run plugin
    :meth:`EdgePlugin.finalize` against the composed graph.

    The caller owns ``symbol_lookup`` because its construction depends
    on which dep export tries are in scope -- the full-graph path
    merges every dep's exports, while the closure-scoped path merges
    only deps inside the requested scope.
    """
    for n, attrs in contrib.base_graph.nodes(data=True):
        target_graph.add_node(n)
        if attrs.get("entrypoint"):
            target_graph.nodes[n]["entrypoint"] = True
    for u, v, attrs in contrib.base_graph.edges(data=True):
        target_graph.add_edge(u, v, **attrs)
    target_graph.graph.setdefault("dead_suites", {}).update(contrib.dead_suites_per_file)
    for src, dst, flags in resolve_edges(contrib.import_edges, symbol_lookup, contrib.base):
        target_graph.add_edge(src, dst, flags=flags)
    if plugins:
        ctx = PluginContext(
            graph=target_graph,
            symbol_lookup=symbol_lookup,
            base=contrib.base,
            project_root=project_root,
        )
        for plugin in plugins:
            if not isinstance(plugin, EdgePlugin):
                raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
            # Materialize before applying so plugins can iterate
            # ctx.graph.nodes without tripping "dictionary changed
            # size during iteration".
            ops = list(plugin.finalize(ctx))
            apply_ops(target_graph, ops)


def _under_any(file: Path, roots: list[Path]) -> bool:
    """True iff ``file`` is equal to or nested under any of ``roots``."""
    f = file.resolve()
    for r in roots:
        if f == r or f.is_relative_to(r):
            return True
    return False


def _infer_project_root(paths: PathMap) -> Path:
    bases = list(paths)
    if not bases:
        return Path.cwd()
    return min(bases, key=lambda p: len(p.parts))


def find_reachable(graph: nx.MultiDiGraph) -> set[SymbolNode]:
    """BFS forward from every node tagged as an entrypoint by a plugin.

    Plugins mark seeds by setting ``graph.nodes[node]["entrypoint"] = True``
    (see :func:`dead_cst.plugins.apply_ops`). There is no longer any
    built-in matching against file paths or FQNs -- that lives in
    :class:`ExplicitEntrypointPlugin`.

    Edges flagged with :data:`EdgeFlags.DEAD_BRANCH` are NOT filtered
    here -- today's behavior, where dead-code references propagate
    liveness through the enclosing decl, is preserved. See
    :func:`find_kept_alive_by_dead_branches` for the strict alternative.
    """
    visited: set[SymbolNode] = set()
    stack = [n for n, attrs in graph.nodes(data=True) if attrs.get("entrypoint")]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph.successors(node))
    return visited


def _find_reachable_strict(graph: nx.MultiDiGraph) -> set[SymbolNode]:
    """Like :func:`find_reachable` but skips ``DEAD_BRANCH``-flagged edges."""
    visited: set[SymbolNode] = set()
    stack = [n for n, attrs in graph.nodes(data=True) if attrs.get("entrypoint")]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for _, succ, attrs in graph.out_edges(node, data=True):
            if attrs.get("flags", EdgeFlags.NONE) & EdgeFlags.DEAD_BRANCH:
                continue
            stack.append(succ)
    return visited


def find_kept_alive_by_dead_branches(graph: nx.MultiDiGraph) -> set[SymbolNode]:
    """Return symbols that would become unreachable if every dead suite were removed.

    Computed as ``find_reachable(graph) -`` strict-mode BFS that skips
    every edge flagged :data:`EdgeFlags.DEAD_BRANCH`. The resulting set
    is the "blast radius" of removing every statically-dead suite in
    the analyzed source -- symbols currently kept alive only through a
    chain that crosses at least one dead-branch reference.

    Used by tooling that reports "if you removed your unreachable
    code, these additional symbols would also become dead." Default
    :func:`find_reachable` is unchanged; this is an opt-in stricter
    pass.
    """
    return find_reachable(graph) - _find_reachable_strict(graph)


def count_nodes(graph: nx.MultiDiGraph, prefix: Path | None) -> dict[str, int]:
    """Count nodes in ``graph`` by ``SymbolNode.type``, optionally restricted by path.

    If ``prefix`` is given, only nodes whose ``path`` is under ``prefix`` are
    counted -- useful for per-base summaries when several packages are
    analysed together. Includes the synthetic ``"synthetic"`` type contributed
    by plugins and third-party-dep markers; the CLI suppresses that key when
    rendering summaries.
    """
    counts = {}
    for node in graph.nodes:
        if prefix and not node.path.is_relative_to(prefix):
            continue
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts


class Analysis:
    """Lazy entrypoint to the dead-cst pipeline.

    Holds the analyzer's config (paths, plugins, resolvers, cache,
    detector, worker count) and memoizes per-base work so multiple
    queries against the same project share the cost. Construction is
    cheap; nothing is read or parsed until you ask.

    Three coarse stages happen on demand:

    1. **Per-base file enumeration + visitor pass** -- driven by
       :meth:`refresh`. Walks each requested base's files, hashes
       them against the cache, runs the visitor + observe pass on
       misses, writes payloads back to the cache. Idempotent and
       scoped: ``refresh(bases=[B])`` touches only ``B``'s file tree.

    2. **Per-base contribution build** -- the per-base trie + a
       base-local graph slice + the unresolved cross-file import set.
       Built once per base from the payloads above, memoized for the
       lifetime of the :class:`Analysis`.

    3. **Cross-base composition** -- merging contributions, running
       :func:`resolve_edges` against the merged tries, running plugin
       :meth:`EdgePlugin.finalize`. Scoped to either the full base set
       (:meth:`materialize_all`) or the "interesting set" of one base
       (:meth:`materialize_closure` / :meth:`PackageView.graph`),
       which is the forward dependency closure of that base's reverse
       (consumer) closure -- the only bases that could keep a decl in
       the target base alive.

    The lazy split lets cheap per-base queries skip stage 3 entirely:
    :meth:`PackageView.modules` and :meth:`PackageView.declarations`
    only need stage 2 for their own base. Reachability queries
    (:meth:`PackageView.dead`, :meth:`Analysis.dead`) trigger stage 3
    over the appropriate scope -- the "interesting set" for a single
    package, or every base for the full graph. Composing a graph is
    much cheaper than recomputing payloads, so per-package queries
    against a warm cache stay fast even on large repos.

    Parameters mirror :func:`build_symbol_graph`; see its docstring for
    detailed semantics. Once constructed, an :class:`Analysis` is
    effectively read-only -- mutating ``paths`` after construction has
    no effect because the configuration is copied. Spin up a fresh
    instance to pick up new search paths or new plugins.
    """

    def __init__(
        self,
        paths: PathMap,
        *,
        plugins: Sequence[EdgePlugin] = (),
        resolvers: Sequence[PathResolver] = (),
        project_root: Path | None = None,
        cache: GraphCache | None = None,
        unreachable_detector: UnreachableRegionDetector | None = None,
        workers: int | None = None,
    ) -> None:
        self._paths: dict[Path, list[Path]] = {b: list(deps) for b, deps in paths.items()}
        self._plugins: tuple[EdgePlugin, ...] = tuple(plugins)
        self._resolvers: tuple[PathResolver, ...] = tuple(resolvers)
        self._cache = cache
        self._workers = workers
        self._project_root: Path = (
            project_root
            if project_root is not None
            else (_infer_project_root(self._paths) if self._paths else Path.cwd())
        )
        self._import_resolver: ImportResolver = _chain_resolvers(self._resolvers)
        self._detector: UnreachableRegionDetector = (
            unreachable_detector
            if unreachable_detector is not None
            else DefaultUnreachableRegionDetector()
        )
        self._ordered_bases: list[Path] | None = None
        self._reverse_closures: dict[Path, frozenset[Path]] = {}
        self._interesting_sets: dict[Path, frozenset[Path]] = {}
        self._base_specs: dict[Path, _BaseSpec] = {}
        self._refreshed: set[Path] = set()
        self._contributions: dict[Path, _BaseContribution] = {}
        self._closure_graphs: dict[Path, nx.MultiDiGraph] = {}
        self._full_graph: nx.MultiDiGraph | None = None

    @property
    def paths(self) -> PathMap:
        """Read-only view of the base -> deps mapping this analysis was built with."""
        return {b: list(deps) for b, deps in self._paths.items()}

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def bases(self) -> list[Path]:
        """Bases in topological order (deps before dependents)."""
        if self._ordered_bases is None:
            self._ordered_bases = order_paths(self._paths)
        return list(self._ordered_bases)

    def reverse_closure(self, base: Path) -> frozenset[Path]:
        """Bases that transitively depend on ``base`` (including ``base`` itself).

        These are the only bases whose source code could import (and
        therefore keep alive) decls under ``base``. A query like "is X
        in ``base`` dead?" only needs entrypoints from this set --
        sibling bases that don't reach ``base`` in the search-paths
        DAG can't reference its decls.
        """
        if base not in self._paths:
            raise KeyError(base)
        if base in self._reverse_closures:
            return self._reverse_closures[base]
        rev: dict[Path, set[Path]] = {b: set() for b in self.bases}
        for b, deps in self._paths.items():
            for dep in deps:
                rev.setdefault(dep, set()).add(b)
        result: set[Path] = {base}
        stack = [base]
        while stack:
            n = stack.pop()
            for parent in rev.get(n, ()):
                if parent not in result:
                    result.add(parent)
                    stack.append(parent)
        closure = frozenset(result)
        self._reverse_closures[base] = closure
        return closure

    def _interesting_set(self, base: Path) -> frozenset[Path]:
        """Bases needed to answer reachability queries about decls in ``base``.

        Computed as the forward (deps) closure of :meth:`reverse_closure`:
        every consumer of ``base`` plus every consumer's transitive
        deps, so we have enough trie data to resolve every cross-base
        import that could lead into ``base``.
        """
        if base in self._interesting_sets:
            return self._interesting_sets[base]
        consumers = self.reverse_closure(base)
        result: set[Path] = set()
        stack: list[Path] = list(consumers)
        while stack:
            b = stack.pop()
            if b in result:
                continue
            result.add(b)
            for dep in self._paths.get(b, []):
                if dep in self._paths and dep not in result:
                    stack.append(dep)
        scope = frozenset(result)
        self._interesting_sets[base] = scope
        return scope

    def refresh(self, bases: Iterable[Path] | None = None) -> Analysis:
        """Update the cache and build per-base contributions for the given bases.

        ``bases=None`` (the default) refreshes every base. Passing a
        subset scopes the file walk + visitor pass to those bases
        only -- sibling bases are untouched. Already-refreshed bases
        are skipped, so calling :meth:`refresh` twice with the same
        argument is cheap.

        Returns ``self`` so callers can chain
        ``Analysis(...).refresh().materialize_all()``.
        """
        targets = list(bases) if bases is not None else self.bases
        unknown = [b for b in targets if b not in self._paths]
        if unknown:
            raise KeyError(f"Unknown bases: {unknown}")
        new_targets = [b for b in targets if b not in self._refreshed]
        if not new_targets:
            return self
        for b in new_targets:
            if b not in self._base_specs:
                fingerprint = self._base_fingerprint(b)
                self._base_specs[b] = _build_base_spec(b, self._paths, self._cache, fingerprint)
        partial_specs = {b: self._base_specs[b] for b in new_targets}
        miss_payloads = _compute_all_miss_payloads(
            base_specs=partial_specs,
            project_root=self._project_root,
            detector=self._detector,
            plugins=self._plugins,
            resolvers=self._resolvers,
            import_resolver=self._import_resolver,
            cache=self._cache,
            workers=self._workers,
        )
        for b in new_targets:
            self._contributions[b] = _build_contribution(b, self._base_specs[b], miss_payloads[b])
            self._refreshed.add(b)
        return self

    def package(self, base: Path) -> PackageView:
        """Return a lazy view onto a single base.

        The returned :class:`PackageView` is cheap; per-base work is
        triggered by its query methods.
        """
        if base not in self._paths:
            raise KeyError(base)
        return PackageView(self, base)

    def packages(self) -> Iterator[PackageView]:
        """Yield a :class:`PackageView` for every base in topological order."""
        for base in self.bases:
            yield PackageView(self, base)

    def materialize_all(self) -> nx.MultiDiGraph:
        """Build the full graph (every base, cross-base resolution, plugins).

        Memoized: the second call returns the same graph object.
        Refreshes every base first, so this is also the trigger for a
        whole-project cache refresh in callers that don't refresh
        explicitly.
        """
        if self._full_graph is not None:
            return self._full_graph
        self.refresh()
        g = nx.MultiDiGraph()
        g.graph["dead_suites"] = {}
        for base in self.bases:
            contrib = self._contributions[base]
            symbol_lookup = self._build_symbol_lookup(base, scope=None)
            _compose_contribution(
                contrib,
                target_graph=g,
                symbol_lookup=symbol_lookup,
                plugins=self._plugins,
                project_root=self._project_root,
            )
        self._full_graph = g
        return g

    def materialize_closure(self, base: Path) -> nx.MultiDiGraph:
        """Build a graph containing every contribution in ``_interesting_set(base)``.

        The result is the smallest graph that gives correct
        reachability answers for decls in ``base``: every consumer of
        ``base`` (so we see every potential alive-keeper) plus every
        consumer's transitive deps (so cross-base imports resolve).

        If :meth:`materialize_all` has already been called, returns
        the full graph instead -- it's a strict superset and cheaper
        than recomputing.
        """
        if self._full_graph is not None:
            return self._full_graph
        if base in self._closure_graphs:
            return self._closure_graphs[base]
        scope = self._interesting_set(base)
        self.refresh(bases=scope)
        g = nx.MultiDiGraph()
        g.graph["dead_suites"] = {}
        for b in self.bases:
            if b not in scope:
                continue
            contrib = self._contributions[b]
            symbol_lookup = self._build_symbol_lookup(b, scope=scope)
            _compose_contribution(
                contrib,
                target_graph=g,
                symbol_lookup=symbol_lookup,
                plugins=self._plugins,
                project_root=self._project_root,
            )
        self._closure_graphs[base] = g
        return g

    def reachable(self) -> set[SymbolNode]:
        """Set of every decl reachable from any entrypoint in the full graph."""
        return find_reachable(self.materialize_all())

    def dead(self) -> Iterator[SymbolNode]:
        """Yield every decl that no entrypoint reaches.

        Excludes ``module`` and ``synthetic`` nodes -- modules stay
        alive as long as anything they contain is alive (handled via
        the parent-module edge), and synthetic nodes are analyzer
        plumbing rather than user-visible decls.
        """
        g = self.materialize_all()
        reachable = find_reachable(g)
        for n in g.nodes:
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def _base_fingerprint(self, base: Path) -> str:
        """Per-base cache fingerprint for ``base``.

        Each base's fingerprint is independent: it covers ``base`` and
        its own search paths but not sibling bases' configs, so cache
        rows for different bases coexist without invalidating each
        other when one base's deps change.
        """
        return compute_fingerprint(
            base=base,
            search_paths=(base, *self._paths.get(base, [])),
            resolvers=self._resolvers,
            plugins=self._plugins,
            unreachable_detector=self._detector,
        )

    def _build_symbol_lookup(self, base: Path, *, scope: frozenset[Path] | None) -> SymbolTrie:
        """Per-base lookup trie: this base's full trie + each in-scope dep's exports.

        ``scope`` bounds which deps' export tries are merged in:
        ``None`` for the full-graph path (every dep), or a
        :meth:`_interesting_set` for closure-scoped materialization.
        Deps must already be refreshed (the caller is responsible for
        calling :meth:`refresh` on the right set first).
        """
        contrib = self._contributions[base]
        lookup = SymbolTrie()
        lookup.merge(contrib.current_trie)
        for dep in self._paths.get(base, []):
            if scope is not None and dep not in scope:
                continue
            dep_contrib = self._contributions.get(dep)
            if dep_contrib is None:
                continue
            lookup.merge(dep_contrib.export_trie)
        return lookup


class PackageView:
    """Lazy view of a single base inside an :class:`Analysis`.

    Cheap to construct (returned by :meth:`Analysis.package`); query
    methods trigger only the work their result depends on. Local
    queries (:meth:`modules`, :meth:`declarations`) only need this
    base's contribution. Cross-base queries (:meth:`importers_of`,
    :meth:`dead`, :meth:`graph`) materialize the
    :meth:`Analysis._interesting_set` for this base.
    """

    __slots__ = ("_analysis", "_base")

    def __init__(self, analysis: Analysis, base: Path) -> None:
        self._analysis = analysis
        self._base = base

    @property
    def base(self) -> Path:
        return self._base

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def reverse_closure(self) -> frozenset[Path]:
        """Bases that transitively depend on this one (including this base)."""
        return self._analysis.reverse_closure(self._base)

    def _contribution(self) -> _BaseContribution:
        self._analysis.refresh(bases=[self._base])
        return self._analysis._contributions[self._base]

    def modules(self) -> Iterator[SymbolNode]:
        """Module nodes for every ``.py`` file in this base.

        Local-only: refreshes this base if needed but never touches
        deps or consumers.
        """
        for n in self._contribution().base_graph.nodes:
            if n.type == "module":
                yield n

    def declarations(self, simple_name: str | None = None) -> Iterator[SymbolNode]:
        """Top-level decls in this base.

        ``simple_name=None`` yields every decl. Pass a string to
        filter to decls whose rightmost dotted segment matches it
        (``"Foo"`` matches ``pkg.mod.Foo`` but not ``pkg.Foo.bar``).
        Local-only.
        """
        for n in self._contribution().base_graph.nodes:
            if n.type in ("module", "synthetic"):
                continue
            if simple_name is not None:
                if n.fqname.rpartition(".")[2] != simple_name:
                    continue
            yield n

    def importers_of(self, target: str) -> set[Path]:
        """Files in this base whose imports reach ``target``.

        ``target`` is matched first as a first-party module fqname,
        then against the synthetic ``[external dist] / [external file]
        / [unresolved]`` markers the resolver creates for non-first-
        party imports. Triggers closure materialization (same scope
        as :meth:`graph`) because cross-base import resolution is
        what populates the predecessors used here.
        """
        g = self._analysis.materialize_closure(self._base)
        target_node: SymbolNode | None = None
        for node in g.nodes:
            if node.type == "module" and node.fqname == target:
                target_node = node
                break
        if target_node is None:
            for prefix in SYNTHETIC_PATH_PREFIXES:
                fq = f"{prefix}{target}"
                for node in g.nodes:
                    if node.type == "synthetic" and node.fqname == fq:
                        target_node = node
                        break
                if target_node is not None:
                    break
        if target_node is None:
            return set()
        return {
            pred.path
            for pred in g.predecessors(target_node)
            if pred.path != target_node.path and pred.path.is_relative_to(self._base)
        }

    def graph(self) -> nx.MultiDiGraph:
        """Materialize and return the closure-scoped graph for this base.

        See :meth:`Analysis.materialize_closure`. The graph is shared
        across queries on the same package -- repeated calls return
        the same object.
        """
        return self._analysis.materialize_closure(self._base)

    def dead(self) -> Iterator[SymbolNode]:
        """Yield decls in this base not reachable from any entrypoint
        in :meth:`reverse_closure`.

        Triggers closure materialization on first call; subsequent
        calls reuse the cached graph. Excludes ``module`` and
        ``synthetic`` nodes (see :meth:`Analysis.dead`).
        """
        g = self._analysis.materialize_closure(self._base)
        reachable = find_reachable(g)
        for n in g.nodes:
            if not n.path.is_relative_to(self._base):
                continue
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n


__all__ = [
    "Analysis",
    "PackageView",
    "build_symbol_graph",
    "count_nodes",
    "find_kept_alive_by_dead_branches",
    "find_reachable",
    "order_paths",
]
