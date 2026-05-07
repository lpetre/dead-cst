from __future__ import annotations

import logging
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
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
    make_payload,
    simple_name,
)
from .resolvers import (
    ImportResolver,
    Package,
    PathResolver,
    clear_path_caches,
)
from .resolvers._core import _validate_packages

logger = logging.getLogger(__name__)


def _bfs_order(seeds: Iterable[Path], neighbors: Mapping[Path, Sequence[Path]]) -> list[Path]:
    """BFS-reachable nodes from ``seeds``, in visit order.

    Cycle-safe via the ``visited`` set. The order is determined by
    the iteration order of ``seeds`` and of each ``neighbors[node]``
    -- pre-sort both for deterministic output.
    """
    visited: set[Path] = set()
    order: list[Path] = []
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        queue.extend(neighbors.get(node, ()))
    return order


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
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    base: Path,
    project_root: Path,
) -> VisitorPayload:
    """Run the visitor + observe pass for a single file and return its payload.

    The caller owns the precomputed FQN entry (built once per base by
    :func:`_build_stale_tasks`) so we can construct
    :class:`MetadataWrapper` directly with ``cache=`` injected,
    skipping :class:`FullRepoManager`'s per-instance ``gen_cache``
    rebuild. Same shape ``FullRepoManager.get_metadata_wrapper_for_path``
    builds, just without re-walking the file list every time.

    Cross-file import resolution moved to
    :func:`dead_cst._edges.resolve_edges`, so this pass is purely a
    function of the file's source -- no ``search_paths`` or resolver
    plumbing is needed here.
    """
    module = cst.parse_module(file.read_text())
    wrapper = MetadataWrapper(
        module,
        unsafe_skip_copy=True,
        cache={FixedFullyQualifiedNameProvider: fqn_entry},
    )
    visitor = SymbolVisitor(file, unreachable_detector=detector, wrapper=wrapper)
    wrapper.visit(visitor)
    base_payload = visitor.to_payload()
    plugin_payload = _run_observe(plugins, file, wrapper.module, base_payload, base, project_root)
    return _merge_payloads(base_payload, plugin_payload)


@dataclass(frozen=True, slots=True)
class _BaseFiles:
    """One base's ``.py`` enumeration partitioned into cache hits and misses.

    Built once per base in :func:`_enumerate_files` and parked on
    :class:`Analysis` so :meth:`Analysis.refresh` can rebuild
    contributions later without re-walking the tree.
    """

    base: Path
    files: tuple[Path, ...]
    hits: dict[Path, VisitorPayload]
    miss_files: tuple[Path, ...]


def _enumerate_files(
    base: Path,
    cache: GraphCache | None,
    fingerprint: str,
) -> _BaseFiles:
    """Walk ``base``'s ``.py`` tree, classify each file as cache hit or miss."""
    files = tuple(sorted(base.rglob("*.py")))
    hits: dict[Path, VisitorPayload] = {}
    miss_files: list[Path] = []
    for file in files:
        payload = cache.get(file, fingerprint) if cache is not None else None
        if payload is None:
            miss_files.append(file)
        else:
            hits[file] = payload
    return _BaseFiles(
        base=base,
        files=files,
        hits=hits,
        miss_files=tuple(miss_files),
    )


@dataclass(frozen=True, slots=True)
class _StaleFile:
    """One stale file ready for the visitor + observe pass.

    ``fqn_entry`` is this file's slice of the per-base FQN cache
    (FQN resolution is base-keyed, hence one ``gen_cache`` call per
    base in :func:`_build_stale_tasks`); the runner injects it into a
    :class:`MetadataWrapper` directly.
    """

    file: Path
    base: Path
    fqn_entry: ModuleNameAndPackage
    project_root: Path


def _rebind_sys_path(search_paths: tuple[Path, ...], baseline: list[str]) -> None:
    """Set ``sys.path`` to ``search_paths + baseline``, deduplicated.

    Used by :meth:`Analysis._materialize` so the resolver in
    :func:`_compose_contribution` -> :func:`resolve_edges` sees this
    base's first-party prefix while classifying trie-miss imports.
    """
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


def _process_task(
    detector: UnreachableRegionDetector,
    plugins: tuple[EdgePlugin, ...],
    task: _StaleFile,
) -> tuple[Path, VisitorPayload]:
    """Run one task; pure (no ``sys.path`` mutation, no resolver call)."""
    payload = _process_one_file(
        task.file,
        fqn_entry=task.fqn_entry,
        detector=detector,
        plugins=plugins,
        base=task.base,
        project_root=task.project_root,
    )
    return task.file, payload


_worker_state: tuple[UnreachableRegionDetector, tuple[EdgePlugin, ...]] | None = None


def _init_worker(
    detector: UnreachableRegionDetector,
    plugins: tuple[EdgePlugin, ...],
) -> None:
    """Pool initializer: stash the worker's detector + plugins."""
    global _worker_state
    _worker_state = (detector, plugins)


def _worker_process_task(task: _StaleFile) -> tuple[Path, VisitorPayload]:
    """Pool task: delegate to :func:`_process_task` against the worker's state."""
    assert _worker_state is not None, "_init_worker must run before _worker_process_task"
    return _process_task(*_worker_state, task)


def _build_stale_tasks(
    base_files: Mapping[Path, _BaseFiles],
    project_root: Path,
) -> list[_StaleFile]:
    """Flatten every base's miss files into one global, deterministic task list.

    One ``gen_cache`` call per base (FQN resolution is base-keyed)
    populates each task's ``fqn_entry``. Sorting on ``(base, file)``
    keeps related tasks together for log readability and makes
    parallel-pool output ordering reproducible.
    """
    tasks: list[_StaleFile] = []
    for base in sorted(base_files):
        bf = base_files[base]
        if not bf.miss_files:
            continue
        fqn_cache = FixedFullyQualifiedNameProvider.gen_cache(
            base, [str(f) for f in bf.miss_files], timeout=5
        )
        tasks.extend(
            _StaleFile(
                file=file,
                base=base,
                fqn_entry=fqn_cache[str(file)],
                project_root=project_root,
            )
            for file in bf.miss_files
        )
    return tasks


def _process_stale_files(
    *,
    tasks: Sequence[_StaleFile],
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    cache: GraphCache | None,
    fingerprint: str,
    workers: int | None,
) -> dict[Path, VisitorPayload]:
    """Run visitor + observe across every task; return ``file -> payload``.

    Both branches use :func:`_process_task` for the per-task work;
    they differ only in whether the runner state lives on the main
    process or in :class:`ProcessPoolExecutor` workers. The pool is
    opt-in (``workers >= 2`` and at least two tasks); below that, the
    in-process path avoids pool startup cost.

    Cache writes happen on the main process as each payload arrives,
    so a partial run still warms the cache for files that completed.
    """
    if not tasks:
        return {}

    out: dict[Path, VisitorPayload] = {}
    use_pool = workers is not None and workers >= 2 and len(tasks) >= 2

    def _record(file: Path, payload: VisitorPayload) -> None:
        out[file] = payload
        if cache is not None:
            cache.put(file, payload, fingerprint)

    if use_pool:
        assert workers is not None
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_init_worker,
            initargs=(detector, tuple(plugins)),
        ) as pool:
            for file, payload in pool.map(_worker_process_task, tasks):
                _record(file, payload)
        return out

    plugins_t = tuple(plugins)
    for task in tasks:
        file, payload = _process_task(detector, plugins_t, task)
        _record(file, payload)
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

    Built once per base by :func:`_build_contribution` and composed
    into a target graph by :func:`_compose_contribution`, which adds
    cross-base edges via :func:`resolve_edges` and runs plugin
    :meth:`EdgePlugin.finalize` against the composed graph.

    ``base_graph.graph["dead_suites"]`` carries this base's per-file
    dead-suite positions; the compose step folds them into the target
    graph's matching key.
    """

    base: Path
    current_trie: SymbolTrie
    export_trie: SymbolTrie
    base_graph: nx.MultiDiGraph
    import_edges: frozenset[tuple[SymbolNode, Import, EdgeFlags]]


def _build_contribution(
    package: Package,
    base_files: _BaseFiles,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> _BaseContribution:
    """Apply ``base_files``' per-file payloads into a base-local graph slice.

    Hits come straight from :class:`_BaseFiles`; the rest are looked
    up in the global ``miss_payloads`` map produced by
    :func:`_process_stale_files`. The base-local ``nx.MultiDiGraph``
    is what makes scope-bounded materialization cheap: composing it
    into the full graph or a closure graph doesn't redo per-file
    apply work. Empty :attr:`Package.exported` means "no restriction"
    (every file in the base is exported to consumers).
    """
    current_trie = SymbolTrie()
    export_trie = SymbolTrie()
    exported = package.exported
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    base_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    base_graph.graph["dead_suites"] = {}
    for file in base_files.files:
        payload = base_files.hits.get(file)
        if payload is None:
            payload = miss_payloads[file]
        file_exported = not exported or _under_any(file, list(exported))
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
        base=package.path,
        current_trie=current_trie,
        export_trie=export_trie,
        base_graph=base_graph,
        import_edges=frozenset(import_edges),
    )


def _compose_contribution(
    contrib: _BaseContribution,
    *,
    target_graph: nx.MultiDiGraph,
    symbol_lookup: SymbolTrie,
    plugins: Sequence[EdgePlugin],
    project_root: Path,
    import_resolver: ImportResolver,
    search_paths: list[Path],
) -> None:
    """Merge ``contrib.base_graph`` into ``target_graph``, stitch
    cross-base imports against ``symbol_lookup``, and run plugin
    :meth:`EdgePlugin.finalize` against the composed graph.

    The caller owns ``symbol_lookup`` because its construction depends
    on which dep export tries are in scope -- the full-graph path
    merges every dep's exports, while the closure-scoped path merges
    only deps inside the requested scope. ``import_resolver`` +
    ``search_paths`` reach :func:`resolve_edges` for the trie-miss
    classification path (stdlib / external dist / external file /
    unresolved); they are unused when every import resolves
    first-party in the trie.
    """
    target_graph.update(
        edges=contrib.base_graph.edges(data=True, keys=True),
        nodes=contrib.base_graph.nodes(data=True),
    )
    target_graph.graph.setdefault("dead_suites", {}).update(contrib.base_graph.graph["dead_suites"])
    for src, dst, flags in resolve_edges(
        contrib.import_edges,
        symbol_lookup,
        contrib.base,
        import_resolver=import_resolver,
        search_paths=search_paths,
    ):
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


def _find_reachable(graph: nx.MultiDiGraph) -> set[SymbolNode]:
    """BFS forward from every node tagged as an entrypoint by a plugin.

    Plugins mark seeds by setting ``graph.nodes[node]["entrypoint"] = True``
    (see :func:`dead_cst.plugins.apply_ops`).

    Edges flagged with :data:`EdgeFlags.DEAD_BRANCH` are NOT filtered
    here -- today's behavior, where dead-code references propagate
    liveness through the enclosing decl, is preserved. See
    :func:`_find_reachable_strict` for the variant that skips them.
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
    """Like :func:`_find_reachable` but skips ``DEAD_BRANCH``-flagged edges."""
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


def _find_kept_alive_by_dead_branches(graph: nx.MultiDiGraph) -> set[SymbolNode]:
    """Symbols kept alive only via at least one ``DEAD_BRANCH`` edge.

    ``_find_reachable(graph) -`` strict-mode BFS that skips every edge
    flagged :data:`EdgeFlags.DEAD_BRANCH`; the difference is the
    "blast radius" of removing every statically-dead suite. Surfaced
    on :class:`Analysis` as :meth:`Analysis.kept_alive_by_dead_branches`
    and on :class:`PackageView` as
    :meth:`PackageView.kept_alive_by_dead_branches`.
    """
    return _find_reachable(graph) - _find_reachable_strict(graph)


def _count_nodes(graph: nx.MultiDiGraph, prefix: Path | None) -> dict[str, int]:
    """Count nodes in ``graph`` by ``SymbolNode.type``, optionally restricted by path.

    If ``prefix`` is given, only nodes whose ``path`` is under ``prefix``
    are counted. Includes the synthetic ``"synthetic"`` type contributed
    by plugins and third-party-dep markers.
    """
    counts: dict[str, int] = {}
    for node in graph.nodes:
        if prefix and not node.path.is_relative_to(prefix):
            continue
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts


class Analysis:
    """Lazy entrypoint to the dead-cst pipeline.

    Holds the analyzer's config (paths, plugins, resolver, cache,
    detector, worker count) and memoizes per-base work so multiple
    queries against the same project share the cost. Construction
    runs the resolver's :meth:`PathResolver.resolve` once to build
    the path map, but no source files are read or parsed until you
    ask -- the visitor pass is gated on :meth:`refresh` /
    :meth:`materialize_all`.

    Three coarse stages happen on demand:

    1. **File enumeration + visitor pass** -- driven by
       :meth:`refresh`. Walks each requested base's files, hashes
       them against the cache, then flattens every base's misses
       into one global stale-file list and runs the visitor +
       observe pass once across all of them (parallel when
       ``workers`` permits). Idempotent and scoped:
       ``refresh(bases=[B])`` walks only ``B``'s file tree.

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

    The configured ``resolver`` is queried twice: once at construction
    time to build the per-base :class:`Package` list (calling
    :meth:`PathResolver.resolve` with ``project_root`` and validating
    the result), and again during edge stitching to classify trie-miss
    imports via :meth:`PathResolver.resolve_import`. Once constructed,
    an :class:`Analysis` is effectively read-only -- spin up a fresh
    instance to pick up a new resolver or new plugins.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        resolver: PathResolver,
        plugins: Sequence[EdgePlugin] = (),
        cache: GraphCache | None = None,
        unreachable_detector: UnreachableRegionDetector | None = None,
        workers: int | None = None,
    ) -> None:
        self._project_root: Path = project_root
        self._packages: tuple[Package, ...] = _validate_packages(resolver.resolve(project_root))
        self._packages_by_path: dict[Path, Package] = {p.path: p for p in self._packages}
        by_name = {p.name: p.path for p in self._packages}
        self._dep_paths_by_base: dict[Path, tuple[Path, ...]] = {
            p.path: tuple(by_name[d] for d in p.deps) for p in self._packages
        }
        # Reverse map: base -> bases that name this one in their `deps`.
        # Pre-sorted by path so consumer-side BFS (used by
        # `reverse_closure` and `bases`) yields a deterministic order.
        consumers: dict[Path, list[Path]] = {p.path: [] for p in self._packages}
        for p in self._packages:
            for dep_name in p.deps:
                consumers[by_name[dep_name]].append(p.path)
        self._consumers_by_base: dict[Path, tuple[Path, ...]] = {
            base: tuple(sorted(cs)) for base, cs in consumers.items()
        }
        self._plugins: tuple[EdgePlugin, ...] = tuple(plugins)
        self._cache = cache
        self._workers = workers
        self._import_resolver: ImportResolver = resolver.resolve_import
        self._detector: UnreachableRegionDetector = (
            unreachable_detector
            if unreachable_detector is not None
            else DefaultUnreachableRegionDetector()
        )
        # One fingerprint per analysis -- the visitor's output is
        # purely a function of the file's source plus the plugin /
        # detector chain, so every base shares the same key.
        self._fingerprint: str = compute_fingerprint(
            plugins=self._plugins,
            unreachable_detector=self._detector,
        )
        self._base_files: dict[Path, _BaseFiles] = {}
        self._contributions: dict[Path, _BaseContribution] = {}
        self._closure_graphs: dict[Path, nx.MultiDiGraph] = {}
        self._full_graph: nx.MultiDiGraph | None = None
        # Memoize closure-walk results -- the package graph is
        # immutable post-construction.
        self._reverse_closures: dict[Path, frozenset[Path]] = {}
        self._interesting_sets: dict[Path, frozenset[Path]] = {}

    @property
    def packages(self) -> tuple[Package, ...]:
        """The :class:`Package` list this analysis was built with."""
        return self._packages

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _dep_paths(self, base: Path) -> tuple[Path, ...]:
        """Precomputed dep paths for the package at ``base``."""
        return self._dep_paths_by_base.get(base, ())

    @cached_property
    def bases(self) -> list[Path]:
        """Deterministic, cycle-tolerant package order.

        BFS forward through :attr:`_consumers_by_base` from packages
        with no deps, so dependencies precede their dependents
        whenever the graph is acyclic. Cycle-trapped packages (none
        of which appear as no-dep seeds) are appended at the end in
        path order. ``_consumers_by_base`` is pre-sorted at
        construction so the BFS visit order is fully determined.
        """
        sorted_paths = sorted(self._packages_by_path)
        seeds = [p for p in sorted_paths if not self._dep_paths_by_base.get(p)]
        order = _bfs_order(seeds, self._consumers_by_base)
        visited = set(order)
        order.extend(p for p in sorted_paths if p not in visited)
        return order

    def reverse_closure(self, base: Path) -> frozenset[Path]:
        """Bases that transitively depend on ``base`` (including ``base`` itself).

        These are the only bases whose source code could import (and
        therefore keep alive) decls under ``base``. A query like "is X
        in ``base`` dead?" only needs entrypoints from this set --
        sibling bases that can't reach ``base`` through the dep graph
        can't reference its decls. Cycle-safe: BFS terminates even
        when ``Package.deps`` cycles.
        """
        if base not in self._packages_by_path:
            raise KeyError(base)
        cached = self._reverse_closures.get(base)
        if cached is None:
            cached = frozenset(_bfs_order([base], self._consumers_by_base))
            self._reverse_closures[base] = cached
        return cached

    def _interesting_set(self, base: Path) -> frozenset[Path]:
        """Bases needed to answer reachability queries about decls in ``base``.

        :meth:`reverse_closure` ∪ each consumer's transitive deps, so
        we have enough trie data to resolve every cross-base import
        that could lead into ``base``.
        """
        cached = self._interesting_sets.get(base)
        if cached is None:
            cached = frozenset(_bfs_order(self.reverse_closure(base), self._dep_paths_by_base))
            self._interesting_sets[base] = cached
        return cached

    def refresh(self, bases: Iterable[Path] | None = None) -> Analysis:
        """Update the cache and build per-base contributions for the given bases.

        ``bases=None`` (the default) refreshes every base. Passing a
        subset scopes the file walk + visitor pass to those bases
        only -- sibling bases are untouched. Already-refreshed bases
        are skipped, so calling :meth:`refresh` twice with the same
        argument is cheap.

        Three steps run in order: (1) walk each new target's tree and
        partition into cache hits / misses, (2) flatten every base's
        misses into a single global stale-file list and run the
        visitor + observe pass on each one (parallel when ``workers``
        permits), (3) apply each base's payloads into a local
        contribution. Step 2 ignores which base each file lives
        under, so a refresh that touches several bases pays for one
        worker pool startup, not one per base.

        Returns ``self`` so callers can chain
        ``Analysis(...).refresh().materialize_all()``.
        """
        targets = list(bases) if bases is not None else self.bases
        unknown = [b for b in targets if b not in self._packages_by_path]
        if unknown:
            raise KeyError(f"Unknown bases: {unknown}")
        new_targets = [b for b in targets if b not in self._contributions]
        if not new_targets:
            return self

        for b in new_targets:
            if b not in self._base_files:
                self._base_files[b] = _enumerate_files(b, self._cache, self._fingerprint)

        pending = {b: self._base_files[b] for b in new_targets}
        tasks = _build_stale_tasks(pending, self._project_root)
        miss_payloads = _process_stale_files(
            tasks=tasks,
            detector=self._detector,
            plugins=self._plugins,
            cache=self._cache,
            fingerprint=self._fingerprint,
            workers=self._workers,
        )

        for b in new_targets:
            self._contributions[b] = _build_contribution(
                self._packages_by_path[b],
                self._base_files[b],
                miss_payloads,
            )
        return self

    def package(self, base: Path) -> PackageView:
        """Return a lazy view onto a single base.

        The returned :class:`PackageView` is cheap; per-base work is
        triggered by its query methods.
        """
        if base not in self._packages_by_path:
            raise KeyError(base)
        return PackageView(self, base)

    def views(self) -> Iterator[PackageView]:
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
        if self._full_graph is None:
            self.refresh()
            self._full_graph = self._materialize(scope=None)
        return self._full_graph

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
        if base not in self._closure_graphs:
            scope = self._interesting_set(base)
            self.refresh(bases=scope)
            self._closure_graphs[base] = self._materialize(scope=scope)
        return self._closure_graphs[base]

    def _materialize(self, *, scope: frozenset[Path] | None) -> nx.MultiDiGraph:
        """Compose every refreshed base in ``scope`` into a fresh graph.

        ``scope=None`` composes every base. Caller is responsible for
        having :meth:`refresh`'d every base in ``scope`` first.

        Cross-file import resolution runs here (in
        :func:`_compose_contribution` -> :func:`resolve_edges`), which
        is where the resolver reads ``sys.path`` /
        :mod:`importlib.metadata`. We rebind ``sys.path`` to each
        base's ``(base, *deps)`` view before composing it and clear
        the resolver LRUs at every transition, restoring the original
        ``sys.path`` on the way out so library callers don't see
        lingering mutations.
        """
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.graph["dead_suites"] = {}
        baseline = list(sys.path)
        last_search_paths: tuple[Path, ...] | None = None
        try:
            for base in self.bases:
                if scope is not None and base not in scope:
                    continue
                search_paths = (base, *self._dep_paths(base))
                if last_search_paths != search_paths:
                    _rebind_sys_path(search_paths, baseline)
                    clear_path_caches()
                    last_search_paths = search_paths
                _compose_contribution(
                    self._contributions[base],
                    target_graph=g,
                    symbol_lookup=self._build_symbol_lookup(base, scope=scope),
                    plugins=self._plugins,
                    project_root=self._project_root,
                    import_resolver=self._import_resolver,
                    search_paths=list(search_paths),
                )
        finally:
            sys.path[:] = baseline
        return g

    def reachable(self) -> set[SymbolNode]:
        """Set of every decl reachable from any entrypoint in the full graph."""
        return _find_reachable(self.materialize_all())

    def dead(self) -> Iterator[SymbolNode]:
        """Yield every decl that no entrypoint reaches.

        Excludes ``module`` and ``synthetic`` nodes -- modules stay
        alive as long as anything they contain is alive (handled via
        the parent-module edge), and synthetic nodes are analyzer
        plumbing rather than user-visible decls.
        """
        g = self.materialize_all()
        reachable = _find_reachable(g)
        for n in g.nodes:
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        """Symbols that would become unreachable if every dead suite were removed.

        Computed as ``reachable() -`` strict-mode BFS that skips every
        edge flagged :data:`EdgeFlags.DEAD_BRANCH`. The resulting set
        is the "blast radius" of removing every statically-dead suite
        in the analyzed source -- symbols currently kept alive only
        through a chain that crosses at least one dead-branch
        reference.

        Used by tooling that reports "if you removed your unreachable
        code, these additional symbols would also become dead." The
        default :meth:`reachable` traversal is unchanged; this is the
        opt-in stricter pass.
        """
        g = self.materialize_all()
        return _find_reachable(g) - _find_reachable_strict(g)

    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]:
        """Count nodes in the full graph by ``SymbolNode.type``.

        ``prefix=None`` (the default) counts every node. Pass a base
        path to scope the count to nodes whose ``path`` is under that
        prefix -- useful for per-base summaries when several bases are
        analysed together.
        """
        return _count_nodes(self.materialize_all(), prefix)

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
        for dep in self._dep_paths(base):
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

    def declarations(self, name: str | None = None) -> Iterator[SymbolNode]:
        """Top-level decls in this base.

        ``name=None`` yields every decl. Pass a string to filter to
        decls whose rightmost dotted segment matches it (``"Foo"``
        matches ``pkg.mod.Foo`` but not ``pkg.Foo.bar``). Local-only.
        """
        for n in self._contribution().base_graph.nodes:
            if n.type in ("module", "synthetic"):
                continue
            if name is not None and simple_name(n.fqname) != name:
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
        scope = self._analysis._interesting_set(self._base)
        ctx = PluginContext(
            graph=self._analysis.materialize_closure(self._base),
            symbol_lookup=self._analysis._build_symbol_lookup(self._base, scope=scope),
            base=self._base,
            project_root=self._analysis.project_root,
        )
        return ctx.importers(target)

    def graph(self) -> nx.MultiDiGraph:
        """Materialize and return the closure-scoped graph for this base.

        See :meth:`Analysis.materialize_closure`. The graph is shared
        across queries on the same package -- repeated calls return
        the same object.
        """
        return self._analysis.materialize_closure(self._base)

    def reachable(self) -> set[SymbolNode]:
        """Set of decls in this base reachable from any entrypoint in
        :meth:`reverse_closure`.

        Triggers closure materialization on first call; subsequent
        calls reuse the cached graph. Filtered to nodes whose ``path``
        is under :attr:`base`, so the result is comparable to
        :meth:`declarations` for "what's alive in this package?"
        questions.
        """
        g = self._analysis.materialize_closure(self._base)
        return {n for n in _find_reachable(g) if n.path.is_relative_to(self._base)}

    def dead(self) -> Iterator[SymbolNode]:
        """Yield decls in this base not reachable from any entrypoint
        in :meth:`reverse_closure`.

        Triggers closure materialization on first call; subsequent
        calls reuse the cached graph. Excludes ``module`` and
        ``synthetic`` nodes (see :meth:`Analysis.dead`).
        """
        g = self._analysis.materialize_closure(self._base)
        reachable = _find_reachable(g)
        for n in g.nodes:
            if not n.path.is_relative_to(self._base):
                continue
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        """Decls in this base kept alive only by dead-branch references.

        Closure-scoped equivalent of :meth:`Analysis.kept_alive_by_dead_branches`,
        filtered to nodes under :attr:`base`.
        """
        g = self._analysis.materialize_closure(self._base)
        diff = _find_reachable(g) - _find_reachable_strict(g)
        return {n for n in diff if n.path.is_relative_to(self._base)}

    def count_nodes(self) -> dict[str, int]:
        """Count nodes contributed by this base, by ``SymbolNode.type``.

        Local-only: doesn't materialize the closure. Counts include
        ``module``, source decls, and any ``synthetic`` nodes plugins
        emitted into this base's contribution during ``observe``.
        """
        return _count_nodes(self._contribution().base_graph, prefix=None)

    def remove_dead_code(self) -> None:
        """Apply the LibCST codemod, deleting every dead decl in this base.

        Materializes the closure, computes reachability, and feeds the
        unreachable subgraph (filtered to this base) to
        :func:`dead_cst.codemod.remove_code`. The transformation is
        destructive -- back the files up first, or run on a clean
        working tree.
        """
        from .codemod import remove_code

        g = self._analysis.materialize_closure(self._base)
        reachable = _find_reachable(g)
        dead_nodes = [n for n in g.nodes if n not in reachable]
        remove_code(g.subgraph(dead_nodes), self._base)


__all__ = [
    "Analysis",
    "PackageView",
]
