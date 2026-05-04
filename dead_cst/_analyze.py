from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import libcst as cst
import networkx as nx
from libcst.metadata import CodeRange, FullRepoManager

from ._branches import (
    DefaultUnreachableRegionDetector,
    UnreachableRegionDetector,
)
from ._cache import GraphCache
from ._edges import resolve_edges
from ._fqn import FixedFullyQualifiedNameProvider
from ._plugins import (
    EdgePlugin,
    ObserveContext,
    PluginContext,
    apply_ops,
)
from ._plugins._core import make_payload
from ._resolvers import (
    ImportResolver,
    PathMap,
    PathResolver,
    default_resolve_import,
    exported_roots,
)
from ._resolvers._imports import safe_resolve_module, temp_sys_path
from ._symbols import EdgeFlags, Import, NodeFlags, SymbolNode, SymbolTrie
from ._visitor import SymbolVisitor, VisitorPayload

logger = logging.getLogger(__name__)


def order_paths(paths: PathMap) -> list[Path]:
    """Topologically sort base paths so dependencies are processed first.

    ``paths`` maps each base directory to the list of other base directories
    it imports from (added to ``sys.path`` while it is processed). The
    returned order ensures that when a base is processed every base it
    depends on has already contributed its symbols to the lookup tables, so
    cross-package import resolution sees them.
    """
    path_order = nx.DiGraph()
    for base, search_paths in paths.items():
        path_order.add_node(base)
        for sp in search_paths:
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

    Each ``.py`` file under each base in ``paths`` is parsed with LibCST;
    modules, classes, functions, top-level variables, and module-level imports
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
    :class:`PluginContext` whose parsed-module cache is primed with the
    modules the analyzer just walked, so plugins never re-read or re-parse.

    Parameters
    ----------
    paths:
        Mapping from base directory to its first-party search-path
        dependencies. For a single-package project, pass ``{root: []}``. For
        a monorepo, list the dependencies so they're added to ``sys.path``
        and resolved as first-party. ``order_paths`` orders the bases.
    plugins:
        Sequence of :class:`EdgePlugin` instances. Plugins emit
        :class:`AddNode`, :class:`AddEdge`, and :class:`RemoveEdge` ops;
        ``AddNode(..., entrypoint=True)`` seeds :func:`find_reachable`.
    resolvers:
        Sequence of :class:`PathResolver` instances whose
        :meth:`~PathResolver.resolve_import` overrides ``name -> path``
        lookups. Tried in order; first non-``None`` answer wins. When
        empty, the analyzer falls back to
        :func:`default_resolve_import` -- the same ``sys.path`` +
        ``importlib`` lookup the shipped resolvers all delegate to.
    project_root:
        Project root used by plugins for path-relative matching. If
        omitted, inferred as the shortest path in ``paths``.
    cache:
        Optional :class:`~dead_cst._cache.GraphCache`. When provided,
        per-file :class:`VisitorPayload` results are looked up by
        content hash and the visitor pass is skipped on cache hit.
        Plugins, edge resolution, and entrypoint seeding all run
        unconditionally on every invocation -- the cache only
        short-circuits the per-file visit. Pass ``None`` (the default)
        to bypass caching entirely.
    unreachable_detector:
        Optional :class:`~dead_cst._branches.UnreachableRegionDetector`
        whose :meth:`find_regions` is invoked once per file to compute
        the set of statically-dead suite positions. Defaults to
        :class:`~dead_cst._branches.DefaultUnreachableRegionDetector`,
        which evaluates only literal-truthiness of ``if`` / ``while``
        tests. Override to fold in domain knowledge (e.g. config flags
        whose value is fixed in production). The returned
        :class:`CodeRange` list feeds :data:`EdgeFlags.DEAD_BRANCH`
        derivation and the per-file ``graph.graph["dead_suites"]``
        reporting map.
    workers:
        When set to an integer ``>= 2``, the per-file visitor + observe
        passes for cache-miss files are dispatched to a
        :class:`~concurrent.futures.ProcessPoolExecutor`. Workers
        return :class:`VisitorPayload` objects to the main process,
        which still owns the SQLite cache write and the trie-stitching
        / edge-resolution stages. ``None`` (default) and ``1`` keep
        the serial path. A fresh pool is built per base so each
        worker's ``sys.path`` and ``FullRepoManager`` match the base
        being processed; cache-warm bases never spin one up.

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
    symbol_graph = nx.MultiDiGraph()
    symbol_graph.graph["dead_suites"] = {}
    base_tries: dict[Path, SymbolTrie] = {}
    export_tries: dict[Path, SymbolTrie] = {}
    root = project_root or _infer_project_root(paths) if paths else Path.cwd()
    import_resolver = _chain_resolvers(resolvers)
    # Materialize once and share across files so all visitors see the
    # same fingerprint-contributing instance.
    detector = (
        unreachable_detector
        if unreachable_detector is not None
        else DefaultUnreachableRegionDetector()
    )
    for base in order_paths(paths):
        logger.debug("Processing base path: %s", base)
        search_paths = [base] + paths.get(base, [])
        safe_resolve_module.cache_clear()
        with temp_sys_path(search_paths):
            base_tries[base] = current_trie = SymbolTrie()
            export_trie = SymbolTrie()
            export_roots = exported_roots(base)
            import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
            files = list(sorted(base.rglob("*.py")))
            hits: dict[Path, VisitorPayload] = {}
            miss_files: list[Path] = []
            for file in files:
                payload = cache.get(file) if cache is not None else None
                if payload is None:
                    miss_files.append(file)
                else:
                    hits[file] = payload

            miss_payloads = _compute_miss_payloads(
                miss_files=miss_files,
                files=files,
                base=base,
                search_paths=search_paths,
                project_root=root,
                import_resolver=import_resolver,
                detector=detector,
                plugins=plugins,
                resolvers=resolvers,
                cache=cache,
                workers=workers,
            )

            for file in files:
                payload = hits.get(file)
                if payload is None:
                    payload = miss_payloads[file]
                # A file's decls go into ``export_trie`` only when the file
                # lives under one of ``base``'s exported dirs (or when the
                # base has no export restriction). This is what hides
                # ``tests/`` from dependents in the typical workspace
                # layout: the member analyzes its own ``tests/`` (decls
                # land in ``current_trie`` and the graph), but consumers'
                # lookup tries never see them.
                file_exported = export_roots is None or _under_any(file, export_roots)
                _apply_payload(
                    payload,
                    current_trie=current_trie,
                    export_trie=export_trie,
                    file_exported=file_exported,
                    symbol_graph=symbol_graph,
                    import_edges=import_edges,
                )

            current_trie.add_module_hierarchy_edges(symbol_graph)
            export_tries[base] = export_trie

            # Per-consumer lookup trie: this base's full trie (everything
            # in scope when resolving its own imports) plus each dep's
            # *exported* trie (what the dep ships to consumers). Deps are
            # processed earlier by topological order, so their export
            # tries already exist.
            symbol_lookup = SymbolTrie()
            symbol_lookup.merge(current_trie)
            for dep in paths.get(base, []):
                symbol_lookup.merge(export_tries.get(dep, base_tries[dep]))

            for src, dst, flags in resolve_edges(import_edges, symbol_lookup, base):
                symbol_graph.add_edge(src, dst, flags=flags)

            # Per-base finalize pass: plugins do graph-only work here
            # (factory walks, transitive subclass closure, pyproject
            # script lookups). No CST access -- per-file CST work
            # already happened in the observe step.
            if plugins:
                ctx = PluginContext(
                    graph=symbol_graph,
                    symbol_lookup=symbol_lookup,
                    base=base,
                    project_root=root,
                )
                for plugin in plugins:
                    if not isinstance(plugin, EdgePlugin):
                        raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
                    # Materialize before applying so plugins can iterate
                    # ctx.graph.nodes without tripping "dictionary changed
                    # size during iteration".
                    ops = list(plugin.finalize(ctx))
                    apply_ops(symbol_graph, ops)

    return symbol_graph


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
    mgr: FullRepoManager,
    search_paths: list[Path],
    import_resolver: ImportResolver,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    base: Path,
    project_root: Path,
) -> VisitorPayload:
    """Run the visitor + observe pass for a single file and return its payload.

    Shared by the serial path inside :func:`build_symbol_graph` and the
    worker entry point used when ``workers >= 2``. The caller owns the
    :class:`FullRepoManager` so it can be reused across files in the
    same base (the FQN cache is precomputed per base) and so workers
    don't rebuild it on every task.
    """
    wrapper = mgr.get_metadata_wrapper_for_path(str(file))
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


@dataclass(frozen=True)
class _WorkerInit:
    """Per-base bundle of state shipped to each :class:`ProcessPoolExecutor` worker.

    Captures everything a worker needs to reproduce the parent's
    visitor environment: the base + search paths (for ``sys.path`` and
    ``FullRepoManager`` construction), the full file list (FQN gen
    cache spans every file in the base), the resolver chain (rebuilt
    via :func:`_chain_resolvers` inside the worker since the resolver
    closure isn't picklable), the detector, the plugin list, and the
    project root used by ``_run_observe``. Frozen so misuse from a
    worker can't leak state back into queued tasks.
    """

    base: Path
    search_paths: tuple[Path, ...]
    files: tuple[Path, ...]
    detector: UnreachableRegionDetector
    plugins: tuple[EdgePlugin, ...]
    resolvers: tuple[PathResolver, ...]
    project_root: Path


@dataclass
class _WorkerCtx:
    """Per-worker mutable state populated by :func:`_init_worker`.

    ``mgr`` is built lazily on the first task so workers that never
    receive one don't pay the construction cost; the first task in a
    worker pays it once and subsequent tasks reuse the same FQN cache.
    """

    init: _WorkerInit
    import_resolver: ImportResolver
    mgr: FullRepoManager | None = field(default=None)


_worker_ctx: _WorkerCtx | None = None


def _init_worker(init: _WorkerInit) -> None:
    """Pool initializer: prime ``sys.path``, the resolver chain, and worker globals.

    Adds ``init.search_paths`` to ``sys.path`` so
    :func:`default_resolve_import` finds first-party modules under the
    current base. Clears :func:`safe_resolve_module`'s LRU cache for
    the same hygiene the parent does between bases. Rebuilds the
    resolver chain locally because :func:`_chain_resolvers` returns a
    closure (not picklable across processes); the ``PathResolver``
    instances themselves are picklable dataclasses and travel via
    ``init``.
    """
    global _worker_ctx
    seen = set(sys.path)
    for p in init.search_paths:
        s = str(p)
        if s not in seen:
            sys.path.insert(0, s)
    safe_resolve_module.cache_clear()
    _worker_ctx = _WorkerCtx(
        init=init,
        import_resolver=_chain_resolvers(init.resolvers),
    )


def _worker_process_file(file: Path) -> tuple[Path, VisitorPayload]:
    """Pool task: run :func:`_process_one_file` for ``file`` against worker state.

    Returns ``(file, payload)`` so :meth:`Executor.map` results can be
    matched back to inputs without relying on submission order.
    Constructs the per-worker :class:`FullRepoManager` lazily on first
    use; subsequent tasks in the same worker reuse it.
    """
    ctx = _worker_ctx
    assert ctx is not None, "_init_worker must run before _worker_process_file"
    if ctx.mgr is None:
        ctx.mgr = FullRepoManager(
            str(ctx.init.base),
            [str(f) for f in ctx.init.files],
            {FixedFullyQualifiedNameProvider},
        )
    payload = _process_one_file(
        file,
        mgr=ctx.mgr,
        search_paths=list(ctx.init.search_paths),
        import_resolver=ctx.import_resolver,
        detector=ctx.init.detector,
        plugins=ctx.init.plugins,
        base=ctx.init.base,
        project_root=ctx.init.project_root,
    )
    return file, payload


def _compute_miss_payloads(
    *,
    miss_files: list[Path],
    files: list[Path],
    base: Path,
    search_paths: list[Path],
    project_root: Path,
    import_resolver: ImportResolver,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    resolvers: Sequence[PathResolver],
    cache: GraphCache | None,
    workers: int | None,
) -> dict[Path, VisitorPayload]:
    """Run visitor + observe for every cache-miss file in a base.

    Splits on the ``workers`` knob:

    * ``None`` / ``1`` / ``len(miss_files) < 2`` -- serial path. Builds
      one :class:`FullRepoManager` for the base and reuses it across
      every miss.
    * ``>= 2`` with at least two misses -- spawns a
      :class:`ProcessPoolExecutor` keyed to this base.
      :func:`_init_worker` ships the resolver chain + detector +
      plugins to each worker, which build their own
      ``FullRepoManager`` on first task. The pool is capped at
      ``min(workers, len(miss_files))`` so a small batch doesn't
      over-provision processes.

    Cache writes happen on the main process as each payload arrives,
    so a partial run still warms the cache for files that completed.
    Iteration order matches ``miss_files`` (sorted) for determinism.
    """
    if not miss_files:
        return {}

    use_pool = workers is not None and workers >= 2 and len(miss_files) >= 2
    miss_payloads: dict[Path, VisitorPayload] = {}

    if not use_pool:
        mgr = FullRepoManager(
            str(base),
            [str(f) for f in files],
            {FixedFullyQualifiedNameProvider},
        )
        for file in miss_files:
            payload = _process_one_file(
                file,
                mgr=mgr,
                search_paths=search_paths,
                import_resolver=import_resolver,
                detector=detector,
                plugins=plugins,
                base=base,
                project_root=project_root,
            )
            miss_payloads[file] = payload
            if cache is not None:
                cache.put(file, payload)
        return miss_payloads

    init = _WorkerInit(
        base=base,
        search_paths=tuple(search_paths),
        files=tuple(files),
        detector=detector,
        plugins=tuple(plugins),
        resolvers=tuple(resolvers),
        project_root=project_root,
    )
    assert workers is not None
    max_workers = min(workers, len(miss_files))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(init,),
    ) as pool:
        for file, payload in pool.map(_worker_process_file, miss_files):
            miss_payloads[file] = payload
            if cache is not None:
                cache.put(file, payload)
    return miss_payloads


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
    (see :func:`dead_cst._plugins.apply_ops`). There is no longer any
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
