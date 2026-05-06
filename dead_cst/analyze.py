from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, cast

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
    PathResolver,
    SourceTree,
    SourceTreeFlags,
    assign_file_to_tree,
    default_resolve_import,
    distribution_lookup,
    editable_distribution_roots,
    safe_resolve_module,
    validate_source_trees,
)
from .resolvers._core import _ValidatedTrees

logger = logging.getLogger(__name__)


def _build_tree_dag(validated: _ValidatedTrees) -> nx.DiGraph:
    """``dep -> consumer`` DAG over :class:`SourceTree` paths.

    Edges go ``S -> T`` where ``S`` is in ``T.search_trees`` (deps
    flow into consumers). Topological order over this DAG drives
    processing -- a consumer's ``symbol_lookup`` requires every dep's
    contribution to be ready before edges are stitched.
    """
    g: nx.DiGraph = nx.DiGraph()
    for path, tree in validated.by_path.items():
        g.add_node(path)
        for ref in tree.search_trees:
            if ref in validated.by_path:
                g.add_edge(ref, path)
    return g


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

    The caller owns the precomputed FQN entry (built once per tree by
    :func:`_collect_tree_specs`) so we can construct
    :class:`MetadataWrapper` directly with ``cache=`` injected,
    skipping :class:`FullRepoManager`'s per-instance ``gen_cache``
    rebuild.
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
class _TreeSpec:
    """Per-:class:`SourceTree` file list partitioned into hits and miss files.

    Built once up front by :func:`_build_tree_spec` so the parallel
    path can flatten every tree's miss files into a single sorted task
    list. ``files`` covers every ``.py`` file longest-prefix-matched to
    this tree (so files in nested trees are routed there instead, not
    duplicated here). ``fqn_cache`` covers only the miss files, since
    hit files never go through the visitor.
    """

    tree: SourceTree
    search_paths: tuple[Path, ...]
    files: tuple[Path, ...]
    hits: dict[Path, VisitorPayload]
    miss_files: tuple[Path, ...]
    fqn_cache: Mapping[str, ModuleNameAndPackage]
    fingerprint: str


def _build_tree_spec(
    tree: SourceTree,
    all_trees: Sequence[SourceTree],
    cache: GraphCache | None,
    fingerprint: str,
) -> _TreeSpec:
    """Build a single :class:`_TreeSpec` for ``tree``.

    Walks ``tree.path`` for ``.py`` files, then routes each file to
    its longest-prefix-matching tree and keeps only the ones owned by
    ``tree``. That filter is what lets a parent tree like
    ``pkg/`` coexist with a nested tree like ``pkg/tests/`` without
    double-walking files.

    ``search_paths = (tree.path, *tree.search_trees)`` -- the tree's
    own path is always its own first-party search root, with declared
    search refs appended.
    """
    search_paths = (tree.path, *tree.search_trees)
    own_files: list[Path] = []
    for candidate in sorted(tree.path.rglob("*.py")):
        owner = assign_file_to_tree(candidate, all_trees)
        if owner is not None and owner.path == tree.path:
            own_files.append(candidate)
    files = tuple(own_files)

    hits: dict[Path, VisitorPayload] = {}
    miss_files: list[Path] = []
    for file in files:
        payload = cache.get(file, fingerprint) if cache is not None else None
        if payload is None:
            miss_files.append(file)
        else:
            hits[file] = payload
    fqn_cache: Mapping[str, ModuleNameAndPackage] = (
        FixedFullyQualifiedNameProvider.gen_cache(
            tree.path, [str(f) for f in miss_files], timeout=5
        )
        if miss_files
        else {}
    )
    return _TreeSpec(
        tree=tree,
        search_paths=search_paths,
        files=files,
        hits=hits,
        miss_files=tuple(miss_files),
        fqn_cache=fqn_cache,
        fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _Task:
    """Per-file unit of work for the visitor + observe pass."""

    file: Path
    tree_path: Path
    search_paths: tuple[Path, ...]
    fqn_entry: ModuleNameAndPackage
    project_root: Path


@dataclass(slots=True)
class _RunnerState:
    """Mutable state for the per-task runner, shared by serial and worker paths."""

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

    :func:`safe_resolve_module` keys on fullname and reads live
    ``sys.path``; clearing avoids stale results across trees with
    different first-party prefixes.
    """
    safe_resolve_module.cache_clear()
    distribution_lookup.cache_clear()
    editable_distribution_roots.cache_clear()


def _process_task(state: _RunnerState, task: _Task) -> tuple[Path, Path, VisitorPayload]:
    """Run one task, transitioning ``sys.path`` + caches if the tree changed."""
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
        base=task.tree_path,
        project_root=task.project_root,
    )
    return task.tree_path, task.file, payload


_worker_state: _RunnerState | None = None


def _init_worker(
    detector: UnreachableRegionDetector,
    plugins: tuple[EdgePlugin, ...],
    resolvers: tuple[PathResolver, ...],
) -> None:
    global _worker_state
    _worker_state = _RunnerState(
        detector=detector,
        plugins=plugins,
        import_resolver=_chain_resolvers(resolvers),
        sys_path_baseline=list(sys.path),
    )


def _worker_process_task(task: _Task) -> tuple[Path, Path, VisitorPayload]:
    assert _worker_state is not None, "_init_worker must run before _worker_process_task"
    return _process_task(_worker_state, task)


def _build_sorted_tasks(tree_specs: dict[Path, _TreeSpec], project_root: Path) -> list[_Task]:
    """Flatten every tree's miss files into one sorted task list.

    Sorting by ``search_paths`` puts same-tree tasks adjacent so a
    runner sees at most one ``sys.path`` transition per tree it
    touches.
    """
    tasks: list[_Task] = [
        _Task(
            file=file,
            tree_path=tree_path,
            search_paths=spec.search_paths,
            fqn_entry=spec.fqn_cache[str(file)],
            project_root=project_root,
        )
        for tree_path, spec in tree_specs.items()
        for file in spec.miss_files
    ]
    tasks.sort(key=lambda t: (t.search_paths, t.file))
    return tasks


def _compute_all_miss_payloads(
    *,
    tree_specs: dict[Path, _TreeSpec],
    project_root: Path,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    resolvers: Sequence[PathResolver],
    import_resolver: ImportResolver,
    cache: GraphCache | None,
    workers: int | None,
) -> dict[Path, dict[Path, VisitorPayload]]:
    """Run visitor + observe for every cache-miss file across every tree."""
    out: dict[Path, dict[Path, VisitorPayload]] = {p: {} for p in tree_specs}
    total_misses = sum(len(s.miss_files) for s in tree_specs.values())
    if total_misses == 0:
        return out

    tasks = _build_sorted_tasks(tree_specs, project_root)
    use_pool = workers is not None and workers >= 2 and total_misses >= 2

    def _record(tree_path: Path, file: Path, payload: VisitorPayload) -> None:
        out[tree_path][file] = payload
        if cache is not None:
            cache.put(file, payload, tree_specs[tree_path].fingerprint)

    if use_pool:
        assert workers is not None
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_init_worker,
            initargs=(detector, tuple(plugins), tuple(resolvers)),
        ) as pool:
            for tree_path, file, payload in pool.map(_worker_process_task, tasks):
                _record(tree_path, file, payload)
        return out

    baseline = list(sys.path)
    state = _RunnerState(
        detector=detector,
        plugins=tuple(plugins),
        import_resolver=import_resolver,
        sys_path_baseline=baseline,
    )
    try:
        for task in tasks:
            tree_path, file, payload = _process_task(state, task)
            _record(tree_path, file, payload)
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
    s_start = (suite.start.line, suite.start.column)
    s_end = (suite.end.line, suite.end.column)
    a_start = (access.start.line, access.start.column)
    a_end = (access.end.line, access.end.column)
    return s_start <= a_start and a_end <= s_end


def _apply_payload(
    payload: VisitorPayload,
    *,
    current_trie: SymbolTrie,
    export_trie: SymbolTrie | None,
    symbol_graph: nx.MultiDiGraph,
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
) -> None:
    """Emit ``payload`` into the in-progress per-tree structures.

    ``export_trie`` is ``None`` for non-exported trees; the apply
    step still populates ``current_trie`` (so within-tree resolution
    works) and the graph, but no decls flow into the package's
    consumer-visible export trie.
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
            if export_trie is not None:
                export_trie.add_declaration(n)

    for src, dst, pos in payload.edges:
        symbol_graph.add_edge(src, dst, flags=flag_for(pos))

    for src, imp, pos in payload.imports:
        import_edges.add((src, imp, flag_for(pos)))

    if payload.dead_suites:
        symbol_graph.graph["dead_suites"][module.path] = payload.dead_suites


@dataclass(slots=True)
class _TreeContribution:
    """One tree's pre-stitched contribution to the symbol graph."""

    tree: SourceTree
    current_trie: SymbolTrie
    export_trie: SymbolTrie  # empty trie when the tree isn't EXPORTED
    base_graph: nx.MultiDiGraph
    import_edges: frozenset[tuple[SymbolNode, Import, EdgeFlags]]


def _build_contribution(
    tree: SourceTree,
    spec: _TreeSpec,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> _TreeContribution:
    """Apply ``spec``'s per-file payloads into a tree-local graph slice."""
    current_trie = SymbolTrie()
    export_trie = SymbolTrie()
    is_exported = bool(tree.flags & SourceTreeFlags.EXPORTED)
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    base_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    base_graph.graph["dead_suites"] = {}
    for file in spec.files:
        payload = spec.hits.get(file)
        if payload is None:
            payload = miss_payloads[file]
        _apply_payload(
            payload,
            current_trie=current_trie,
            export_trie=export_trie if is_exported else None,
            symbol_graph=base_graph,
            import_edges=import_edges,
        )
    current_trie.add_module_hierarchy_edges(base_graph)
    return _TreeContribution(
        tree=tree,
        current_trie=current_trie,
        export_trie=export_trie,
        base_graph=base_graph,
        import_edges=frozenset(import_edges),
    )


def _compose_contribution(
    contrib: _TreeContribution,
    *,
    target_graph: nx.MultiDiGraph,
    symbol_lookup: SymbolTrie,
    plugins: Sequence[EdgePlugin],
    project_root: Path,
) -> None:
    """Merge ``contrib.base_graph`` into ``target_graph``, stitch
    cross-tree imports against ``symbol_lookup``, and run plugin
    :meth:`EdgePlugin.finalize` against the composed graph.
    """
    target_graph.update(
        edges=contrib.base_graph.edges(data=True, keys=True),
        nodes=contrib.base_graph.nodes(data=True),
    )
    target_graph.graph.setdefault("dead_suites", {}).update(contrib.base_graph.graph["dead_suites"])
    for src, dst, flags in resolve_edges(contrib.import_edges, symbol_lookup, contrib.tree.path):
        target_graph.add_edge(src, dst, flags=flags)
    if plugins:
        ctx = PluginContext(
            graph=target_graph,
            symbol_lookup=symbol_lookup,
            base=contrib.tree.path,
            project_root=project_root,
        )
        for plugin in plugins:
            if not isinstance(plugin, EdgePlugin):
                raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
            ops = list(plugin.finalize(ctx))
            apply_ops(target_graph, ops)


def _infer_project_root(trees: Sequence[SourceTree]) -> Path:
    if not trees:
        return Path.cwd()
    return min((t.path for t in trees), key=lambda p: len(p.parts))


def _trees_from_mapping(paths: Mapping[Path, Sequence[Path]]) -> list[SourceTree]:
    """Convert a ``{path: [search_paths]}`` shorthand into a tree list.

    Each path becomes its own ``EXPORTED`` tree. Package names are
    derived from the final path component (with an index suffix when
    two paths share a name) so they're unique across the resulting
    list. Search-path entries are passed through as ``search_trees``;
    paths that aren't keys are simply absent from the resolver's
    layout, which validation will reject -- the shorthand is a
    convenience for the homogeneous case where every search ref is
    another tree in the same mapping.
    """
    used: dict[str, int] = {}

    def _pkg(path: Path) -> str:
        stem = path.name or "root"
        if stem in used:
            used[stem] += 1
            return f"{stem}_{used[stem]}"
        used[stem] = 0
        return stem

    pkg_for: dict[Path, str] = {}
    for p in paths:
        pkg_for[p.resolve()] = _pkg(p)

    out: list[SourceTree] = []
    for p, deps in paths.items():
        out.append(
            SourceTree(
                path=p.resolve(),
                package=pkg_for[p.resolve()],
                flags=SourceTreeFlags.EXPORTED,
                search_trees=tuple(d.resolve() for d in deps),
            )
        )
    return out


def _find_reachable(graph: nx.MultiDiGraph) -> set[SymbolNode]:
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
    return _find_reachable(graph) - _find_reachable_strict(graph)


def _count_nodes(graph: nx.MultiDiGraph, prefix: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in graph.nodes:
        if prefix and not node.path.is_relative_to(prefix):
            continue
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts


class Analysis:
    """Lazy entrypoint to the dead-cst pipeline.

    Holds the analyzer's config (source trees, plugins, resolvers,
    cache, detector, worker count) and memoizes per-tree work so
    multiple queries against the same project share the cost.
    Construction is cheap; nothing is read or parsed until you ask.

    Three coarse stages happen on demand:

    1. **Per-tree file enumeration + visitor pass** -- driven by
       :meth:`refresh`. Walks each requested tree's files, hashes
       them against the cache, runs the visitor + observe pass on
       misses, writes payloads back to the cache.

    2. **Per-tree contribution build** -- the per-tree trie + a
       tree-local graph slice + the unresolved cross-file import set.

    3. **Cross-tree composition** -- merging contributions, running
       :func:`resolve_edges` against the merged tries, running plugin
       :meth:`EdgePlugin.finalize`. Scoped to either every tree
       (:meth:`materialize_all`) or the "interesting set" of one tree
       (:meth:`materialize_closure` / :meth:`PackageView.graph`).

    The lazy split lets cheap per-package queries skip stage 3
    entirely for purely local questions like
    :meth:`PackageView.modules`.
    """

    def __init__(
        self,
        source_trees: Sequence[SourceTree] | Mapping[Path, Sequence[Path]],
        *,
        plugins: Sequence[EdgePlugin] = (),
        resolvers: Sequence[PathResolver] = (),
        project_root: Path | None = None,
        cache: GraphCache | None = None,
        unreachable_detector: UnreachableRegionDetector | None = None,
        workers: int | None = None,
    ) -> None:
        trees: Sequence[SourceTree]
        if isinstance(source_trees, Mapping):
            # ``{path: [search_paths]}`` shorthand: each key becomes its
            # own ``EXPORTED`` :class:`SourceTree`. Search paths are
            # interpreted as references to other trees in the same dict
            # by path identity. The package name is derived from each
            # path's final component (with an index suffix when names
            # collide). Useful for tests and the common single-tree
            # case; the canonical input is the list-of-trees form.
            trees = _trees_from_mapping(cast(Mapping[Path, Sequence[Path]], source_trees))
        else:
            trees = source_trees
        self._validated = validate_source_trees(trees)
        self._plugins: tuple[EdgePlugin, ...] = tuple(plugins)
        self._resolvers: tuple[PathResolver, ...] = tuple(resolvers)
        self._cache = cache
        self._workers = workers
        self._project_root: Path = (
            project_root
            if project_root is not None
            else (
                _infer_project_root(self._validated.trees) if self._validated.trees else Path.cwd()
            )
        )
        self._import_resolver: ImportResolver = _chain_resolvers(self._resolvers)
        self._detector: UnreachableRegionDetector = (
            unreachable_detector
            if unreachable_detector is not None
            else DefaultUnreachableRegionDetector()
        )
        self._dep_graph: nx.DiGraph | None = None
        self._ordered_paths: list[Path] | None = None
        self._tree_specs: dict[Path, _TreeSpec] = {}
        self._contributions: dict[Path, _TreeContribution] = {}
        self._closure_graphs: dict[Path, nx.MultiDiGraph] = {}
        self._full_graph: nx.MultiDiGraph | None = None

    @property
    def source_trees(self) -> list[SourceTree]:
        """The validated tree list this analysis was built with."""
        return list(self._validated.trees)

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _dep_dag(self) -> nx.DiGraph:
        if self._dep_graph is None:
            self._dep_graph = _build_tree_dag(self._validated)
        return self._dep_graph

    @property
    def tree_paths(self) -> list[Path]:
        """Tree paths in topological order (deps before consumers)."""
        if self._ordered_paths is None:
            self._ordered_paths = list(nx.topological_sort(self._dep_dag()))
        return list(self._ordered_paths)

    @property
    def packages(self) -> list[str]:
        """Unique package names across the configured trees."""
        seen: dict[str, None] = {}
        for t in self._validated.trees:
            seen.setdefault(t.package, None)
        return list(seen)

    def reverse_closure(self, tree_path: Path) -> frozenset[Path]:
        """Tree paths that transitively depend on ``tree_path`` (inclusive)."""
        if tree_path not in self._validated.by_path:
            raise KeyError(tree_path)
        return frozenset({tree_path}) | frozenset(nx.descendants(self._dep_dag(), tree_path))

    def _interesting_set(self, tree_path: Path) -> frozenset[Path]:
        """Tree paths needed to answer reachability for decls in ``tree_path``."""
        dag = self._dep_dag()
        scope: set[Path] = set()
        for consumer in self.reverse_closure(tree_path):
            scope.add(consumer)
            scope |= nx.ancestors(dag, consumer)
        return frozenset(scope)

    def _package_interesting_set(self, package: str) -> frozenset[Path]:
        """Union of :meth:`_interesting_set` over every tree in ``package``."""
        scope: set[Path] = set()
        for tree in self._validated.by_package.get(package, []):
            scope |= self._interesting_set(tree.path)
        return frozenset(scope)

    def refresh(self, tree_paths: Iterable[Path] | None = None) -> Analysis:
        """Update the cache and build per-tree contributions for the given trees.

        ``tree_paths=None`` (the default) refreshes every tree.
        Already-refreshed trees are skipped, so calling :meth:`refresh`
        twice with the same argument is cheap.
        """
        targets = list(tree_paths) if tree_paths is not None else self.tree_paths
        unknown = [p for p in targets if p not in self._validated.by_path]
        if unknown:
            raise KeyError(f"Unknown tree paths: {unknown}")
        new_targets = [p for p in targets if p not in self._contributions]
        if not new_targets:
            return self
        all_trees = list(self._validated.trees)
        for p in new_targets:
            if p not in self._tree_specs:
                tree = self._validated.by_path[p]
                self._tree_specs[p] = _build_tree_spec(
                    tree, all_trees, self._cache, self._tree_fingerprint(tree)
                )
        partial_specs = {p: self._tree_specs[p] for p in new_targets}
        miss_payloads = _compute_all_miss_payloads(
            tree_specs=partial_specs,
            project_root=self._project_root,
            detector=self._detector,
            plugins=self._plugins,
            resolvers=self._resolvers,
            import_resolver=self._import_resolver,
            cache=self._cache,
            workers=self._workers,
        )
        for p in new_targets:
            tree = self._validated.by_path[p]
            self._contributions[p] = _build_contribution(
                tree, self._tree_specs[p], miss_payloads[p]
            )
        return self

    def package(self, name: str) -> PackageView:
        """Return a lazy view onto a single logical package."""
        if name not in self._validated.by_package:
            raise KeyError(name)
        return PackageView(self, name)

    def package_views(self) -> Iterator[PackageView]:
        """Yield a :class:`PackageView` for every package."""
        for name in self.packages:
            yield PackageView(self, name)

    def materialize_all(self) -> nx.MultiDiGraph:
        """Build the full graph (every tree, cross-tree resolution, plugins)."""
        if self._full_graph is None:
            self.refresh()
            self._full_graph = self._materialize(scope=None)
        return self._full_graph

    def materialize_closure(self, tree_path: Path) -> nx.MultiDiGraph:
        """Build a graph containing every contribution in the
        :meth:`_interesting_set` of ``tree_path``."""
        if self._full_graph is not None:
            return self._full_graph
        if tree_path not in self._closure_graphs:
            scope = self._interesting_set(tree_path)
            self.refresh(tree_paths=scope)
            self._closure_graphs[tree_path] = self._materialize(scope=scope)
        return self._closure_graphs[tree_path]

    def _materialize(self, *, scope: frozenset[Path] | None) -> nx.MultiDiGraph:
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.graph["dead_suites"] = {}
        for path in self.tree_paths:
            if scope is not None and path not in scope:
                continue
            _compose_contribution(
                self._contributions[path],
                target_graph=g,
                symbol_lookup=self._build_symbol_lookup(path, scope=scope),
                plugins=self._plugins,
                project_root=self._project_root,
            )
        return g

    def reachable(self) -> set[SymbolNode]:
        return _find_reachable(self.materialize_all())

    def dead(self) -> Iterator[SymbolNode]:
        g = self.materialize_all()
        reachable = _find_reachable(g)
        for n in g.nodes:
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        g = self.materialize_all()
        return _find_reachable(g) - _find_reachable_strict(g)

    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]:
        return _count_nodes(self.materialize_all(), prefix)

    def _tree_fingerprint(self, tree: SourceTree) -> str:
        """Per-tree cache fingerprint."""
        return compute_fingerprint(
            base=tree.path,
            search_paths=(tree.path, *tree.search_trees),
            resolvers=self._resolvers,
            plugins=self._plugins,
            unreachable_detector=self._detector,
        )

    def _build_symbol_lookup(self, tree_path: Path, *, scope: frozenset[Path] | None) -> SymbolTrie:
        """Per-tree lookup trie: this tree's full trie + each in-scope
        search ref's export trie.

        The search refs are the tree's ``search_trees`` (paths to
        EXPORTED trees of other packages, or another tree in the same
        package). Only the search refs' *export* tries flow in -- so a
        tests/ tree that lists lib/ in its search refs sees lib/'s
        consumer-visible surface, not its private internals.
        """
        contrib = self._contributions[tree_path]
        lookup = SymbolTrie()
        lookup.merge(contrib.current_trie)
        tree = self._validated.by_path[tree_path]
        for ref in tree.search_trees:
            if scope is not None and ref not in scope:
                continue
            ref_contrib = self._contributions.get(ref)
            if ref_contrib is None:
                continue
            lookup.merge(ref_contrib.export_trie)
        return lookup


class PackageView:
    """Lazy view of a logical package inside an :class:`Analysis`.

    A package is one or more :class:`SourceTree` entries sharing a
    package name. Queries aggregate over every tree in the package;
    cross-package queries (:meth:`importers_of`, :meth:`dead`,
    :meth:`graph`) materialize the union of every tree's interesting
    set.
    """

    __slots__ = ("_analysis", "_name")

    def __init__(self, analysis: Analysis, name: str) -> None:
        self._analysis = analysis
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    @property
    def trees(self) -> list[SourceTree]:
        """Trees that share this package's name, in declaration order."""
        return list(self._analysis._validated.by_package[self._name])

    @property
    def exported_tree(self) -> SourceTree | None:
        """The package's single ``EXPORTED`` tree, if any."""
        return self._analysis._validated.exported_for.get(self._name)

    def reverse_closure(self) -> frozenset[Path]:
        """Tree paths that transitively depend on any tree in this package."""
        out: set[Path] = set()
        for tree in self.trees:
            out |= self._analysis.reverse_closure(tree.path)
        return frozenset(out)

    def _under_any_tree(self, path: Path) -> bool:
        return any(path.is_relative_to(t.path) for t in self.trees)

    def modules(self) -> Iterator[SymbolNode]:
        """Module nodes for every ``.py`` file in any tree of this package."""
        self._analysis.refresh(tree_paths=[t.path for t in self.trees])
        for tree in self.trees:
            for n in self._analysis._contributions[tree.path].base_graph.nodes:
                if n.type == "module":
                    yield n

    def declarations(self, name: str | None = None) -> Iterator[SymbolNode]:
        """Top-level decls in any tree of this package."""
        self._analysis.refresh(tree_paths=[t.path for t in self.trees])
        for tree in self.trees:
            for n in self._analysis._contributions[tree.path].base_graph.nodes:
                if n.type in ("module", "synthetic"):
                    continue
                if name is not None and simple_name(n.fqname) != name:
                    continue
                yield n

    def importers_of(self, target: str) -> set[Path]:
        """Files in this package whose imports reach ``target``.

        Triggers closure materialization. Aggregates across every
        tree in the package.
        """
        scope = self._analysis._package_interesting_set(self._name)
        self._analysis.refresh(tree_paths=scope)
        graph = self._analysis._materialize(scope=scope)
        out: set[Path] = set()
        for tree in self.trees:
            ctx = PluginContext(
                graph=graph,
                symbol_lookup=self._analysis._build_symbol_lookup(tree.path, scope=scope),
                base=tree.path,
                project_root=self._analysis.project_root,
            )
            out |= ctx.importers(target)
        return out

    def graph(self) -> nx.MultiDiGraph:
        """Materialize and return the closure-scoped graph for this package."""
        scope = self._analysis._package_interesting_set(self._name)
        self._analysis.refresh(tree_paths=scope)
        return self._analysis._materialize(scope=scope)

    def reachable(self) -> set[SymbolNode]:
        """Decls in this package reachable from any entrypoint in
        :meth:`reverse_closure`."""
        g = self.graph()
        return {n for n in _find_reachable(g) if self._under_any_tree(n.path)}

    def dead(self) -> Iterator[SymbolNode]:
        """Yield decls in this package not reachable from any entrypoint."""
        g = self.graph()
        reachable = _find_reachable(g)
        for n in g.nodes:
            if not self._under_any_tree(n.path):
                continue
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        g = self.graph()
        diff = _find_reachable(g) - _find_reachable_strict(g)
        return {n for n in diff if self._under_any_tree(n.path)}

    def count_nodes(self) -> dict[str, int]:
        """Count nodes contributed by this package, by ``SymbolNode.type``."""
        self._analysis.refresh(tree_paths=[t.path for t in self.trees])
        counts: dict[str, int] = {}
        for tree in self.trees:
            sub = _count_nodes(self._analysis._contributions[tree.path].base_graph, prefix=None)
            for k, v in sub.items():
                counts[k] = counts.get(k, 0) + v
        return counts

    def remove_dead_code(self) -> None:
        """Apply the LibCST codemod, deleting every dead decl in this package."""
        from .codemod import remove_code

        g = self.graph()
        reachable = _find_reachable(g)
        dead_nodes = [n for n in g.nodes if n not in reachable]
        sub = g.subgraph(dead_nodes)
        for tree in self.trees:
            remove_code(sub, tree.path)


__all__ = [
    "Analysis",
    "PackageView",
]
