from __future__ import annotations

import enum
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
    make_payload,
    simple_name,
)
from .resolvers import (
    ImportResolver,
    Package,
    PathResolver,
    assign_file_to_package,
    default_resolve_import,
    distribution_lookup,
    editable_distribution_roots,
    export_search_root,
    is_exported_file,
    safe_resolve_module,
    validate_packages,
)
from .resolvers._core import _ValidatedPackages

logger = logging.getLogger(__name__)


class Phase(enum.Enum):
    EXPORTS = "exports"
    INTERNALS = "internals"


def _build_pkg_dag(validated: _ValidatedPackages) -> nx.DiGraph:
    """``dep -> consumer`` DAG over package names.

    Edges go ``D -> C`` where ``D`` is in ``C.deps`` (deps flow into
    consumers). Topological order over this DAG drives phase-1
    processing -- a consumer's export-trie stitch requires every
    dep's export trie to be ready before edges resolve.
    """
    g: nx.DiGraph = nx.DiGraph()
    for name, pkg in validated.by_name.items():
        g.add_node(name)
        for dep in pkg.deps:
            g.add_edge(dep, name)
    return g


def _chain_resolvers(resolvers: Sequence[PathResolver]) -> ImportResolver:
    """Compose ``resolvers`` into one ``name -> path`` callable."""
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
    """Invoke each plugin's :meth:`EdgePlugin.observe` and collect contributions."""
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
    """Run the visitor + observe pass for a single file and return its payload."""
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
class _PhaseSpec:
    """Per-(package, phase) file plan.

    Each :class:`Package` contributes up to two phase specs: one for
    its exported files (phase 1) and one for its internal files
    (phase 2). The two specs have different ``search_paths``
    (visibility into other packages) and therefore different cache
    fingerprints.
    """

    package: Package
    phase: Phase
    search_paths: tuple[Path, ...]
    files: tuple[Path, ...]
    hits: dict[Path, VisitorPayload]
    miss_files: tuple[Path, ...]
    fqn_cache: Mapping[str, ModuleNameAndPackage]
    fingerprint: str

    @property
    def key(self) -> tuple[str, Phase]:
        return (self.package.name, self.phase)


def _phase_search_paths(
    pkg: Package, phase: Phase, validated: _ValidatedPackages
) -> tuple[Path, ...]:
    """Sys.path entries used during ``phase`` for ``pkg``.

    Phase 1 (exports) sees ``pkg``'s own exported sys.path entry plus
    each dep's exported sys.path entry. Phase 2 (internals) sees
    ``pkg.path`` (so top-level dirs like ``tests/`` resolve), ``pkg``'s
    exported sys.path entry, and every other package's exported entry
    -- the all-to-all phase-2 visibility that lets non-exported code
    have apparent cross-package cycles without violating the deps DAG.
    """
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path | None) -> None:
        if p is None or p in seen:
            return
        paths.append(p)
        seen.add(p)

    own_export = export_search_root(pkg)
    if phase is Phase.EXPORTS:
        add(own_export)
        for dep_name in pkg.deps:
            add(export_search_root(validated.by_name[dep_name]))
    else:
        add(pkg.path)
        add(own_export)
        for other in validated.packages:
            if other.name == pkg.name:
                continue
            add(export_search_root(other))
    return tuple(paths)


def _partition_files(pkg: Package, all_pkgs: Sequence[Package]) -> tuple[list[Path], list[Path]]:
    """Walk ``pkg.path`` for ``.py`` files and split into (exported, internal).

    Files routed to other packages by longest-prefix-match (a deeper
    nested package directory) are dropped here -- they show up under
    that package instead.
    """
    exported: list[Path] = []
    internal: list[Path] = []
    for candidate in sorted(pkg.path.rglob("*.py")):
        owner = assign_file_to_package(candidate, all_pkgs)
        if owner is None or owner.name != pkg.name:
            continue
        if is_exported_file(candidate, pkg):
            exported.append(candidate)
        else:
            internal.append(candidate)
    return exported, internal


def _build_phase_spec(
    pkg: Package,
    phase: Phase,
    files: list[Path],
    search_paths: tuple[Path, ...],
    cache: GraphCache | None,
    fingerprint: str,
) -> _PhaseSpec:
    files_tuple = tuple(files)
    hits: dict[Path, VisitorPayload] = {}
    miss_files: list[Path] = []
    for file in files_tuple:
        payload = cache.get(file, fingerprint) if cache is not None else None
        if payload is None:
            miss_files.append(file)
        else:
            hits[file] = payload
    fqn_cache: Mapping[str, ModuleNameAndPackage] = (
        FixedFullyQualifiedNameProvider.gen_cache(pkg.path, [str(f) for f in miss_files], timeout=5)
        if miss_files
        else {}
    )
    return _PhaseSpec(
        package=pkg,
        phase=phase,
        search_paths=search_paths,
        files=files_tuple,
        hits=hits,
        miss_files=tuple(miss_files),
        fqn_cache=fqn_cache,
        fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _Task:
    """Per-file unit of work for the visitor + observe pass."""

    file: Path
    phase_key: tuple[str, Phase]
    base: Path
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
    """Drop the ``sys.path``-derived resolver caches."""
    safe_resolve_module.cache_clear()
    distribution_lookup.cache_clear()
    editable_distribution_roots.cache_clear()


def _process_task(
    state: _RunnerState, task: _Task
) -> tuple[tuple[str, Phase], Path, VisitorPayload]:
    """Run one task, transitioning ``sys.path`` + caches if the search path changed."""
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
    return task.phase_key, task.file, payload


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


def _worker_process_task(task: _Task) -> tuple[tuple[str, Phase], Path, VisitorPayload]:
    assert _worker_state is not None, "_init_worker must run before _worker_process_task"
    return _process_task(_worker_state, task)


def _build_sorted_tasks(
    phase_specs: dict[tuple[str, Phase], _PhaseSpec], project_root: Path
) -> list[_Task]:
    """Flatten every phase's miss files into one sorted task list.

    Sorting by ``search_paths`` keeps same-search-path tasks adjacent
    so a runner sees at most one ``sys.path`` transition per
    (package, phase) it touches.
    """
    tasks: list[_Task] = [
        _Task(
            file=file,
            phase_key=key,
            base=spec.package.path,
            search_paths=spec.search_paths,
            fqn_entry=spec.fqn_cache[str(file)],
            project_root=project_root,
        )
        for key, spec in phase_specs.items()
        for file in spec.miss_files
    ]
    tasks.sort(key=lambda t: (t.search_paths, t.file))
    return tasks


def _compute_all_miss_payloads(
    *,
    phase_specs: dict[tuple[str, Phase], _PhaseSpec],
    project_root: Path,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    resolvers: Sequence[PathResolver],
    import_resolver: ImportResolver,
    cache: GraphCache | None,
    workers: int | None,
) -> dict[tuple[str, Phase], dict[Path, VisitorPayload]]:
    """Run visitor + observe for every cache-miss file across every phase spec."""
    out: dict[tuple[str, Phase], dict[Path, VisitorPayload]] = {k: {} for k in phase_specs}
    total_misses = sum(len(s.miss_files) for s in phase_specs.values())
    if total_misses == 0:
        return out

    tasks = _build_sorted_tasks(phase_specs, project_root)
    use_pool = workers is not None and workers >= 2 and total_misses >= 2

    def _record(key: tuple[str, Phase], file: Path, payload: VisitorPayload) -> None:
        out[key][file] = payload
        if cache is not None:
            cache.put(file, payload, phase_specs[key].fingerprint)

    if use_pool:
        assert workers is not None
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_init_worker,
            initargs=(detector, tuple(plugins), tuple(resolvers)),
        ) as pool:
            for key, file, payload in pool.map(_worker_process_task, tasks):
                _record(key, file, payload)
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
            key, file, payload = _process_task(state, task)
            _record(key, file, payload)
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
    """Emit ``payload`` into the in-progress per-package phase structures.

    ``export_trie`` is non-``None`` only for phase 1 (exports). Phase
    2 still populates ``current_trie`` (so within-package resolution
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
class _PhaseContribution:
    """One (package, phase) pre-stitched contribution to the symbol graph."""

    package: Package
    phase: Phase
    current_trie: SymbolTrie
    export_trie: SymbolTrie  # populated only for phase 1; empty for phase 2
    base_graph: nx.MultiDiGraph
    import_edges: frozenset[tuple[SymbolNode, Import, EdgeFlags]]


def _build_phase_contribution(
    spec: _PhaseSpec,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> _PhaseContribution:
    """Apply ``spec``'s per-file payloads into a phase-local graph slice."""
    current_trie = SymbolTrie()
    export_trie = SymbolTrie()
    is_exports = spec.phase is Phase.EXPORTS
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
            export_trie=export_trie if is_exports else None,
            symbol_graph=base_graph,
            import_edges=import_edges,
        )
    current_trie.add_module_hierarchy_edges(base_graph)
    return _PhaseContribution(
        package=spec.package,
        phase=spec.phase,
        current_trie=current_trie,
        export_trie=export_trie,
        base_graph=base_graph,
        import_edges=frozenset(import_edges),
    )


def _compose_contribution(
    contrib: _PhaseContribution,
    *,
    target_graph: nx.MultiDiGraph,
    symbol_lookup: SymbolTrie,
    plugins: Sequence[EdgePlugin],
    project_root: Path,
) -> None:
    """Merge ``contrib.base_graph`` into ``target_graph``, stitch
    cross-package imports against ``symbol_lookup``, and run plugin
    :meth:`EdgePlugin.finalize` against the composed graph.
    """
    target_graph.update(
        edges=contrib.base_graph.edges(data=True, keys=True),
        nodes=contrib.base_graph.nodes(data=True),
    )
    target_graph.graph.setdefault("dead_suites", {}).update(contrib.base_graph.graph["dead_suites"])
    for src, dst, flags in resolve_edges(contrib.import_edges, symbol_lookup, contrib.package.path):
        target_graph.add_edge(src, dst, flags=flags)
    if plugins:
        ctx = PluginContext(
            graph=target_graph,
            symbol_lookup=symbol_lookup,
            base=contrib.package.path,
            project_root=project_root,
        )
        for plugin in plugins:
            if not isinstance(plugin, EdgePlugin):
                raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
            ops = list(plugin.finalize(ctx))
            apply_ops(target_graph, ops)


def _warn_if_project_venv_inactive(project_root: Path) -> None:
    """Warn when the project has a sibling ``.venv``/``venv`` that
    isn't the running interpreter."""
    candidate = next(
        (project_root / name for name in (".venv", "venv") if (project_root / name).is_dir()),
        None,
    )
    if candidate is None:
        return
    try:
        if Path(sys.prefix).resolve() == candidate.resolve():
            return
    except OSError:
        return
    logger.warning(
        "dead-cst is not running inside %s. Third-party imports "
        "may classify as [unresolved] and framework plugin lookups "
        "may fail. Re-run via 'uv run dead-cst ...' (or activate "
        "the venv) for full classification.",
        candidate,
    )


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

    Constructed with a ``project_root`` and a list of
    :class:`PathResolver`\\s. The analyzer calls each resolver's
    :meth:`PathResolver.resolve` to derive the project's
    :class:`Package` layout; the validated package list is exposed as
    :attr:`packages`. Holds the analyzer's config (resolvers, plugins,
    cache, detector, worker count) and memoizes per-(package, phase)
    work so multiple queries against the same project share the cost.
    Construction is cheap; nothing is read or parsed until you ask.

    Stages:

    1. **Per-(package, phase) file enumeration + visitor pass** --
       driven by :meth:`refresh`. Walks each package's ``path``,
       partitions ``.py`` files into exported vs internal, hashes
       against the cache, runs the visitor + observe pass on misses,
       writes payloads back to the cache.

    2. **Per-(package, phase) contribution build** -- a phase-local
       trie + graph slice + the unresolved cross-file import set.

    3. **Two-phase composition** -- phase 1 composes exports
       contributions in topological order over ``deps``, stitching
       cross-package imports against (own export trie under
       construction + each dep's already-built export trie). Phase 2
       composes internals contributions in any order, stitching
       against (the package's full surface + every package's export
       trie). Plugin :meth:`EdgePlugin.finalize` runs once per
       composition.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        resolvers: Sequence[PathResolver] = (),
        plugins: Sequence[EdgePlugin] = (),
        cache: GraphCache | None = None,
        unreachable_detector: UnreachableRegionDetector | None = None,
        workers: int | None = None,
    ) -> None:
        self._project_root: Path = project_root.resolve()
        self._resolvers: tuple[PathResolver, ...] = tuple(resolvers)
        pkgs: list[Package] = []
        for r in self._resolvers:
            pkgs.extend(r.resolve(self._project_root))
        self._validated = validate_packages(pkgs)
        self._plugins: tuple[EdgePlugin, ...] = tuple(plugins)
        self._cache = cache
        self._workers = workers
        self._import_resolver: ImportResolver = _chain_resolvers(self._resolvers)
        self._detector: UnreachableRegionDetector = (
            unreachable_detector
            if unreachable_detector is not None
            else DefaultUnreachableRegionDetector()
        )
        self._dep_graph: nx.DiGraph | None = None
        self._phase_specs: dict[tuple[str, Phase], _PhaseSpec] = {}
        self._contributions: dict[tuple[str, Phase], _PhaseContribution] = {}
        self._full_graph: nx.MultiDiGraph | None = None
        _warn_if_project_venv_inactive(self._project_root)

    @property
    def packages(self) -> list[Package]:
        """The validated package list this analysis was built with."""
        return list(self._validated.packages)

    @property
    def package_names(self) -> list[str]:
        """Package names in declaration order."""
        return [p.name for p in self._validated.packages]

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _dep_dag(self) -> nx.DiGraph:
        if self._dep_graph is None:
            self._dep_graph = _build_pkg_dag(self._validated)
        return self._dep_graph

    def reverse_closure(self, package: str) -> frozenset[str]:
        """Packages that transitively depend on ``package`` (inclusive)."""
        if package not in self._validated.by_name:
            raise KeyError(package)
        dag = self._dep_dag()
        out: set[str] = {package}
        for descendant in nx.descendants(dag, package):
            out.add(descendant)
        return frozenset(out)

    def refresh(self, packages: Iterable[str] | None = None) -> Analysis:
        """Walk the cache and build per-phase contributions.

        ``packages=None`` (the default) refreshes every package. The
        argument exists so callers can narrow the file walk to a
        subset of packages; phase-2 visibility still resolves against
        every other package's export trie at composition time, so
        composition is always full-graph.
        """
        names = list(packages) if packages is not None else self.package_names
        unknown = [n for n in names if n not in self._validated.by_name]
        if unknown:
            raise KeyError(f"Unknown packages: {unknown}")
        all_pkgs = list(self._validated.packages)
        new_keys: list[tuple[str, Phase]] = []
        for name in names:
            pkg = self._validated.by_name[name]
            exported_files, internal_files = _partition_files(pkg, all_pkgs)
            for phase, files in (
                (Phase.EXPORTS, exported_files),
                (Phase.INTERNALS, internal_files),
            ):
                if not files:
                    continue
                key = (name, phase)
                if key in self._contributions:
                    continue
                if key not in self._phase_specs:
                    search = _phase_search_paths(pkg, phase, self._validated)
                    fingerprint = compute_fingerprint(
                        base=pkg.path,
                        search_paths=search,
                        resolvers=self._resolvers,
                        plugins=self._plugins,
                        unreachable_detector=self._detector,
                    )
                    self._phase_specs[key] = _build_phase_spec(
                        pkg, phase, files, search, self._cache, fingerprint
                    )
                new_keys.append(key)
        if not new_keys:
            return self
        partial_specs = {key: self._phase_specs[key] for key in new_keys}
        miss_payloads = _compute_all_miss_payloads(
            phase_specs=partial_specs,
            project_root=self._project_root,
            detector=self._detector,
            plugins=self._plugins,
            resolvers=self._resolvers,
            import_resolver=self._import_resolver,
            cache=self._cache,
            workers=self._workers,
        )
        for key in new_keys:
            self._contributions[key] = _build_phase_contribution(
                self._phase_specs[key], miss_payloads[key]
            )
        return self

    def package(self, name: str) -> PackageView:
        """Return a lazy view onto a single logical package."""
        if name not in self._validated.by_name:
            raise KeyError(name)
        return PackageView(self, name)

    def package_views(self) -> Iterator[PackageView]:
        """Yield a :class:`PackageView` for every package."""
        for name in self.package_names:
            yield PackageView(self, name)

    def materialize_all(self) -> nx.MultiDiGraph:
        """Build the full graph (every package, both phases, plugins)."""
        if self._full_graph is None:
            self.refresh()
            self._full_graph = self._materialize_full()
        return self._full_graph

    def materialize_closure(self, package: str) -> nx.MultiDiGraph:
        """Build the full graph; ``package`` is just validated.

        Phase-2's all-to-all visibility makes a closure-scoped graph
        unsound -- any package's internals could keep ``package``'s
        decls alive -- so this returns the same graph as
        :meth:`materialize_all`. Kept as a stable API surface for
        callers that previously asked for a closure-scoped graph.
        """
        if package not in self._validated.by_name:
            raise KeyError(package)
        return self.materialize_all()

    def _phase_contributions(self, name: str) -> Iterator[_PhaseContribution]:
        """Yield this package's contributions in phase order, skipping empty ones."""
        for phase in (Phase.EXPORTS, Phase.INTERNALS):
            contrib = self._contributions.get((name, phase))
            if contrib is not None:
                yield contrib

    def _materialize_full(self) -> nx.MultiDiGraph:
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.graph["dead_suites"] = {}

        # Built once and shared across every phase-2 lookup. Phase-2
        # visibility for any package is "every package's export trie",
        # so the union is identical for every package -- we just merge
        # the package's own combined trie on top, per-package, below.
        all_exports = SymbolTrie()
        for contrib in self._contributions.values():
            if contrib.phase is Phase.EXPORTS:
                all_exports.merge(contrib.export_trie)

        for phase in (Phase.EXPORTS, Phase.INTERNALS):
            for pkg in self._validated.topo_order:
                contrib = self._contributions.get((pkg.name, phase))
                if contrib is None:
                    continue
                _compose_contribution(
                    contrib,
                    target_graph=g,
                    symbol_lookup=self._build_phase_lookup(pkg, phase, all_exports),
                    plugins=self._plugins,
                    project_root=self._project_root,
                )
        return g

    def _build_phase_lookup(
        self, pkg: Package, phase: Phase, all_exports: SymbolTrie
    ) -> SymbolTrie:
        """Per-(package, phase) symbol lookup trie used for edge stitching.

        Phase 1 sees the package's own exports (under construction)
        plus each dep's already-built export trie. Phase 2 sees the
        package's internal surface plus ``all_exports`` -- the
        prebuilt union of every package's export tries (including
        ``pkg``'s own), so non-exported code cycles across packages
        without violating the deps DAG.
        """
        contrib = self._contributions[(pkg.name, phase)]
        lookup = SymbolTrie()
        lookup.merge(contrib.current_trie)
        if phase is Phase.EXPORTS:
            for dep_name in pkg.deps:
                dep_contrib = self._contributions.get((dep_name, Phase.EXPORTS))
                if dep_contrib is not None:
                    lookup.merge(dep_contrib.export_trie)
            return lookup
        lookup.merge(all_exports)
        return lookup

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


class PackageView:
    """Lazy view of a logical package inside an :class:`Analysis`."""

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
    def package(self) -> Package:
        return self._analysis._validated.by_name[self._name]

    def reverse_closure(self) -> frozenset[str]:
        """Packages that transitively depend on this one (inclusive)."""
        return self._analysis.reverse_closure(self._name)

    def _under_path(self, path: Path) -> bool:
        try:
            return path == self.package.path or path.is_relative_to(self.package.path)
        except ValueError:
            return False

    def modules(self) -> Iterator[SymbolNode]:
        """Module nodes for every ``.py`` file in this package."""
        self._analysis.refresh(packages=[self._name])
        for contrib in self._analysis._phase_contributions(self._name):
            for n in contrib.base_graph.nodes:
                if n.type == "module":
                    yield n

    def declarations(self, name: str | None = None) -> Iterator[SymbolNode]:
        """Top-level decls in this package."""
        self._analysis.refresh(packages=[self._name])
        for contrib in self._analysis._phase_contributions(self._name):
            for n in contrib.base_graph.nodes:
                if n.type in ("module", "synthetic"):
                    continue
                if name is not None and simple_name(n.fqname) != name:
                    continue
                yield n

    def importers_of(self, target: str) -> set[Path]:
        """Files that import ``target`` (anywhere in the workspace)."""
        graph = self._analysis.materialize_all()
        ctx = PluginContext(
            graph=graph,
            symbol_lookup=SymbolTrie(),  # plugin's importers() doesn't use lookup
            base=self.package.path,
            project_root=self._analysis.project_root,
        )
        return ctx.importers(target)

    def graph(self) -> nx.MultiDiGraph:
        """Materialize and return the full workspace graph."""
        return self._analysis.materialize_all()

    def reachable(self) -> set[SymbolNode]:
        """Decls in this package reachable from any entrypoint."""
        g = self.graph()
        return {n for n in _find_reachable(g) if self._under_path(n.path)}

    def dead(self) -> Iterator[SymbolNode]:
        """Yield decls in this package not reachable from any entrypoint."""
        g = self.graph()
        reachable = _find_reachable(g)
        for n in g.nodes:
            if not self._under_path(n.path):
                continue
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        g = self.graph()
        diff = _find_reachable(g) - _find_reachable_strict(g)
        return {n for n in diff if self._under_path(n.path)}

    def count_nodes(self) -> dict[str, int]:
        """Count nodes contributed by this package, by ``SymbolNode.type``."""
        self._analysis.refresh(packages=[self._name])
        counts: dict[str, int] = {}
        for contrib in self._analysis._phase_contributions(self._name):
            sub = _count_nodes(contrib.base_graph, prefix=None)
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
        remove_code(sub, self.package.path)


__all__ = [
    "Analysis",
    "PackageView",
]
