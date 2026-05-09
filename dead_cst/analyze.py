from __future__ import annotations

import logging
import sys
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import networkx as nx

from ._edges import resolve_edges
from ._refresh import (
    PackageContribution,
    PackageFiles,
    build_contribution,
    build_stale_tasks,
    enumerate_files,
    process_stale_files,
)
from .branches import (
    DefaultUnreachableRegionDetector,
    UnreachableRegionDetector,
)
from .cache import GraphCache, compute_fingerprint
from .graph import EdgeFlags, NodeFlags, SymbolNode, SymbolTrie
from .plugins import (
    EdgePlugin,
    PluginContext,
    apply_ops,
)
from .plugins._core import simple_name
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


def _rebind_sys_path(search_paths: tuple[Path, ...], baseline: list[str]) -> None:
    """Set ``sys.path`` to ``search_paths + baseline``, deduplicated.

    Used by :meth:`Analysis._materialize` so the resolver in
    :func:`_compose_contribution` -> :func:`resolve_edges` sees this
    package's first-party prefix while classifying trie-miss imports.
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


def _compose_contribution(
    contrib: PackageContribution,
    *,
    target_graph: nx.MultiDiGraph,
    symbol_lookup: SymbolTrie,
    plugins: Sequence[EdgePlugin],
    project_root: Path,
    import_resolver: ImportResolver,
    search_paths: list[Path],
) -> None:
    """Merge ``contrib.package_graph`` into ``target_graph``, stitch
    cross-package imports against ``symbol_lookup``, and run plugin
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
        edges=contrib.package_graph.edges(data=True, keys=True),
        nodes=contrib.package_graph.nodes(data=True),
    )
    target_graph.graph.setdefault("dead_suites", {}).update(
        contrib.package_graph.graph["dead_suites"]
    )
    target_graph.add_edges_from(
        (src, dst, {"flags": flags})
        for src, dst, flags in resolve_edges(
            contrib.import_edges,
            symbol_lookup,
            contrib.package.path,
            import_resolver=import_resolver,
            search_paths=search_paths,
        )
    )
    if plugins:
        ctx = PluginContext(
            graph=target_graph,
            symbol_lookup=symbol_lookup,
            package=contrib.package,
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


def _find_reachable_excluding(graph: nx.MultiDiGraph, exclude_flags: NodeFlags) -> set[SymbolNode]:
    """Like :func:`_find_reachable` but skips entrypoints with any of ``exclude_flags``.

    The ``SymbolNode`` (the graph key) carries its own flag bits, so
    membership is checked directly off the node. ``exclude_flags`` may
    combine multiple bits (``NodeFlags.TESTCASE | NodeFlags.NOQA``) to
    drop several entrypoint classes in one pass.
    """
    visited: set[SymbolNode] = set()
    stack = [
        n
        for n, attrs in graph.nodes(data=True)
        if attrs.get("entrypoint") and not (n.flags & exclude_flags)
    ]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph.successors(node))
    return visited


def _find_kept_alive_by_flags_only(graph: nx.MultiDiGraph, flags: NodeFlags) -> set[SymbolNode]:
    """Symbols reachable only from entrypoints carrying any of ``flags``.

    ``_find_reachable(graph) -`` :func:`_find_reachable_excluding(graph, flags)`;
    the difference is the "blast radius" of dropping every entrypoint with
    any of those flag bits. Surfaced on :class:`Analysis` and
    :class:`PackageView` as ``kept_alive_by_flags_only(flags)`` -- pass
    :data:`NodeFlags.TESTCASE` for "blast radius of dropping the test
    suite", :data:`NodeFlags.NOQA` for "blast radius of removing every
    F401 pin", or both ORed together.
    """
    return _find_reachable(graph) - _find_reachable_excluding(graph, flags)


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
    detector, worker count) and memoizes per-package work so multiple
    queries against the same project share the cost. Construction
    runs the resolver's :meth:`PathResolver.resolve` once to build
    the path map, but no source files are read or parsed until you
    ask -- the visitor pass is gated on :meth:`refresh` /
    :meth:`materialize_all`.

    Three coarse stages happen on demand:

    1. **File enumeration + visitor pass** -- driven by
       :meth:`refresh`. Walks each requested package's files, hashes
       them against the cache, then flattens every package's misses
       into one global stale-file list and runs the visitor +
       observe pass once across all of them (parallel when
       ``workers`` permits). Idempotent and scoped:
       ``refresh(packages=[p])`` walks only ``p``'s file tree.

    2. **Per-package contribution build** -- the per-package trie + a
       package-local graph slice + the unresolved cross-file import
       set. Built once per package from the payloads above, memoized
       for the lifetime of the :class:`Analysis`.

    3. **Cross-package composition** -- merging contributions, running
       :func:`resolve_edges` against the merged tries, running plugin
       :meth:`EdgePlugin.finalize`. Scoped to either the full package
       set (:meth:`materialize_all`) or the "interesting set" of one
       package (:meth:`materialize_closure` / :meth:`PackageView.graph`),
       which is the forward dependency closure of that package's
       reverse (consumer) closure -- the only packages that could keep
       a decl in the target package alive.

    The lazy split lets cheap per-package queries skip stage 3 entirely:
    :meth:`PackageView.modules` and :meth:`PackageView.declarations`
    only need stage 2 for their own package. Reachability queries
    (:meth:`PackageView.dead`, :meth:`Analysis.dead`) trigger stage 3
    over the appropriate scope -- the "interesting set" for a single
    package, or every package for the full graph. Composing a graph is
    much cheaper than recomputing payloads, so per-package queries
    against a warm cache stay fast even on large repos.

    The configured ``resolver`` is queried twice: once at construction
    time to build the per-package :class:`Package` list (calling
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
        validated = _validate_packages(resolver.resolve(project_root))
        self._packages_by_path: dict[Path, Package] = {p.path: p for p in validated}
        by_name = {p.name: p.path for p in validated}
        self._dep_paths_by_package: dict[Path, tuple[Path, ...]] = {
            p.path: tuple(by_name[d] for d in p.deps) for p in validated
        }
        # Reverse map: package path -> packages that name this one in
        # their ``deps``. Pre-sorted by path so consumer-side BFS (used
        # by ``reverse_closure`` and ``packages``) yields a deterministic
        # order.
        consumers: dict[Path, list[Path]] = {p.path: [] for p in validated}
        for p in validated:
            for dep_name in p.deps:
                consumers[by_name[dep_name]].append(p.path)
        self._consumers_by_package: dict[Path, tuple[Path, ...]] = {
            path: tuple(sorted(cs)) for path, cs in consumers.items()
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
        # detector chain, so every package shares the same key.
        self._fingerprint: str = compute_fingerprint(
            plugins=self._plugins,
            unreachable_detector=self._detector,
        )
        self._package_files: dict[Path, PackageFiles] = {}
        self._contributions: dict[Path, PackageContribution] = {}
        self._closure_graphs: dict[Path, nx.MultiDiGraph] = {}
        self._full_graph: nx.MultiDiGraph | None = None
        # Memoize closure-walk results -- the package graph is
        # immutable post-construction.
        self._reverse_closures: dict[Path, frozenset[Path]] = {}
        self._interesting_sets: dict[Path, frozenset[Path]] = {}

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _dep_paths(self, package: Path) -> tuple[Path, ...]:
        """Precomputed dep paths for the package at ``package``."""
        return self._dep_paths_by_package.get(package, ())

    @cached_property
    def packages(self) -> tuple[Package, ...]:
        """Deterministic, cycle-tolerant package order.

        Dependencies precede their dependents whenever the package
        graph is acyclic. Packages trapped in dep cycles are appended
        at the end in path order. The traversal is fully deterministic
        across runs.
        """
        sorted_paths = sorted(self._packages_by_path)
        seeds = [p for p in sorted_paths if not self._dep_paths_by_package.get(p)]
        order = _bfs_order(seeds, self._consumers_by_package)
        visited = set(order)
        order.extend(p for p in sorted_paths if p not in visited)
        return tuple(self._packages_by_path[p] for p in order)

    def reverse_closure(self, package: Path) -> frozenset[Path]:
        """Packages that transitively depend on ``package`` (including itself).

        These are the only packages whose source code could import (and
        therefore keep alive) decls under ``package``. A query like
        "is X in ``package`` dead?" only needs entrypoints from this
        set -- sibling packages that can't reach ``package`` through
        the dep graph can't reference its decls. Cycle-safe: BFS
        terminates even when ``Package.deps`` cycles.
        """
        if package not in self._packages_by_path:
            raise KeyError(package)
        cached = self._reverse_closures.get(package)
        if cached is None:
            cached = frozenset(_bfs_order([package], self._consumers_by_package))
            self._reverse_closures[package] = cached
        return cached

    def _interesting_set(self, package: Path) -> frozenset[Path]:
        """Packages needed to answer reachability queries about decls in ``package``.

        :meth:`reverse_closure` ∪ each consumer's transitive deps, so
        we have enough trie data to resolve every cross-package import
        that could lead into ``package``.
        """
        cached = self._interesting_sets.get(package)
        if cached is None:
            cached = frozenset(
                _bfs_order(self.reverse_closure(package), self._dep_paths_by_package)
            )
            self._interesting_sets[package] = cached
        return cached

    def refresh(self, packages: Iterable[Path] | None = None) -> Analysis:
        """Update the cache and build per-package contributions.

        ``packages=None`` (the default) refreshes every package.
        Passing a subset scopes the file walk + visitor pass to those
        packages only -- siblings are untouched. Already-refreshed
        packages are skipped, so calling :meth:`refresh` twice with the
        same argument is cheap.

        Three steps run in order: (1) walk each new target's tree and
        partition into cache hits / misses, (2) flatten every package's
        misses into a single global stale-file list and run the
        visitor + observe pass on each one (parallel when ``workers``
        permits), (3) apply each package's payloads into a local
        contribution. Step 2 ignores which package each file lives
        under, so a refresh that touches several packages pays for one
        worker pool startup, not one per package.

        Returns ``self`` so callers can chain
        ``Analysis(...).refresh().materialize_all()``.
        """
        targets = list(packages) if packages is not None else [p.path for p in self.packages]
        unknown = [p for p in targets if p not in self._packages_by_path]
        if unknown:
            raise KeyError(f"Unknown packages: {unknown}")
        new_targets = [p for p in targets if p not in self._contributions]
        if not new_targets:
            return self

        for path in new_targets:
            if path not in self._package_files:
                self._package_files[path] = enumerate_files(
                    self._packages_by_path[path], self._cache, self._fingerprint
                )

        pending = {p: self._package_files[p] for p in new_targets}
        tasks = build_stale_tasks(pending, self._project_root)
        miss_payloads = process_stale_files(
            tasks=tasks,
            detector=self._detector,
            plugins=self._plugins,
            cache=self._cache,
            fingerprint=self._fingerprint,
            workers=self._workers,
        )

        for path in new_targets:
            self._contributions[path] = build_contribution(
                self._packages_by_path[path],
                self._package_files[path],
                miss_payloads,
            )
        return self

    def package(self, path: Path) -> PackageView:
        """Return a lazy view onto a single package.

        The returned :class:`PackageView` is cheap; per-package work is
        triggered by its query methods.
        """
        if path not in self._packages_by_path:
            raise KeyError(path)
        return PackageView(self, self._packages_by_path[path])

    def views(self) -> Iterator[PackageView]:
        """Yield a :class:`PackageView` for every package in :attr:`packages` order."""
        for package in self.packages:
            yield PackageView(self, package)

    def materialize_all(self) -> nx.MultiDiGraph:
        """Build the full graph (every package, cross-package resolution, plugins).

        Memoized: the second call returns the same graph object.
        Refreshes every package first, so this is also the trigger
        for a whole-project cache refresh in callers that don't refresh
        explicitly.
        """
        if self._full_graph is None:
            self.refresh()
            self._full_graph = self._materialize(scope=None)
        return self._full_graph

    def materialize_closure(self, package: Path) -> nx.MultiDiGraph:
        """Build a graph containing every contribution in ``_interesting_set(package)``.

        The result is the smallest graph that gives correct
        reachability answers for decls in ``package``: every consumer
        of ``package`` (so we see every potential alive-keeper) plus
        every consumer's transitive deps (so cross-package imports
        resolve).

        If :meth:`materialize_all` has already been called, returns
        the full graph instead -- it's a strict superset and cheaper
        than recomputing.
        """
        if self._full_graph is not None:
            return self._full_graph
        if package not in self._closure_graphs:
            scope = self._interesting_set(package)
            self.refresh(packages=scope)
            self._closure_graphs[package] = self._materialize(scope=scope)
        return self._closure_graphs[package]

    def _materialize(self, *, scope: frozenset[Path] | None) -> nx.MultiDiGraph:
        """Compose every refreshed package in ``scope`` into a fresh graph.

        ``scope=None`` composes every package. Caller is responsible
        for having :meth:`refresh`'d every package in ``scope`` first.

        Cross-file import resolution runs here (in
        :func:`_compose_contribution` -> :func:`resolve_edges`), which
        is where the resolver reads ``sys.path`` /
        :mod:`importlib.metadata`. We rebind ``sys.path`` to each
        package's ``(path, *deps)`` view before composing it and clear
        the resolver LRUs at every transition, restoring the original
        ``sys.path`` on the way out so library callers don't see
        lingering mutations.
        """
        from ._progress import progress

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.graph["dead_suites"] = {}
        baseline = list(sys.path)
        last_search_paths: tuple[Path, ...] | None = None
        target_paths = [p.path for p in self.packages if scope is None or p.path in scope]
        try:
            for path in progress(
                target_paths,
                total=len(target_paths),
                desc="Reconciling packages",
                unit="package",
            ):
                search_paths = (path, *self._dep_paths(path))
                if last_search_paths != search_paths:
                    _rebind_sys_path(search_paths, baseline)
                    clear_path_caches()
                    last_search_paths = search_paths
                _compose_contribution(
                    self._contributions[path],
                    target_graph=g,
                    symbol_lookup=self._build_symbol_lookup(path, scope=scope),
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

    def kept_alive_by_flags_only(self, flags: NodeFlags) -> set[SymbolNode]:
        """Symbols reachable only from entrypoints carrying any of ``flags``.

        Computed as ``reachable() -`` BFS that excludes every entrypoint
        whose flags intersect the argument. The resulting set is the
        "blast radius" of dropping every entrypoint with any of those
        flag bits. The default :meth:`reachable` traversal is
        unchanged; this is the opt-in stricter pass.

        Common flag arguments:

        * :data:`NodeFlags.TESTCASE` -- "if you deleted your tests,
          these symbols would also become dead" (pytest / unittest
          discovery seeds).
        * :data:`NodeFlags.NOQA` -- "if I removed every
          ``# noqa: F401`` pin, what would actually become dead?"
          (imports the visitor preserved because of a per-line or
          file-level ruff/pyflakes directive).
        * ``NodeFlags.TESTCASE | NodeFlags.NOQA`` -- combine in one
          pass.
        """
        g = self.materialize_all()
        return _find_kept_alive_by_flags_only(g, flags)

    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]:
        """Count nodes in the full graph by ``SymbolNode.type``.

        ``prefix=None`` (the default) counts every node. Pass a
        package path to scope the count to nodes whose ``path`` is
        under that prefix -- useful for per-package summaries when
        several packages are analysed together.
        """
        return _count_nodes(self.materialize_all(), prefix)

    def _build_symbol_lookup(self, package: Path, *, scope: frozenset[Path] | None) -> SymbolTrie:
        """Per-package lookup trie: this package's full trie + each in-scope dep's exports.

        ``scope`` bounds which deps' export tries are merged in:
        ``None`` for the full-graph path (every dep), or a
        :meth:`_interesting_set` for closure-scoped materialization.
        Deps must already be refreshed (the caller is responsible for
        calling :meth:`refresh` on the right set first).
        """
        contrib = self._contributions[package]
        lookup = SymbolTrie()
        lookup.merge(contrib.current_trie)
        for dep in self._dep_paths(package):
            if scope is not None and dep not in scope:
                continue
            dep_contrib = self._contributions.get(dep)
            if dep_contrib is None:
                continue
            lookup.merge(dep_contrib.export_trie)
        return lookup


class PackageView:
    """Lazy view of a single package inside an :class:`Analysis`.

    Cheap to construct (returned by :meth:`Analysis.package`); query
    methods trigger only the work their result depends on. Local
    queries (:meth:`modules`, :meth:`declarations`) only need this
    package's contribution. Cross-package queries (:meth:`importers_of`,
    :meth:`dead`, :meth:`graph`) materialize the
    :meth:`Analysis._interesting_set` for this package.
    """

    __slots__ = ("_analysis", "_package")

    def __init__(self, analysis: Analysis, package: Package) -> None:
        self._analysis = analysis
        self._package = package

    @property
    def package(self) -> Package:
        return self._package

    @property
    def path(self) -> Path:
        """Convenience for ``view.package.path``."""
        return self._package.path

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def reverse_closure(self) -> frozenset[Path]:
        """Packages that transitively depend on this one (including itself)."""
        return self._analysis.reverse_closure(self._package.path)

    def _contribution(self) -> PackageContribution:
        self._analysis.refresh(packages=[self._package.path])
        return self._analysis._contributions[self._package.path]

    def modules(self) -> Iterator[SymbolNode]:
        """Module nodes for every ``.py`` file in this package.

        Local-only: refreshes this package if needed but never
        touches deps or consumers.
        """
        for n in self._contribution().package_graph.nodes:
            if n.type == "module":
                yield n

    def declarations(self, name: str | None = None) -> Iterator[SymbolNode]:
        """Top-level decls in this package.

        ``name=None`` yields every decl. Pass a string to filter to
        decls whose rightmost dotted segment matches it (``"Foo"``
        matches ``pkg.mod.Foo`` but not ``pkg.Foo.bar``). Local-only.
        """
        for n in self._contribution().package_graph.nodes:
            if n.type in ("module", "synthetic"):
                continue
            if name is not None and simple_name(n.fqname) != name:
                continue
            yield n

    def importers_of(self, target: str) -> set[Path]:
        """Files in this package whose imports reach ``target``.

        ``target`` is matched first as a first-party module fqname,
        then against the synthetic ``[external dist] / [external file]
        / [unresolved]`` markers the resolver creates for non-first-
        party imports. Triggers closure materialization (same scope
        as :meth:`graph`) because cross-package import resolution is
        what populates the predecessors used here.
        """
        scope = self._analysis._interesting_set(self._package.path)
        ctx = PluginContext(
            graph=self._analysis.materialize_closure(self._package.path),
            symbol_lookup=self._analysis._build_symbol_lookup(self._package.path, scope=scope),
            package=self._package,
            project_root=self._analysis.project_root,
        )
        return ctx.importers(target)

    def graph(self) -> nx.MultiDiGraph:
        """Materialize and return the closure-scoped graph for this package.

        See :meth:`Analysis.materialize_closure`. The graph is shared
        across queries on the same package -- repeated calls return
        the same object.
        """
        return self._analysis.materialize_closure(self._package.path)

    def reachable(self) -> set[SymbolNode]:
        """Set of decls in this package reachable from any entrypoint in
        :meth:`reverse_closure`.

        Triggers closure materialization on first call; subsequent
        calls reuse the cached graph. Filtered to nodes whose ``path``
        is under :attr:`path`, so the result is comparable to
        :meth:`declarations` for "what's alive in this package?"
        questions.
        """
        g = self._analysis.materialize_closure(self._package.path)
        return {n for n in _find_reachable(g) if n.path.is_relative_to(self._package.path)}

    def dead(self) -> Iterator[SymbolNode]:
        """Yield decls in this package not reachable from any entrypoint
        in :meth:`reverse_closure`.

        Triggers closure materialization on first call; subsequent
        calls reuse the cached graph. Excludes ``module`` and
        ``synthetic`` nodes (see :meth:`Analysis.dead`).
        """
        g = self._analysis.materialize_closure(self._package.path)
        reachable = _find_reachable(g)
        for n in g.nodes:
            if not n.path.is_relative_to(self._package.path):
                continue
            if n.type in ("module", "synthetic"):
                continue
            if n not in reachable:
                yield n

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        """Decls in this package kept alive only by dead-branch references.

        Closure-scoped equivalent of :meth:`Analysis.kept_alive_by_dead_branches`,
        filtered to nodes under :attr:`path`.
        """
        g = self._analysis.materialize_closure(self._package.path)
        diff = _find_reachable(g) - _find_reachable_strict(g)
        return {n for n in diff if n.path.is_relative_to(self._package.path)}

    def kept_alive_by_flags_only(self, flags: NodeFlags) -> set[SymbolNode]:
        """Decls in this package kept alive only by entrypoints carrying any of ``flags``.

        Closure-scoped equivalent of :meth:`Analysis.kept_alive_by_flags_only`,
        filtered to nodes under :attr:`path`. See that method for the
        common flag arguments (``TESTCASE``, ``NOQA``, or both).
        """
        g = self._analysis.materialize_closure(self._package.path)
        diff = _find_kept_alive_by_flags_only(g, flags)
        return {n for n in diff if n.path.is_relative_to(self._package.path)}

    def count_nodes(self) -> dict[str, int]:
        """Count nodes contributed by this package, by ``SymbolNode.type``.

        Local-only: doesn't materialize the closure. Counts include
        ``module``, source decls, and any ``synthetic`` nodes plugins
        emitted into this package's contribution during ``observe``.
        """
        return _count_nodes(self._contribution().package_graph, prefix=None)

    def remove_dead_code(self) -> None:
        """Apply the LibCST codemod, deleting every dead decl in this package.

        Materializes the closure, computes reachability, and feeds the
        unreachable subgraph (filtered to this package) to
        :func:`dead_cst.codemod.remove_code`. The transformation is
        destructive -- back the files up first, or run on a clean
        working tree.
        """
        from .codemod import remove_code

        g = self._analysis.materialize_closure(self._package.path)
        reachable = _find_reachable(g)
        dead_nodes = [n for n in g.nodes if n not in reachable]
        remove_code(g.subgraph(dead_nodes), self._package.path)


__all__ = [
    "Analysis",
    "PackageView",
]
