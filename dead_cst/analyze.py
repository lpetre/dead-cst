from __future__ import annotations

import logging
import sys
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from libcst.metadata import CodeRange

from ._edges import resolve_edges
from ._graphstore import SymbolGraph
from ._package import PackageContribution, build_contribution
from ._refresh import (
    PackageFiles,
    build_stale_tasks,
    enumerate_files,
    process_stale_files,
)
from .branches import (
    DefaultUnreachableRegionDetector,
    UnreachableRegionDetector,
)
from .cache import GraphCache, compute_fingerprint
from .graph import EdgeFlags, NodeFlags, SymbolNode, SymbolTrie, _claim_edge
from .plugins import (
    SYNTHETIC_PATH_PREFIXES,
    EdgePlugin,
    PluginContext,
    apply_ops,
)
from .plugins._core import simple_name
from .resolvers import (
    ImportResolver,
    Package,
    PathResolver,
    clear_module_specs_cache,
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
    target_graph: SymbolGraph,
    symbol_lookup: SymbolTrie,
    plugins: Sequence[EdgePlugin],
    project_root: Path,
    import_resolver: ImportResolver,
    search_paths: list[Path],
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]],
) -> None:
    """Merge ``contrib``'s raw nodes / edges into ``target_graph``,
    stitch cross-package imports against ``symbol_lookup``, and run
    plugin :meth:`EdgePlugin.finalize` against the composed graph.

    The caller owns ``symbol_lookup`` because its construction depends
    on which dep export tries are in scope -- the full-graph path
    merges every dep's exports, while the closure-scoped path merges
    only deps inside the requested scope. ``import_resolver`` +
    ``search_paths`` reach :func:`resolve_edges` for the trie-miss
    classification path (stdlib / external dist / external file /
    unresolved); they are unused when every import resolves
    first-party in the trie.

    ``emitted`` is owned by :meth:`Analysis._materialize` so the dedup
    window spans every package in one compose pass -- cross-package
    duplicates (e.g. two packages re-exporting the same external)
    collapse to one edge instead of accumulating as parallel
    multigraph edges.
    """
    for node in contrib.nodes:
        target_graph.add(node)
    for src, dst, flags in contrib.edges:
        if _claim_edge(emitted, src, dst, flags):
            target_graph.add_edge(src, dst, flags)
    for src, dst, flags in resolve_edges(
        contrib.import_edges,
        symbol_lookup,
        contrib.package.path,
        emitted,
        import_resolver=import_resolver,
        search_paths=search_paths,
    ):
        target_graph.add_edge(src, dst, flags)
    if plugins:
        ctx = PluginContext(
            graph=target_graph,
            symbol_lookup=symbol_lookup,
            contribution=contrib,
            project_root=project_root,
        )
        for plugin in plugins:
            if not isinstance(plugin, EdgePlugin):
                raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
            # Materialize before applying so plugins can iterate
            # ctx.graph.nodes without tripping "dictionary changed
            # size during iteration".
            ops = list(plugin.finalize(ctx))
            apply_ops(target_graph, ops, emitted)


_NON_DECL_TYPES: frozenset[str] = frozenset({"module", "synthetic"})


def _entrypoint_seeds(graph: SymbolGraph, exclude_flags: NodeFlags = NodeFlags.NONE) -> list[int]:
    """Indices of nodes carrying :data:`NodeFlags.ENTRYPOINT`, optionally minus ``exclude_flags``.

    The default seed source for :func:`_find_reachable`. Returns
    rustworkx node indices, not :class:`SymbolNode`\\s -- the BFS in
    :func:`_find_reachable` runs in index space, and going through
    :class:`SymbolNode` and back is a wasted round-trip. Passing
    ``exclude_flags`` drops any entrypoint whose flags intersect it --
    the building block for :func:`_find_kept_alive_by_flags_only`.
    """
    return [
        graph.index(n)
        for n in graph.nodes
        if n.flags & NodeFlags.ENTRYPOINT and not (n.flags & exclude_flags)
    ]


def _find_reachable(
    graph: SymbolGraph,
    seeds: Iterable[int],
    *,
    prefix: Path | None = None,
    skip_dead_branches: bool = False,
) -> set[SymbolNode]:
    """BFS forward from ``seeds`` (rustworkx node indices).

    ``skip_dead_branches=True`` filters :data:`EdgeFlags.DEAD_BRANCH`
    edges from traversal -- the diff against the default traversal is
    the "blast radius" of removing every statically-dead suite.

    ``prefix`` filters the returned set to nodes whose ``path`` lies
    under it; the BFS still traverses the full graph, so transitive
    reachability through nodes outside ``prefix`` is preserved.
    """
    raw = graph.raw
    visited_idx: set[int] = set()
    stack: list[int] = list(seeds)
    while stack:
        i = stack.pop()
        if i in visited_idx:
            continue
        visited_idx.add(i)
        if skip_dead_branches:
            for _, dst_i, payload in raw.out_edges(i):
                if not (payload & EdgeFlags.DEAD_BRANCH):
                    stack.append(dst_i)
        else:
            stack.extend(raw.successor_indices(i))
    visited = {raw[i] for i in visited_idx}
    if prefix is None:
        return visited
    return {n for n in visited if n.path.is_relative_to(prefix)}


def _find_kept_alive_by_dead_branches(
    graph: SymbolGraph, *, prefix: Path | None = None
) -> set[SymbolNode]:
    """Symbols kept alive only via at least one ``DEAD_BRANCH`` edge.

    ``_find_reachable(graph) -`` strict-mode BFS that skips every edge
    flagged :data:`EdgeFlags.DEAD_BRANCH`; the difference is the
    "blast radius" of removing every statically-dead suite. Surfaced
    on :class:`Analysis` as :meth:`Analysis.kept_alive_by_dead_branches`
    and on :class:`PackageView` as
    :meth:`PackageView.kept_alive_by_dead_branches`.
    """
    seeds = _entrypoint_seeds(graph)
    return _find_reachable(graph, seeds, prefix=prefix) - _find_reachable(
        graph, seeds, prefix=prefix, skip_dead_branches=True
    )


def _find_kept_alive_by_flags_only(
    graph: SymbolGraph, flags: NodeFlags, *, prefix: Path | None = None
) -> set[SymbolNode]:
    """Symbols reachable only from entrypoints carrying any of ``flags``.

    Diff between full reachability and reachability with every
    ``flags``-tagged entrypoint dropped -- the "blast radius" of
    removing those entrypoints. Surfaced on :class:`Analysis` and
    :class:`PackageView` as ``kept_alive_by_flags_only(flags)``.
    """
    all_seeds = _entrypoint_seeds(graph)
    kept_seeds = [i for i in all_seeds if not (graph.node(i).flags & flags)]
    return _find_reachable(graph, all_seeds, prefix=prefix) - _find_reachable(
        graph, kept_seeds, prefix=prefix
    )


def _iter_dead(graph: SymbolGraph, *, prefix: Path | None = None) -> Iterator[SymbolNode]:
    """Yield every decl in ``graph`` no entrypoint reaches.

    Excludes ``module`` and ``synthetic`` nodes (see
    :data:`_NON_DECL_TYPES`). ``prefix`` restricts the iteration to
    nodes whose ``path`` lies under it.
    """
    reachable = _find_reachable(graph, _entrypoint_seeds(graph))
    for n in graph.nodes:
        if prefix is not None and not n.path.is_relative_to(prefix):
            continue
        if n.type in _NON_DECL_TYPES:
            continue
        if n not in reachable:
            yield n


def _count_nodes(nodes: Iterable[SymbolNode], prefix: Path | None) -> dict[str, int]:
    """Count ``nodes`` by ``SymbolNode.type``, optionally restricted by path.

    If ``prefix`` is given, only nodes whose ``path`` is under ``prefix``
    are counted. Includes the synthetic ``"synthetic"`` type contributed
    by plugins and third-party-dep markers.
    """
    counts: dict[str, int] = {}
    for node in nodes:
        if prefix and not node.path.is_relative_to(prefix):
            continue
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts


def _count_nodes_by_prefix(
    nodes: Iterable[SymbolNode], prefixes: Sequence[Path]
) -> dict[Path, dict[str, int]]:
    """One-pass equivalent of :func:`_count_nodes` for many prefixes.

    A naive ``[_count_nodes(nodes, p) for p in prefixes]`` re-walks
    every node of the full graph for every prefix; the CLI's text/JSON
    output paths do this twice (once for the full graph, once for the
    unreachable subgraph) and it dominates report-formatting time on
    large workspaces. We bucket nodes by ``node.path`` first so each
    unique file pays the prefix-matching cost once regardless of how
    many declarations live in it.

    Semantics match :func:`_count_nodes`: a node contributes to every
    prefix it ``is_relative_to`` -- nested prefixes both pick it up.
    """
    by_path: dict[Path, dict[str, int]] = {}
    for node in nodes:
        bucket = by_path.get(node.path)
        if bucket is None:
            bucket = {}
            by_path[node.path] = bucket
        bucket[node.type] = bucket.get(node.type, 0) + 1

    result: dict[Path, dict[str, int]] = {p: {} for p in prefixes}
    for path, bucket in by_path.items():
        for prefix in prefixes:
            if not path.is_relative_to(prefix):
                continue
            target = result[prefix]
            for kind, count in bucket.items():
                target[kind] = target.get(kind, 0) + count
    return result


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
    :meth:`PackageView.declarations` and :meth:`PackageView.count_nodes`
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
        # Per-package because ``Package.exported`` enters the fingerprint.
        self._fingerprints: dict[Path, str] = {
            p.path: compute_fingerprint(
                plugins=self._plugins,
                unreachable_detector=self._detector,
                package=p,
            )
            for p in validated
        }
        self._package_files: dict[Path, PackageFiles] = {}
        self._contributions: dict[Path, PackageContribution] = {}
        self._closure_graphs: dict[Path, SymbolGraph] = {}
        self._full_graph: SymbolGraph | None = None
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
                    self._packages_by_path[path], self._cache, self._fingerprints[path]
                )

        pending = {p: self._package_files[p] for p in new_targets}
        tasks = build_stale_tasks(pending, self._project_root, self._fingerprints)
        miss_payloads = process_stale_files(
            tasks=tasks,
            detector=self._detector,
            plugins=self._plugins,
            cache=self._cache,
            workers=self._workers,
        )

        for path in new_targets:
            pf = self._package_files[path]
            self._contributions[path] = build_contribution(
                self._packages_by_path[path],
                pf,
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

    def materialize_all(self) -> SymbolGraph:
        """Build the full graph (every package, cross-package resolution, plugins).

        Memoized: the second call returns the same graph object.
        Refreshes every package first, so this is also the trigger
        for a whole-project cache refresh in callers that don't refresh
        explicitly.
        """
        if self._full_graph is None:
            self.refresh()
            self._full_graph = self._materialize(included=frozenset(p.path for p in self.packages))
        return self._full_graph

    def materialize_closure(self, package: Path) -> SymbolGraph:
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
            included = self._interesting_set(package)
            self.refresh(packages=included)
            self._closure_graphs[package] = self._materialize(included=included)
        return self._closure_graphs[package]

    def _materialize(
        self,
        *,
        included: frozenset[Path],
    ) -> SymbolGraph:
        """Compose every package in ``included`` into a fresh graph.

        Caller is responsible for having :meth:`refresh`'d every
        package in ``included`` first; ``_interesting_set`` is closed
        under transitive deps, so passing one of those (or the full
        package set for ``materialize_all``) is enough.

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

        g = SymbolGraph()
        baseline = list(sys.path)
        last_search_paths: tuple[Path, ...] | None = None
        target_paths = [p.path for p in self.packages if p.path in included]
        # One dedup set spans the whole compose pass so cross-package
        # duplicates (e.g. two packages both importing the same external
        # dist) collapse to one edge. ``_compose_contribution`` populates
        # it from contribution edges, ``resolve_edges`` (called through
        # the same path), and plugin ``AddEdge`` ops.
        emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = set()
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
                    # Only the first-party prefix moved -- dist caches in
                    # ``_imports`` are keyed on the site-packages slice of
                    # ``sys.path`` and survive the transition for free, so
                    # we just refresh the fullname-keyed module-spec cache.
                    clear_module_specs_cache()
                    last_search_paths = search_paths
                _compose_contribution(
                    self._contributions[path],
                    target_graph=g,
                    symbol_lookup=self._build_symbol_lookup(path),
                    plugins=self._plugins,
                    project_root=self._project_root,
                    import_resolver=self._import_resolver,
                    search_paths=list(search_paths),
                    emitted=emitted,
                )
        finally:
            sys.path[:] = baseline
        return g

    def reachable(self) -> set[SymbolNode]:
        """Set of every decl reachable from any entrypoint in the full graph."""
        g = self.materialize_all()
        return _find_reachable(g, _entrypoint_seeds(g))

    def dead(self) -> Iterator[SymbolNode]:
        """Yield every decl that no entrypoint reaches.

        Excludes ``module`` and ``synthetic`` nodes -- modules stay
        alive as long as anything they contain is alive (handled via
        the parent-module edge), and synthetic nodes are analyzer
        plumbing rather than user-visible decls.
        """
        return _iter_dead(self.materialize_all())

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
        return _find_kept_alive_by_dead_branches(self.materialize_all())

    def kept_alive_by_flags_only(self, flags: NodeFlags) -> set[SymbolNode]:
        """Symbols reachable only from entrypoints carrying any of ``flags``.

        Opt-in "blast radius" query: the diff between full reachability
        and reachability with those entrypoints dropped. Pass
        :data:`NodeFlags.TESTCASE` for "what dies if the test suite
        goes", :data:`NodeFlags.NOQA` for "what dies if every
        ``# noqa: F401`` pin is removed", or any OR-combination. See
        :class:`NodeFlags` for the full list.
        """
        return _find_kept_alive_by_flags_only(self.materialize_all(), flags)

    def dead_suites(self) -> Mapping[Path, tuple[CodeRange, ...]]:
        """Statically-dead suite positions, merged across every package.

        Triggers :meth:`refresh` on first call so the contributions are
        populated. Each key is the absolute file path; the value is the
        tuple of dead-suite :class:`CodeRange`s the detector recorded
        for that file. Files with no dead suites are omitted.
        """
        self.refresh()
        merged: dict[Path, tuple[CodeRange, ...]] = {}
        for contrib in self._contributions.values():
            merged.update(contrib.dead_suites)
        return merged

    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]:
        """Count nodes in the full graph by ``SymbolNode.type``.

        ``prefix=None`` (the default) counts every node. Pass a
        package path to scope the count to nodes whose ``path`` is
        under that prefix -- useful for per-package summaries when
        several packages are analysed together.
        """
        return _count_nodes(self.materialize_all().nodes, prefix)

    def _build_symbol_lookup(self, package: Path) -> SymbolTrie:
        """Per-package lookup trie: this package's full trie + each dep's exports.

        ``merge`` pulls in every entry from this package's own trie
        (the package sees itself fully). ``merge_exported`` filters
        each dep's trie to entries flagged :data:`NodeFlags.EXPORTED`,
        so dep-internal decls stay invisible to the consumer.

        Deps must already be refreshed; both
        :meth:`materialize_all` and :meth:`materialize_closure`
        guarantee this because :meth:`_interesting_set` is closed
        under transitive deps.
        """
        contrib = self._contributions[package]
        lookup = SymbolTrie()
        lookup.merge(contrib.trie)
        for dep in self._dep_paths(package):
            lookup.merge_exported(self._contributions[dep].trie)
        return lookup


class PackageView:
    """Lazy view of a single package inside an :class:`Analysis`.

    Cheap to construct (returned by :meth:`Analysis.package`); query
    methods trigger only the work their result depends on. Local
    queries (:meth:`declarations`, :meth:`count_nodes`) only need this
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

    def declarations(self, name: str | None = None) -> Iterator[SymbolNode]:
        """Top-level decls in this package.

        ``name=None`` yields every decl. Pass a string to filter to
        decls whose rightmost dotted segment matches it (``"Foo"``
        matches ``pkg.mod.Foo`` but not ``pkg.Foo.bar``). Local-only.
        """
        for n in self._contribution().nodes:
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

        Stdlib imports (``[stdlib] <target>``) are not surfaced as
        synthetic nodes by the resolver, so this method cannot match
        on them.
        """
        package_path = self._package.path
        graph = self._analysis.materialize_closure(package_path)
        symbol_lookup = self._analysis._build_symbol_lookup(package_path)

        # First-party module lookup goes through the trie; on miss,
        # scan synthetic markers (one pass over graph.nodes covering all
        # three prefixes).
        trie_node = symbol_lookup._get(target.split("."))
        target_node: SymbolNode | None = trie_node.module if trie_node else None
        if target_node is None:
            wanted = {f"{prefix}{target}" for prefix in SYNTHETIC_PATH_PREFIXES}
            for n in graph.nodes:
                if n.type == "synthetic" and n.fqname in wanted:
                    target_node = n
                    break
        if target_node is None:
            return set()

        # Exclude same-file predecessors -- for a first-party module
        # node, every decl inside the module is a predecessor (via the
        # standard ``decl -> module`` edge), but we want *importers* of
        # the module, not its contents.
        target_path = target_node.path
        raw = graph.raw
        result: set[Path] = set()
        for i in raw.predecessor_indices(graph.index(target_node)):
            pred_path = raw[i].path
            if pred_path != target_path and pred_path.is_relative_to(package_path):
                result.add(pred_path)
        return result

    def graph(self) -> SymbolGraph:
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
        return _find_reachable(g, _entrypoint_seeds(g), prefix=self._package.path)

    def dead(self) -> Iterator[SymbolNode]:
        """Yield decls in this package not reachable from any entrypoint
        in :meth:`reverse_closure`.

        Triggers closure materialization on first call; subsequent
        calls reuse the cached graph. Excludes ``module`` and
        ``synthetic`` nodes (see :meth:`Analysis.dead`).
        """
        g = self._analysis.materialize_closure(self._package.path)
        return _iter_dead(g, prefix=self._package.path)

    def kept_alive_by_dead_branches(self) -> set[SymbolNode]:
        """Decls in this package kept alive only by dead-branch references.

        Closure-scoped equivalent of :meth:`Analysis.kept_alive_by_dead_branches`,
        filtered to nodes under :attr:`path`.
        """
        g = self._analysis.materialize_closure(self._package.path)
        return _find_kept_alive_by_dead_branches(g, prefix=self._package.path)

    def kept_alive_by_flags_only(self, flags: NodeFlags) -> set[SymbolNode]:
        """Decls in this package kept alive only by entrypoints carrying any of ``flags``.

        Closure-scoped equivalent of :meth:`Analysis.kept_alive_by_flags_only`,
        filtered to nodes under :attr:`path`. See that method for the
        common flag arguments (``TESTCASE``, ``NOQA``, or both).
        """
        g = self._analysis.materialize_closure(self._package.path)
        return _find_kept_alive_by_flags_only(g, flags, prefix=self._package.path)

    def dead_suites(self) -> Mapping[Path, tuple[CodeRange, ...]]:
        """Statically-dead suite positions for files in this package.

        Local-only: reads straight from this package's
        :class:`PackageContribution` (triggers a refresh of just this
        package). Files with no dead suites are omitted.
        """
        return dict(self._contribution().dead_suites)

    def count_nodes(self) -> dict[str, int]:
        """Count nodes contributed by this package, by ``SymbolNode.type``.

        Local-only: doesn't materialize the closure. Counts include
        ``module``, source decls, and any ``synthetic`` nodes plugins
        emitted into this package's contribution during ``observe``.
        """
        return _count_nodes(self._contribution().nodes, prefix=None)

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
        reachable = _find_reachable(g, _entrypoint_seeds(g))
        dead_nodes = [n for n in g.nodes if n not in reachable]
        remove_code(g.subgraph(dead_nodes), self._package.path)


__all__ = [
    "Analysis",
    "PackageView",
]
