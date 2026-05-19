from __future__ import annotations

import logging
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from ._graphstore import SymbolGraph
from .graph import KEEPALIVE_DEFAULT, EdgeFlags, SymbolNode
from .resolvers import Package, PathResolver
from .resolvers._core import _validate_packages


def _path_under(node_path: str, prefix: Path) -> bool:
    """Return True if ``node_path`` (the rust-side string) lives under ``prefix``."""
    return Path(node_path).is_relative_to(prefix)


logger = logging.getLogger(__name__)


def _bfs_order(seeds: Iterable[Path], neighbors: Mapping[Path, Sequence[Path]]) -> list[Path]:
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


_NON_DECL_TYPES: frozenset[str] = frozenset({"module", "synthetic"})


def _keepalive_seeds(graph: SymbolGraph, seed_flags: int) -> list[int]:
    """Indices of every node whose flags intersect ``seed_flags``.

    Each bit in ``seed_flags`` is a keepalive criterion -- a node
    carrying any of them is a reachability seed. Pass
    :data:`KEEPALIVE_DEFAULT` for the standard behaviour, or a narrower
    mask to ask focused questions (e.g. ``NodeFlags.ENTRYPOINT`` alone
    excludes tests / noqa pins / notebooks from the seed set).
    """
    return [graph.index(n) for n in graph.nodes if n.flags & seed_flags]


def _find_reachable(
    graph: SymbolGraph,
    seeds: Iterable[int],
    *,
    prefix: Path | None = None,
    skip_dead_branches: bool = False,
) -> set[SymbolNode]:
    visited_idx: set[int] = set()
    stack: list[int] = list(seeds)
    while stack:
        i = stack.pop()
        if i in visited_idx:
            continue
        visited_idx.add(i)
        if skip_dead_branches:
            for _, dst_i, payload in graph.out_edges(i):
                if not (payload & EdgeFlags.DEAD_BRANCH):
                    stack.append(dst_i)
        else:
            stack.extend(graph.successor_indices(i))
    visited = {graph.node(i) for i in visited_idx}
    if prefix is None:
        return visited
    return {n for n in visited if _path_under(n.path, prefix)}


def _find_kept_alive_by_dead_branches(
    graph: SymbolGraph,
    *,
    seed_flags: int = KEEPALIVE_DEFAULT,
    prefix: Path | None = None,
) -> set[SymbolNode]:
    seeds = _keepalive_seeds(graph, seed_flags)
    return _find_reachable(graph, seeds, prefix=prefix) - _find_reachable(
        graph, seeds, prefix=prefix, skip_dead_branches=True
    )


def _find_kept_alive_by_flags_only(
    graph: SymbolGraph,
    flags: int,
    *,
    seed_flags: int = KEEPALIVE_DEFAULT,
    prefix: Path | None = None,
) -> set[SymbolNode]:
    """Diff between ``reachable(seed_flags)`` and
    ``reachable(seed_flags & ~flags)`` -- the blast radius of dropping
    every seed whose flags carry any bit in ``flags``."""
    all_seeds = _keepalive_seeds(graph, seed_flags)
    kept_seeds = _keepalive_seeds(graph, seed_flags & ~flags)
    return _find_reachable(graph, all_seeds, prefix=prefix) - _find_reachable(
        graph, kept_seeds, prefix=prefix
    )


def _iter_dead(
    graph: SymbolGraph,
    *,
    seed_flags: int = KEEPALIVE_DEFAULT,
    prefix: Path | None = None,
) -> Iterator[SymbolNode]:
    reachable = _find_reachable(graph, _keepalive_seeds(graph, seed_flags))
    for n in graph.nodes:
        if prefix is not None and not _path_under(n.path, prefix):
            continue
        if n.kind in _NON_DECL_TYPES:
            continue
        if n not in reachable:
            yield n


def _count_nodes(nodes: Iterable[SymbolNode], prefix: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        if prefix and not _path_under(node.path, prefix):
            continue
        counts[node.kind] = counts.get(node.kind, 0) + 1
    return counts


def _count_nodes_by_prefix(
    nodes: Iterable[SymbolNode], prefixes: Sequence[Path]
) -> dict[Path, dict[str, int]]:
    by_path: dict[Path, dict[str, int]] = {}
    for node in nodes:
        # ``node.path`` is the raw string from the rust side; pre-intern
        # to a Path once per unique value.
        path = Path(node.path)
        bucket = by_path.get(path)
        if bucket is None:
            bucket = {}
            by_path[path] = bucket
        bucket[node.kind] = bucket.get(node.kind, 0) + 1

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

    Builds the project's symbol graph via the rust backend
    (:mod:`dead_cst._native`), which uses ty's ``SemanticIndex`` to
    resolve every cross-file reference in one pass. Construction reads
    the resolver's :meth:`PathResolver.resolve` to enumerate first-party
    packages; no source is read or parsed until you ask a question
    (:meth:`materialize_all`, :meth:`reachable`, :meth:`dead`).
    """

    def __init__(
        self,
        project_root: Path,
        *,
        resolver: PathResolver,
        plugins: Sequence[object] = (),
        show_progress: bool = False,
    ) -> None:
        self._project_root: Path = project_root
        validated = _validate_packages(resolver.resolve(project_root))
        self._packages_by_path: dict[Path, Package] = {p.path: p for p in validated}
        by_name = {p.name: p.path for p in validated}
        self._dep_paths_by_package: dict[Path, tuple[Path, ...]] = {
            p.path: tuple(by_name[d] for d in p.deps) for p in validated
        }
        consumers: dict[Path, list[Path]] = {p.path: [] for p in validated}
        for p in validated:
            for dep_name in p.deps:
                consumers[by_name[dep_name]].append(p.path)
        self._consumers_by_package: dict[Path, tuple[Path, ...]] = {
            path: tuple(sorted(cs)) for path, cs in consumers.items()
        }
        self._plugins: tuple[object, ...] = tuple(plugins)
        self._show_progress: bool = show_progress
        self._full_graph: SymbolGraph | None = None
        self._reverse_closures: dict[Path, frozenset[Path]] = {}

    @property
    def project_root(self) -> Path:
        return self._project_root

    @cached_property
    def packages(self) -> tuple[Package, ...]:
        sorted_paths = sorted(self._packages_by_path)
        seeds = [p for p in sorted_paths if not self._dep_paths_by_package.get(p)]
        order = _bfs_order(seeds, self._consumers_by_package)
        visited = set(order)
        order.extend(p for p in sorted_paths if p not in visited)
        return tuple(self._packages_by_path[p] for p in order)

    def reverse_closure(self, package: Path) -> frozenset[Path]:
        if package not in self._packages_by_path:
            raise KeyError(package)
        cached = self._reverse_closures.get(package)
        if cached is None:
            cached = frozenset(_bfs_order([package], self._consumers_by_package))
            self._reverse_closures[package] = cached
        return cached

    def package(self, path: Path) -> PackageView:
        if path not in self._packages_by_path:
            raise KeyError(path)
        return PackageView(self, self._packages_by_path[path])

    def views(self) -> Iterator[PackageView]:
        for package in self.packages:
            yield PackageView(self, package)

    def materialize_all(self) -> SymbolGraph:
        """Build the project-wide graph (memoized).

        Routes through the rust backend
        (:func:`dead_cst._native.materialize_project`), which uses ty's
        ``SemanticIndex`` to assemble the graph in one pass. Each
        registered plugin's ``run(ctx)`` is invoked during this pass.
        """
        if self._full_graph is not None:
            return self._full_graph
        from ._backend import materialize_project

        # Pass each first-party package's path as a src_root so the
        # rust backend mounts files at the right module fqname
        # (``pkg_a/A/__init__.py`` -> ``A``, not ``pkg_a.A``).
        src_roots = tuple(p.path for p in self.packages)
        self._full_graph = materialize_project(
            self._project_root,
            self._plugins,
            src_roots=src_roots,
            show_progress=self._show_progress,
        )
        return self._full_graph

    def materialize_closure(self, package: Path) -> SymbolGraph:
        """Return the project-wide graph.

        The rust backend builds the whole project at once via ty's
        Salsa db; there is no cheaper per-package closure path. Kept
        for API parity with :class:`PackageView` callers.
        """
        return self.materialize_all()

    def reachable(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> set[SymbolNode]:
        """Set of every decl reachable from any seed in ``seed_flags``.

        ``seed_flags`` defaults to :data:`KEEPALIVE_DEFAULT` (every
        keepalive bit ORed together). Pass a subset to scope the
        question -- e.g. ``seed_flags=NodeFlags.ENTRYPOINT`` excludes
        tests, ``noqa``-pinned imports, and notebooks from the seed set.
        """
        g = self.materialize_all()
        return _find_reachable(g, _keepalive_seeds(g, seed_flags))

    def dead(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> Iterator[SymbolNode]:
        """Yield every decl that no seed in ``seed_flags`` reaches."""
        return _iter_dead(self.materialize_all(), seed_flags=seed_flags)

    def kept_alive_by_dead_branches(
        self, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[SymbolNode]:
        return _find_kept_alive_by_dead_branches(self.materialize_all(), seed_flags=seed_flags)

    def kept_alive_by_flags_only(
        self, flags: int, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[SymbolNode]:
        return _find_kept_alive_by_flags_only(self.materialize_all(), flags, seed_flags=seed_flags)

    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]:
        return _count_nodes(self.materialize_all().nodes, prefix)


class PackageView:
    """Lazy view onto a single package within an :class:`Analysis`."""

    __slots__ = ("_analysis", "_package")

    def __init__(self, analysis: Analysis, package: Package) -> None:
        self._analysis = analysis
        self._package = package

    @property
    def package(self) -> Package:
        return self._package

    @property
    def path(self) -> Path:
        return self._package.path

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def reverse_closure(self) -> frozenset[Path]:
        return self._analysis.reverse_closure(self._package.path)

    def declarations(self, name: str | None = None) -> Iterator[SymbolNode]:
        for n in self._analysis.materialize_all().nodes:
            if not _path_under(n.path, self._package.path):
                continue
            if n.kind in ("module", "synthetic"):
                continue
            if name is not None and n.fqname.rpartition(".")[2] != name:
                continue
            yield n

    def graph(self) -> SymbolGraph:
        return self._analysis.materialize_all()

    def reachable(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> set[SymbolNode]:
        g = self._analysis.materialize_all()
        return _find_reachable(g, _keepalive_seeds(g, seed_flags), prefix=self._package.path)

    def dead(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> Iterator[SymbolNode]:
        return _iter_dead(
            self._analysis.materialize_all(),
            seed_flags=seed_flags,
            prefix=self._package.path,
        )

    def kept_alive_by_dead_branches(
        self, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[SymbolNode]:
        return _find_kept_alive_by_dead_branches(
            self._analysis.materialize_all(),
            seed_flags=seed_flags,
            prefix=self._package.path,
        )

    def kept_alive_by_flags_only(
        self, flags: int, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[SymbolNode]:
        return _find_kept_alive_by_flags_only(
            self._analysis.materialize_all(),
            flags,
            seed_flags=seed_flags,
            prefix=self._package.path,
        )

    def count_nodes(self) -> dict[str, int]:
        return _count_nodes(
            (
                n
                for n in self._analysis.materialize_all().nodes
                if _path_under(n.path, self._package.path)
            ),
            prefix=None,
        )

    def remove_dead_code(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> None:
        from .codemod import remove_code

        g = self._analysis.materialize_all()
        reachable = _find_reachable(g, _keepalive_seeds(g, seed_flags))
        dead_nodes = [
            n for n in g.nodes if n not in reachable and _path_under(n.path, self._package.path)
        ]
        remove_code(g.subgraph(dead_nodes), self._package.path)


__all__ = [
    "Analysis",
    "PackageView",
]
