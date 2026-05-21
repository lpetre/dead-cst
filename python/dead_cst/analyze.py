from __future__ import annotations
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence

from .graph import KEEPALIVE_DEFAULT, EdgeFlags, SymbolNode
from .resolvers import Package, PathResolver
from .resolvers._core import _validate_packages

if TYPE_CHECKING:
    from dead_cst import _native as native

    from .plugins import Plugin


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


def _iter_dead(
    ctx: native.ProjectContext,
    reachable: set[SymbolNode],
) -> Iterator[SymbolNode]:
    for n in ctx.nodes():
        if n.kind in _NON_DECL_TYPES:
            continue
        if n not in reachable:
            yield n


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
        plugins: Sequence[Plugin] = (),
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
        self._plugins: tuple[Plugin, ...] = tuple(plugins)
        self._show_progress: bool = show_progress
        # Held past ``materialize_all`` so the rust BFS queries
        # (:meth:`reachable`, :meth:`dead`, :meth:`descendants`,
        # :meth:`ancestors`) and node/edge enumeration can run against
        # the live context without re-building the project graph.
        self._ctx: native.ProjectContext | None = None
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

    def materialize_all(self) -> native.ProjectContext:
        """Build the project-wide graph (memoized).

        Uses ty's ``SemanticIndex`` (via :mod:`dead_cst._native`) to
        assemble the graph in one pass. Each registered plugin's
        ``run(ctx)`` is invoked during this pass.

        Returns the live :class:`native.ProjectContext`. Bulk
        reachability queries on the analysis (:meth:`reachable`,
        :meth:`dead`, :meth:`descendants`, :meth:`ancestors`) delegate
        to its rust BFS; ``ctx.nodes()`` / ``ctx.edges()`` enumerate
        the graph without copying into a Python adjacency list.
        """
        if self._ctx is not None:
            return self._ctx
        from dead_cst import _native

        from .plugins import Plugin

        src_roots = [str(e) for p in self.packages for e in p.exported_paths]
        owned_paths = [str(p.path) for p in self.packages]
        # Per-package env roots. Order matters: package's own
        # ``exported_paths`` first (longest-match priority for fqname
        # derivation), then ``path`` (catch-all for non-exported owned
        # files like src-layout ``tests/``), then deps' ``exported_paths``
        # (so cross-package imports resolve into deps but not non-deps).
        env_roots_per_package: list[list[str]] = []
        for p in self.packages:
            dep_exports = (
                str(e)
                for dep_path in self._dep_paths_by_package.get(p.path, ())
                for e in self._packages_by_path[dep_path].exported_paths
            )
            entries = [*(str(e) for e in p.exported_paths), str(p.path), *dep_exports]
            env_roots_per_package.append(list(dict.fromkeys(entries)))
        ctx = _native.ProjectContext(
            str(self._project_root),
            src_roots=src_roots or None,
            package_owned_paths=owned_paths or None,
            package_env_roots=env_roots_per_package or None,
            show_progress=self._show_progress,
        )
        for plugin in self._plugins:
            # Catch ``Pluign()`` typos and bare dicts before the rust
            # side silently drops them.
            if not isinstance(plugin, Plugin):
                raise TypeError(
                    f"Expected a dead_cst.plugins.Plugin instance, got "
                    f"{type(plugin).__name__!r}: {plugin!r}"
                )
            ctx.add_plugin(plugin)
        ctx.materialize()
        self._ctx = ctx
        return ctx

    def reachable(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> set[SymbolNode]:
        """Set of every decl reachable from any seed in ``seed_flags``.

        ``seed_flags`` defaults to :data:`KEEPALIVE_DEFAULT` (every
        keepalive bit ORed together). Pass a subset to scope the
        question -- e.g. ``seed_flags=NodeFlags.ENTRYPOINT`` excludes
        tests, ``noqa``-pinned imports, and notebooks from the seed set.

        Delegates to :meth:`native.ProjectContext.reachable` — one FFI
        hop, no Python-side adjacency walk.
        """
        ctx = self.materialize_all()
        return set(ctx.reachable(seed_flags=seed_flags))

    def dead(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> Iterator[SymbolNode]:
        """Yield every decl that no seed in ``seed_flags`` reaches."""
        ctx = self.materialize_all()
        return _iter_dead(ctx, self.reachable(seed_flags=seed_flags))

    def descendants(self, root: SymbolNode, *, skip_flags: int = 0) -> list[SymbolNode]:
        """Forward closure from ``root`` (rust BFS, single FFI hop)."""
        ctx = self.materialize_all()
        return list(ctx.descendants(root, skip_flags=skip_flags))

    def ancestors(self, decl: SymbolNode, *, skip_flags: int = 0) -> list[SymbolNode]:
        """Reverse closure into ``decl`` (rust BFS, single FFI hop)."""
        ctx = self.materialize_all()
        return list(ctx.ancestors(decl, skip_flags=skip_flags))

    def kept_alive_by_dead_branches(
        self, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[SymbolNode]:
        """Decls reachable only via ``EdgeFlags.DEAD_BRANCH`` edges.

        Diff of the default closure against the strict closure that
        skips dead-branch edges — same BFS path both queries take in
        rust.
        """
        ctx = self.materialize_all()
        full = set(ctx.reachable(seed_flags=seed_flags))
        strict = set(ctx.reachable(seed_flags=seed_flags, skip_flags=EdgeFlags.DEAD_BRANCH))
        return full - strict

    def kept_alive_by_flags_only(
        self, flags: int, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[SymbolNode]:
        """Blast radius of dropping every seed whose flags carry any
        bit in ``flags`` — the diff between ``reachable(seed_flags)``
        and ``reachable(seed_flags & ~flags)``."""
        ctx = self.materialize_all()
        full = set(ctx.reachable(seed_flags=seed_flags))
        without = set(ctx.reachable(seed_flags=seed_flags & ~flags))
        return full - without


__all__ = [
    "Analysis",
]
