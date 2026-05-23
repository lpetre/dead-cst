from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

from .graph import KEEPALIVE_DEFAULT, EdgeFlags, SymbolNode

if TYPE_CHECKING:
    from dead_cst import _native as native

    from .plugins import Plugin


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


class Analysis:
    """Lazy entrypoint to the dead-cst pipeline.

    Builds the project's symbol graph via the rust backend
    (:mod:`dead_cst._native`), which uses ty's ``SemanticIndex`` to
    resolve every cross-file reference in one pass.

    The caller is responsible for setting up a venv (or supplying a
    pre-existing one) with editable ``.pth`` entries pointing at
    each first-party member's published source dir. ty reads those
    ``.pth`` files when walking ``site-packages`` and uses them as
    additional module-resolution search paths, which is how
    ``from libx import foo`` correctly resolves to
    ``packages/libx/src/libx/__init__.py`` (and how the file at
    that path correctly mounts as module ``libx`` rather than
    ``packages.libx.src.libx``). For uv workspaces,
    ``uv sync --all-packages`` produces exactly this layout.

    With no ``venv`` argument, ty's auto-discovery picks up the
    project root as the first-party search path -- fine for
    single-package projects but ignores any multi-member layout
    info that a venv would have encoded.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        venv: Path | None = None,
        plugins: Sequence[Plugin] = (),
        show_progress: bool = False,
    ) -> None:
        self._project_root: Path = project_root
        self._venv: Path | None = venv
        self._plugins: tuple[Plugin, ...] = tuple(plugins)
        self._show_progress: bool = show_progress
        # Buffered until ``materialize_all`` constructs the ctx.
        # ``None`` means "use the rust-side default" (32 MiB at the
        # time of writing — see ``DEFAULT_RAYON_STACK_SIZE`` in the
        # rust crate).
        self._stack_size: int | None = None
        # Held past ``materialize_all`` so the rust BFS queries
        # (:meth:`reachable`, :meth:`dead`, :meth:`descendants`,
        # :meth:`ancestors`) and node/edge enumeration can run against
        # the live context without re-building the project graph.
        self._ctx: native.ProjectContext | None = None

    def set_stack_size(self, bytes_: int) -> None:
        """Override the rayon worker stack size (bytes) used by the
        populate phase. Call BEFORE :meth:`materialize_all`; calls
        after the graph is materialized have no effect on the
        already-built graph.

        Defaults to 32 MiB on the rust side, which is generous for
        typical Python code. Raise this if you see a stack overflow
        on deeply-nested generated code (e.g. protobuf modules,
        ML-generated ASTs, or large nested literal dicts) — the
        declared size is virtual address space on Linux, so going
        much higher costs no resident memory unless actually used.
        """
        if bytes_ <= 0:
            raise ValueError(f"stack_size must be > 0, got {bytes_}")
        self._stack_size = bytes_

    @property
    def project_root(self) -> Path:
        return self._project_root

    def materialize_all(self) -> native.ProjectContext:
        """Build the project-wide graph (memoized).

        Returns the live :class:`native.ProjectContext`. Bulk
        reachability queries on the analysis delegate to the rust
        BFS; ``ctx.nodes()`` / ``ctx.edges()`` enumerate the graph
        without copying into a Python adjacency list.
        """
        if self._ctx is not None:
            return self._ctx
        from dead_cst import _native

        from .plugins import Plugin

        # When a venv is provided, suppress ty's auto-discovery of
        # ``env.root`` (which would put ``project_root`` in static
        # search paths, shadowing the .pth-derived per-member paths
        # that come via ``python_env``). With ``env.root = []``,
        # static_paths is empty; ty falls through to
        # ``dynamic_resolution_paths`` (site-packages + .pth) for
        # first-party resolution.
        venv_str = str(self._venv) if self._venv is not None else None
        suppress_autodetect = self._venv is not None
        ctx = _native.ProjectContext(
            str(self._project_root),
            src_roots=[] if suppress_autodetect else None,
            python_env=venv_str,
            show_progress=self._show_progress,
        )
        if self._stack_size is not None:
            ctx.set_stack_size(self._stack_size)
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
        """Set of every decl reachable from any seed in ``seed_flags``."""
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
        """Decls reachable only via ``EdgeFlags.DEAD_BRANCH`` edges."""
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
