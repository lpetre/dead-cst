from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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


def _serial_mode() -> bool:
    """``True`` if ``DEAD_CST_PLUGINS_SERIAL=1`` is set in the env.

    Kill switch for the concurrent plugin executor — falls back to
    the rust-side serial loop. Useful for debugging plugin races,
    flaky CI environments, or comparing serial vs parallel timings.
    """
    return os.environ.get("DEAD_CST_PLUGINS_SERIAL", "") == "1"


def _plugin_worker_count(n_plugins: int) -> int:
    """Worker count for the plugin :class:`ThreadPoolExecutor`.

    Resolves to ``DEAD_CST_PLUGIN_WORKERS`` if set (clamped to
    ``[1, n_plugins]``), otherwise ``min(n_plugins, cpu_count or 4)``.
    Never returns more workers than plugins — extra workers just
    idle.
    """
    raw = os.environ.get("DEAD_CST_PLUGIN_WORKERS", "")
    if raw:
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError(f"DEAD_CST_PLUGIN_WORKERS must be an integer, got {raw!r}") from exc
        if requested < 1:
            raise ValueError(f"DEAD_CST_PLUGIN_WORKERS must be >= 1, got {requested}")
        return min(requested, n_plugins)
    return min(n_plugins, os.cpu_count() or 4)


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
        # ``None`` means "no override" — the rust side uses rayon's
        # global pool with rayon's own default stack.
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

        With no override set, the populate phase uses rayon's global
        pool with rayon's own default stack (2 MiB unless
        ``RAYON_STACK_SIZE`` / ``RUST_MIN_STACK`` are set
        process-wide), which is sufficient for typical Python code.
        Call this on projects with deeply-nested generated code
        (e.g. protobuf modules, ML-generated ASTs, or large nested
        literal dicts) that stack-overflow at the default — the
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

        # Pre-graph plugin hook. Plugins may scan ``project_root`` for
        # config files / framework manifests / etc. before any graph
        # construction happens. Type-validate each plugin here so
        # ``Pluign()`` typos and bare dicts fail with a clean
        # ``TypeError`` instead of being silently dropped by the rust
        # ``add_plugin`` loop below.
        for plugin in self._plugins:
            if not isinstance(plugin, Plugin):
                raise TypeError(
                    f"Expected a dead_cst.plugins.Plugin instance, got "
                    f"{type(plugin).__name__!r}: {plugin!r}"
                )
            plugin.prepare(self._project_root)

        # Always keep ``project_root`` in ty's static search paths
        # alongside any ``.pth``-derived dynamic paths from the venv.
        # ``helpers::canonical_module_for_file`` (rust side) does a
        # specificity-aware reverse lookup so a deep ``.pth`` path
        # still wins the fqname for files it covers, while the
        # project root becomes a safe fallback for single-package
        # setups whose editable install uses a PEP 660 ``MetaPathFinder``
        # (no flat ``.pth``) — the case that #222 reproduced as
        # ``[unresolved]`` synthetics.
        venv_str = str(self._venv) if self._venv is not None else None
        ctx = _native.ProjectContext(
            str(self._project_root),
            python_env=venv_str,
            show_progress=self._show_progress,
        )
        if self._stack_size is not None:
            ctx.set_stack_size(self._stack_size)

        # ``DEAD_CST_PLUGINS_SERIAL=1`` (or no plugins / a single
        # plugin) keeps the rust-side serial loop. Otherwise we drive
        # plugins from a ``ThreadPoolExecutor`` so any plugin time
        # spent in GIL-releasing rust queries (``find_decorated_decls``,
        # ``find_subclasses_of_class``, ``find_handler_decorators``, ...)
        # overlaps across workers.
        #
        # Both paths satisfy the *frozen-graph* contract: every
        # plugin's ``run(ctx)`` observes the same base-graph state.
        # Ops collected from each plugin land in registration order
        # via a single end-of-pass :meth:`apply_ops_batched` call —
        # a plugin's own emissions are invisible to its own queries,
        # and to every other plugin's queries, until the apply pass
        # runs.
        if _serial_mode() or len(self._plugins) <= 1:
            for plugin in self._plugins:
                ctx.add_plugin(plugin)
            ctx.materialize()
        else:
            ctx.build_only()
            workers = _plugin_worker_count(len(self._plugins))
            # Plugins are scheduled in registration order; results
            # ``.result()`` waits in the same order. Errors propagate
            # via the future's ``.result()`` call. We collect each
            # plugin's ops into a :class:`CollectedOps` handle off
            # the worker thread, then fold them all into the graph
            # in registration order via one ``apply_ops_batched``
            # call on the main thread.
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="dead-cst-plugin",
            ) as pool:
                futures = [pool.submit(ctx.run_plugin_collect, p) for p in self._plugins]
                collected = [fut.result() for fut in futures]
            ctx.apply_ops_batched(collected)
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
