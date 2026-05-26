from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, TypedDict

from .graph import KEEPALIVE_DEFAULT, EdgeFlags, SymbolNode

if TYPE_CHECKING:
    from dead_cst import _native as native

    from .plugins import Plugin


#: Public, ordered list of progress phases. The polling thread fires
#: ``phase_start`` / ``phase_progress`` / ``phase_end`` events keyed on
#: these names; callbacks can use the order to drive a multi-bar UI or
#: a single rolling progress line. Order matches the rust pipeline.
PROGRESS_PHASES: tuple[str, ...] = ("enum", "populate", "assemble", "fqname", "plugins")

# Discriminants the rust side stores in ``ProgressCounters.phase``.
# Mirror ``src/progress.rs``'s ``PHASE_*`` constants; the polling
# thread maps these back to ``PROGRESS_PHASES`` names.
_PHASE_NONE = 0
_PHASE_ID_TO_NAME: dict[int, str] = {
    1: "enum",
    2: "populate",
    3: "assemble",
    4: "fqname",
    5: "plugins",
}

#: Polling interval (seconds) for the progress thread. ~100 ms is
#: small enough that a UI bar feels smooth and large enough that we
#: don't drown the user's callback in events for sub-second builds.
PROGRESS_POLL_INTERVAL_S: float = 0.1

ProgressEvent = str
ProgressCallback = Callable[..., None]


class ProgressSnapshot(TypedDict):
    """Shape returned by :meth:`native.ProjectContext.read_progress_snapshot`
    and :meth:`native.ProgressHandle.snapshot`.

    Every counter field is an integer in either microseconds (the
    ``*_elapsed_us`` keys) or item count (``*_done`` / ``*_total``).
    The ``phase`` discriminant maps to one of
    :data:`PROGRESS_PHASES` via :data:`_PHASE_ID_TO_NAME`.
    ``plugin_states`` carries one ``(name, started_us, finished_us)``
    triple per registered plugin in registration order; both
    timestamps are ``0`` until stamped (``started_us == 0`` ⇔ not
    yet running, ``finished_us == 0`` ⇔ still running).
    """

    phase: int
    finished: bool
    enum_done: int
    enum_total: int
    enum_elapsed_us: int
    populate_done: int
    populate_total: int
    populate_elapsed_us: int
    assemble_done: int
    assemble_total: int
    assemble_elapsed_us: int
    fqname_done: int
    fqname_total: int
    fqname_elapsed_us: int
    plugins_done: int
    plugins_total: int
    plugins_elapsed_us: int
    plugin_states: list[tuple[str, int, int]]


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


def _safe_invoke_callback(cb: ProgressCallback, event: str, **kwargs: Any) -> None:
    """Invoke the user callback, swallowing-and-warning on any exception.

    A buggy callback must never deadlock the build — the rust pipeline
    is oblivious to the polling thread, but a raise that propagates
    out of the polling thread's loop would silently kill it, leaving
    later events on the floor. Warn once per exception with
    :func:`warnings.warn` (stacklevel pointing past the helper) and
    keep polling.
    """
    try:
        cb(event, **kwargs)
    except Exception as exc:  # noqa: BLE001 — see docstring
        warnings.warn(
            f"dead-cst progress_callback raised {type(exc).__name__}: {exc!r} "
            f"on event {event!r}; suppressing and continuing",
            RuntimeWarning,
            stacklevel=2,
        )


class _ProgressPoller:
    """Polling thread driver that translates rust-side atomic counters
    into structured Python callback events.

    Lifecycle: created in :meth:`Analysis.materialize_all` right before
    the rust build; ``start()`` spawns a daemon thread that reads a
    counter snapshot every :data:`PROGRESS_POLL_INTERVAL_S` seconds.
    On each tick the poller fires ``phase_start`` / ``phase_progress``
    / ``phase_end`` events for any phase that advanced. When the rust
    side stamps ``finished=True`` (or :meth:`stop` is called) the
    thread flushes any pending ``*_end`` events and exits.

    The poller calls into rust via
    :meth:`native.ProjectContext.read_progress_snapshot`, which is a
    plain atomic-load → dict conversion. The user callback runs on
    the poller's thread, *not* on the main thread — callbacks that
    touch shared state need their own synchronisation.
    """

    def __init__(
        self,
        ctx: native.ProjectContext,
        callback: ProgressCallback,
        interval_s: float = PROGRESS_POLL_INTERVAL_S,
        *,
        plugin_names: Sequence[str] = (),
    ) -> None:
        self._ctx = ctx
        # Pre-mint the borrow-free progress handle so polls during a
        # long ``materialize`` call don't race with the context's
        # ``borrow_mut`` token. Created up-front (outside the build)
        # so the poller doesn't have to re-enter the context lock.
        self._handle: Any = ctx.progress_handle()
        self._cb = callback
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-phase running state: which phases have we started /
        # ended so far?  Built as we go so a phase that skips
        # forward (e.g. enum -> assemble) still fires the missing
        # start/end pair.
        self._started: set[str] = set()
        self._ended: set[str] = set()
        # Last-seen ``*_done`` count per phase. Lets us emit a final
        # ``phase_progress`` flush when the phase ends, so callbacks
        # that gate UI on ``current == total`` see the terminal state.
        self._last_progress: dict[str, int] = {}
        # Plugin start/finish bookkeeping. ``plugins_done`` is an
        # atomic counter on the rust side; the poller infers which
        # plugin index just finished by diffing snapshots.
        self._plugin_names: tuple[str, ...] = tuple(plugin_names)
        self._plugins_signalled_start: set[int] = set()
        self._plugins_signalled_end: set[int] = set()
        self._plugins_phase_total: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="dead-cst-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the polling thread to exit, join it, then run one
        final dispatch on the caller's thread. Idempotent.

        The final dispatch is what makes sub-100 ms builds work: the
        rust pipeline can finish before the polling thread completes
        its first 100 ms wait, so without an explicit drain at stop
        time the callback would never see any events. Joining the
        poller first keeps the dispatch single-threaded; ``_dispatch``
        is otherwise lock-free.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Final drain on the caller's thread. The rust side has
        # already stamped ``finished=True`` (since materialize_all
        # exits before stop() is called), so this dispatch picks up
        # every phase the poller missed because of timing.
        try:
            snap = self._handle.snapshot()
        except Exception:  # noqa: BLE001
            return
        self._dispatch(snap)

    def _run(self) -> None:
        # Loop until the rust side flips ``finished`` or ``stop()``
        # is called. We read one final snapshot after the rust side
        # signals done so any trailing increments (e.g. the last
        # populate file or the final fqname node) show up before we
        # emit ``phase_end``.
        finished_seen = False
        while True:
            try:
                snap = self._handle.snapshot()
            except Exception as exc:  # noqa: BLE001
                # If the rust side disappears (ctx GC'd, native crash),
                # warn and exit. Don't try to fire more events.
                warnings.warn(
                    f"dead-cst progress poller lost ctx: {type(exc).__name__}: {exc!r}",
                    RuntimeWarning,
                    stacklevel=1,
                )
                return
            self._dispatch(snap)
            if finished_seen:
                return
            if snap["finished"]:
                # One more iteration so we catch any final increments
                # that landed between the last poll and the rust-side
                # ``mark_finished``.
                finished_seen = True
                continue
            if self._stop.wait(self._interval):
                # External stop — the ``stop()`` caller will do a
                # final drain dispatch on its own thread; we exit
                # without one here to avoid racing against it.
                return

    def _dispatch(self, snap: ProgressSnapshot) -> None:
        # Map snapshot fields to per-phase (done, total, elapsed_us).
        phase_stats: dict[str, tuple[int, int, int]] = {
            "enum": (
                snap["enum_done"],
                snap["enum_total"],
                snap["enum_elapsed_us"],
            ),
            "populate": (
                snap["populate_done"],
                snap["populate_total"],
                snap["populate_elapsed_us"],
            ),
            "assemble": (
                snap["assemble_done"],
                snap["assemble_total"],
                snap["assemble_elapsed_us"],
            ),
            "fqname": (
                snap["fqname_done"],
                snap["fqname_total"],
                snap["fqname_elapsed_us"],
            ),
            "plugins": (
                snap["plugins_done"],
                snap["plugins_total"],
                snap["plugins_elapsed_us"],
            ),
        }
        current_phase_id = snap["phase"]

        for name in PROGRESS_PHASES:
            done, total, elapsed_us = phase_stats[name]
            phase_id = next(
                (pid for pid, n in _PHASE_ID_TO_NAME.items() if n == name),
                _PHASE_NONE,
            )
            already_started = name in self._started
            already_ended = name in self._ended

            # ``phase_start`` fires when the rust side advances past
            # this phase's ID, or when this phase is the current one,
            # or (defensively) when there's any progress data for a
            # not-yet-started phase. ``finished=True`` with non-zero
            # totals forces us to flush any phases we never observed
            # mid-flight.
            should_start = (
                not already_started
                and (
                    current_phase_id > phase_id
                    or current_phase_id == phase_id
                    or done > 0
                    or total > 0
                    or (snap["finished"] and (total > 0 or elapsed_us > 0))
                )
                # Only fire ``phase_start`` once the rust side has
                # actually begun the phase. Zero everything = no
                # signal yet.
                and (current_phase_id >= phase_id or done > 0 or total > 0)
            )
            if should_start:
                self._started.add(name)
                _safe_invoke_callback(
                    self._cb,
                    "phase_start",
                    phase=name,
                    total=(total if total > 0 else None),
                )
                if name == "plugins":
                    self._plugins_phase_total = total if total > 0 else None

            if name in self._started and not already_ended:
                last = self._last_progress.get(name, -1)
                if done != last:
                    self._last_progress[name] = done
                    _safe_invoke_callback(
                        self._cb,
                        "phase_progress",
                        phase=name,
                        current=done,
                        total=(total if total > 0 else None),
                    )

                # Plugin start/end synthesis. The rust side
                # publishes a per-plugin slot snapshot — one
                # ``(name, started_us, finished_us)`` triple per
                # registered plugin, in registration order. The
                # poller diffs them against ``_plugin_states_seen``
                # so each transition fires exactly one event.
                if name == "plugins":
                    self._emit_plugin_events_from_slots(snap, total)

            # ``phase_end`` fires when the rust side has moved past
            # this phase (current_phase_id > phase_id), or when the
            # whole pipeline is marked finished. ``elapsed_us`` of
            # zero is fine — the rust side guarantees a finish_phase
            # write before stamping the next phase.
            should_end = (
                name in self._started
                and not already_ended
                and (current_phase_id > phase_id or snap["finished"])
            )
            if should_end:
                self._ended.add(name)
                # If we never reported total=done, do one last flush
                # before emitting ``phase_end``. Idempotency-safe;
                # callbacks should already handle ``current == total``.
                if total > 0 and self._last_progress.get(name) != total:
                    self._last_progress[name] = total
                    _safe_invoke_callback(
                        self._cb,
                        "phase_progress",
                        phase=name,
                        current=total,
                        total=total,
                    )
                _safe_invoke_callback(
                    self._cb,
                    "phase_end",
                    phase=name,
                    elapsed_ms=elapsed_us // 1000,
                )

    def _emit_plugin_events_from_slots(self, snap: ProgressSnapshot, total: int) -> None:
        """Diff the rust per-plugin slot snapshot against
        ``_plugins_signalled_{start,end}`` and fire one event per
        observed transition.

        Each slot is a ``(name, started_us, finished_us)`` triple:

        * ``started_us > 0`` and ``idx`` not in signalled-start →
          fire ``plugin_start(name=..., index=idx, total=...)``.
        * ``finished_us > 0`` and ``idx`` not in signalled-end →
          fire ``plugin_end(name=..., elapsed_ms=...)`` with the
          per-plugin elapsed time computed from the slot pair.

        Slot order is registration order, so ``idx`` is meaningful;
        completion order across slots can interleave freely under the
        parallel ``ThreadPoolExecutor`` pass — each slot writes only
        to its own atomic so attribution is exact.
        """
        states = snap["plugin_states"]
        plugin_total = total if total > 0 else len(states)
        for idx, (name, started_us, finished_us) in enumerate(states):
            if started_us > 0 and idx not in self._plugins_signalled_start:
                self._plugins_signalled_start.add(idx)
                _safe_invoke_callback(
                    self._cb,
                    "plugin_start",
                    name=name,
                    index=idx,
                    total=plugin_total,
                )
            if finished_us > 0 and idx not in self._plugins_signalled_end:
                self._plugins_signalled_end.add(idx)
                elapsed_us = finished_us - started_us if finished_us >= started_us else 0
                _safe_invoke_callback(
                    self._cb,
                    "plugin_end",
                    name=name,
                    ops_emitted=0,
                    elapsed_ms=elapsed_us // 1000,
                )


class _DispatchShim:
    """Internal: wraps a :class:`DispatchAppPlugin` so the harness can
    drive its ``policy(ctx, gathered)`` through the same threaded
    fan-out it uses for every other plugin.

    The shim is two-phase: at construction time it holds a slot index
    into a shared fused-gather result and a ``Future`` that the worker
    thread will block on when ``run(ctx)`` is called. The gather runs
    as its own task in the same :class:`ThreadPoolExecutor`, so it
    overlaps with every non-dispatch plugin instead of serializing
    before them on the main thread.

    Inactive dispatch plugins (``_is_active(ctx)`` returned ``False``
    upstream) carry ``slot=None`` and ``gather_future=None``; their
    ``run(ctx)`` is a no-op so progress accounting stays
    one-shim-per-plugin.

    Used only by :meth:`Analysis.materialize_all`'s dispatch-batching
    path; never appears in user code.
    """

    name = "DispatchShim"
    version = 1

    def __init__(
        self,
        plugin: Any,
        slot: int | None,
        gather_future: Any | None,
    ) -> None:
        self._plugin = plugin
        self._slot = slot
        self._gather_future = gather_future

    def prepare(self, project_root: Path) -> None:
        pass

    def run(self, ctx: native.ProjectContext) -> Iterator[Any]:
        if self._slot is None or self._gather_future is None:
            return iter(())
        # Block until the gather worker has populated every slot. The
        # gather releases the GIL inside every rust query it issues
        # (``find_subclasses_indices``, ``find_constructions``, …),
        # so non-dispatch plugins in sibling workers overlap with it.
        gathered_list = self._gather_future.result()
        gathered = gathered_list[self._slot]
        if gathered is None:
            return iter(())
        return iter(self._plugin.policy(ctx, gathered))


def _build_dispatch_schedule(
    ctx: native.ProjectContext,
    plugins: Sequence[Any],
    gather_future: Any | None,
    active_slot_by_id: dict[int, int],
) -> list[Any]:
    """Return a per-plugin work list with every
    :class:`DispatchAppPlugin` replaced by a :class:`_DispatchShim`
    bound to its slot in ``gather_future``.

    Position-preserving: ``len(out) == len(plugins)``. Non-dispatch
    plugins pass through unchanged. Dispatch plugins not in
    ``active_slot_by_id`` (because ``_is_active(ctx)`` returned
    ``False``) get a no-op shim so progress accounting stays
    one-shim-per-plugin.
    """
    from .plugins.decl_shapes import DispatchAppPlugin

    return [
        _DispatchShim(p, active_slot_by_id.get(id(p)), gather_future)
        if isinstance(p, DispatchAppPlugin)
        else p
        for p in plugins
    ]


def _make_indicatif_callback() -> ProgressCallback:
    """Default progress callback for ``show_progress=True``.

    Logs each phase transition to stderr in a fixed-width format —
    not as fancy as the indicatif multi-bar setup, but it's pure
    Python and visible everywhere indicatif is (TTYs, redirected
    stderr, CI logs). Indicatif itself is still running on the rust
    side; this callback complements rather than replaces it.
    """
    import sys

    def _cb(event: str, **kwargs: Any) -> None:
        if event == "phase_start":
            phase = kwargs.get("phase", "?")
            total = kwargs.get("total")
            tag = f"  {phase:<10}"
            if total is None:
                print(f"[dead-cst] {tag} start", file=sys.stderr)
            else:
                print(f"[dead-cst] {tag} start (total={total})", file=sys.stderr)
        elif event == "phase_end":
            phase = kwargs.get("phase", "?")
            elapsed_ms = kwargs.get("elapsed_ms", 0)
            tag = f"  {phase:<10}"
            print(f"[dead-cst] {tag} done in {elapsed_ms} ms", file=sys.stderr)
        # phase_progress / plugin_* are deliberately swallowed: too
        # noisy for the default stderr-text fallback. Users wanting a
        # bar should plug indicatif's *Python* binding (or rich's
        # Progress) into their own callback.

    return _cb


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
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if progress_callback is not None and show_progress:
            raise ValueError(
                "Pass either show_progress=True or progress_callback=..., "
                "not both. show_progress=True installs a default callback."
            )
        self._project_root: Path = project_root
        self._venv: Path | None = venv
        self._plugins: tuple[Plugin, ...] = tuple(plugins)
        self._show_progress: bool = show_progress
        # ``show_progress=True`` installs a default stderr-text
        # callback so the rust-side indicatif bars get a Python-side
        # echo. The two paths don't conflict — rust prints to its
        # MultiProgress draw target, Python prints per-phase headers.
        self._progress_callback: ProgressCallback | None = progress_callback or (
            _make_indicatif_callback() if show_progress else None
        )
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

        self._drive_build(ctx)
        self._ctx = ctx
        return ctx

    def re_materialize(self, dirty_files: Sequence[Path | str]) -> native.ProjectContext:
        """Incrementally re-run materialize against the existing ctx.

        Notifies salsa that ``dirty_files`` have changed (via
        :meth:`native.ProjectContext.sync_paths`), then re-drives the
        build + plugin pipeline on the same :class:`native.ProjectContext`.
        Salsa's per-file cache survives across calls, so unchanged files
        skip parsing / ``file_to_nodes`` / ``file_to_edges`` /
        ``file_to_ref_edges`` recomputation; cross-file importers
        invalidate transitively through salsa's auto-tracked reads.

        The assemble pass and plugin pass still run unconditionally —
        they're cheap O(N) walks over a warm salsa cache. Plugin
        ``prepare`` is *not* re-run; plugins are assumed unchanged
        across the lifetime of the :class:`Analysis`.

        Returns the same (now-rebuilt) :class:`native.ProjectContext`
        instance that :meth:`materialize_all` returned. Callers that
        cached :class:`SymbolNode` objects from the previous build must
        re-fetch them — node identities are rebuilt by the assemble
        pass.

        Raises :class:`RuntimeError` if :meth:`materialize_all` hasn't
        been called yet (there's no ctx to re-build).
        """
        if self._ctx is None:
            raise RuntimeError(
                "re_materialize() requires a prior materialize_all() call to construct the ctx"
            )
        self._ctx.sync_paths([str(p) for p in dirty_files])
        self._ctx.reset_progress()
        self._drive_build(self._ctx)
        return self._ctx

    def _drive_build(self, ctx: native.ProjectContext) -> None:
        """Run the build + plugin pipeline on an already-constructed ctx.

        Factored out of :meth:`materialize_all` so
        :meth:`re_materialize` can drive a second build on the same
        ctx without duplicating the dispatch / progress / plugin-pool
        plumbing. Plugin ``prepare`` is *not* invoked here — that's a
        one-shot hook owned by :meth:`materialize_all`.
        """
        # The rust-serial path below calls ``ctx.add_plugin(p)`` per
        # plugin, which appends. Clear first so a re_materialize doesn't
        # stack a second copy of every plugin on top of the first run's
        # registrations.
        ctx.clear_plugins()

        # Spin up the progress polling thread if the user asked for
        # callbacks. The thread reads rust-side atomic counters at
        # ~100 ms and fires structured events. Joined in a ``finally``
        # so a build failure still drains the queue.
        poller: _ProgressPoller | None = None
        if self._progress_callback is not None:
            plugin_names = tuple(type(p).__qualname__ for p in self._plugins)
            poller = _ProgressPoller(
                ctx,
                self._progress_callback,
                plugin_names=plugin_names,
            )
            poller.start()

        # The plugin loop has two modes:
        #
        # * **Rust-side serial loop** — ``ctx.materialize()`` drives the
        #   build pass and every registered plugin's ``run(ctx)`` in
        #   one C call. Used when ``DEAD_CST_PLUGINS_SERIAL=1`` or
        #   ``len(plugins) <= 1`` AND no plugin needs a between-build-
        #   and-run hook (the dispatch-batching path does).
        # * **Python-driven loop** — ``ctx.build_only()`` runs the
        #   build pass; Python then drives plugin ``run(ctx)`` calls
        #   itself, either through a :class:`ThreadPoolExecutor` (so
        #   GIL-releasing rust queries overlap) or serially when
        #   ``DEAD_CST_PLUGINS_SERIAL=1``. When any registered plugin
        #   is a :class:`DispatchAppPlugin`, the harness submits one
        #   shared ``_gather_batched`` task into the same pool and
        #   binds each dispatch plugin's :class:`_DispatchShim` to its
        #   slot in the gather's future. The gather overlaps with
        #   every non-dispatch plugin in sibling workers (its rust
        #   queries release the GIL); the dispatch shims block on the
        #   future inside their own ``run(ctx)``, so they only consume
        #   CPU once the gather is ready.
        #
        # Both paths satisfy the *frozen-graph* contract: every
        # plugin's ``run(ctx)`` observes the same base-graph state.
        # Ops collected from each plugin land in registration order
        # via a single end-of-pass :meth:`apply_ops_batched` call —
        # a plugin's own emissions are invisible to its own queries,
        # and to every other plugin's queries, until the apply pass
        # runs.
        try:
            from .plugins.decl_shapes import DispatchAppPlugin, _gather_batched

            has_dispatch = any(isinstance(p, DispatchAppPlugin) for p in self._plugins)
            use_rust_serial = not has_dispatch and (_serial_mode() or len(self._plugins) <= 1)
            if use_rust_serial:
                for plugin in self._plugins:
                    ctx.add_plugin(plugin)
                ctx.materialize()
            else:
                ctx.build_only()

                # Identify active dispatch plugins on the main thread
                # (the ``_is_active`` probe is cheap — just an
                # ``ImportQuery.exists()`` per app module). Build the
                # spec list + slot map now, but defer the actual gather
                # to a pool worker below.
                active_specs: list[Any] = []
                active_slot_by_id: dict[int, int] = {}
                if has_dispatch:
                    for p in self._plugins:
                        if isinstance(p, DispatchAppPlugin) and p._is_active(ctx):
                            active_slot_by_id[id(p)] = len(active_specs)
                            active_specs.append(p.spec)

                # Worker count: regular plugins + one reserved slot
                # for the gather task when active. The +1 keeps the
                # gather from starving the non-dispatch plugins it's
                # meant to overlap with — without it, a dispatch shim
                # blocked on the gather future would occupy a worker
                # the gather itself wants.
                n_work = len(self._plugins) + (1 if active_specs else 0)
                workers = 1 if _serial_mode() else _plugin_worker_count(n_work)

                # Progress names track the *original* plugins so users see
                # ``FlaskPlugin`` etc. — shims are an implementation detail.
                plugin_names = [type(p).__qualname__ for p in self._plugins]
                ctx.progress_plugins_start(plugin_names)
                try:
                    # Plugins are scheduled in registration order;
                    # ``.result()`` waits in the same order. Errors
                    # propagate via the future. Each worker collects
                    # its plugin's ops into a :class:`CollectedOps`
                    # handle (no graph mutation); the main thread
                    # folds every handle into the graph in registration
                    # order via one ``apply_ops_batched`` call.
                    #
                    # The per-plugin ``progress_plugin_started`` /
                    # ``progress_plugin_finished`` stamps live in this
                    # worker so the rust per-plugin counter slabs see
                    # the actual completion order (the polling thread
                    # uses those to attribute ``plugin_end`` events to
                    # the right plugin even when futures resolve out of
                    # registration order).
                    def _run_collect(idx: int, plugin: Any):
                        ctx.progress_plugin_started(idx)
                        try:
                            return ctx.run_plugin_collect(plugin)
                        finally:
                            ctx.progress_plugin_finished(idx)
                            ctx.progress_plugin_done()

                    with ThreadPoolExecutor(
                        max_workers=workers,
                        thread_name_prefix="dead-cst-plugin",
                    ) as pool:
                        # Submit the gather first so it has a head
                        # start on the per-plugin tasks below — the
                        # gather is single-purpose work (every active
                        # dispatch plugin needs its slice), so it
                        # gets exclusive use of the +1 worker.
                        gather_future: Any | None = (
                            pool.submit(_gather_batched, ctx, active_specs)
                            if active_specs
                            else None
                        )
                        scheduled = (
                            _build_dispatch_schedule(
                                ctx, self._plugins, gather_future, active_slot_by_id
                            )
                            if has_dispatch
                            else list(self._plugins)
                        )
                        futures = [
                            pool.submit(_run_collect, idx, p) for idx, p in enumerate(scheduled)
                        ]
                        collected = [fut.result() for fut in futures]
                    ctx.apply_ops_batched(collected)
                finally:
                    ctx.progress_plugins_finish()
        except BaseException:
            # Make sure the polling thread doesn't outlive a failed
            # build. ``mark_progress_finished`` flips the rust-side
            # atomic so the poller exits on its next tick.
            ctx.mark_progress_finished()
            raise
        finally:
            if poller is not None:
                poller.stop()

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
    "PROGRESS_PHASES",
    "PROGRESS_POLL_INTERVAL_S",
    "ProgressCallback",
]
