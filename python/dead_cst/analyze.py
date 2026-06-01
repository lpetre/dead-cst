from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, TypedDict

from .graph import KEEPALIVE_DEFAULT, EdgeFlags

if TYPE_CHECKING:
    from dead_cst import _native as native


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
_DECL_KINDS: tuple[str, ...] = ("function", "class", "variable", "import", "type_alias")


def _iter_dead_indices(
    ctx: native.ProjectContext,
    reachable: set[int],
) -> Iterator[int]:
    """Yield positional indices of every decl-kind node not in
    ``reachable``. ``module`` / ``synthetic`` nodes are filtered out
    via the kind whitelist (synthetic markers, anchor modules — they
    aren't "dead" in any meaningful sense).
    """
    for kind in _DECL_KINDS:
        for idx in ctx.indices_where(kind=kind):
            if idx not in reachable:
                yield idx


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
        plugins: Sequence[native.NativePlugin] = (),
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
        self._plugins: tuple[native.NativePlugin, ...] = tuple(plugins)
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

        # Pre-graph plugin hook. Plugins may scan ``project_root`` for
        # config files / framework manifests / etc. before any graph
        # construction happens. Type-validate each plugin here so a
        # bare dict or a ``Pluign()`` typo fails with a clean
        # ``TypeError`` instead of being silently dropped by the rust
        # ``add_plugin`` loop below.
        for plugin in self._plugins:
            if not isinstance(plugin, _native.NativePlugin):
                raise TypeError(
                    f"Expected a dead_cst._native.NativePlugin instance, "
                    f"got {type(plugin).__name__!r}: {plugin!r}"
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

    def re_materialize(self, events: Sequence[native.ChangeEvent]) -> native.ProjectContext:
        """Incrementally re-run materialize against the existing ctx.

        ``events`` is the list of :class:`native.ChangeEvent`\\s to
        apply before rebuilding. Build it from any source you trust:
        an LSP integration consuming ``didChangeWatchedFiles``, a
        file-watcher that emits ``Changed`` / ``Created`` / ``Deleted``,
        or — when you have no precise list — a single
        ``ChangeEvent.rescan()`` (or the equivalent
        ``ctx.detect_changes()`` helper, which is the same call).

        :meth:`native.ProjectContext.apply_changes` forwards the events
        to ty's ``ProjectDatabase::apply_changes``, which only bumps
        salsa revisions for files whose mtime / size actually changed,
        registers ``Created`` paths, drops ``Deleted`` ones, and on
        ``Rescan`` re-walks the project and reloads metadata. Salsa's
        per-file cache for unaffected files survives, so unchanged
        files skip parsing / ``file_to_nodes`` / ``file_to_edges`` /
        ``file_to_ref_edges`` recomputation; cross-file importers
        invalidate transitively through salsa's auto-tracked reads.
        The assemble pass and plugin pass run unconditionally — cheap
        O(N) walks over the warm cache. Plugin ``prepare`` is *not*
        re-run; plugins are assumed unchanged across the lifetime of
        the :class:`Analysis`.

        Returns the same (now-rebuilt) :class:`native.ProjectContext`
        instance that :meth:`materialize_all` returned. Callers that
        cached :class:`SymbolNode` objects from the previous build
        must re-fetch them — node identities are rebuilt by the
        assemble pass.

        Raises :class:`RuntimeError` if :meth:`materialize_all` hasn't
        been called yet.
        """
        if self._ctx is None:
            raise RuntimeError(
                "re_materialize() requires a prior materialize_all() call to construct the ctx"
            )
        self._ctx.apply_changes(list(events))
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
            plugin_names = tuple(p.name for p in self._plugins)
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
        #   ``len(plugins) <= 1``.
        # * **Python-driven loop** — ``ctx.build_only()`` runs the
        #   build pass; Python then drives plugin ``run(ctx)`` calls
        #   itself through a :class:`ThreadPoolExecutor` (so
        #   GIL-releasing rust queries overlap).
        #
        # Both paths satisfy the *frozen-graph* contract: every
        # plugin's ``run(ctx)`` observes the same base-graph state.
        # Ops collected from each plugin land in registration order
        # via a single end-of-pass :meth:`apply_ops_batched` call —
        # a plugin's own emissions are invisible to its own queries,
        # and to every other plugin's queries, until the apply pass
        # runs.
        try:
            use_rust_serial = _serial_mode() or len(self._plugins) <= 1
            if use_rust_serial:
                for plugin in self._plugins:
                    ctx.add_plugin(plugin)
                ctx.materialize()
            else:
                # Register every plugin before ``build_only`` so the
                # build can fold per-file native plugins inline — it
                # reads the registered set to find their ids, warms the
                # per-file ops in the parallel fan-out, and replays them
                # during assembly. Project-wide plugins still run below
                # via the executor; per-file plugins no-op there since
                # they're already applied.
                for plugin in self._plugins:
                    ctx.add_plugin(plugin)
                ctx.build_only()

                workers = 1 if _serial_mode() else _plugin_worker_count(len(self._plugins))

                plugin_names = [p.name for p in self._plugins]
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
                        futures = [
                            pool.submit(_run_collect, idx, p) for idx, p in enumerate(self._plugins)
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

    def reachable(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> set[int]:
        """Positional indices of every decl reachable from any seed in
        ``seed_flags``. Pair with :meth:`native.ProjectContext.nodes_at`
        / :meth:`native.ProjectContext.node_attrs` to materialise the
        underlying ``SymbolNode``\\ s or batched attribute snapshots
        on demand.
        """
        ctx = self.materialize_all()
        return set(ctx.reachable_indices(seed_flags=seed_flags))

    def dead(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> Iterator[int]:
        """Yield positional indices of every decl that no seed in
        ``seed_flags`` reaches. Skips ``module`` and ``synthetic``
        nodes (markers/anchors, not "dead" in the actionable sense).
        """
        ctx = self.materialize_all()
        return _iter_dead_indices(ctx, self.reachable(seed_flags=seed_flags))

    def descendants(self, root_idx: int, *, skip_flags: int = 0) -> list[int]:
        """Forward closure from ``root_idx`` (rust BFS, single FFI
        hop). Returns positional indices into
        :meth:`native.ProjectContext.nodes`.
        """
        ctx = self.materialize_all()
        return list(ctx.descendants_indices(root_idx, skip_flags=skip_flags))

    def ancestors(self, decl_idx: int, *, skip_flags: int = 0) -> list[int]:
        """Reverse closure into ``decl_idx`` (rust BFS, single FFI
        hop). Returns positional indices into
        :meth:`native.ProjectContext.nodes`.
        """
        ctx = self.materialize_all()
        return list(ctx.ancestors_indices(decl_idx, skip_flags=skip_flags))

    def kept_alive_by_dead_branches(self, *, seed_flags: int = KEEPALIVE_DEFAULT) -> set[int]:
        """Indices reachable only via ``EdgeFlags.DEAD_BRANCH`` edges."""
        ctx = self.materialize_all()
        full = set(ctx.reachable_indices(seed_flags=seed_flags))
        strict = set(ctx.reachable_indices(seed_flags=seed_flags, skip_flags=EdgeFlags.DEAD_BRANCH))
        return full - strict

    def kept_alive_by_flags_only(
        self, flags: int, *, seed_flags: int = KEEPALIVE_DEFAULT
    ) -> set[int]:
        """Blast radius of dropping every seed whose flags carry any
        bit in ``flags`` — the diff between
        ``reachable_indices(seed_flags)`` and
        ``reachable_indices(seed_flags & ~flags)``."""
        ctx = self.materialize_all()
        full = set(ctx.reachable_indices(seed_flags=seed_flags))
        without = set(ctx.reachable_indices(seed_flags=seed_flags & ~flags))
        return full - without


__all__ = [
    "Analysis",
    "PROGRESS_PHASES",
    "PROGRESS_POLL_INTERVAL_S",
    "ProgressCallback",
]
