"""Coverage for :class:`Analysis`'s structured progress-callback API.

Counters live on the rust side as relaxed atomics; a Python polling
thread reads them every ~100 ms and fires events. These tests assert
the event taxonomy, the show_progress mutex, and that callback
exceptions don't deadlock the build.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest

from dead_cst import Analysis
from dead_cst.analyze import PROGRESS_PHASES


def _write(tmp_path: Path, files: dict[str, str]) -> None:
    for name, src in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def _materialize_with_callback(
    tmp_path: Path,
    cb: Any,
    *,
    files: dict[str, str] | None = None,
    **kwargs: Any,
) -> Analysis:
    files = files or {
        "a.py": "def f(): pass",
        "b.py": "from a import f\nf()",
        "c.py": "x = 1\ny = x + 1",
    }
    _write(tmp_path, files)
    analysis = Analysis(tmp_path, progress_callback=cb, **kwargs)
    analysis.materialize_all()
    return analysis


def test_progress_callback_emits_full_event_sequence(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def cb(event: str, **kwargs: Any) -> None:
        events.append((event, kwargs))

    _materialize_with_callback(tmp_path, cb)

    # Every phase should bracket: start ... progress* ... end.
    seen = {name: {"start": False, "end": False} for name in PROGRESS_PHASES}
    for event, kwargs in events:
        if event == "phase_start":
            assert kwargs["phase"] in PROGRESS_PHASES
            seen[kwargs["phase"]]["start"] = True
        elif event == "phase_end":
            assert kwargs["phase"] in PROGRESS_PHASES
            assert "elapsed_ms" in kwargs
            seen[kwargs["phase"]]["end"] = True
        elif event == "phase_progress":
            assert kwargs["phase"] in PROGRESS_PHASES
            assert "current" in kwargs
            assert "total" in kwargs

    # The plugin phase fires even when zero plugins were registered —
    # rust still stamps it to make the event taxonomy uniform.
    for name in ("enum", "populate", "assemble", "fqname", "plugins"):
        assert seen[name]["start"], f"missing phase_start for {name!r}"
        assert seen[name]["end"], f"missing phase_end for {name!r}"


def test_progress_callback_phase_ordering(tmp_path: Path) -> None:
    """phase_start events fire in the documented pipeline order."""
    starts: list[str] = []

    def cb(event: str, **kwargs: Any) -> None:
        if event == "phase_start":
            starts.append(kwargs["phase"])

    _materialize_with_callback(tmp_path, cb)

    # Filter to the canonical phase list (in case a phase repeated,
    # which it shouldn't).
    seen_order = [p for p in starts if p in PROGRESS_PHASES]
    # Each phase starts at most once.
    assert len(set(seen_order)) == len(seen_order)
    # Order matches PROGRESS_PHASES.
    indices = [PROGRESS_PHASES.index(p) for p in seen_order]
    assert indices == sorted(indices)


def test_phase_progress_carries_count_and_total(tmp_path: Path) -> None:
    """phase_progress fires with sensible counts and (where known) totals."""
    progresses: list[dict[str, Any]] = []

    def cb(event: str, **kwargs: Any) -> None:
        if event == "phase_progress":
            progresses.append(kwargs)

    _materialize_with_callback(tmp_path, cb)

    # Populate / assemble / fqname / plugins always carry a non-None total.
    # enum reports total once the scan completes.
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for p in progresses:
        by_phase.setdefault(p["phase"], []).append(p)

    for name in ("populate", "assemble", "fqname"):
        events = by_phase.get(name, [])
        assert events, f"no phase_progress events for {name!r}"
        # current monotonically non-decreasing.
        currents = [e["current"] for e in events]
        assert currents == sorted(currents)
        # final event reaches the total.
        last = events[-1]
        assert last["current"] == last["total"]


def test_show_progress_and_callback_are_mutually_exclusive(tmp_path: Path) -> None:
    def cb(event: str, **kwargs: Any) -> None:
        pass

    with pytest.raises(ValueError, match="show_progress=True or progress_callback"):
        Analysis(tmp_path, progress_callback=cb, show_progress=True)


def test_show_progress_installs_default_callback(tmp_path: Path) -> None:
    """show_progress=True should silently install a stderr-text default
    callback; passing it shouldn't raise and should still drive the
    build to completion.
    """
    _write(tmp_path, {"a.py": "x = 1"})
    analysis = Analysis(tmp_path, show_progress=True)
    ctx = analysis.materialize_all()
    # Sanity check the build actually ran.
    nodes = list(ctx.nodes())
    assert nodes


def test_callback_exception_does_not_deadlock(tmp_path: Path) -> None:
    """If the callback raises on every event, the polling thread must
    swallow the exception (via warnings.warn) and keep going so the
    build finishes.
    """
    saw_call: list[str] = []

    def cb(event: str, **kwargs: Any) -> None:
        saw_call.append(event)
        raise RuntimeError("oh no")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analysis = _materialize_with_callback(tmp_path, cb)
        ctx = analysis.materialize_all()  # idempotent — no re-run.

    # The build completed.
    assert list(ctx.nodes())
    # The callback was definitely called at least once.
    assert saw_call
    # At least one RuntimeWarning surfaced for the swallowed raise.
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "progress_callback raised RuntimeError" in str(w.message)
        for w in caught
    )


def test_progress_callback_with_concurrent_plugins(tmp_path: Path) -> None:
    """Concurrent plugin pass (>1 plugin) emits plugin_start / plugin_end
    for each plugin via the Python-side ThreadPoolExecutor wrapper.
    """
    from dead_cst.plugins import MainBlockPlugin, ModuleDundersPlugin

    events: list[tuple[str, dict[str, Any]]] = []

    def cb(event: str, **kwargs: Any) -> None:
        events.append((event, kwargs))

    _write(tmp_path, {"a.py": "def f(): pass\nf()"})
    analysis = Analysis(
        tmp_path,
        plugins=[
            MainBlockPlugin(),
            ModuleDundersPlugin(),
        ],
        progress_callback=cb,
    )
    analysis.materialize_all()

    plugin_starts = [k for (e, k) in events if e == "plugin_start"]
    plugin_ends = [k for (e, k) in events if e == "plugin_end"]
    assert len(plugin_starts) == 2
    assert len(plugin_ends) == 2
    names = [k["name"] for k in plugin_starts]
    # The two plugin class qualnames should both have surfaced.
    assert "MainBlockPlugin" in names
    assert "ModuleDundersPlugin" in names
    # With per-plugin counter slabs, each ``plugin_end`` carries the
    # plugin's actual name (not the registration-order approximation
    # the old global-counter path used) and a real elapsed_ms.
    end_names = [k["name"] for k in plugin_ends]
    assert sorted(end_names) == sorted(names)
    # Each plugin's slot index in ``plugin_start`` matches its
    # registration order.
    indices = sorted(k["index"] for k in plugin_starts)
    assert indices == [0, 1]


def test_progress_snapshot_dict_shape(tmp_path: Path) -> None:
    """``read_progress_snapshot`` exposes the documented integer keys."""
    _write(tmp_path, {"a.py": "x = 1"})
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()
    snap = ctx.read_progress_snapshot()
    expected_keys = {
        "phase",
        "finished",
        "enum_done",
        "enum_total",
        "enum_elapsed_us",
        "populate_done",
        "populate_total",
        "populate_elapsed_us",
        "assemble_done",
        "assemble_total",
        "assemble_elapsed_us",
        "fqname_done",
        "fqname_total",
        "fqname_elapsed_us",
        "plugins_done",
        "plugins_total",
        "plugins_elapsed_us",
    }
    assert expected_keys.issubset(snap.keys())
    assert snap["finished"] is True
    assert snap["enum_total"] >= 1
