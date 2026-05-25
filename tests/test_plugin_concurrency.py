"""Concurrent plugin execution sanity checks.

These tests don't measure speed — they prove that the Python-side
:class:`concurrent.futures.ThreadPoolExecutor` driving plugins
actually runs them in parallel, and that the kill-switch env var
falls back to serial. Threading primitives (a :class:`threading.Barrier`)
gate the parallel-claim test so it never flakes on slow CI.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pytest

from dead_cst import Analysis
from dead_cst import _native as native
from dead_cst.plugins import Plugin


class _BarrierPlugin(Plugin):
    """Plugin that ``barrier.wait()``s inside ``run`` and records its
    thread id. Combined with a barrier of two participants, two such
    plugins running on different threads both reach ``wait`` and
    proceed; one stuck on a single-threaded executor would block
    indefinitely (the barrier's timeout surfaces the failure).
    """

    def __init__(self, barrier: threading.Barrier, observed: list[int]) -> None:
        self.barrier = barrier
        self.observed = observed

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
        self.observed.append(threading.get_ident())
        # If we're not actually parallel, the second plugin never
        # reaches ``wait`` (the first never returns) and the barrier
        # times out. The 5 s budget is comfortable on every CI we
        # ship to and small enough that a real serial bug doesn't
        # hang for minutes.
        self.barrier.wait(timeout=5.0)
        return ()


def test_plugins_run_concurrently(tmp_path: Path) -> None:
    """Two plugins on a two-thread barrier finish only if executed in
    parallel — single-threaded execution times out on the barrier."""
    (tmp_path / "a.py").write_text("x = 1\n")

    barrier = threading.Barrier(2)
    observed: list[int] = []
    plugins = [
        _BarrierPlugin(barrier, observed),
        _BarrierPlugin(barrier, observed),
    ]
    Analysis(tmp_path, plugins=plugins).materialize_all()

    assert len(observed) == 2
    assert observed[0] != observed[1], (
        f"plugins ran on the same thread {observed[0]} — executor did not actually parallelize"
    )


def test_serial_env_falls_back(tmp_path: Path) -> None:
    """``DEAD_CST_PLUGINS_SERIAL=1`` skips the executor — the
    barrier-based parallel test would deadlock under serial, so we
    use a thread-id collector instead: same thread means serial."""
    (tmp_path / "a.py").write_text("x = 1\n")

    observed: list[int] = []

    class _RecorderPlugin(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            observed.append(threading.get_ident())
            return ()

    plugins = [_RecorderPlugin(), _RecorderPlugin(), _RecorderPlugin()]
    with patch.dict(os.environ, {"DEAD_CST_PLUGINS_SERIAL": "1"}):
        Analysis(tmp_path, plugins=plugins).materialize_all()

    assert len(observed) == 3
    assert len(set(observed)) == 1, (
        f"serial mode should run all plugins on one thread, got {set(observed)}"
    )


def test_single_plugin_uses_serial_path(tmp_path: Path) -> None:
    """One-plugin Analysis hits the rust-side serial loop directly
    (no executor overhead). Confirmed by the plugin landing on the
    caller's thread."""
    (tmp_path / "a.py").write_text("x = 1\n")

    caller_thread = threading.get_ident()
    observed: list[int] = []

    class _RecorderPlugin(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            observed.append(threading.get_ident())
            return ()

    Analysis(tmp_path, plugins=[_RecorderPlugin()]).materialize_all()

    assert observed == [caller_thread]


@pytest.mark.parametrize("workers", ["1", "2", "8"])
def test_worker_count_env(tmp_path: Path, workers: str) -> None:
    """``DEAD_CST_PLUGIN_WORKERS`` caps the pool size. We just verify
    the call doesn't blow up — actual pool-size enforcement is a
    cpython-internal concern."""
    (tmp_path / "a.py").write_text("x = 1\n")

    class _NoopPlugin(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            return ()

    plugins = [_NoopPlugin(), _NoopPlugin(), _NoopPlugin()]
    with patch.dict(os.environ, {"DEAD_CST_PLUGIN_WORKERS": workers}):
        Analysis(tmp_path, plugins=plugins).materialize_all()
