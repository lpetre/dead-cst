"""Tests for the ``workers`` parameter on :func:`build_symbol_graph`.

The serial and parallel paths must produce identical graphs and warm
the cache identically. Workers communicate by pickling
:class:`VisitorPayload` blobs back to the main process; the SQLite
cache write still happens on the main process so failure semantics
match the serial path.
"""

from __future__ import annotations

import signal
import textwrap
from pathlib import Path

import pytest

from dead_cst.cache import (
    CACHE_DIR_NAME,
    GraphCache,
    compute_fingerprint,
)
from dead_cst.graph import VisitorPayload
from dead_cst.plugins import ObserveContext, PluginContext


def _fp() -> str:
    return compute_fingerprint()


class _RaiseOnFilePlugin:
    """EdgePlugin that raises on a single file by name (works under fork *and* spawn)."""

    name = "raise_on_file"
    version = 1

    def __init__(self, *, target_name: str) -> None:
        self.target_name = target_name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RaiseOnFilePlugin) and other.target_name == self.target_name

    def __hash__(self) -> int:
        return hash((self.name, self.target_name))

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        if ctx.path.name == self.target_name:
            raise RuntimeError(f"boom: {ctx.path.name}")
        return None

    def finalize(self, ctx: PluginContext):
        return ()


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def _node_set(graph) -> set[tuple[str, str]]:
    return {(n.fqname, n.type) for n in graph.nodes}


def _edge_set(graph) -> set[tuple[str, str]]:
    return {(s.fqname, d.fqname) for s, d in graph.edges()}


def _multi_file_layout() -> dict[str, str]:
    return {
        "pkg/__init__.py": "",
        "pkg/a.py": """
            def f():
                pass

            def g():
                return f()
        """,
        "pkg/b.py": """
            from .a import g

            def h():
                return g()
        """,
        "pkg/c.py": """
            from .b import h

            def main():
                return h()
        """,
        "pkg/d.py": """
            VALUE = 42

            def lookup():
                return VALUE
        """,
    }


def test_parallel_matches_serial(tmp_path, make_analysis):
    """``workers=2`` and the default serial path return the same graph."""
    _write(tmp_path, _multi_file_layout())
    serial = make_analysis().materialize_all()
    parallel = make_analysis(workers=2).materialize_all()
    assert _node_set(parallel) == _node_set(serial)
    assert _edge_set(parallel) == _edge_set(serial)


def test_parallel_warms_cache(tmp_path, make_analysis):
    """Parallel runs write payloads to the cache so a follow-up run skips workers."""
    _write(tmp_path, _multi_file_layout())
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = _fp()

    with GraphCache(db) as cache:
        cold = make_analysis(cache=cache, workers=2).materialize_all()

    files = sorted(tmp_path.rglob("*.py"))
    with GraphCache(db) as cache:
        for f in files:
            assert cache.get(f, fp) is not None, f"{f} missing from cache after parallel run"

    with GraphCache(db) as cache:
        warm = make_analysis(cache=cache).materialize_all()
    assert _node_set(warm) == _node_set(cold)
    assert _edge_set(warm) == _edge_set(cold)


def test_parallel_falls_back_to_serial_for_single_miss(tmp_path, make_analysis, monkeypatch):
    """``workers=2`` with a single cache-miss skips the pool (overhead floor)."""
    _write(tmp_path, _multi_file_layout())
    db = tmp_path / CACHE_DIR_NAME / "cache.db"

    with GraphCache(db) as cache:
        make_analysis(cache=cache).materialize_all()

    (tmp_path / "pkg" / "a.py").write_text("def f():\n    return 1\n")

    from dead_cst import _refresh

    calls = []
    real = _refresh.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("max_workers"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_refresh, "ProcessPoolExecutor", _spy)
    with GraphCache(db) as cache:
        make_analysis(cache=cache, workers=4).materialize_all()
    assert calls == [], "single-miss run should not spawn a pool"


def test_parallel_pool_capped_at_total_task_count(tmp_path, make_analysis, monkeypatch):
    """The pool caps ``max_workers`` at the total number of cache-miss tasks."""
    _write(tmp_path, _multi_file_layout())

    from dead_cst import _refresh

    seen = []
    real = _refresh.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("max_workers"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_refresh, "ProcessPoolExecutor", _spy)
    make_analysis(workers=64).materialize_all()
    # Five files in _multi_file_layout(); pool should be capped at 5.
    assert seen == [5]


@pytest.mark.parametrize("workers", [None, 1])
def test_workers_none_or_one_keeps_serial_path(tmp_path, make_analysis, monkeypatch, workers):
    """``workers=None`` and ``workers=1`` never spawn a pool."""
    _write(tmp_path, _multi_file_layout())
    from dead_cst import _refresh

    calls = []
    real = _refresh.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(_refresh, "ProcessPoolExecutor", _spy)
    make_analysis(workers=workers).materialize_all()
    assert calls == []


def test_multi_base_uses_one_pool(tmp_path, make_analysis, monkeypatch):
    """Two bases share a single persistent pool, not one per base."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(
        base_a,
        {
            "pkg/__init__.py": "",
            "pkg/x.py": "def x(): pass\n",
            "pkg/y.py": "from .x import x\ndef y(): return x()\n",
        },
    )
    _write(
        base_b,
        {
            "pkg/__init__.py": "",
            "pkg/p.py": "def p(): pass\n",
            "pkg/q.py": "from .p import p\ndef q(): return p()\n",
        },
    )

    from dead_cst import _refresh

    pool_calls = []
    real = _refresh.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        pool_calls.append(kwargs.get("max_workers"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_refresh, "ProcessPoolExecutor", _spy)
    make_analysis(["a", "b"], workers=2).materialize_all()
    assert len(pool_calls) == 1, f"expected exactly one pool across both bases, got {pool_calls!r}"


def test_multi_base_parallel_matches_serial(tmp_path, make_analysis):
    """Multi-base parallel runs produce the same graph as the serial path."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(
        base_a,
        {
            "pkg/__init__.py": "",
            "pkg/x.py": "def x(): pass\n",
            "pkg/y.py": "from .x import x\ndef y(): return x()\n",
        },
    )
    _write(
        base_b,
        {
            "pkg/__init__.py": "",
            "pkg/p.py": "def p(): pass\n",
            "pkg/q.py": "from .p import p\ndef q(): return p()\n",
        },
    )
    serial = make_analysis(["a", "b"]).materialize_all()
    parallel = make_analysis(["a", "b"], workers=2).materialize_all()
    assert _node_set(parallel) == _node_set(serial)
    assert _edge_set(parallel) == _edge_set(serial)


def test_tasks_sorted_by_package(tmp_path, make_analysis, monkeypatch):
    """Worker tasks are submitted ``(package_path, file)``-sorted so each package's files are contiguous.

    The visitor pass no longer touches ``sys.path``, so task order is
    purely an aesthetic / determinism concern -- but keeping
    same-package files adjacent stays readable in logs and stable across
    runs. ``as_completed`` reorders results by completion, so we assert
    on submission order via the ``submit`` spy rather than the consumer
    iterator.
    """
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(
        base_a,
        {
            "pkg/__init__.py": "",
            "pkg/x.py": "def x(): pass\n",
            "pkg/y.py": "def y(): pass\n",
        },
    )
    _write(
        base_b,
        {
            "pkg/__init__.py": "",
            "pkg/p.py": "def p(): pass\n",
            "pkg/q.py": "def q(): pass\n",
        },
    )

    from dead_cst import _refresh

    submitted: list = []
    real_submit = _refresh.ProcessPoolExecutor.submit  # bound on the class

    def _spy_submit(self, fn, *args, **kwargs):
        # First positional arg is the StaleFile task.
        if args:
            submitted.append(args[0].package.path)
        return real_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(_refresh.ProcessPoolExecutor, "submit", _spy_submit)
    make_analysis(["a", "b"], workers=2).materialize_all()

    assert submitted, "expected at least one task submitted to the pool"
    runs = [submitted[0]]
    for p in submitted[1:]:
        if p != runs[-1]:
            runs.append(p)
    assert len(runs) == len(set(runs)), (
        f"expected each package's tasks contiguous, got order {submitted!r}"
    )


@pytest.mark.parametrize("workers", [None, 2])
def test_failures_aggregate_at_end(tmp_path, make_analysis, workers):
    """One task's failure surfaces as an ExceptionGroup; the rest still cache-warm."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/good.py": "def good(): pass\n",
            "pkg/bad.py": "def bad(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"

    with GraphCache(db) as cache:
        with pytest.raises(ExceptionGroup) as excinfo:
            make_analysis(
                cache=cache,
                workers=workers,
                plugins=[_RaiseOnFilePlugin(target_name="bad.py")],
            ).materialize_all()

    group = excinfo.value
    assert "1/" in str(group)
    messages = [str(e) for e in group.exceptions]
    assert any("boom: bad.py" in m for m in messages), messages

    fp = compute_fingerprint(plugins=[_RaiseOnFilePlugin(target_name="bad.py")])
    with GraphCache(db) as cache:
        assert cache.get(tmp_path / "pkg" / "good.py", fp) is not None, (
            "good.py payload should be cache-warmed despite bad.py failing"
        )
        assert cache.get(tmp_path / "pkg" / "bad.py", fp) is None, (
            "bad.py raised, so its payload must not be cached"
        )


@pytest.mark.parametrize("workers", [None, 2])
def test_debug_logs_per_file_status(tmp_path, make_analysis, caplog, workers):
    """At ``DEBUG`` level each completion emits one ``[i/N] ok|FAILED <file>`` record."""
    _write(tmp_path, _multi_file_layout())
    with caplog.at_level("DEBUG", logger="dead_cst._refresh"):
        make_analysis(workers=workers).materialize_all()
    lines = [r.getMessage() for r in caplog.records if r.name == "dead_cst._refresh"]
    progress_lines = [ln for ln in lines if ("] ok " in ln or "] FAILED " in ln)]
    assert len(progress_lines) == 5, progress_lines
    assert all(ln.startswith(f"[{i}/5]") for i, ln in enumerate(progress_lines, 1)), progress_lines


def test_pool_installs_and_restores_signal_handlers(tmp_path, make_analysis, monkeypatch):
    """The pool wraps execution in SIGTERM/SIGINT handlers and restores them on exit."""
    _write(tmp_path, _multi_file_layout())

    from dead_cst import _refresh

    installed: list[tuple[int, object]] = []
    real_signal = signal.signal

    def _spy(sig, handler):
        installed.append((int(sig), handler))
        return real_signal(sig, handler)

    monkeypatch.setattr(_refresh.signal, "signal", _spy)

    sigterm_before = signal.getsignal(signal.SIGTERM)
    sigint_before = signal.getsignal(signal.SIGINT)

    make_analysis(workers=2).materialize_all()

    sigs = [s for s, _ in installed]
    assert int(signal.SIGTERM) in sigs, sigs
    assert int(signal.SIGINT) in sigs, sigs
    assert sigs.count(int(signal.SIGTERM)) == 2, sigs
    assert sigs.count(int(signal.SIGINT)) == 2, sigs
    assert signal.getsignal(signal.SIGTERM) is sigterm_before
    assert signal.getsignal(signal.SIGINT) is sigint_before


def test_pool_cancel_flag_raises_keyboard_interrupt(tmp_path, make_analysis, monkeypatch):
    """Tripping the SIGTERM handler synchronously raises ``KeyboardInterrupt``."""
    _write(tmp_path, _multi_file_layout())

    from dead_cst import _refresh

    real_signal = signal.signal
    captured: dict[str, object] = {}

    def _spy(sig, handler):
        if int(sig) == int(signal.SIGTERM) and "handler" not in captured:
            captured["handler"] = handler
            handler(int(sig), None)
        return real_signal(sig, handler)

    monkeypatch.setattr(_refresh.signal, "signal", _spy)

    with pytest.raises(KeyboardInterrupt):
        make_analysis(workers=2).materialize_all()
