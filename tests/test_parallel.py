"""Tests for the ``workers`` parameter on :func:`build_symbol_graph`.

The serial and parallel paths must produce identical graphs and warm
the cache identically. Workers communicate by pickling
:class:`VisitorPayload` blobs back to the main process; the SQLite
cache write still happens on the main process so failure semantics
match the serial path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst.cache import (
    CACHE_DIR_NAME,
    GraphCache,
    compute_fingerprint,
)


def _fp() -> str:
    return compute_fingerprint()


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
