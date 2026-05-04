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

from dead_cst import build_symbol_graph
from dead_cst._cache import (
    CACHE_DIR_NAME,
    GraphCache,
    compute_fingerprint,
)


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


def test_parallel_matches_serial(tmp_path):
    """``workers=2`` and the default serial path return the same graph."""
    _write(tmp_path, _multi_file_layout())
    serial = build_symbol_graph({tmp_path: []})
    parallel = build_symbol_graph({tmp_path: []}, workers=2)
    assert _node_set(parallel) == _node_set(serial)
    assert _edge_set(parallel) == _edge_set(serial)


def test_parallel_warms_cache(tmp_path):
    """Parallel runs write payloads to the cache so a follow-up run skips workers."""
    _write(tmp_path, _multi_file_layout())
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[])

    with GraphCache(db, fingerprint=fp) as cache:
        cold = build_symbol_graph({tmp_path: []}, cache=cache, workers=2)

    # Every .py file under tmp_path should now be cached.
    files = sorted(tmp_path.rglob("*.py"))
    with GraphCache(db, fingerprint=fp) as cache:
        for f in files:
            assert cache.get(f) is not None, f"{f} missing from cache after parallel run"

    # And a follow-up serial run sees the same graph.
    with GraphCache(db, fingerprint=fp) as cache:
        warm = build_symbol_graph({tmp_path: []}, cache=cache)
    assert _node_set(warm) == _node_set(cold)
    assert _edge_set(warm) == _edge_set(cold)


def test_parallel_falls_back_to_serial_for_single_miss(tmp_path, monkeypatch):
    """``workers=2`` with a single cache-miss skips the pool (overhead floor)."""
    _write(tmp_path, _multi_file_layout())
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[])

    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)

    # Edit one file so exactly one miss survives.
    (tmp_path / "pkg" / "a.py").write_text("def f():\n    return 1\n")

    from dead_cst import _analyze

    calls = []
    real = _analyze.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("max_workers"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_analyze, "ProcessPoolExecutor", _spy)
    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache, workers=4)
    assert calls == [], "single-miss run should not spawn a pool"


def test_parallel_pool_capped_at_miss_count(tmp_path, monkeypatch):
    """The pool caps ``max_workers`` at ``len(miss_files)``."""
    _write(tmp_path, _multi_file_layout())

    from dead_cst import _analyze

    seen = []
    real = _analyze.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("max_workers"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_analyze, "ProcessPoolExecutor", _spy)
    build_symbol_graph({tmp_path: []}, workers=64)
    # Five files in _multi_file_layout(); pool should be capped at 5.
    assert seen and all(w <= 5 for w in seen)


@pytest.mark.parametrize("workers", [None, 1])
def test_workers_none_or_one_keeps_serial_path(tmp_path, monkeypatch, workers):
    """``workers=None`` and ``workers=1`` never spawn a pool."""
    _write(tmp_path, _multi_file_layout())
    from dead_cst import _analyze

    calls = []
    real = _analyze.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(_analyze, "ProcessPoolExecutor", _spy)
    build_symbol_graph({tmp_path: []}, workers=workers)
    assert calls == []
