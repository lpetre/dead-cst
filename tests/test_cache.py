"""Tests for the SQLite-backed :class:`GraphCache`.

The cache short-circuits the per-file visitor pass when a file's
SHA-256 content hash matches what's already on disk; the rest of the
analyzer (per-base ``resolve_edges``, plugin pass) runs every
invocation. These tests cover:

* the on-disk schema and fingerprint reconciliation,
* per-file hit / miss / mtime-but-no-content-change behavior,
* graph equivalence between cached and uncached runs,
* invalidation on path-map / resolver-chain changes,
* the CLI ``--no-cache`` flag and ``cache clear`` subcommand.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dead_cst import build_symbol_graph
from dead_cst._cache import (
    CACHE_DIR_NAME,
    GraphCache,
    clear_cache,
    compute_fingerprint,
    default_cache_path,
    file_hash,
)
from dead_cst._resolvers import ManualResolver
from dead_cst._visitor import VisitorPayload
from dead_cst.cli import app


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_for_equal_inputs(tmp_path):
    """Two equal call signatures hash to the same fingerprint."""
    a = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    b = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    assert a == b


def test_fingerprint_changes_with_paths(tmp_path):
    """Adding a base or dep flips the fingerprint."""
    a = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    b = compute_fingerprint(paths={tmp_path: [tmp_path / "dep"]}, resolvers=[])
    assert a != b


def test_fingerprint_changes_with_resolvers(tmp_path):
    """Adding a resolver name changes the fingerprint."""
    a = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    b = compute_fingerprint(paths={tmp_path: []}, resolvers=[ManualResolver(specs=[])])
    assert a != b


# ---------------------------------------------------------------------------
# GraphCache lifecycle
# ---------------------------------------------------------------------------


def test_open_creates_schema_and_records_fingerprint(tmp_path):
    """Opening on a fresh path creates the tables and stores the fingerprint."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    cache = GraphCache(db, fingerprint="fp1")
    cache.close()
    assert db.exists()
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "meta" in tables
        assert "file_cache" in tables
        row = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        assert row == ("fp1",)
    finally:
        conn.close()


def test_open_writes_gitignore(tmp_path):
    """The cache directory gets a wildcard .gitignore so it stays out of VCS."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    GraphCache(db, fingerprint="fp1").close()
    gi = db.parent / ".gitignore"
    assert gi.exists()
    assert gi.read_text() == "*\n"


def test_fingerprint_mismatch_wipes_file_cache(tmp_path):
    """Opening with a new fingerprint drops every cached row."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    cache = GraphCache(db, fingerprint="fp1")
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    payload = VisitorPayload(nodes=(), edges=(), imports=(), dead_suites=())
    cache.put(file, payload)
    cache.close()

    cache2 = GraphCache(db, fingerprint="fp2")
    assert cache2.get(file) is None
    cache2.close()


def test_get_returns_payload_on_hit(tmp_path):
    """A second :meth:`get` after :meth:`put` returns the same payload."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    payload = VisitorPayload(nodes=(), edges=(), imports=(), dead_suites=())
    with GraphCache(db, fingerprint="fp") as cache:
        cache.put(file, payload)
        restored = cache.get(file)
    assert restored == payload


def test_get_invalidates_on_content_change(tmp_path):
    """Editing the file invalidates its row even before the next ``put``."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    payload = VisitorPayload(nodes=(), edges=(), imports=(), dead_suites=())
    with GraphCache(db, fingerprint="fp") as cache:
        cache.put(file, payload)
        file.write_text("x = 2\n")
        assert cache.get(file) is None


def test_get_returns_none_for_missing_file(tmp_path):
    """An unreadable / missing file simply misses; no exception."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db, fingerprint="fp") as cache:
        assert cache.get(tmp_path / "nope.py") is None


def test_corrupt_blob_is_dropped(tmp_path):
    """Unreadable rows are deleted on access so the cache self-heals."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    cache = GraphCache(db, fingerprint="fp")
    h = file_hash(file)
    cache._conn.execute(
        "INSERT INTO file_cache(path, content_hash, payload) VALUES(?, ?, ?)",
        (str(file), h, b"not-a-pickle"),
    )
    cache._conn.commit()

    assert cache.get(file) is None
    row = cache._conn.execute("SELECT 1 FROM file_cache WHERE path=?", (str(file),)).fetchone()
    assert row is None
    cache.close()


def test_default_cache_path_under_root(tmp_path):
    p = default_cache_path(tmp_path)
    assert p == tmp_path / CACHE_DIR_NAME / "cache.db"


# ---------------------------------------------------------------------------
# build_symbol_graph integration
# ---------------------------------------------------------------------------


def _node_set(graph):
    return {(n.fqname, n.type) for n in graph.nodes}


def _edge_set(graph):
    return {(s.fqname, d.fqname) for s, d in graph.edges()}


def test_build_symbol_graph_cached_matches_uncached(tmp_path):
    """Cached and uncached runs produce the same nodes and edges."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
                def f(): pass
                def g(): return f()
            """,
            "pkg/b.py": """
                from .a import g
                def h(): return g()
            """,
        },
    )
    cold = build_symbol_graph({tmp_path: []})

    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    with GraphCache(db, fingerprint=fp) as cache:
        first = build_symbol_graph({tmp_path: []}, cache=cache)
    with GraphCache(db, fingerprint=fp) as cache:
        warm = build_symbol_graph({tmp_path: []}, cache=cache)

    assert _node_set(first) == _node_set(cold)
    assert _node_set(warm) == _node_set(cold)
    assert _edge_set(first) == _edge_set(cold)
    assert _edge_set(warm) == _edge_set(cold)


def test_warm_run_skips_visitor(tmp_path, monkeypatch):
    """A warm run with no edits doesn't construct the visitor."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)

    # Patch SymbolVisitor at the call site (it's imported by name in
    # _analyze) and assert the warm run never instantiates it.
    from dead_cst import _analyze

    calls = []
    real = _analyze.SymbolVisitor

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(_analyze, "SymbolVisitor", _spy)
    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)
    assert calls == []


def test_edited_file_re_runs_visitor(tmp_path, monkeypatch):
    """Editing one file invalidates only that file's cached payload."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
            "pkg/b.py": "def g(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)

    (tmp_path / "pkg" / "a.py").write_text("def f(): return 1\n")

    from dead_cst import _analyze

    visited: list[Path] = []
    real = _analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(_analyze, "SymbolVisitor", _spy)
    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)

    assert visited == [tmp_path / "pkg" / "a.py"]


def test_fingerprint_change_forces_full_rebuild(tmp_path, monkeypatch):
    """Re-opening with a new fingerprint forces every file through the visitor."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
            "pkg/b.py": "def g(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp1 = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    with GraphCache(db, fingerprint=fp1) as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)

    from dead_cst import _analyze

    visited: list[Path] = []
    real = _analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(_analyze, "SymbolVisitor", _spy)
    with GraphCache(db, fingerprint="changed") as cache:
        build_symbol_graph({tmp_path: []}, cache=cache)
    # All three files (init + a + b) get re-visited under the new fingerprint.
    assert {p.name for p in visited} == {"__init__.py", "a.py", "b.py"}


def test_plugins_still_run_on_warm_cache(tmp_path):
    """Plugin pass executes every run, even when every file hits the cache.

    Plugins aren't part of the fingerprint -- swapping them between
    runs reuses cached payloads and only re-runs the plugin pass. This
    test confirms a plugin's contributions land in the warm-run graph.
    """
    from dead_cst import MainBlockPlugin

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
                def main(): pass

                if __name__ == "__main__":
                    main()
            """,
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[])

    # Cold run with the plugin -- populates the cache and the plugin's
    # synthetic entrypoint.
    with GraphCache(db, fingerprint=fp) as cache:
        cold = build_symbol_graph({tmp_path: []}, plugins=[MainBlockPlugin()], cache=cache)
    cold_entrypoints = {n.fqname for n, a in cold.nodes(data=True) if a.get("entrypoint")}

    # Warm run -- visitor is skipped, but the plugin must still run and
    # mark the same entrypoint.
    with GraphCache(db, fingerprint=fp) as cache:
        warm = build_symbol_graph({tmp_path: []}, plugins=[MainBlockPlugin()], cache=cache)
    warm_entrypoints = {n.fqname for n, a in warm.nodes(data=True) if a.get("entrypoint")}

    assert cold_entrypoints == warm_entrypoints
    assert cold_entrypoints  # plugin actually contributed something


# ---------------------------------------------------------------------------
# clear_cache helper
# ---------------------------------------------------------------------------


def test_clear_cache_returns_true_when_present(tmp_path):
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    GraphCache(db, fingerprint="fp").close()
    assert clear_cache(db) is True
    assert not db.exists()


def test_clear_cache_returns_false_when_missing(tmp_path):
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    assert clear_cache(db) is False


def test_clear_cache_removes_cache_dir_when_empty(tmp_path):
    """The .dead-cst-cache dir is removed iff it's empty after cleanup."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    GraphCache(db, fingerprint="fp").close()
    cache_dir = db.parent
    assert cache_dir.exists()
    clear_cache(db)
    # Cache was the only thing in the dir (plus the .gitignore we
    # ourselves dropped); both should be gone.
    assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_analyze_creates_cache_file(runner, tmp_path):
    """``dead-cst analyze`` populates the default cache path."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
        },
    )
    result = runner.invoke(app, ["analyze", str(tmp_path), "-e", "re:.*"])
    assert result.exit_code in (0, 1)
    assert (tmp_path / CACHE_DIR_NAME / "cache.db").exists()


def test_cli_analyze_no_cache_skips_db(runner, tmp_path):
    """``--no-cache`` neither reads nor writes the on-disk database."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
        },
    )
    result = runner.invoke(app, ["analyze", str(tmp_path), "--no-cache", "-e", "re:.*"])
    assert result.exit_code in (0, 1)
    assert not (tmp_path / CACHE_DIR_NAME / "cache.db").exists()


def test_cli_cache_clear_removes_db(runner, tmp_path):
    """``dead-cst cache clear`` deletes the database and reports it."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
        },
    )
    runner.invoke(app, ["analyze", str(tmp_path), "-e", "re:.*"])
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    assert db.exists()

    result = runner.invoke(app, ["cache", "clear", str(tmp_path)])
    assert result.exit_code == 0
    assert "Removed" in result.stdout
    assert not db.exists()


def test_cli_cache_clear_when_missing(runner, tmp_path):
    """Clearing a nonexistent cache reports it without erroring."""
    result = runner.invoke(app, ["cache", "clear", str(tmp_path)])
    assert result.exit_code == 0
    assert "No cache found" in result.stdout
