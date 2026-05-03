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


def test_fingerprint_subclasses_with_distinct_names_distinct(tmp_path):
    """Two ``LiteralListPlugin`` subclasses with distinct ``name`` produce
    distinct fingerprints, even when their other config differs.

    This guards the abstract-base contract: the bases deliberately omit
    ``name`` / ``version`` so subclasses must declare them. Each
    subclass owns its own cache namespace via its unique ``name``.
    """
    from dataclasses import dataclass

    from dead_cst import LiteralListPlugin

    @dataclass(kw_only=True)
    class A(LiteralListPlugin):
        owner_fqname: str = "pkg.a"
        variable_name: str = "X"
        name: str = "a"
        version: int = 1700000000

    @dataclass(kw_only=True)
    class B(LiteralListPlugin):
        owner_fqname: str = "pkg.b"
        variable_name: str = "Y"
        name: str = "b"
        version: int = 1700000000

    fp_a = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=[A()])
    fp_b = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=[B()])
    assert fp_a != fp_b


def test_fingerprint_changes_when_unreachable_detector_changes(tmp_path):
    """Swapping the unreachable-region detector flips the fingerprint.

    Detectors are folded into each cached payload's ``dead_suites``
    list, so a detector swap must invalidate the file_cache. Both the
    default and custom detectors satisfy ``Cacheable`` via their
    ``(name, version)`` pair.
    """
    from dataclasses import dataclass

    from libcst.metadata import CodeRange, MetadataWrapper

    @dataclass(frozen=True)
    class Custom:
        name: str = "custom"
        version: int = 1

        def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]:
            return []

    fp_default = compute_fingerprint(paths={tmp_path: []}, resolvers=[])
    fp_custom = compute_fingerprint(
        paths={tmp_path: []}, resolvers=[], unreachable_detector=Custom()
    )
    assert fp_default != fp_custom


def test_fingerprint_changes_when_unreachable_detector_version_bumped(tmp_path):
    """Bumping a detector's ``version`` invalidates the cache key."""
    from dataclasses import dataclass

    from libcst.metadata import CodeRange, MetadataWrapper

    @dataclass(frozen=True)
    class Custom:
        name: str = "custom"
        version: int = 1

        def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]:
            return []

    fp_v1 = compute_fingerprint(paths={tmp_path: []}, resolvers=[], unreachable_detector=Custom())
    fp_v2 = compute_fingerprint(
        paths={tmp_path: []}, resolvers=[], unreachable_detector=Custom(version=2)
    )
    assert fp_v1 != fp_v2


def test_fingerprint_changes_when_plugin_version_bumped(tmp_path):
    """Bumping a plugin's epoch ``version`` invalidates the cache key.

    Versions are Unix epoch ints by convention -- the convention's
    point is that two simultaneous bumps merge with ``max()`` semantics
    instead of colliding on a re-used label.
    """
    from dataclasses import dataclass

    from dead_cst import LiteralListPlugin

    @dataclass(kw_only=True)
    class P(LiteralListPlugin):
        owner_fqname: str = "pkg"
        variable_name: str = "X"
        name: str = "p"
        version: int = 1700000000

    fp_old = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=[P()])
    fp_new = compute_fingerprint(
        paths={tmp_path: []}, resolvers=[], plugins=[P(version=1700000001)]
    )
    assert fp_old != fp_new


def test_abstract_base_requires_name_and_version():
    """Direct instantiation of ``LiteralListPlugin`` /
    ``DecoratedDeclPlugin`` must fail -- both are abstract bases that
    leave ``name`` / ``version`` to concrete subclasses so the cache
    fingerprint is always well-defined."""
    import pytest

    from dead_cst import DecoratedDeclPlugin, LiteralListPlugin

    with pytest.raises(TypeError):
        LiteralListPlugin()
    with pytest.raises(TypeError):
        DecoratedDeclPlugin()


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


def test_warm_run_with_plugins_parses_zero_files(tmp_path, monkeypatch):
    """A warm run with the full builtin plugin set never parses any file.

    The two-pass plugin protocol bakes per-file observe contributions
    into the cached :class:`VisitorPayload`, so on a cache hit the
    analyzer skips both the visitor and the per-plugin observe. The
    per-base ``finalize`` step is graph-only (no CST access). This
    test pins that contract: even with every builtin plugin enabled,
    a warm run never instantiates ``FullRepoManager`` and never calls
    ``cst.parse_module`` from inside the analyzer.
    """
    import libcst as cst

    from dead_cst import (
        ClickPlugin,
        FastAPIPlugin,
        FlaskPlugin,
        InitSubclassPlugin,
        MainBlockPlugin,
        ModuleDundersPlugin,
        ProjectScriptsPlugin,
        PytestPlugin,
        TyperPlugin,
        UnittestPlugin,
    )
    from dead_cst import _analyze

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
                def main(): pass

                if __name__ == "__main__":
                    main()
            """,
            "pkg/b.py": """
                __version__ = "1.0"

                def f(): pass
            """,
        },
    )
    plugins = [
        MainBlockPlugin(),
        ProjectScriptsPlugin(),
        ModuleDundersPlugin(),
        PytestPlugin(),
        UnittestPlugin(),
        FastAPIPlugin(),
        FlaskPlugin(),
        TyperPlugin(),
        ClickPlugin(),
        InitSubclassPlugin(),
    ]
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=plugins)

    # Cold run populates the cache and exercises every observe step.
    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, plugins=plugins, cache=cache)

    visitor_calls: list[object] = []
    mgr_calls: list[object] = []
    parse_calls: list[object] = []
    real_visitor = _analyze.SymbolVisitor
    real_mgr = _analyze.FullRepoManager
    real_parse = cst.parse_module

    def _visitor_spy(*args, **kwargs):
        visitor_calls.append(args)
        return real_visitor(*args, **kwargs)

    def _mgr_spy(*args, **kwargs):
        mgr_calls.append(args)
        return real_mgr(*args, **kwargs)

    def _parse_spy(*args, **kwargs):
        parse_calls.append(args)
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(_analyze, "SymbolVisitor", _visitor_spy)
    monkeypatch.setattr(_analyze, "FullRepoManager", _mgr_spy)
    monkeypatch.setattr(cst, "parse_module", _parse_spy)

    with GraphCache(db, fingerprint=fp) as cache:
        build_symbol_graph({tmp_path: []}, plugins=plugins, cache=cache)

    assert visitor_calls == []
    assert mgr_calls == []
    assert parse_calls == []


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


def test_plugin_contributions_survive_warm_cache(tmp_path):
    """Plugin observe contributions are baked into the cached payload.

    The two-pass protocol folds each plugin's per-file ``observe``
    output into the cached :class:`VisitorPayload`. On warm runs the
    visitor is skipped *and* observe is skipped -- the cached payload
    already carries the plugin nodes/edges. The per-base ``finalize``
    runs every analysis (graph-only, no CST). The end-to-end live
    set must match the cold run.
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
    plugins = [MainBlockPlugin()]
    fp = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=plugins)

    with GraphCache(db, fingerprint=fp) as cache:
        cold = build_symbol_graph({tmp_path: []}, plugins=plugins, cache=cache)
    cold_entrypoints = {n.fqname for n, a in cold.nodes(data=True) if a.get("entrypoint")}

    with GraphCache(db, fingerprint=fp) as cache:
        warm = build_symbol_graph({tmp_path: []}, plugins=plugins, cache=cache)
    warm_entrypoints = {n.fqname for n, a in warm.nodes(data=True) if a.get("entrypoint")}

    assert cold_entrypoints == warm_entrypoints
    assert cold_entrypoints  # plugin actually contributed something


def test_plugin_version_bump_invalidates_cache(tmp_path, monkeypatch):
    """Bumping a plugin's ``version`` invalidates its cached observe output."""
    from dead_cst import MainBlockPlugin

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": "def main(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"

    plugins_v1 = [MainBlockPlugin()]
    fp_v1 = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=plugins_v1)
    with GraphCache(db, fingerprint=fp_v1) as cache:
        build_symbol_graph({tmp_path: []}, plugins=plugins_v1, cache=cache)

    # Bump the plugin's version: should invalidate the file_cache so
    # the next run re-visits every file (and re-runs observe).
    bumped = MainBlockPlugin()
    bumped.version = "2"
    fp_v2 = compute_fingerprint(paths={tmp_path: []}, resolvers=[], plugins=[bumped])
    assert fp_v1 != fp_v2

    from dead_cst import _analyze

    visited: list[Path] = []
    real = _analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(_analyze, "SymbolVisitor", _spy)
    with GraphCache(db, fingerprint=fp_v2) as cache:
        build_symbol_graph({tmp_path: []}, plugins=[bumped], cache=cache)
    assert {p.name for p in visited} == {"__init__.py", "m.py"}


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
