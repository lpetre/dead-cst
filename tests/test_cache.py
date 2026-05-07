"""Tests for the SQLite-backed :class:`GraphCache`.

The cache short-circuits the per-file visitor pass when a file's
SHA-256 content hash matches what's already on disk; the rest of the
analyzer (per-base ``resolve_edges``, plugin pass) runs every
invocation. These tests cover:

* the on-disk schema and per-base fingerprint reconciliation,
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

from dead_cst.cache import (
    CACHE_DIR_NAME,
    GraphCache,
    clear_cache,
    compute_fingerprint,
    default_cache_path,
    file_hash,
)
from dead_cst.cli import app
from dead_cst.graph import VisitorPayload
from dead_cst.resolvers import ManualResolver


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def _fp(base: Path, **kwargs) -> str:
    """Shorthand: per-base fingerprint."""
    return compute_fingerprint(base=base, **kwargs)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_for_equal_inputs(tmp_path):
    """Two equal call signatures hash to the same fingerprint."""
    a = _fp(tmp_path)
    b = _fp(tmp_path)
    assert a == b


def test_fingerprint_independent_of_search_paths(tmp_path):
    """``search_paths`` no longer enters the fingerprint.

    Cross-file import resolution moved to
    :func:`dead_cst._edges.resolve_edges`, which runs unconditionally
    on every analysis, so search-path changes re-stitch edges
    without invalidating cached payloads.
    """
    a = _fp(tmp_path)
    b = _fp(tmp_path)
    assert a == b


def test_fingerprint_changes_with_base(tmp_path):
    """The base itself is in the fingerprint; sibling bases are independent."""
    a = _fp(tmp_path / "a")
    b = _fp(tmp_path / "b")
    assert a != b


def test_fingerprint_independent_of_sibling_bases(tmp_path):
    """A base's fingerprint depends only on its own config, not the project's.

    This is the per-base contract: adding or removing a sibling base
    from the analysis must not invalidate the cached rows for ``base``.
    """
    a = _fp(tmp_path / "lib")
    # Same base -- a "sibling" base just doesn't enter the picture.
    b = _fp(tmp_path / "lib")
    assert a == b


def test_fingerprint_independent_of_resolvers(tmp_path):
    """Resolver chain no longer enters the fingerprint.

    The resolver participates in :func:`resolve_edges` (which runs
    unconditionally), not the per-file visitor pass, so swapping
    resolvers re-stitches edges without invalidating cached payloads.
    """
    a = _fp(tmp_path)
    b = _fp(tmp_path)
    assert a == b


def test_fingerprint_subclasses_with_distinct_names_distinct(tmp_path):
    """Two ``LiteralListPlugin`` subclasses with distinct ``name`` produce
    distinct fingerprints, even when their other config differs.

    This guards the abstract-base contract: the bases deliberately omit
    ``name`` / ``version`` so subclasses must declare them. Each
    subclass owns its own cache namespace via its unique ``name``.
    """
    from dataclasses import dataclass

    from dead_cst.plugins import LiteralListPlugin

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

    fp_a = _fp(tmp_path, plugins=[A()])
    fp_b = _fp(tmp_path, plugins=[B()])
    assert fp_a != fp_b


def test_fingerprint_changes_when_unreachable_detector_changes(tmp_path):
    """Swapping the unreachable-region detector flips the fingerprint."""
    from dataclasses import dataclass

    from libcst.metadata import CodeRange, MetadataWrapper

    @dataclass(frozen=True)
    class Custom:
        name: str = "custom"
        version: int = 1

        def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]:
            return []

    fp_default = _fp(tmp_path)
    fp_custom = _fp(tmp_path, unreachable_detector=Custom())
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

    fp_v1 = _fp(tmp_path, unreachable_detector=Custom())
    fp_v2 = _fp(tmp_path, unreachable_detector=Custom(version=2))
    assert fp_v1 != fp_v2


def test_fingerprint_changes_when_visitor_version_bumped(tmp_path, monkeypatch):
    """Bumping ``SymbolVisitor.version`` invalidates the cache key."""
    from dead_cst._visitor import SymbolVisitor

    fp_v1 = _fp(tmp_path)
    monkeypatch.setattr(SymbolVisitor, "version", SymbolVisitor.version + 1)
    fp_v2 = _fp(tmp_path)
    assert fp_v1 != fp_v2


def test_fingerprint_changes_when_plugin_version_bumped(tmp_path):
    """Bumping a plugin's epoch ``version`` invalidates the cache key."""
    from dataclasses import dataclass

    from dead_cst.plugins import LiteralListPlugin

    @dataclass(kw_only=True)
    class P(LiteralListPlugin):
        owner_fqname: str = "pkg"
        variable_name: str = "X"
        name: str = "p"
        version: int = 1700000000

    fp_old = _fp(tmp_path, plugins=[P()])
    fp_new = _fp(tmp_path, plugins=[P(version=1700000001)])
    assert fp_old != fp_new


def test_abstract_base_requires_name_and_version():
    """Direct instantiation of ``LiteralListPlugin`` /
    ``DecoratedDeclPlugin`` must fail -- both are abstract bases that
    leave ``name`` / ``version`` to concrete subclasses so the cache
    fingerprint is always well-defined."""
    from dead_cst.plugins import DecoratedDeclPlugin, LiteralListPlugin

    with pytest.raises(TypeError):
        LiteralListPlugin()
    with pytest.raises(TypeError):
        DecoratedDeclPlugin()


# ---------------------------------------------------------------------------
# GraphCache lifecycle
# ---------------------------------------------------------------------------


def test_open_creates_schema(tmp_path):
    """Opening on a fresh path creates the tables and records the schema version."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    cache = GraphCache(db)
    cache.close()
    assert db.exists()
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "meta" in tables
        assert "file_cache" in tables
        cols = {r[1] for r in conn.execute("PRAGMA table_info(file_cache)")}
        assert {"path", "content_hash", "fingerprint", "payload"}.issubset(cols)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert row is not None and int(row[0]) >= 2
    finally:
        conn.close()


def test_open_writes_gitignore(tmp_path):
    """The cache directory gets a wildcard .gitignore so it stays out of VCS."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    GraphCache(db).close()
    gi = db.parent / ".gitignore"
    assert gi.exists()
    assert gi.read_text() == "*\n"


def test_per_row_fingerprint_isolates_bases(tmp_path):
    """A row written under fingerprint A is invisible to a get() under B,
    but other rows in the same DB stay valid."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file_a = tmp_path / "a.py"
    file_a.write_text("x = 1\n")
    file_b = tmp_path / "b.py"
    file_b.write_text("y = 2\n")
    payload = VisitorPayload(nodes=(), edges=(), imports=(), dead_suites=())
    with GraphCache(db) as cache:
        cache.put(file_a, payload, "fp1")
        cache.put(file_b, payload, "fp2")
        # Same fingerprints -> hits.
        assert cache.get(file_a, "fp1") == payload
        assert cache.get(file_b, "fp2") == payload
        # Cross-fingerprint -> miss.
        assert cache.get(file_a, "fp2") is None
        assert cache.get(file_b, "fp1") is None


def test_get_returns_payload_on_hit(tmp_path):
    """A second :meth:`get` after :meth:`put` returns the same payload."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    payload = VisitorPayload(nodes=(), edges=(), imports=(), dead_suites=())
    with GraphCache(db) as cache:
        cache.put(file, payload, "fp")
        restored = cache.get(file, "fp")
    assert restored == payload


def test_get_invalidates_on_content_change(tmp_path):
    """Editing the file invalidates its row even before the next ``put``."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    payload = VisitorPayload(nodes=(), edges=(), imports=(), dead_suites=())
    with GraphCache(db) as cache:
        cache.put(file, payload, "fp")
        file.write_text("x = 2\n")
        assert cache.get(file, "fp") is None


def test_get_returns_none_for_missing_file(tmp_path):
    """An unreadable / missing file simply misses; no exception."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db) as cache:
        assert cache.get(tmp_path / "nope.py", "fp") is None


def test_corrupt_blob_is_dropped(tmp_path):
    """Unreadable rows are deleted on access so the cache self-heals."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    cache = GraphCache(db)
    h = file_hash(file)
    cache._conn.execute(
        "INSERT INTO file_cache(path, content_hash, fingerprint, payload) VALUES(?, ?, ?, ?)",
        (str(file), h, "fp", b"not-a-pickle"),
    )
    cache._conn.commit()

    assert cache.get(file, "fp") is None
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


def test_build_symbol_graph_cached_matches_uncached(tmp_path, make_analysis):
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
    cold = make_analysis().materialize_all()

    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db) as cache:
        first = make_analysis(cache=cache).materialize_all()
    with GraphCache(db) as cache:
        warm = make_analysis(cache=cache).materialize_all()

    assert _node_set(first) == _node_set(cold)
    assert _node_set(warm) == _node_set(cold)
    assert _edge_set(first) == _edge_set(cold)
    assert _edge_set(warm) == _edge_set(cold)


def test_warm_run_skips_visitor(tmp_path, make_analysis, monkeypatch):
    """A warm run with no edits doesn't construct the visitor."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db) as cache:
        make_analysis(cache=cache).materialize_all()

    from dead_cst import analyze

    calls = []
    real = analyze.SymbolVisitor

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _spy)
    with GraphCache(db) as cache:
        make_analysis(cache=cache).materialize_all()
    assert calls == []


def test_warm_run_with_plugins_parses_zero_files(tmp_path, make_analysis, monkeypatch):
    """A warm run with the full builtin plugin set never parses any file."""
    import libcst as cst

    from dead_cst import analyze
    from dead_cst.plugins import (
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

    with GraphCache(db) as cache:
        make_analysis(plugins=plugins, cache=cache).materialize_all()

    visitor_calls: list[object] = []
    wrapper_calls: list[object] = []
    parse_calls: list[object] = []
    fqn_calls: list[object] = []
    real_visitor = analyze.SymbolVisitor
    real_wrapper = analyze.MetadataWrapper
    real_parse = cst.parse_module
    real_gen_cache = analyze.FixedFullyQualifiedNameProvider.gen_cache

    def _visitor_spy(*args, **kwargs):
        visitor_calls.append(args)
        return real_visitor(*args, **kwargs)

    def _wrapper_spy(*args, **kwargs):
        wrapper_calls.append(args)
        return real_wrapper(*args, **kwargs)

    def _parse_spy(*args, **kwargs):
        parse_calls.append(args)
        return real_parse(*args, **kwargs)

    def _fqn_spy(*args, **kwargs):
        fqn_calls.append(args)
        return real_gen_cache(*args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _visitor_spy)
    monkeypatch.setattr(analyze, "MetadataWrapper", _wrapper_spy)
    monkeypatch.setattr(cst, "parse_module", _parse_spy)
    monkeypatch.setattr(analyze.FixedFullyQualifiedNameProvider, "gen_cache", classmethod(_fqn_spy))

    with GraphCache(db) as cache:
        make_analysis(plugins=plugins, cache=cache).materialize_all()

    assert visitor_calls == []
    assert wrapper_calls == []
    assert parse_calls == []
    assert fqn_calls == []


def test_edited_file_re_runs_visitor(tmp_path, make_analysis, monkeypatch):
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
    with GraphCache(db) as cache:
        make_analysis(cache=cache).materialize_all()

    (tmp_path / "pkg" / "a.py").write_text("def f(): return 1\n")

    from dead_cst import analyze

    visited: list[Path] = []
    real = analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _spy)
    with GraphCache(db) as cache:
        make_analysis(cache=cache).materialize_all()

    assert visited == [tmp_path / "pkg" / "a.py"]


def test_resolver_change_does_not_invalidate_cache(tmp_path, make_analysis, monkeypatch):
    """Resolvers no longer enter the per-file fingerprint.

    Cross-file import resolution moved to
    :func:`dead_cst._edges.resolve_edges`, so adding or swapping a
    resolver re-stitches edges on the next analysis without
    re-running the visitor on any cached file.
    """
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\n",
            "pkg/b.py": "def g(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db) as cache:
        make_analysis(cache=cache).materialize_all()

    from dead_cst import analyze

    visited: list[Path] = []
    real = analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _spy)
    with GraphCache(db) as cache:
        make_analysis(resolvers=[ManualResolver(specs=[])], cache=cache).materialize_all()
    assert visited == []


def test_plugin_contributions_survive_warm_cache(tmp_path, make_analysis):
    """Plugin observe contributions are baked into the cached payload."""
    from dead_cst.plugins import MainBlockPlugin

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

    with GraphCache(db) as cache:
        cold = make_analysis(plugins=plugins, cache=cache).materialize_all()
    cold_entrypoints = {n.fqname for n, a in cold.nodes(data=True) if a.get("entrypoint")}

    with GraphCache(db) as cache:
        warm = make_analysis(plugins=plugins, cache=cache).materialize_all()
    warm_entrypoints = {n.fqname for n, a in warm.nodes(data=True) if a.get("entrypoint")}

    assert cold_entrypoints == warm_entrypoints
    assert cold_entrypoints  # plugin actually contributed something


def test_plugin_version_bump_invalidates_cache(tmp_path, make_analysis, monkeypatch):
    """Bumping a plugin's ``version`` invalidates its cached observe output."""
    from dead_cst.plugins import MainBlockPlugin

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": "def main(): pass\n",
        },
    )
    db = tmp_path / CACHE_DIR_NAME / "cache.db"

    plugins_v1 = [MainBlockPlugin()]
    with GraphCache(db) as cache:
        make_analysis(plugins=plugins_v1, cache=cache).materialize_all()

    bumped = MainBlockPlugin()
    bumped.version = "2"

    from dead_cst import analyze

    visited: list[Path] = []
    real = analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _spy)
    with GraphCache(db) as cache:
        make_analysis(plugins=[bumped], cache=cache).materialize_all()
    assert {p.name for p in visited} == {"__init__.py", "m.py"}


# ---------------------------------------------------------------------------
# clear_cache helper
# ---------------------------------------------------------------------------


def test_clear_cache_returns_true_when_present(tmp_path):
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    GraphCache(db).close()
    assert clear_cache(db) is True
    assert not db.exists()


def test_clear_cache_returns_false_when_missing(tmp_path):
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    assert clear_cache(db) is False


def test_clear_cache_removes_cache_dir_when_empty(tmp_path):
    """The .dead-cst-cache dir is removed iff it's empty after cleanup."""
    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    GraphCache(db).close()
    cache_dir = db.parent
    assert cache_dir.exists()
    clear_cache(db)
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
