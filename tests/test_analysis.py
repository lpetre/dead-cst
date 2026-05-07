"""Tests for :class:`dead_cst.Analysis` and :class:`dead_cst.PackageView`.

Pins the lazy / scoped behavior promised by the new entry-point API:

* construction is cheap (no filesystem walk, no parsing),
* :meth:`Analysis.refresh` is idempotent and base-scoped,
* per-base queries (``modules``, ``declarations``) don't materialize
  cross-base state,
* :meth:`Analysis.materialize_all` produces the same graph as
  :func:`build_symbol_graph`,
* per-base ``dead`` answers are equal to the slice of the full ``dead``
  set restricted to that base,
* the per-base cache fingerprint isolates sibling bases (changing one
  base's deps does not invalidate other bases' rows).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst import Analysis
from dead_cst.cache import CACHE_DIR_NAME, GraphCache
from dead_cst.plugins import ExplicitEntrypointPlugin
from dead_cst.resolvers import ManualResolver


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


# ---------------------------------------------------------------------------
# Construction is cheap; refresh drives all I/O.
# ---------------------------------------------------------------------------


def test_construction_does_no_filesystem_walk(tmp_path, make_analysis, monkeypatch):
    """Constructing :class:`Analysis` must not touch disk."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    rglob_calls: list[Path] = []
    real = Path.rglob

    def _spy(self, pattern):
        rglob_calls.append(self)
        return real(self, pattern)

    monkeypatch.setattr(Path, "rglob", _spy)
    make_analysis()
    assert rglob_calls == []


def test_refresh_is_idempotent(tmp_path, make_analysis):
    """A second :meth:`refresh` over the same bases re-uses the cached spec."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    a = make_analysis().refresh()
    contributions_before = dict(a._contributions)
    a.refresh()
    # Same instance objects -- nothing was rebuilt.
    for base, contrib in contributions_before.items():
        assert a._contributions[base] is contrib


def test_refresh_rejects_unknown_base(tmp_path, make_analysis):
    """Refreshing a base not produced by any resolver errors quickly."""
    _write(tmp_path, {"pkg/__init__.py": ""})
    a = make_analysis()
    with pytest.raises(KeyError):
        a.refresh(bases=[tmp_path / "nope"])


# ---------------------------------------------------------------------------
# Per-base queries are local and fast.
# ---------------------------------------------------------------------------


def test_package_modules_local_only(tmp_path):
    """:meth:`PackageView.modules` returns this base's modules only."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(base_a, {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    _write(base_b, {"pkg/__init__.py": "", "pkg/m.py": "def g(): pass\n"})
    a = Analysis(tmp_path, resolvers=[ManualResolver(specs=["a", "b"])])
    pkg_a = a.package(base_a)
    a_paths = {n.path for n in pkg_a.modules()}
    assert a_paths == {(base_a / "pkg" / "__init__.py"), (base_a / "pkg" / "m.py")}


def test_package_declarations_filter_by_simple_name(tmp_path, make_analysis):
    """``simple_name`` matches only the rightmost dotted segment."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": "def Foo(): pass\nclass Foo: pass\nbar = 1\n",
        },
    )
    a = make_analysis()
    pv = a.package(tmp_path)
    foos = list(pv.declarations("Foo"))
    assert {n.fqname for n in foos} == {"pkg.m.Foo"}
    assert {n.type for n in foos} == {"function", "class"}


def test_local_query_doesnt_materialize_full_graph(tmp_path):
    """``pkg.modules()`` populates only this base's contribution; full graph
    is still un-materialized."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(base_a, {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    _write(base_b, {"pkg/__init__.py": "", "pkg/m.py": "def g(): pass\n"})
    a = Analysis(tmp_path, resolvers=[ManualResolver(specs=["a", "b"])])
    list(a.package(base_a).modules())
    assert base_a in a._contributions
    assert base_b not in a._contributions
    assert a._full_graph is None


# ---------------------------------------------------------------------------
# Reverse closure + interesting set scope reachability queries.
# ---------------------------------------------------------------------------


def test_reverse_closure_includes_self_and_consumers(tmp_path):
    """``app -> core -> lib``: lib's reverse closure is {lib, core, app}."""
    lib = tmp_path / "lib"
    core = tmp_path / "core"
    app = tmp_path / "app"
    for d in (lib, core, app):
        d.mkdir()
    a = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["lib", "core:lib", "app:core"])],
    )
    assert a.reverse_closure(lib) == frozenset({lib, core, app})
    assert a.reverse_closure(core) == frozenset({core, app})
    assert a.reverse_closure(app) == frozenset({app})


def test_package_dead_uses_closure_only(tmp_path):
    """A pkg.dead() materialization only refreshes the interesting set,
    not unrelated sibling bases.

    Layout: ``app -> core``; ``other`` is a sibling that doesn't depend
    on ``core``. ``core.dead()`` must not refresh ``other``.
    """
    core = tmp_path / "core"
    app = tmp_path / "app"
    other = tmp_path / "other"
    _write(core, {"pkg/__init__.py": "", "pkg/m.py": "def used(): pass\ndef dead(): pass\n"})
    _write(
        app,
        {
            "pkg/__init__.py": "",
            "pkg/main.py": "from pkg.m import used\nused()\n",
        },
    )
    _write(other, {"pkg/__init__.py": "", "pkg/m.py": "def x(): pass\n"})
    a = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["core", "app:core", "other"])],
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.main"])],
    )
    list(a.package(core).dead())
    assert other not in a._contributions


# ---------------------------------------------------------------------------
# Reachability + dead semantics.
# ---------------------------------------------------------------------------


def test_package_dead_matches_full_dead_slice(tmp_path):
    """For each base, ``pkg.dead()`` equals ``analysis.dead()`` filtered
    to that base."""
    core = tmp_path / "core"
    app = tmp_path / "app"
    _write(core, {"pkg/__init__.py": "", "pkg/m.py": "def used(): pass\ndef dead(): pass\n"})
    _write(
        app,
        {
            "pkg/__init__.py": "",
            "pkg/main.py": "from pkg.m import used\nused()\n",
        },
    )
    resolvers = [ManualResolver(specs=["core", "app:core"])]
    a_full = Analysis(
        tmp_path,
        resolvers=resolvers,
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.main"])],
    )
    full_dead_in_core = {n.fqname for n in a_full.dead() if n.path.is_relative_to(core)}
    a_pkg = Analysis(
        tmp_path,
        resolvers=resolvers,
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.main"])],
    )
    pkg_dead = {n.fqname for n in a_pkg.package(core).dead()}
    assert pkg_dead == full_dead_in_core


# ---------------------------------------------------------------------------
# Per-base cache fingerprint: changing one base's deps doesn't invalidate
# sibling bases' rows.
# ---------------------------------------------------------------------------


def test_dep_changes_do_not_invalidate_visitor_cache(tmp_path, monkeypatch):
    """Changing a base's deps re-stitches edges but does not re-run the visitor.

    ``search_paths`` left the per-file fingerprint when import
    resolution moved to :func:`dead_cst._edges.resolve_edges`, so
    swapping a base's deps just reshuffles the stitching pass --
    every cached :class:`VisitorPayload` for that base stays valid.
    """
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    extra = tmp_path / "extra"
    extra.mkdir()
    _write(base_a, {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    _write(base_b, {"pkg/__init__.py": "", "pkg/m.py": "def g(): pass\n"})

    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db) as cache:
        Analysis(
            tmp_path,
            resolvers=[ManualResolver(specs=["a", "b"])],
            cache=cache,
        ).materialize_all()

    from dead_cst import analyze

    visited: list[Path] = []
    real = analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _spy)
    with GraphCache(db) as cache:
        Analysis(
            tmp_path,
            resolvers=[ManualResolver(specs=["a:extra", "b"])],
            cache=cache,
        ).materialize_all()
    assert visited == []
