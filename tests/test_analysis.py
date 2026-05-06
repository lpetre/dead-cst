"""Tests for :class:`dead_cst.Analysis` and :class:`dead_cst.PackageView`.

Pins the lazy / scoped behavior promised by the entry-point API:

* construction is cheap (no filesystem walk, no parsing),
* :meth:`Analysis.refresh` is idempotent and tree-scoped,
* per-package queries (``modules``, ``declarations``) don't
  materialize cross-tree state,
* :meth:`Analysis.materialize_all` produces the same graph as a fresh
  full analysis,
* per-package ``dead`` answers equal the slice of the full ``dead``
  set restricted to that package,
* the per-tree cache fingerprint isolates sibling trees.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst import Analysis
from dead_cst.cache import CACHE_DIR_NAME, GraphCache
from dead_cst.plugins import ExplicitEntrypointPlugin
from conftest import build_trees


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def test_construction_does_no_filesystem_walk(tmp_path, monkeypatch):
    """Constructing :class:`Analysis` must not touch disk."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    rglob_calls: list[Path] = []
    real = Path.rglob

    def _spy(self, pattern):
        rglob_calls.append(self)
        return real(self, pattern)

    monkeypatch.setattr(Path, "rglob", _spy)
    Analysis(build_trees({tmp_path: []}))
    assert rglob_calls == []


def test_refresh_is_idempotent(tmp_path):
    """A second :meth:`refresh` over the same trees re-uses the cached spec."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    a = Analysis(build_trees({tmp_path: []})).refresh()
    contributions_before = dict(a._contributions)
    a.refresh()
    for path, contrib in contributions_before.items():
        assert a._contributions[path] is contrib


def test_refresh_rejects_unknown_package(tmp_path):
    """Refreshing a package that wasn't configured errors quickly."""
    _write(tmp_path, {"pkg/__init__.py": ""})
    a = Analysis(build_trees({tmp_path: []}))
    with pytest.raises(KeyError):
        a.refresh(packages=["nope"])


def test_package_modules_local_only(tmp_path):
    """:meth:`PackageView.modules` returns this package's modules only."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(base_a, {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    _write(base_b, {"pkg/__init__.py": "", "pkg/m.py": "def g(): pass\n"})
    a = Analysis(build_trees({base_a: [], base_b: []}))
    pkg_a = a.package("a")
    a_paths = {n.path for n in pkg_a.modules()}
    assert a_paths == {(base_a / "pkg" / "__init__.py"), (base_a / "pkg" / "m.py")}


def test_package_declarations_filter_by_simple_name(tmp_path):
    """``simple_name`` matches only the rightmost dotted segment."""
    base = tmp_path
    _write(
        base,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": "def Foo(): pass\nclass Foo: pass\nbar = 1\n",
        },
    )
    a = Analysis(build_trees({base: []}))
    pkg_name = base.name
    pv = a.package(pkg_name)
    foos = list(pv.declarations("Foo"))
    assert {n.fqname for n in foos} == {"pkg.m.Foo"}
    assert {n.type for n in foos} == {"function", "class"}


def test_local_query_doesnt_materialize_full_graph(tmp_path):
    """``pkg.modules()`` populates only this package's contributions; full graph
    is still un-materialized."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    _write(base_a, {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    _write(base_b, {"pkg/__init__.py": "", "pkg/m.py": "def g(): pass\n"})
    a = Analysis(build_trees({base_a: [], base_b: []}))
    list(a.package("a").modules())
    assert base_a.resolve() in a._contributions
    assert base_b.resolve() not in a._contributions
    assert a._full_graph is None


def test_reverse_closure_includes_self_and_consumers(tmp_path):
    """``app -> core -> lib``: lib's reverse closure is {lib, core, app}."""
    lib = tmp_path / "lib"
    core = tmp_path / "core"
    app = tmp_path / "app"
    for d in (lib, core, app):
        d.mkdir()
    a = Analysis(build_trees({lib: [], core: [lib], app: [core]}))
    assert a.reverse_closure("lib") == frozenset({"lib", "core", "app"})
    assert a.reverse_closure("core") == frozenset({"core", "app"})
    assert a.reverse_closure("app") == frozenset({"app"})


def test_package_dead_uses_closure_only(tmp_path):
    """A pkg.dead() materialization only refreshes the interesting set,
    not unrelated sibling trees."""
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
        build_trees({core: [], app: [core], other: []}),
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.main"])],
    )
    list(a.package("core").dead())
    assert other.resolve() not in a._contributions


def test_package_dead_matches_full_dead_slice(tmp_path):
    """For each package, ``pkg.dead()`` equals ``analysis.dead()`` filtered
    to that package."""
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
    a_full = Analysis(
        build_trees({core: [], app: [core]}),
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.main"])],
    )
    full_dead_in_core = {n.fqname for n in a_full.dead() if n.path.is_relative_to(core)}
    a_pkg = Analysis(
        build_trees({core: [], app: [core]}),
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.main"])],
    )
    pkg_dead = {n.fqname for n in a_pkg.package("core").dead()}
    assert pkg_dead == full_dead_in_core


def test_per_tree_fingerprint_isolates_siblings(tmp_path, monkeypatch):
    """Changing one tree's search refs invalidates *only* that tree's rows."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    extra = tmp_path / "extra"
    _write(extra, {"pkg/__init__.py": "", "pkg/x.py": "def x(): pass\n"})
    _write(base_a, {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    _write(base_b, {"pkg/__init__.py": "", "pkg/m.py": "def g(): pass\n"})

    db = tmp_path / CACHE_DIR_NAME / "cache.db"
    with GraphCache(db) as cache:
        Analysis(build_trees({base_a: [], base_b: []}), cache=cache).materialize_all()

    from dead_cst import analyze

    visited: list[Path] = []
    real = analyze.SymbolVisitor

    def _spy(path, *args, **kwargs):
        visited.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(analyze, "SymbolVisitor", _spy)
    # Add a search-tree ref to base_a only. base_b's fingerprint is
    # unchanged, so its rows stay valid; base_a's rows are invalidated.
    with GraphCache(db) as cache:
        Analysis(
            build_trees({base_a: [extra], base_b: [], extra: []}), cache=cache
        ).materialize_all()
    visited_under_a = {p for p in visited if p.is_relative_to(base_a)}
    visited_under_b = {p for p in visited if p.is_relative_to(base_b)}
    assert visited_under_a, "base_a should re-visit after fingerprint change"
    assert not visited_under_b, "base_b should not re-visit -- its fingerprint is unchanged"
