"""Tests for :class:`dead_cst.Analysis` and :class:`dead_cst.PackageView`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst.plugins import ExplicitEntrypointPlugin


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def test_package_declarations_filter_by_simple_name(tmp_path, make_analysis):
    """Filter by simple name (rightmost dotted segment)."""
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


def test_reverse_closure_includes_self_and_consumers(tmp_path, make_analysis):
    """``app -> core -> lib``: lib's reverse closure is {lib, core, app}."""
    lib = tmp_path / "lib"
    core = tmp_path / "core"
    app = tmp_path / "app"
    for d in (lib, core, app):
        d.mkdir()
    a = make_analysis(["lib", "core:lib", "app:core"])
    assert a.reverse_closure(lib) == frozenset({lib, core, app})
    assert a.reverse_closure(core) == frozenset({core, app})
    assert a.reverse_closure(app) == frozenset({app})


def test_cycle_in_deps_is_tolerated(tmp_path, make_analysis):
    """``a <-> b`` cycles in :attr:`Package.deps` don't crash the analyzer."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    for d in (a_dir, b_dir):
        d.mkdir()
    analysis = make_analysis(["a:b", "b:a"])

    assert {p.path for p in analysis.packages} == {a_dir, b_dir}
    assert analysis.reverse_closure(a_dir) == frozenset({a_dir, b_dir})
    assert analysis.reverse_closure(b_dir) == frozenset({a_dir, b_dir})


def test_package_dead_matches_full_dead_slice(tmp_path, make_analysis):
    """For each package, ``pkg.dead()`` equals ``analysis.dead()`` filtered to that package."""
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
    plugins = [ExplicitEntrypointPlugin(specs=["pkg.main"])]
    a_full = make_analysis(["core", "app:core"], plugins=plugins)
    full_dead_in_core = {n.fqname for n in a_full.dead() if n.path.is_relative_to(core)}
    a_pkg = make_analysis(["core", "app:core"], plugins=plugins)
    pkg_dead = {n.fqname for n in a_pkg.package(core).dead()}
    assert pkg_dead == full_dead_in_core
