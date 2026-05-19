"""Tests for :class:`dead_cst.Analysis`."""

from __future__ import annotations

import textwrap
from pathlib import Path


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


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
