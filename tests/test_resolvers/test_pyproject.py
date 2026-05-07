"""Tests for :class:`dead_cst.resolvers.pyproject.PyprojectResolver`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst.resolvers import PyprojectResolver


def test_pyproject_resolver_reads_explicit_packages(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"

            [[tool.dead-cst.packages]]
            path = "."
            name = "x"
            exported = ["src"]
            deps = []
        """).strip()
    )

    result = PyprojectResolver().resolve(tmp_path)
    assert len(result) == 1
    pkg = result[0]
    assert pkg.path == tmp_path.resolve()
    assert pkg.name == "x"
    assert pkg.exported == ((tmp_path / "src").resolve(),)
    assert pkg.deps == ()


def test_pyproject_resolver_explicit_with_deps(tmp_path: Path):
    """Two explicit packages with a dep edge."""
    (tmp_path / "core").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "ws"

            [[tool.dead-cst.packages]]
            path = "core"
            name = "core"
            exported = ["."]

            [[tool.dead-cst.packages]]
            path = "app"
            name = "app"
            exported = ["."]
            deps = ["core"]
        """).strip()
    )

    result = PyprojectResolver().resolve(tmp_path)
    by_name = {p.name: p for p in result}
    assert by_name["core"].deps == ()
    assert by_name["app"].deps == ("core",)


def test_pyproject_resolver_src_layout_fallback(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    result = PyprojectResolver().resolve(tmp_path)
    assert len(result) == 1
    pkg = result[0]
    assert pkg.path == tmp_path.resolve()
    assert pkg.name == "x"
    assert pkg.exported == ((tmp_path / "src").resolve(),)


def test_pyproject_resolver_no_pyproject(tmp_path: Path):
    assert PyprojectResolver().resolve(tmp_path) == []
