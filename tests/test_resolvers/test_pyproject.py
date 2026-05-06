"""Tests for :class:`dead_cst.resolvers.pyproject.PyprojectResolver`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst.resolvers import PyprojectResolver, SourceTreeFlags


def test_pyproject_resolver_reads_explicit_trees(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"

            [[tool.dead-cst.trees]]
            path = "src"
            package = "x"
            exported = true

            [[tool.dead-cst.trees]]
            path = "tests"
            package = "x"
            search_trees = ["src"]
        """).strip()
    )

    result = PyprojectResolver().resolve(tmp_path)
    assert [(t.path, t.package, bool(t.flags & SourceTreeFlags.EXPORTED)) for t in result] == [
        ((tmp_path / "src").resolve(), "x", True),
        ((tmp_path / "tests").resolve(), "x", False),
    ]
    assert result[1].search_trees == ((tmp_path / "src").resolve(),)


def test_pyproject_resolver_src_layout_fallback(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    result = PyprojectResolver().resolve(tmp_path)
    assert [t.path for t in result] == [(tmp_path / "src").resolve()]
    assert result[0].flags & SourceTreeFlags.EXPORTED
    assert result[0].package == "x"


def test_pyproject_resolver_src_layout_includes_tests(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    result = PyprojectResolver().resolve(tmp_path)
    assert [(t.path, bool(t.flags & SourceTreeFlags.EXPORTED)) for t in result] == [
        ((tmp_path / "src").resolve(), True),
        ((tmp_path / "tests").resolve(), False),
    ]
    # tests tree's search ref points back at src
    assert result[1].search_trees == ((tmp_path / "src").resolve(),)


def test_pyproject_resolver_no_pyproject(tmp_path: Path):
    assert PyprojectResolver().resolve(tmp_path) == []
