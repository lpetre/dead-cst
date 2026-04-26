"""Tests for :class:`dead_cst._resolvers.pyproject.PyprojectResolver`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst import PyprojectResolver


def test_pyproject_resolver_reads_paths_section(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"

            [tool.dead-cst]
            paths = [
              { base = "src", deps = ["tests"] }
            ]
        """).strip()
    )

    result = PyprojectResolver().resolve(tmp_path)
    assert result == {
        (tmp_path / "src").resolve(): [(tmp_path / "tests").resolve()],
    }


def test_pyproject_resolver_src_layout_fallback(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    result = PyprojectResolver().resolve(tmp_path)
    assert result == {(tmp_path / "src").resolve(): []}


def test_pyproject_resolver_no_pyproject(tmp_path: Path):
    assert PyprojectResolver().resolve(tmp_path) == {}
