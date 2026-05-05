"""Tests for :func:`dead_cst.resolvers.exported_roots` backend dispatch."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst.resolvers import exported_roots


def test_exported_roots_no_pyproject(tmp_path: Path):
    assert exported_roots(tmp_path) is None


def test_exported_roots_src_layout_overrides_backend(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["should_be_ignored"]
        """).strip()
    )
    assert exported_roots(tmp_path) == [(tmp_path / "src").resolve()]


def test_exported_roots_hatch_packages(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo", "bar"]
        """).strip()
    )
    assert exported_roots(tmp_path) == [
        (tmp_path / "foo").resolve(),
        (tmp_path / "bar").resolve(),
    ]


def test_exported_roots_setuptools_packages(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"
            [build-system]
            build-backend = "setuptools.build_meta"
            [tool.setuptools]
            packages = ["foo", "foo.bar"]
        """).strip()
    )
    assert exported_roots(tmp_path) == [
        (tmp_path / "foo").resolve(),
        (tmp_path / "foo" / "bar").resolve(),
    ]


def test_exported_roots_poetry_packages(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"
            [build-system]
            build-backend = "poetry.core.masonry.api"
            [[tool.poetry.packages]]
            include = "foo"
            from = "lib"
        """).strip()
    )
    assert exported_roots(tmp_path) == [(tmp_path / "lib" / "foo").resolve()]


def test_exported_roots_flit_module(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"
            [build-system]
            build-backend = "flit_core.buildapi"
            [tool.flit.module]
            name = "foo"
        """).strip()
    )
    assert exported_roots(tmp_path) == [(tmp_path / "foo").resolve()]


def test_exported_roots_name_match_fallback(tmp_path: Path):
    (tmp_path / "my_pkg").mkdir()
    (tmp_path / "my_pkg" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "my-pkg"
            [build-system]
            build-backend = "hatchling.build"
        """).strip()
    )
    # Hyphen in [project].name normalizes to underscore for the dir match.
    assert exported_roots(tmp_path) == [(tmp_path / "my_pkg").resolve()]


def test_exported_roots_name_match_requires_init_py(tmp_path: Path):
    (tmp_path / "my_pkg").mkdir()  # no __init__.py
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "my_pkg"
            [build-system]
            build-backend = "hatchling.build"
        """).strip()
    )
    assert exported_roots(tmp_path) is None


def test_exported_roots_unknown_backend_with_no_match_returns_none(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "x"
            [build-system]
            build-backend = "some.unknown.backend"
        """).strip()
    )
    assert exported_roots(tmp_path) is None
