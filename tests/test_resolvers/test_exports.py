"""Tests for :func:`dead_cst.resolvers.exported_roots` and
:func:`dead_cst.resolvers.exported_tree_root`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst.resolvers import exported_roots, exported_tree_root


def test_exported_roots_no_pyproject(tmp_path: Path):
    assert exported_roots(tmp_path) is None


def test_exported_roots_src_layout_overrides_backend(tmp_path: Path):
    (tmp_path / "src" / "foo").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["nope"]
        """).strip()
    )
    assert exported_roots(tmp_path) == [(tmp_path / "src").resolve()]


def test_exported_roots_hatch_packages(tmp_path: Path):
    (tmp_path / "foo").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo", "tools"]
        """).strip()
    )
    assert exported_roots(tmp_path) == [
        (tmp_path / "foo").resolve(),
        (tmp_path / "tools").resolve(),
    ]


def test_exported_roots_setuptools_packages(tmp_path: Path):
    (tmp_path / "foo" / "bar").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo"
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
    (tmp_path / "lib" / "foo").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo"
            [build-system]
            build-backend = "poetry.core.masonry.api"
            [tool.poetry]
            packages = [{ include = "foo", from = "lib" }]
        """).strip()
    )
    assert exported_roots(tmp_path) == [(tmp_path / "lib" / "foo").resolve()]


def test_exported_roots_flit_module(tmp_path: Path):
    (tmp_path / "foo").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo"
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
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "my-pkg"
        """).strip()
    )
    assert exported_roots(tmp_path) == [(tmp_path / "my_pkg").resolve()]


def test_exported_roots_name_match_requires_init_py(tmp_path: Path):
    (tmp_path / "my_pkg").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "my-pkg"
        """).strip()
    )
    assert exported_roots(tmp_path) is None


def test_exported_roots_unknown_backend_with_no_match_returns_none(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [build-system]
            build-backend = "weird.unknown"
        """).strip()
    )
    assert exported_roots(tmp_path) is None


def test_exported_tree_root_no_pyproject(tmp_path: Path):
    assert exported_tree_root(tmp_path) is None


def test_exported_tree_root_src_layout_returns_src(tmp_path: Path):
    (tmp_path / "src" / "foo").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "foo"\n')
    assert exported_tree_root(tmp_path) == (tmp_path / "src").resolve()


def test_exported_tree_root_flat_packages_returns_project_root(tmp_path: Path):
    (tmp_path / "foo").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo"]
        """).strip()
    )
    assert exported_tree_root(tmp_path) == tmp_path.resolve()


def test_exported_tree_root_nested_packages_returns_project_root(tmp_path: Path):
    """``packages = ["foo/a"]`` ships the import-name ``foo.a``; the
    sys.path entry that resolves it is the project root, not
    ``project_root/foo``."""
    (tmp_path / "foo" / "a").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo-a"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo/a"]
        """).strip()
    )
    assert exported_tree_root(tmp_path) == tmp_path.resolve()


def test_exported_tree_root_no_match_falls_back_to_project_root(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "foo"\n')
    assert exported_tree_root(tmp_path) == tmp_path.resolve()
