"""Tests for :mod:`dead_cst._resolvers`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst import (
    PyprojectResolver,
    VenvResolver,
    load_resolver,
    merge_paths,
)


def test_venv_resolver_finds_site_packages(tmp_path: Path):
    venv = tmp_path / ".venv"
    sp = venv / "lib" / "python3.13" / "site-packages"
    sp.mkdir(parents=True)

    result = VenvResolver().resolve(tmp_path)
    assert tmp_path.resolve() in result
    assert sp.resolve() in result[tmp_path.resolve()]


def test_venv_resolver_missing_returns_empty(tmp_path: Path):
    # Passing an explicit (nonexistent) venv_dir skips the active-venv probe.
    assert VenvResolver(venv_dir="nope").resolve(tmp_path) == {}


def test_venv_resolver_custom_dir(tmp_path: Path):
    venv = tmp_path / "envs" / "myenv"
    sp = venv / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)

    result = VenvResolver(venv_dir="envs/myenv").resolve(tmp_path)
    assert sp.resolve() in result[tmp_path.resolve()]


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


def test_merge_paths_unions_deps():
    base = Path("/a")
    d1 = Path("/b")
    d2 = Path("/c")
    result = merge_paths({base: [d1]}, {base: [d2]}, {base: [d1]})
    assert result == {base: [d1, d2]}


def test_merge_paths_drops_self():
    base = Path("/a")
    result = merge_paths({base: [base]})
    assert result == {base: []}


def test_load_resolver_known():
    assert isinstance(load_resolver("venv"), VenvResolver)
    assert isinstance(load_resolver("pyproject"), PyprojectResolver)


def test_load_resolver_unknown_raises():
    with pytest.raises(KeyError):
        load_resolver("does-not-exist")
