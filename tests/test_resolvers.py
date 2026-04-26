"""Tests for :mod:`dead_cst._resolvers`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst import (
    PyprojectResolver,
    UvWorkspaceResolver,
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


def _write_uv_workspace(tmp_path: Path, *, with_src: bool = True) -> None:
    """Lay out a two-member uv workspace (core + app, app deps on core)."""
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "ws"
            version = "0.0.0"
            [tool.uv.workspace]
            members = ["packages/*"]
        """).strip()
    )
    for name in ("core", "app"):
        member_dir = tmp_path / "packages" / name
        if with_src:
            (member_dir / "src" / name).mkdir(parents=True)
        else:
            member_dir.mkdir(parents=True)
    (tmp_path / "uv.lock").write_text(
        textwrap.dedent("""
            version = 1
            revision = 3
            requires-python = ">=3.11"

            [manifest]
            members = ["app", "core", "ws"]

            [[package]]
            name = "app"
            version = "0.0.0"
            source = { editable = "packages/app" }
            dependencies = [
                { name = "core" },
            ]

            [[package]]
            name = "core"
            version = "0.0.0"
            source = { editable = "packages/core" }

            [[package]]
            name = "ws"
            version = "0.0.0"
            source = { virtual = "." }
        """).strip()
    )


def test_uv_workspace_resolver_src_layout(tmp_path: Path):
    _write_uv_workspace(tmp_path)

    result = UvWorkspaceResolver().resolve(tmp_path)

    core_src = (tmp_path / "packages" / "core" / "src").resolve()
    app_src = (tmp_path / "packages" / "app" / "src").resolve()
    assert result == {core_src: [], app_src: [core_src]}


def test_uv_workspace_resolver_flat_layout(tmp_path: Path):
    _write_uv_workspace(tmp_path, with_src=False)

    result = UvWorkspaceResolver().resolve(tmp_path)

    core_dir = (tmp_path / "packages" / "core").resolve()
    app_dir = (tmp_path / "packages" / "app").resolve()
    assert result == {core_dir: [], app_dir: [core_dir]}


def test_uv_workspace_resolver_skips_virtual_root(tmp_path: Path):
    _write_uv_workspace(tmp_path)

    result = UvWorkspaceResolver().resolve(tmp_path)

    # The "ws" package has source = { virtual = "." } and must not appear.
    assert tmp_path.resolve() not in result


def test_uv_workspace_resolver_no_lockfile(tmp_path: Path):
    assert UvWorkspaceResolver().resolve(tmp_path) == {}


def test_uv_workspace_resolver_ignores_non_workspace_deps(tmp_path: Path):
    """Deps that aren't workspace members (e.g. regular PyPI deps) are dropped
    silently -- they don't have a source dir under our control."""
    _write_uv_workspace(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text(
        lock.read_text().replace(
            'dependencies = [\n    { name = "core" },\n]',
            'dependencies = [\n    { name = "core" },\n    { name = "requests" },\n]',
        )
    )

    result = UvWorkspaceResolver().resolve(tmp_path)
    core_src = (tmp_path / "packages" / "core" / "src").resolve()
    app_src = (tmp_path / "packages" / "app" / "src").resolve()
    assert result[app_src] == [core_src]


def test_uv_workspace_resolver_explicit_lock_path(tmp_path: Path):
    _write_uv_workspace(tmp_path)
    moved = tmp_path / "stash" / "uv.lock"
    moved.parent.mkdir()
    moved.write_text((tmp_path / "uv.lock").read_text())
    (tmp_path / "uv.lock").unlink()

    result = UvWorkspaceResolver(lock_path=moved).resolve(tmp_path)
    assert result  # non-empty -- lock_path override took effect


def test_load_resolver_known():
    assert isinstance(load_resolver("venv"), VenvResolver)
    assert isinstance(load_resolver("pyproject"), PyprojectResolver)
    assert isinstance(load_resolver("uv_workspace"), UvWorkspaceResolver)


def test_load_resolver_unknown_raises():
    with pytest.raises(KeyError):
        load_resolver("does-not-exist")
