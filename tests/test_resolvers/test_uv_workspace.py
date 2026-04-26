"""Tests for :class:`dead_cst._resolvers.uv_workspace.UvWorkspaceResolver`.

Includes an end-to-end regression test for the cross-member ``tests/``
collision that motivated :func:`dead_cst._resolvers.exported_roots`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst import UvWorkspaceResolver


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


def test_uv_workspace_flat_layout_with_tests_dirs(tmp_path: Path):
    """Regression for the AssertionError in the issue: two members with
    ``tests/`` packages used to collide when their tries were merged.

    With per-consumer export scoping, the dep's ``tests/`` is never
    merged into the consumer's lookup trie, so analysis succeeds and the
    real cross-member ``import foo.c.mod`` resolves."""
    from dead_cst import build_symbol_graph

    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "ws"
            version = "0.1"
            [tool.uv.workspace]
            members = ["pkg_a", "libc"]
        """).strip()
    )
    (tmp_path / "uv.lock").write_text(
        textwrap.dedent("""
            version = 1
            [[package]]
            name = "pkg-a"
            version = "0.1"
            source = { editable = "pkg_a" }
            dependencies = [{ name = "libc" }]

            [[package]]
            name = "libc"
            version = "0.1"
            source = { editable = "libc" }
        """).strip()
    )

    pkg_a = tmp_path / "pkg_a"
    libc = tmp_path / "libc"
    (pkg_a / "tests").mkdir(parents=True)
    (libc / "tests").mkdir(parents=True)
    (libc / "foo" / "c").mkdir(parents=True)

    (pkg_a / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "pkg_a"
            version = "0.1"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["pkg_a"]
        """).strip()
    )
    (pkg_a / "pkg_a").mkdir()
    (pkg_a / "pkg_a" / "__init__.py").write_text("")
    (pkg_a / "pkg_a" / "app.py").write_text("from foo.c.mod import y\n")
    (pkg_a / "tests" / "__init__.py").write_text("")
    (pkg_a / "tests" / "conftest.py").write_text("import pytest\n")

    (libc / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "libc"
            version = "0.1"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo"]
        """).strip()
    )
    (libc / "foo" / "__init__.py").write_text("")
    (libc / "foo" / "c" / "__init__.py").write_text("")
    (libc / "foo" / "c" / "mod.py").write_text("y = 1\n")
    (libc / "tests" / "__init__.py").write_text("")
    (libc / "tests" / "conftest.py").write_text("import pytest\n")

    paths = UvWorkspaceResolver().resolve(tmp_path)
    # No AssertionError -- this used to crash before the fix.
    graph = build_symbol_graph(paths)

    # Both members' tests modules exist as distinct nodes (full graph picture).
    tests_modules = [n for n in graph.nodes if n.type == "module" and n.fqname == "tests"]
    assert {n.path for n in tests_modules} == {
        (pkg_a / "tests" / "__init__.py").resolve(),
        (libc / "tests" / "__init__.py").resolve(),
    }

    # The cross-member import resolved: pkg_a/app.py -> libc/foo/c/mod.y
    mod_y = next(n for n in graph.nodes if n.fqname == "foo.c.mod.y" and n.type == "variable")
    app_import = next(
        n
        for n in graph.nodes
        if n.type == "import" and n.path == (pkg_a / "pkg_a" / "app.py").resolve()
    )
    assert graph.has_edge(app_import, mod_y)
