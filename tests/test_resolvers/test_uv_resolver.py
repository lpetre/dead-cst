"""Tests for :class:`dead_cst.contrib.uv_resolver.UvResolver`."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst.resolvers import UvResolver


def _write_uv(tmp_path: Path, *, with_src: bool = True) -> None:
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


def _by_name(packages):
    return {p.name: p for p in packages}


def test_uv_resolver_src_layout(tmp_path: Path):
    _write_uv(tmp_path)

    result = UvResolver().resolve(tmp_path)
    by = _by_name(result)

    core_dir = (tmp_path / "packages" / "core").resolve()
    app_dir = (tmp_path / "packages" / "app").resolve()
    assert by["core"].path == core_dir
    assert by["core"].exported == ((core_dir / "src").resolve(),)
    assert by["core"].deps == ()
    assert by["app"].path == app_dir
    assert by["app"].exported == ((app_dir / "src").resolve(),)
    assert by["app"].deps == ("core",)


def test_uv_resolver_flat_layout(tmp_path: Path):
    _write_uv(tmp_path, with_src=False)

    result = UvResolver().resolve(tmp_path)
    by = _by_name(result)

    core_dir = (tmp_path / "packages" / "core").resolve()
    app_dir = (tmp_path / "packages" / "app").resolve()
    assert by["core"].path == core_dir
    assert by["app"].path == app_dir
    assert by["app"].deps == ("core",)


def test_uv_resolver_skips_virtual_root(tmp_path: Path):
    _write_uv(tmp_path)

    result = UvResolver().resolve(tmp_path)

    # The "ws" package has source = { virtual = "." } and must not appear.
    assert tmp_path.resolve() not in {p.path for p in result}


def test_uv_resolver_includes_virtual_members(tmp_path: Path):
    """Virtual members (apps/services that don't ship as wheels) are first-party
    code and must be analyzed alongside editable members."""
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "ws"
            version = "0.0.0"
            [tool.uv.workspace]
            members = ["apps/*", "libs/*"]
        """).strip()
    )
    for kind, name in (("apps", "app-a"), ("libs", "lib-a")):
        (tmp_path / kind / name / "src" / name.replace("-", "_")).mkdir(parents=True)
    (tmp_path / "uv.lock").write_text(
        textwrap.dedent("""
            version = 1
            revision = 3
            requires-python = ">=3.11"

            [manifest]
            members = ["app-a", "lib-a", "ws"]

            [[package]]
            name = "app-a"
            version = "0.0.0"
            source = { virtual = "apps/app-a" }
            dependencies = [
                { name = "lib-a" },
            ]

            [[package]]
            name = "lib-a"
            version = "0.0.0"
            source = { editable = "libs/lib-a" }

            [[package]]
            name = "ws"
            version = "0.0.0"
            source = { virtual = "." }
        """).strip()
    )

    result = UvResolver().resolve(tmp_path)
    by = _by_name(result)
    lib_dir = (tmp_path / "libs" / "lib-a").resolve()
    app_dir = (tmp_path / "apps" / "app-a").resolve()
    assert by["lib-a"].path == lib_dir
    assert by["app-a"].path == app_dir
    assert by["app-a"].deps == ("lib-a",)


def test_uv_resolver_no_lockfile(tmp_path: Path):
    assert UvResolver().resolve(tmp_path) == []


def test_uv_resolver_ignores_non_workspace_deps(tmp_path: Path):
    """Deps that aren't workspace members are dropped silently -- they
    don't have a source dir under our control."""
    _write_uv(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text(
        lock.read_text().replace(
            'dependencies = [\n    { name = "core" },\n]',
            'dependencies = [\n    { name = "core" },\n    { name = "requests" },\n]',
        )
    )

    result = UvResolver().resolve(tmp_path)
    by = _by_name(result)
    assert by["app"].deps == ("core",)


def test_uv_resolver_explicit_lock_path(tmp_path: Path):
    _write_uv(tmp_path)
    moved = tmp_path / "stash" / "uv.lock"
    moved.parent.mkdir()
    moved.write_text((tmp_path / "uv.lock").read_text())
    (tmp_path / "uv.lock").unlink()

    result = UvResolver(lock_path=moved).resolve(tmp_path)
    assert result  # non-empty -- lock_path override took effect


def test_uv_src_layout_walks_tests_dir(tmp_path: Path):
    """src-layout members include sibling ``tests/`` etc. as internal
    files (everything under ``path`` not under ``exported``), so they
    get walked by phase 2."""
    from dead_cst import Analysis

    _write_uv(tmp_path)
    member = tmp_path / "packages" / "core"
    (member / "src" / "core" / "__init__.py").write_text("def used(): pass\n")
    (member / "tests").mkdir()
    (member / "tests" / "__init__.py").write_text("")
    (member / "tests" / "test_core.py").write_text("from core import used\nused()\n")

    graph = Analysis(tmp_path, resolvers=[UvResolver()]).materialize_all()

    # tests/test_core.py was walked (it's in the graph as a module).
    test_module = next(
        (n for n in graph.nodes if n.fqname == "tests.test_core" and n.type == "module"),
        None,
    )
    assert test_module is not None, "tests/test_core.py should be walked"
    assert test_module.path == (member / "tests" / "test_core.py").resolve()


def test_uv_flat_layout_with_tests_dirs(tmp_path: Path):
    """Two members with ``tests/`` packages used to collide when their tries were merged.

    With per-package export tries (and only the EXPORTED tree's decls in the
    export trie), the dep's ``tests/`` is never merged into the consumer's
    lookup, so analysis succeeds and the cross-member ``import foo.c.mod``
    resolves."""
    from dead_cst import Analysis

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

    graph = Analysis(tmp_path, resolvers=[UvResolver()]).materialize_all()

    # The cross-member import resolved: pkg_a/app.py -> libc/foo/c/mod.y
    mod_y = next(n for n in graph.nodes if n.fqname == "foo.c.mod.y" and n.type == "variable")
    app_import = next(
        n
        for n in graph.nodes
        if n.type == "import" and n.path == (pkg_a / "pkg_a" / "app.py").resolve()
    )
    assert graph.has_edge(app_import, mod_y)


def test_uv_shared_namespace_package(tmp_path: Path):
    """PEP 420 namespace shared across workspace members."""
    from dead_cst import Analysis

    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "ws"
            version = "0.1"
            [tool.uv.workspace]
            members = ["packages/*"]
        """).strip()
    )
    (tmp_path / "uv.lock").write_text(
        textwrap.dedent("""
            version = 1
            [[package]]
            name = "foo-b"
            version = "0.1"
            source = { editable = "packages/foo-b" }
            dependencies = [{ name = "foo-a" }]

            [[package]]
            name = "foo-a"
            version = "0.1"
            source = { editable = "packages/foo-a" }
        """).strip()
    )

    foo_a = tmp_path / "packages" / "foo-a"
    foo_b = tmp_path / "packages" / "foo-b"
    (foo_a / "foo" / "a").mkdir(parents=True)
    (foo_b / "foo" / "b").mkdir(parents=True)

    (foo_a / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo-a"
            version = "0.1"
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo/a"]
        """).strip()
    )
    (foo_a / "foo" / "a" / "__init__.py").write_text("value = 42\n")

    (foo_b / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "foo-b"
            version = "0.1"
            dependencies = ["foo-a"]
            [build-system]
            build-backend = "hatchling.build"
            [tool.hatch.build.targets.wheel]
            packages = ["foo/b"]
        """).strip()
    )
    (foo_b / "foo" / "b" / "__init__.py").write_text("from foo.a import value\n\nresult = value\n")

    graph = Analysis(tmp_path, resolvers=[UvResolver()]).materialize_all()

    value = next(n for n in graph.nodes if n.fqname == "foo.a.value" and n.type == "variable")
    result = next(n for n in graph.nodes if n.fqname == "foo.b.result" and n.type == "variable")
    assert value.path == (foo_a / "foo" / "a" / "__init__.py").resolve()
    assert result.path == (foo_b / "foo" / "b" / "__init__.py").resolve()

    b_import = next(
        n
        for n in graph.nodes
        if n.type == "import" and n.path == (foo_b / "foo" / "b" / "__init__.py").resolve()
    )
    assert graph.has_edge(b_import, value)
