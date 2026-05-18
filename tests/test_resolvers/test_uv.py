"""Tests for :class:`dead_cst.contrib.uv.UvResolver`.

Includes an end-to-end regression test for the cross-member ``tests/``
collision that motivated :func:`dead_cst.resolvers.exported_roots`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst.contrib.uv import MissingVenvError
from dead_cst.resolvers import UvResolver


def _make_fake_venv(workspace_root: Path) -> Path:
    """Create a minimal ``.venv/lib/pythonX.Y/site-packages`` and return it.

    ``UvResolver`` requires a populated venv -- in real usage,
    ``uv sync --all-packages`` puts one at the workspace root. Tests
    create the directory structure the resolver looks for and return
    the resolved ``site-packages`` path so assertions can include it
    in expected dep lists.
    """
    sp = workspace_root / ".venv" / "lib" / "python3.13" / "site-packages"
    sp.mkdir(parents=True)
    return sp.resolve()


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


def test_uv_resolver_src_layout(tmp_path: Path):
    _write_uv_workspace(tmp_path)
    _make_fake_venv(tmp_path)

    result = UvResolver().resolve(tmp_path)
    by_name = {p.name: p for p in result}

    core_src = (tmp_path / "packages" / "core" / "src").resolve()
    app_src = (tmp_path / "packages" / "app" / "src").resolve()
    assert by_name["core"].path == core_src
    assert by_name["core"].deps == ()
    assert by_name["app"].path == app_src
    assert by_name["app"].deps == ("core",)


def test_uv_resolver_flat_layout(tmp_path: Path):
    _write_uv_workspace(tmp_path, with_src=False)
    _make_fake_venv(tmp_path)

    result = UvResolver().resolve(tmp_path)
    by_name = {p.name: p for p in result}

    core_dir = (tmp_path / "packages" / "core").resolve()
    app_dir = (tmp_path / "packages" / "app").resolve()
    assert by_name["core"].path == core_dir
    assert by_name["core"].deps == ()
    assert by_name["app"].path == app_dir
    assert by_name["app"].deps == ("core",)


def test_uv_resolver_skips_virtual_root(tmp_path: Path):
    _write_uv_workspace(tmp_path)
    _make_fake_venv(tmp_path)

    result = UvResolver().resolve(tmp_path)

    # The "ws" package has source = { virtual = "." } and must not appear.
    assert tmp_path.resolve() not in {p.path for p in result}


def test_uv_resolver_missing_venv_raises(tmp_path: Path, monkeypatch):
    """Workspace with no synced ``.venv`` raises an actionable error
    instead of silently producing wrong results downstream."""
    import sys

    _write_uv_workspace(tmp_path)
    # Override the active-venv fallback so the resolver can't accidentally
    # find one outside the workspace (e.g. the test runner's own venv).
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)

    with pytest.raises(MissingVenvError, match="uv sync"):
        UvResolver().resolve(tmp_path)


def test_uv_resolver_includes_virtual_members(tmp_path: Path):
    """Virtual members (apps/services that don't ship as wheels) are first-party
    code and must be analyzed alongside editable members.

    Regression for the silent drop documented in issue #32: a workspace mixing
    ``apps/*`` (virtual) with ``libs/*`` (editable) used to skip the apps,
    causing libraries whose only consumers were apps to be reported as dead.
    """
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

    _make_fake_venv(tmp_path)
    result = UvResolver().resolve(tmp_path)
    by_name = {p.name: p for p in result}

    lib_src = (tmp_path / "libs" / "lib-a" / "src").resolve()
    app_src = (tmp_path / "apps" / "app-a" / "src").resolve()
    assert by_name["lib-a"].path == lib_src
    assert by_name["lib-a"].deps == ()
    assert by_name["app-a"].path == app_src
    assert by_name["app-a"].deps == ("lib-a",)


def test_uv_resolver_no_lockfile(tmp_path: Path):
    # No lockfile => not a uv workspace => silent no-op (don't raise on the
    # missing venv, since the resolver isn't applicable here).
    assert UvResolver().resolve(tmp_path) == ()


def test_uv_resolver_ignores_non_workspace_deps(tmp_path: Path):
    """Deps that aren't workspace members (e.g. regular PyPI deps) are dropped
    silently -- they don't have a source dir under our control."""
    _write_uv_workspace(tmp_path)
    _make_fake_venv(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text(
        lock.read_text().replace(
            'dependencies = [\n    { name = "core" },\n]',
            'dependencies = [\n    { name = "core" },\n    { name = "requests" },\n]',
        )
    )

    result = UvResolver().resolve(tmp_path)
    by_name = {p.name: p for p in result}
    assert by_name["app"].deps == ("core",)


def test_uv_resolver_explicit_lock_path(tmp_path: Path):
    _write_uv_workspace(tmp_path)
    _make_fake_venv(tmp_path)
    moved = tmp_path / "stash" / "uv.lock"
    moved.parent.mkdir()
    moved.write_text((tmp_path / "uv.lock").read_text())
    (tmp_path / "uv.lock").unlink()

    result = UvResolver(lock_path=moved).resolve(tmp_path)
    assert result  # non-empty -- lock_path override took effect


def test_uv_workspace_flat_layout_with_tests_dirs(tmp_path: Path):
    """Regression for the AssertionError in the issue: two members with
    ``tests/`` packages used to collide when their tries were merged.

    With per-consumer export scoping, the dep's ``tests/`` is never
    merged into the consumer's lookup trie, so analysis succeeds and the
    real cross-member ``import foo.c.mod`` resolves."""
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

    _make_fake_venv(tmp_path)
    # No AssertionError -- this used to crash before the fix.
    graph = Analysis(tmp_path, resolver=UvResolver()).materialize_all()

    # The cross-member import resolved: pkg_a/app.py -> libc/foo/c/mod.y
    mod_y = next(n for n in graph.nodes if n.fqname == "foo.c.mod.y" and n.type == "variable")
    app_import = next(
        n
        for n in graph.nodes
        if n.type == "import" and n.path == (pkg_a / "pkg_a" / "app.py").resolve()
    )
    assert graph.raw.has_edge(graph.index(app_import), graph.index(mod_y))


def test_uv_workspace_shared_namespace_package(tmp_path: Path):
    """PEP 420 namespace shared across workspace members (Google-style monorepo).

    Two distributions contribute submodules to the same top-level ``foo``
    namespace -- ``foo-a`` ships ``foo/a/`` and ``foo-b`` ships ``foo/b/``,
    with no ``foo/__init__.py`` in either. A cross-member import
    ``from foo.a import value`` from inside ``foo.b`` must resolve through
    PEP 420 namespace merging of the two ``foo/`` dirs on the search path.
    """
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

    # Note: no foo/__init__.py in either member -- PEP 420 namespace package.
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

    _make_fake_venv(tmp_path)
    resolver = UvResolver()
    packages = resolver.resolve(tmp_path)
    by_name = {p.name: p for p in packages}
    foo_a_dir = foo_a.resolve()
    foo_b_dir = foo_b.resolve()
    assert by_name["foo-a"].path == foo_a_dir
    assert by_name["foo-a"].deps == ()
    assert by_name["foo-b"].path == foo_b_dir
    assert by_name["foo-b"].deps == ("foo-a",)

    graph = Analysis(tmp_path, resolver=resolver).materialize_all()

    # foo.a.value (in foo-a) and foo.b.result (in foo-b) both made it into
    # the graph as distinct variables under the shared ``foo`` namespace.
    value = next(n for n in graph.nodes if n.fqname == "foo.a.value" and n.type == "variable")
    result = next(n for n in graph.nodes if n.fqname == "foo.b.result" and n.type == "variable")
    assert value.path == (foo_a / "foo" / "a" / "__init__.py").resolve()
    assert result.path == (foo_b / "foo" / "b" / "__init__.py").resolve()

    # The cross-member ``from foo.a import value`` in foo-b resolved across
    # the namespace boundary back to foo-a's declaration.
    b_import = next(
        n
        for n in graph.nodes
        if n.type == "import" and n.path == (foo_b / "foo" / "b" / "__init__.py").resolve()
    )
    assert graph.raw.has_edge(graph.index(b_import), graph.index(value))
