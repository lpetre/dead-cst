"""Tests for the native ``unittest`` plugin (``NativePlugin.unittest``)."""

from __future__ import annotations


from dead_cst import _native as native
from dead_cst.graph import NodeFlags


def test_unittest_plugin_marks_testcase_subclass(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class MyThings(unittest.TestCase):
                def test_one(self): pass

            class Helper:
                pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.things.MyThings" in reached
    assert "pkg.things.Helper" not in reached


def test_unittest_plugin_handles_from_import(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            from unittest import TestCase

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_handles_aliased_imports(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/aliased_module.py": """
            import unittest as ut

            class ModAliased(ut.TestCase):
                def test_one(self): pass
            """,
            "pkg/aliased_class.py": """
            from unittest import TestCase as TC

            class ClsAliased(TC):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.aliased_module.ModAliased" in reached
    assert "pkg.aliased_class.ClsAliased" in reached


def test_unittest_plugin_marks_async_testcase(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class MyAsync(unittest.IsolatedAsyncioTestCase):
                async def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    assert "pkg.things.MyAsync" in reachable_fqnames(graph)


def test_unittest_plugin_marks_module_hooks(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            def setUpModule(): pass
            def tearDownModule(): pass
            def load_tests(loader, tests, pattern): return tests
            def helper(): pass

            class MyThings(unittest.TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.things.setUpModule" in reached
    assert "pkg.things.tearDownModule" in reached
    assert "pkg.things.load_tests" in reached
    assert "pkg.things.helper" not in reached


def test_unittest_plugin_ignores_unrelated_classes(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class TestCase:
                pass

            # Inherits from a local class also named TestCase, not unittest's.
            class NotReally(TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    reached = reachable_fqnames(graph)
    # Bare ``TestCase`` is the locally-defined class (we never imported the
    # name from unittest), so the subclass is not picked up.
    assert "pkg.things.NotReally" not in reached


def test_unittest_plugin_skips_files_not_importing_unittest(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            class TestCase:
                pass

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    assert "pkg.things.MyThings" not in reachable_fqnames(graph)


def test_unittest_plugin_resolves_through_star_import(build_plugin_graph, reachable_fqnames):
    """ty's type hierarchy follows star imports, so ``class X(TestCase)``
    after ``from unittest import *`` resolves.
    """
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            from unittest import *

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("unittest")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "UnittestPlugin"


def test_native_unittest_full_walk(build_plugin_graph, reachable_fqnames):
    """``NativePlugin.unittest()`` walks the whole ``TestCase`` hierarchy
    (sync + async + cross-module subclass) and pins module lifecycle hooks,
    while leaving unrelated decls dead."""
    files = {
        "pkg/__init__.py": "",
        "pkg/things.py": """
        import unittest

        class MyThings(unittest.TestCase):
            def test_one(self): pass

        class AsyncThings(unittest.IsolatedAsyncioTestCase):
            async def test_two(self): pass

        class Helper: pass

        def setUpModule(): pass
        def tearDownModule(): pass
        def load_tests(loader, tests, pattern): return tests
        """,
        "pkg/derived.py": """
        from pkg.things import MyThings

        class Derived(MyThings):
            def test_three(self): pass
        """,
        "pkg/regular.py": "def untouched(): pass",
    }
    reached = reachable_fqnames(build_plugin_graph(files, [native.NativePlugin.unittest()]))
    assert {
        "pkg.things.MyThings",
        "pkg.things.AsyncThings",
        "pkg.things.setUpModule",
        "pkg.things.tearDownModule",
        "pkg.things.load_tests",
        "pkg.derived.Derived",
    } <= reached
    assert "pkg.things.Helper" not in reached
    assert "pkg.regular.untouched" not in reached


def test_unittest_plugin_tags_seeds_as_testcase(build_plugin_graph):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class MyThings(unittest.TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    seeds = [n for n in graph.nodes() if n.flags & NodeFlags.TESTCASE]
    assert seeds, "expected unittest plugin to seed at least one TESTCASE node"


def test_unittest_plugin_marks_subclass_via_local_mixin(build_plugin_graph, reachable_fqnames):
    """A test class inheriting from a project-local TestCase subclass is alive."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/base.py": """
            import unittest

            class ProjectTestCase(unittest.TestCase):
                pass
            """,
            "tests/things.py": """
            from tests.base import ProjectTestCase

            class MyThings(ProjectTestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.base.ProjectTestCase" in reached
    assert "tests.things.MyThings" in reached


def test_unittest_plugin_marks_subclass_via_reexport(build_plugin_graph, reachable_fqnames):
    """A class extending a re-exported ``TestCase`` (``from pkg.bases import TestCase``) is alive."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/bases.py": "from unittest import TestCase\n",
            "pkg/things.py": """
            from pkg.bases import TestCase

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_marks_three_level_subclass_chain(build_plugin_graph, reachable_fqnames):
    """Subclass of a subclass of a project mixin still resolves."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/a.py": """
            import unittest

            class L1(unittest.TestCase):
                pass
            """,
            "tests/b.py": """
            from tests.a import L1

            class L2(L1):
                pass
            """,
            "tests/c.py": """
            from tests.b import L2

            class L3(L2):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.a.L1" in reached
    assert "tests.b.L2" in reached
    assert "tests.c.L3" in reached


def test_unittest_plugin_marks_subclass_via_module_alias(
    build_plugin_graph, reachable_fqnames, monkeypatch
):
    """A module-level alias of an imported ``TestCase`` (``Base = TestCase``)
    resolves through the uniform binder ladder — no ``find_references`` walk."""
    monkeypatch.delenv("DEAD_CST_SUBCLASS_REF_FALLBACK", raising=False)
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            from unittest import TestCase

            Base = TestCase

            class MyThings(Base):
                def test_one(self): pass
            """,
        },
        [native.NativePlugin.unittest()],
    )
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_relative_reexport_needs_ref_fallback(
    build_plugin_graph, reachable_fqnames, monkeypatch
):
    """A ``TestCase`` re-exported through a *relative* import can't be resolved
    by the static binder ladder (``collect_all_imports_local`` doesn't apply
    relative-import resolution), so the subclass is found only when the opt-in
    ``DEAD_CST_SUBCLASS_REF_FALLBACK`` ``find_references`` walk is enabled."""
    files = {
        "pkg/__init__.py": "",
        "pkg/bases.py": "from unittest import TestCase\n",
        "pkg/things.py": """
        from .bases import TestCase

        class MyThings(TestCase):
            def test_one(self): pass
        """,
    }

    monkeypatch.delenv("DEAD_CST_SUBCLASS_REF_FALLBACK", raising=False)
    reached_off = reachable_fqnames(build_plugin_graph(files, [native.NativePlugin.unittest()]))
    assert "pkg.things.MyThings" not in reached_off

    monkeypatch.setenv("DEAD_CST_SUBCLASS_REF_FALLBACK", "1")
    reached_on = reachable_fqnames(build_plugin_graph(files, [native.NativePlugin.unittest()]))
    assert "pkg.things.MyThings" in reached_on
