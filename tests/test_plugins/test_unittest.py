"""Tests for :class:`UnittestPlugin`."""

from __future__ import annotations


from dead_cst.graph import NodeFlags
from dead_cst.plugins import UnittestPlugin


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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
    )
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("unittest")
    assert isinstance(plugin, UnittestPlugin)


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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
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
        [UnittestPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.a.L1" in reached
    assert "tests.b.L2" in reached
    assert "tests.c.L3" in reached
