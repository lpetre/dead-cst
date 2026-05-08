"""Tests for :class:`UnittestPlugin`."""

from __future__ import annotations

from dead_cst.plugins import UnittestPlugin


def test_unittest_plugin_marks_testcase_subclass(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class MyThings(unittest.TestCase):
                def test_one(self): pass

            class Helper:
                pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.things.MyThings" in reached
    assert "pkg.things.Helper" not in reached


def test_unittest_plugin_handles_from_import(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            from unittest import TestCase

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_handles_aliased_imports(make_analysis, write_files, reachable_fqnames):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.aliased_module.ModAliased" in reached
    assert "pkg.aliased_class.ClsAliased" in reached


def test_unittest_plugin_marks_async_testcase(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class MyAsync(unittest.IsolatedAsyncioTestCase):
                async def test_one(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    assert "pkg.things.MyAsync" in reachable_fqnames(graph)


def test_unittest_plugin_marks_module_hooks(make_analysis, write_files, reachable_fqnames):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.things.setUpModule" in reached
    assert "pkg.things.tearDownModule" in reached
    assert "pkg.things.load_tests" in reached
    assert "pkg.things.helper" not in reached


def test_unittest_plugin_ignores_unrelated_classes(make_analysis, write_files, reachable_fqnames):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    # Bare ``TestCase`` is the locally-defined class (we never imported the
    # name from unittest), so the subclass is not picked up.
    assert "pkg.things.NotReally" not in reached


def test_unittest_plugin_skips_files_not_importing_unittest(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            class TestCase:
                pass

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    assert "pkg.things.MyThings" not in reachable_fqnames(graph)


def test_unittest_plugin_skips_pure_star_import(make_analysis, write_files, reachable_fqnames):
    # Documented limitation: the resolver doesn't surface stdlib star
    # imports as graph nodes, so the prefilter can't see them. Users
    # should ``from unittest import TestCase`` instead.
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            from unittest import *

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    assert "pkg.things.MyThings" not in reachable_fqnames(graph)


def test_unittest_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("unittest")
    assert isinstance(plugin, UnittestPlugin)


def test_unittest_plugin_tags_seeds_as_testcase(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/things.py": """
            import unittest

            class MyThings(unittest.TestCase):
                def test_one(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    seeds = [n for n, attrs in graph.nodes(data=True) if attrs.get("entrypoint")]
    assert seeds, "expected unittest plugin to seed at least one entrypoint"
    for seed in seeds:
        assert graph.nodes[seed].get("testcase"), seed.fqname


def test_unittest_plugin_marks_subclass_via_local_mixin(
    make_analysis, write_files, reachable_fqnames
):
    """A test class inheriting from a project-local TestCase subclass is alive."""
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "tests.base.ProjectTestCase" in reached
    assert "tests.things.MyThings" in reached


def test_unittest_plugin_marks_subclass_via_reexport(make_analysis, write_files, reachable_fqnames):
    """A class extending a re-exported ``TestCase`` (``from pkg.bases import TestCase``) is alive."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/bases.py": "from unittest import TestCase\n",
            "pkg/things.py": """
            from pkg.bases import TestCase

            class MyThings(TestCase):
                def test_one(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_marks_three_level_subclass_chain(
    make_analysis, write_files, reachable_fqnames
):
    """Subclass of a subclass of a project mixin still resolves."""
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "tests.a.L1" in reached
    assert "tests.b.L2" in reached
    assert "tests.c.L3" in reached
