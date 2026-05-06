"""Tests for :class:`UnittestPlugin`."""

from __future__ import annotations

from dead_cst import Analysis
from dead_cst.plugins import UnittestPlugin


def test_unittest_plugin_marks_testcase_subclass(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.things.MyThings" in reached
    assert "pkg.things.Helper" not in reached


def test_unittest_plugin_handles_from_import(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.things.MyThings" in reachable_fqnames(graph)


def test_unittest_plugin_handles_aliased_imports(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.aliased_module.ModAliased" in reached
    assert "pkg.aliased_class.ClsAliased" in reached


def test_unittest_plugin_marks_async_testcase(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.things.MyAsync" in reachable_fqnames(graph)


def test_unittest_plugin_marks_module_hooks(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.things.setUpModule" in reached
    assert "pkg.things.tearDownModule" in reached
    assert "pkg.things.load_tests" in reached
    assert "pkg.things.helper" not in reached


def test_unittest_plugin_ignores_unrelated_classes(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    reached = reachable_fqnames(graph)
    # Bare ``TestCase`` is the locally-defined class (we never imported the
    # name from unittest), so the subclass is not picked up.
    assert "pkg.things.NotReally" not in reached


def test_unittest_plugin_skips_files_not_importing_unittest(
    tmp_path, write_files, reachable_fqnames
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.things.MyThings" not in reachable_fqnames(graph)


def test_unittest_plugin_skips_pure_star_import(tmp_path, write_files, reachable_fqnames):
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
    graph = Analysis(
        {tmp_path: []},
        plugins=[UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.things.MyThings" not in reachable_fqnames(graph)


def test_unittest_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("unittest")
    assert isinstance(plugin, UnittestPlugin)
