"""End-to-end tests for :data:`NodeFlags.TESTCASE` and ``kept_alive_by_flags_only(NodeFlags.TESTCASE)``.

Test plugins (pytest, unittest) stamp their synthetic seed nodes with
``ENTRYPOINT | TESTCASE``. Default :func:`find_reachable` treats those
seeds the same as any other entrypoint; the flag-taking blast-radius
query returns production code currently kept alive only because tests
still touch it.
"""

from __future__ import annotations

from dead_cst import NodeFlags
from dead_cst.analyze import (
    _entrypoint_seeds,
    _find_kept_alive_by_flags_only,
    _find_reachable as find_reachable,
)
from dead_cst.plugins import PytestPlugin, UnittestPlugin


def find_reachable_excluding_tests(graph):
    return find_reachable(graph, _entrypoint_seeds(graph, NodeFlags.TESTCASE))


def find_kept_alive_by_tests_only(graph):
    return _find_kept_alive_by_flags_only(graph, NodeFlags.TESTCASE)


def test_test_only_helper_is_kept_alive_by_tests(make_analysis, write_files):
    """A helper exercised only by tests shows up in
    ``kept_alive_by_tests_only`` but not in the strict reachable set."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": """
            def helper(): return 1
            """,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import helper

            def test_helper(): assert helper() == 1
            """,
        }
    )
    graph = make_analysis(plugins=[PytestPlugin()]).materialize_all()
    helper = next(n for n in graph.nodes if n.fqname == "pkg.lib.helper")
    assert helper in find_reachable(graph, _entrypoint_seeds(graph))
    assert helper not in find_reachable_excluding_tests(graph)
    assert helper in find_kept_alive_by_tests_only(graph)


def test_production_only_decl_survives_strict_pass(make_analysis, write_files):
    """A decl reachable from a non-test entrypoint is not in
    ``kept_alive_by_tests_only``."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": """
            def helper(): return 1

            if __name__ == "__main__":
                helper()
            """,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import helper

            def test_helper(): assert helper() == 1
            """,
        }
    )
    from dead_cst.plugins import MainBlockPlugin

    graph = make_analysis(plugins=[MainBlockPlugin(), PytestPlugin()]).materialize_all()
    helper = next(n for n in graph.nodes if n.fqname == "pkg.lib.helper")
    # ``MainBlockPlugin`` keeps it alive without any test seed.
    assert helper in find_reachable_excluding_tests(graph)
    assert helper not in find_kept_alive_by_tests_only(graph)


def test_unittest_kept_alive_by_tests(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1\n",
            "pkg/things.py": """
            import unittest
            from pkg.lib import helper

            class MyThings(unittest.TestCase):
                def test_one(self): assert helper() == 1
            """,
        }
    )
    graph = make_analysis(plugins=[UnittestPlugin()]).materialize_all()
    helper = next(n for n in graph.nodes if n.fqname == "pkg.lib.helper")
    assert helper in find_reachable(graph, _entrypoint_seeds(graph))
    assert helper in find_kept_alive_by_tests_only(graph)


def test_analysis_method_returns_strict_diff(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1\n",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import helper

            def test_helper(): assert helper() == 1
            """,
        }
    )
    analysis = make_analysis(plugins=[PytestPlugin()])
    blast = analysis.kept_alive_by_flags_only(NodeFlags.TESTCASE)
    fqnames = {n.fqname for n in blast}
    assert "pkg.lib.helper" in fqnames


def test_package_view_kept_alive_by_tests_only(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1\n",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import helper

            def test_helper(): assert helper() == 1
            """,
        }
    )
    analysis = make_analysis(plugins=[PytestPlugin()])
    package_path = analysis.packages[0].path
    blast = analysis.package(package_path).kept_alive_by_flags_only(NodeFlags.TESTCASE)
    fqnames = {n.fqname for n in blast}
    assert "pkg.lib.helper" in fqnames
    # Filtered to nodes under this package.
    for n in blast:
        assert n.path.is_relative_to(package_path)
