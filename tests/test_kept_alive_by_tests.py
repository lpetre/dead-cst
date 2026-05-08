"""End-to-end tests for :data:`NodeFlags.TESTCASE` and the
``kept_alive_by_tests_only`` queries.

Test plugins (pytest, unittest) stamp their synthetic seed nodes with
:data:`NodeFlags.TESTCASE` (in addition to :data:`NodeFlags.ENTRYPOINT`).
The analyzer mirrors that into ``graph.nodes[seed]["testcase"] = True``.

Default :func:`find_reachable` ignores the flag -- test seeds keep their
targets alive the same as any other entrypoint. The strict pass
:func:`find_reachable_excluding_tests` skips those seeds, and the diff
:func:`find_kept_alive_by_tests_only` is the "blast radius" of dropping
the test suite: production code that's currently kept alive only because
tests still touch it.
"""

from __future__ import annotations

from dead_cst.analyze import (
    _find_kept_alive_by_tests_only as find_kept_alive_by_tests_only,
    _find_reachable as find_reachable,
    _find_reachable_excluding_tests as find_reachable_excluding_tests,
)
from dead_cst.plugins import PytestPlugin, UnittestPlugin


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
    assert helper in find_reachable(graph)
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
    assert helper in find_reachable(graph)
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
    blast = analysis.kept_alive_by_tests_only()
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
    blast = analysis.package(package_path).kept_alive_by_tests_only()
    fqnames = {n.fqname for n in blast}
    assert "pkg.lib.helper" in fqnames
    # Filtered to nodes under this package.
    for n in blast:
        assert n.path.is_relative_to(package_path)
