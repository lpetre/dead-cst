"""End-to-end tests for the ``test/testcase`` plugin flag and
``kept_alive_by_flags_only(test/testcase)``.

Test plugins (pytest, unittest) declare and stamp their seed
nodes with the registered ``test/testcase`` flag (resolved by name via
``node_flag("test/testcase")``). Default reachability treats those seeds
the same as any other entrypoint; the flag-taking blast-radius query
returns production code currently kept alive only because tests still
touch it.
"""

from __future__ import annotations

from dead_cst import _native as native


def find_reachable_excluding_tests(graph):
    testcase = graph.node_flag("test/testcase") or 0
    return set(graph.reachable(seed_flags=graph.default_seed_mask() & ~testcase))


def find_kept_alive_by_tests_only(graph):
    full = set(graph.reachable(seed_flags=graph.default_seed_mask()))
    return full - find_reachable_excluding_tests(graph)


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
    graph = make_analysis(plugins=[native.NativePlugin.pytest()]).materialize_all()
    helper = next(n for n in graph.nodes() if n.fqname == "pkg.lib.helper")
    assert helper in set(graph.reachable(seed_flags=graph.default_seed_mask()))
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
    graph = make_analysis(
        plugins=[native.NativePlugin.main_block(), native.NativePlugin.pytest()]
    ).materialize_all()
    helper = next(n for n in graph.nodes() if n.fqname == "pkg.lib.helper")
    # The main_block plugin keeps it alive without any test seed.
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
    graph = make_analysis(plugins=[native.NativePlugin.unittest()]).materialize_all()
    helper = next(n for n in graph.nodes() if n.fqname == "pkg.lib.helper")
    assert helper in set(graph.reachable(seed_flags=graph.default_seed_mask()))
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
    analysis = make_analysis(plugins=[native.NativePlugin.pytest()])
    blast = analysis.kept_alive_by_flags_only(analysis.node_flag("test/testcase"))
    ctx = analysis.materialize_all()
    fqnames = {a.fqname for a in ctx.node_attrs(list(blast))}
    assert "pkg.lib.helper" in fqnames
