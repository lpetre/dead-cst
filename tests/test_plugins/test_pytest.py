"""Tests for :class:`PytestPlugin`."""

from __future__ import annotations

from dead_cst.graph import NodeFlags
from dead_cst.contrib import PytestPlugin


def test_pytest_plugin_marks_test_functions(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/test_things.py": """
            def test_one(): pass
            def test_two(): pass
            def helper(): pass
            """,
        },
        [PytestPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.test_things.test_one" in reached
    assert "tests.test_things.test_two" in reached
    # ``helper`` is not a test function and is not referenced from one
    assert "tests.test_things.helper" not in reached


def test_pytest_plugin_recognizes_underscore_test_suffix(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/things_test.py": "def test_one(): pass",
        },
        [PytestPlugin()],
    )
    assert "pkg.things_test.test_one" in reachable_fqnames(graph)


def test_pytest_plugin_marks_test_classes(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/test_cls.py": """
            class TestThing:
                def test_a(self): pass
            class Helper:
                pass
            """,
        },
        [PytestPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.test_cls.TestThing" in reached
    assert "tests.test_cls.Helper" not in reached


def test_pytest_plugin_marks_conftest_decls(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/conftest.py": """
            import pytest

            @pytest.fixture
            def my_fixture():
                return 1

            def pytest_collection_modifyitems(config, items):
                pass

            collect_ignore = ["legacy.py"]
            """,
        },
        [PytestPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.conftest.my_fixture" in reached
    assert "tests.conftest.pytest_collection_modifyitems" in reached
    assert "tests.conftest.collect_ignore" in reached


def test_pytest_plugin_marks_decorated_fixtures_outside_conftest(
    build_plugin_graph, reachable_fqnames
):
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest
            from pytest import fixture

            @pytest.fixture
            def bare_fixture():
                return 1

            @pytest.fixture(scope="module")
            def parametrized_fixture():
                return 2

            @fixture
            def imported_fixture():
                return 3

            def not_a_fixture():
                return 4
            """,
        },
        [PytestPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.fixtures.bare_fixture" in reached
    assert "tests.fixtures.parametrized_fixture" in reached
    assert "tests.fixtures.imported_fixture" in reached
    assert "tests.fixtures.not_a_fixture" not in reached


def test_pytest_plugin_ignores_non_test_modules(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/utils.py": """
            def test_helper(): pass
            class TestData: pass
            """,
        },
        [PytestPlugin()],
    )
    reached = reachable_fqnames(graph)
    # ``utils.py`` isn't a pytest-discovered file even though its symbols
    # match the test_*/Test* naming
    assert "pkg.utils.test_helper" not in reached
    assert "pkg.utils.TestData" not in reached


def test_pytest_plugin_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("pytest")
    assert isinstance(plugin, PytestPlugin)


def test_pytest_plugin_tags_seeds_as_testcase(build_plugin_graph):
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/test_things.py": "def test_one(): pass",
            "tests/conftest.py": """
            import pytest

            @pytest.fixture
            def my_fixture(): return 1
            """,
        },
        [PytestPlugin()],
    )
    seeds = [n for n in graph.nodes() if n.flags & NodeFlags.TESTCASE]
    assert seeds, "expected pytest plugin to seed at least one TESTCASE node"
