"""Tests for the native ``pytest`` plugin."""

from __future__ import annotations

from dead_cst import _native as native


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
        [native.NativePlugin.pytest()],
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
        [native.NativePlugin.pytest()],
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
        [native.NativePlugin.pytest()],
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
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.conftest.my_fixture" in reached
    assert "tests.conftest.pytest_collection_modifyitems" in reached
    assert "tests.conftest.collect_ignore" in reached


def test_pytest_plugin_marks_decorated_fixtures_outside_conftest(
    build_plugin_graph, reachable_fqnames
):
    """Every ``@pytest.fixture``-decorated function is unconditionally
    alive via the synthetic ``<pytest:fixtures>:<module>`` seed,
    regardless of whether any test parameter mentions it. The
    parameter-name edges are *additive* — they make the dependency
    queryable without changing the alive set."""
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
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.fixtures.bare_fixture" in reached
    assert "tests.fixtures.parametrized_fixture" in reached
    assert "tests.fixtures.imported_fixture" in reached
    assert "tests.fixtures.not_a_fixture" not in reached


def test_pytest_plugin_emits_test_to_fixture_edge(build_plugin_graph):
    """The ``test → fixture`` edge is the load-bearing artifact for
    the new fixture model. Verify it's actually in the graph (not just
    that reachability happens to land right)."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest

            @pytest.fixture
            def my_fixture():
                return 1
            """,
            "tests/test_x.py": """
            from tests.fixtures import my_fixture  # noqa: F401

            def test_uses(my_fixture):
                assert my_fixture == 1
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    test_idx = by_fqname["tests.test_x.test_uses"]
    fixture_idx = by_fqname["tests.fixtures.my_fixture"]
    assert (test_idx, fixture_idx, 0) in [(s, d, f) for (s, d, f) in graph.edges()]


def test_pytest_plugin_unrelated_param_no_edge(build_plugin_graph):
    """Parameter names that don't match any project fixture (pytest
    builtins like ``tmp_path``, third-party plugin fixtures like
    ``mocker``, free-form ``parametrize`` names) are silently ignored —
    no spurious edges to unrelated decls that happen to share the
    name."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/test_x.py": """
            def test_uses_builtin(tmp_path, capsys):
                pass
            """,
        },
        [native.NativePlugin.pytest()],
    )
    # No fixture ⇒ no edge out of test_uses_builtin.
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    test_idx = by_fqname["tests.test_x.test_uses_builtin"]
    out_edges = [(s, d, f) for (s, d, f) in graph.edges() if s == test_idx]
    # Only the module-anchor edge (test → its module).
    assert all(nodes[d].kind == "module" for (_, d, _) in out_edges)


def test_pytest_plugin_class_method_pulls_fixture_alive(build_plugin_graph, reachable_fqnames):
    """A fixture used only by a ``Test*`` class method must stay alive.
    We don't have method-level graph nodes, so the class is the
    rendezvous point — ``class → fixture`` edges by name match."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest

            @pytest.fixture
            def class_only_fixture():
                return 1
            """,
            "tests/test_cls.py": """
            from tests.fixtures import class_only_fixture  # noqa: F401

            class TestThing:
                def test_method(self, class_only_fixture):
                    assert class_only_fixture == 1
            """,
        },
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.fixtures.class_only_fixture" in reached


def test_pytest_plugin_emits_class_to_fixture_edge(build_plugin_graph):
    """The ``class → fixture`` edge is the load-bearing artifact for
    Test* class fixture deps. Verify it's actually in the graph."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest

            @pytest.fixture
            def my_fixture():
                return 1
            """,
            "tests/test_cls.py": """
            from tests.fixtures import my_fixture  # noqa: F401

            class TestThing:
                def test_a(self, my_fixture): pass
                def test_b(self): pass
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    cls_idx = by_fqname["tests.test_cls.TestThing"]
    fixture_idx = by_fqname["tests.fixtures.my_fixture"]
    assert (cls_idx, fixture_idx, 0) in [(s, d, f) for (s, d, f) in graph.edges()]


def test_pytest_plugin_class_self_cls_excluded(build_plugin_graph):
    """``self`` and ``cls`` parameter names must NOT produce edges
    even if a fixture happens to share the name."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest

            @pytest.fixture
            def self():
                return 1

            @pytest.fixture
            def cls():
                return 2
            """,
            "tests/test_cls.py": """
            from tests.fixtures import self, cls  # noqa: F401

            class TestThing:
                def test_a(self): pass
                @classmethod
                def test_b(cls): pass
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    cls_idx = by_fqname["tests.test_cls.TestThing"]
    bad_targets = {by_fqname["tests.fixtures.self"], by_fqname["tests.fixtures.cls"]}
    edges = [(s, d, f) for (s, d, f) in graph.edges() if s == cls_idx]
    assert not any(d in bad_targets for (_, d, _) in edges)


def test_pytest_plugin_fixture_name_kwarg_alias(build_plugin_graph):
    """``@pytest.fixture(name="alias")`` binds the fixture under the
    alias, not its function name. The ``test → fixture`` edge must
    resolve through the alias — verify the edge is present in the
    graph. (Reachability of the fixture itself is guaranteed by the
    unconditional fixture seed, so we check the edge directly.)"""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest

            @pytest.fixture(name="alias")
            def some_fn():
                return 1
            """,
            "tests/test_uses_alias.py": """
            from tests.fixtures import some_fn  # noqa: F401

            def test_takes_alias(alias):
                assert alias == 1
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    test_idx = by_fqname["tests.test_uses_alias.test_takes_alias"]
    fixture_idx = by_fqname["tests.fixtures.some_fn"]
    assert (test_idx, fixture_idx, 0) in [(s, d, f) for (s, d, f) in graph.edges()]


def test_pytest_plugin_conftest_fixture_unused_stays_alive(build_plugin_graph, reachable_fqnames):
    """Fixtures defined in a ``conftest.py`` are still seeded alive via
    the conftest-seeds-everything rule (they're often referenced by
    tests we don't model precisely). The test → fixture edge is
    *additive* — it pulls non-conftest fixtures alive when used."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/conftest.py": """
            import pytest

            @pytest.fixture
            def conftest_fixture():
                return 1
            """,
        },
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    # No test even uses it — conftest seed still keeps it alive.
    assert "tests.conftest.conftest_fixture" in reached


def test_pytest_plugin_ignores_non_test_modules(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/utils.py": """
            def test_helper(): pass
            class TestData: pass
            """,
        },
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    # ``utils.py`` isn't a pytest-discovered file even though its symbols
    # match the test_*/Test* naming
    assert "pkg.utils.test_helper" not in reached
    assert "pkg.utils.TestData" not in reached


def test_pytest_plugin_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("pytest")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "PytestPlugin"


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
        [native.NativePlugin.pytest()],
    )
    testcase = graph.node_flag("test/testcase")
    assert testcase is not None, "pytest plugin should register the test/testcase flag"
    seeds = [n for n in graph.nodes() if n.flags & testcase]
    assert seeds, "expected pytest plugin to seed at least one test/testcase node"
