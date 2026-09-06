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
    alive via the ``test/fixture`` flag, regardless of whether any test
    parameter mentions it. The parameter-name edges are *additive* —
    they make the dependency queryable without changing the alive set."""
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
    ``test/fixture`` flag, so we check the edge directly.)"""
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
    """Fixtures defined in a ``conftest.py`` are still kept alive via the
    ``test/fixture`` flag (conftest decls are flagged wholesale — they're
    often referenced by tests we don't model precisely). The test →
    fixture edge is *additive* — it pulls non-conftest fixtures alive
    when used."""
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


def test_pytest_plugin_flag_split_testcase_vs_fixture(build_plugin_graph):
    """Genuine tests carry ``test/testcase``; fixtures and conftest decls
    carry the provisional ``test/fixture`` flag instead. The two are
    distinct bits and never overlap on a node."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/test_things.py": """
            import pytest

            def test_one(): pass

            @pytest.fixture
            def local_fixture(): return 1
            """,
            "tests/conftest.py": """
            import pytest

            @pytest.fixture
            def my_fixture(): return 1

            def pytest_configure(config): pass
            """,
        },
        [native.NativePlugin.pytest()],
    )
    testcase = graph.node_flag("test/testcase")
    fixture = graph.node_flag("test/fixture")
    assert testcase is not None, "pytest plugin should register test/testcase"
    assert fixture is not None, "pytest plugin should register test/fixture"
    assert testcase != fixture, "the two flags must be distinct bits"

    by_fqname = {n.fqname: n for n in graph.nodes()}

    # Genuine test → testcase only.
    test_one = by_fqname["tests.test_things.test_one"]
    assert test_one.flags & testcase
    assert not test_one.flags & fixture

    # Fixtures (in a test module or a conftest) → fixture only.
    for fq in ("tests.test_things.local_fixture", "tests.conftest.my_fixture"):
        node = by_fqname[fq]
        assert node.flags & fixture, f"{fq} should carry test/fixture"
        assert not node.flags & testcase, f"{fq} should not carry test/testcase"

    # Non-fixture conftest decls → fixture (conftest is flagged wholesale).
    configure = by_fqname["tests.conftest.pytest_configure"]
    assert configure.flags & fixture
    assert not configure.flags & testcase


def test_pytest_plugin_fixture_flag_is_measurable(build_plugin_graph):
    """Dropping the ``test/fixture`` bit from the seed mask removes exactly
    the fixture-kept-alive set — the blast-radius query the provisional
    flag exists to support."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/conftest.py": """
            import pytest

            @pytest.fixture
            def only_fixture(): return 1
            """,
        },
        [native.NativePlugin.pytest()],
    )
    fixture = graph.node_flag("test/fixture")
    assert fixture is not None
    seed = graph.default_seed_mask()
    assert seed & fixture, "test/fixture is a default-on seed"

    with_fixture = {n.fqname for n in graph.reachable(seed_flags=seed)}
    without_fixture = {n.fqname for n in graph.reachable(seed_flags=seed & ~fixture)}
    assert "tests.conftest.only_fixture" in with_fixture
    assert "tests.conftest.only_fixture" not in without_fixture


def test_pytest_plugin_matches_fixture_anywhere_in_decorator_stack(
    build_plugin_graph, reachable_fqnames
):
    """``decorated_decls`` reads the *whole* decorator list, so a
    ``@pytest.fixture`` is matched whether it sits on top, in the middle,
    or at the bottom of a stack -- including behind decorators the
    matcher cannot classify (``@(lambda f: f)``, ``@noop()(noop)``)."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import functools
            import pytest
            from pytest import fixture

            def noop(f):
                return f

            @pytest.fixture
            @noop
            def match_on_top():
                return 1

            @noop
            @pytest.fixture(scope="module")
            @functools.lru_cache(maxsize=None)
            def match_in_middle():
                return 2

            @noop
            @functools.wraps(noop)
            @fixture
            def match_at_bottom():
                return 3

            @noop()(noop)
            @(lambda f: f)
            @pytest.fixture
            def match_behind_unclassifiable():
                return 4

            @noop
            @functools.lru_cache(maxsize=None)
            def not_a_fixture():
                return 5
            """,
        },
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.fixtures.match_on_top" in reached
    assert "tests.fixtures.match_in_middle" in reached
    assert "tests.fixtures.match_at_bottom" in reached
    assert "tests.fixtures.match_behind_unclassifiable" in reached
    assert "tests.fixtures.not_a_fixture" not in reached


def test_pytest_plugin_reads_kwargs_from_matching_decorator_in_stack(build_plugin_graph):
    """``decorated_decls_with_args`` returns the kwargs of the decorator
    that *matched*, not of whichever decorator happens to be first in the
    stack: the ``name=`` alias on the inner ``@pytest.fixture`` is what
    the ``test -> fixture`` edge is keyed on."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import functools
            import pytest

            def noop(name=None):
                return lambda f: f

            @noop(name="decoy")
            @functools.lru_cache(maxsize=None)
            @pytest.fixture(name="renamed")
            def underlying():
                return 4
            """,
            "tests/test_x.py": """
            def test_uses(renamed):
                assert renamed == 4
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    test_idx = by_fqname["tests.test_x.test_uses"]
    fixture_idx = by_fqname["tests.fixtures.underlying"]
    assert (test_idx, fixture_idx, 0) in [(s, d, f) for (s, d, f) in graph.edges()]


def test_pytest_plugin_matches_fixture_through_builder_chain(build_plugin_graph, reachable_fqnames):
    """A decorator is classified by its *head* call, so a builder-style
    suffix (``@fixture().something(...)``) is peeled and the decl matched
    exactly as if it were ``@fixture()``. A ``fixture`` that only appears
    in the suffix (``@foo().fixture()``) is not the head and does not
    match."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest
            from pytest import fixture

            def foo(*a, **k):
                return foo

            @fixture().bar()
            def chained_bare_head():
                return 1

            @fixture(scope="module").bar(16).baz
            def chained_called_head():
                return 2

            @pytest.fixture().bar()
            def chained_attr_head():
                return 3

            @foo().fixture()
            def fixture_only_in_suffix():
                return 4
            """,
        },
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.fixtures.chained_bare_head" in reached
    assert "tests.fixtures.chained_called_head" in reached
    assert "tests.fixtures.chained_attr_head" in reached
    assert "tests.fixtures.fixture_only_in_suffix" not in reached


def test_pytest_plugin_reads_kwargs_from_head_call_of_builder_chain(build_plugin_graph):
    """``decorated_decls_with_args`` captures the kwargs of the *head*
    call, not of the builder suffix: ``name=`` on ``fixture(...)`` keys
    the ``test -> fixture`` edge even with ``.bar(name="decoy")`` after it."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            from pytest import fixture

            @fixture(name="renamed").bar(name="decoy")
            def underlying():
                return 4
            """,
            "tests/test_x.py": """
            def test_uses(renamed):
                assert renamed == 4

            def test_decoy(decoy):
                assert decoy == 4
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    fixture_idx = by_fqname["tests.fixtures.underlying"]
    edges = [(s, d) for (s, d, _) in graph.edges()]
    assert (by_fqname["tests.test_x.test_uses"], fixture_idx) in edges
    assert (by_fqname["tests.test_x.test_decoy"], fixture_idx) not in edges


def test_pytest_plugin_folds_f_string_fixture_name(build_plugin_graph):
    """``decorated_decls_with_args`` folds a ``name=`` kwarg built from a
    module constant, so ``@fixture(name=f"{PREFIX}_conn")`` publishes
    ``db_conn`` and the ``test -> fixture`` edge lands."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            from pytest import fixture

            PREFIX = "db"

            @fixture(name=f"{PREFIX}_conn")
            def underlying():
                return 4
            """,
            "tests/test_x.py": """
            def test_uses(db_conn):
                assert db_conn == 4
            """,
        },
        [native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    edges = [(s, d) for (s, d, _) in graph.edges()]
    assert (by_fqname["tests.test_x.test_uses"], by_fqname["tests.fixtures.underlying"]) in edges
