"""Negative tests that document known gaps in the decorator matcher
(``decorated_decls`` / ``decorated_decls_with_args``).

Each case asserts the *current* behaviour and includes a comment about
the ideal behaviour. When the matcher is improved these tests will
start failing -- that is the signal to flip the assertions and promote
them into the relevant plugin test module (``test_pytest`` /
``test_celery``).
"""

from __future__ import annotations

from dead_cst import _native as native


def test_limitation_chained_call_decorator_is_never_matched(build_plugin_graph, reachable_fqnames):
    """A decorator whose expression is a *call chain* (``@foo().bar()``)
    is dropped from the decorator row entirely, so neither ``foo`` nor
    ``bar`` can match.

    ``decorator_descriptor`` (``runtime/src/file_extraction.rs``) strips
    only the outermost call and hands ``fixture().bar`` to
    ``collapse_attribute_chain``, which returns ``None`` as soon as the
    chain passes through a call -- the ``filter_map`` then discards the
    decorator. Ideally the chain would be peeled with
    ``peel_value_chain`` and the plugin would decide which segment it
    keys on; either ``@fixture().bar()`` (match the root) or
    ``@foo().fixture()`` (match the last call) could then be matched.
    Today both decls are reported dead, while the stacked equivalent
    (``@fixture`` over ``@bar``) is alive."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest
            from pytest import fixture

            def foo(*a, **k):
                return foo

            def bar(*a, **k):
                return bar

            @fixture
            @bar
            def stacked():
                return 1

            @fixture().bar()
            def chained_root_match():
                return 2

            @foo().fixture()
            def chained_tail_match():
                return 3

            @pytest.fixture().bar()
            def chained_attr_root_match():
                return 4
            """,
        },
        [native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.fixtures.stacked" in reached
    # Ideal: all three chained forms are matched and alive.
    assert "tests.fixtures.chained_root_match" not in reached
    assert "tests.fixtures.chained_tail_match" not in reached
    assert "tests.fixtures.chained_attr_root_match" not in reached


def test_limitation_same_file_decorator_is_never_matched(build_plugin_graph, reachable_fqnames):
    """A decorator *defined in the same file* as the decl is never
    matched, even when that file's module is the one the plugin targets.

    ``find_decorated_decls_core`` (``runtime/src/project.rs``) builds its
    local-name map with ``imports_local_from_facts``, which only walks
    ``from X import y`` / ``import X`` facts. A file that defines
    ``shared_task`` itself has no such fact, so the file is skipped (or,
    when other matching imports exist, the local name is simply absent
    from the map). Ideally, when the file's own module name equals one of
    the queried ``modules``, its top-level ``def``/``class`` names would
    seed the map. Today the same-file decl is reported dead while the
    imported forms in a sibling file are alive."""
    graph = build_plugin_graph(
        {
            "celery.py": """
            def shared_task(f):
                return f

            @shared_task
            def same_file_task():
                return 1
            """,
            "other.py": """
            from celery import shared_task
            import celery

            @shared_task
            def imported_task():
                return 2

            @celery.shared_task
            def attr_task():
                return 3
            """,
        },
        [native.NativePlugin.celery()],
    )
    reached = reachable_fqnames(graph)
    assert "other.imported_task" in reached
    assert "other.attr_task" in reached
    # Ideal: ``celery.same_file_task`` is matched and alive.
    assert "celery.same_file_task" not in reached
