"""Tests for the top-level-only declaration model.

``dead-cst`` tracks only top-level declarations (module-level functions,
classes, and variables). Anything declared inside another declaration
-- nested functions, nested classes, methods, class-body variables --
is deliberately not given its own node. References made from inside
those nested scopes are attributed to the enclosing top-level
declaration.

These tests pin down that design so regressions surface clearly.
"""

import pytest


@pytest.mark.parametrize(
    "src, expected_edges",
    [
        pytest.param(
            """
            def helper(): return 1
            def outer():
                def inner():
                    return helper()
            """,
            {
                "mod.helper -> mod",
                "mod.outer -> mod",
                "mod.outer -> mod.helper",
            },
            id="nested-function-reference-attributes-to-outer-function",
        ),
        pytest.param(
            """
            def helper(): return 1
            def outer():
                def inner():
                    return helper()
                inner()
            """,
            # ``inner`` is never a node, so calling it from ``outer`` is
            # a no-op for the graph -- the edge the user cares about is
            # ``outer -> helper``.
            {
                "mod.helper -> mod",
                "mod.outer -> mod",
                "mod.outer -> mod.helper",
            },
            id="call-to-nested-function-does-not-create-intermediate-node",
        ),
        pytest.param(
            """
            def helper(): pass
            class C:
                def m(self): helper()
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.helper",
                "mod.helper -> mod",
            },
            id="method-reference-attributes-to-enclosing-class",
        ),
        pytest.param(
            """
            def helper(): pass
            class C:
                @classmethod
                def m(cls): helper()
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.helper",
                "mod.helper -> mod",
            },
            id="classmethod-reference-attributes-to-enclosing-class",
        ),
        pytest.param(
            """
            class C:
                def m(self): pass
            def f(): C().m()
            """,
            # ``C.m`` is not a node, so the method call only produces
            # ``f -> C``. Keeping ``C`` alive is sufficient to keep
            # ``m`` alive as part of its source.
            {
                "mod.C -> mod",
                "mod.f -> mod",
                "mod.f -> mod.C",
            },
            id="method-call-links-to-class-only",
        ),
        pytest.param(
            """
            class A:
                def m(self): pass
            class B(A):
                def m(self): super().m()
            """,
            # ``B.m -> A.m`` is not expressible; the class-level
            # inheritance edge ``B -> A`` carries the dependency.
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.B -> mod.A",
            },
            id="super-call-relies-on-class-level-inheritance-edge",
        ),
        pytest.param(
            """
            def helper(): pass
            class Out:
                class Inner:
                    x = helper
            """,
            # Neither ``Out.Inner`` nor its members are nodes; the
            # reference to ``helper`` folds up into ``Out``.
            {
                "mod.Out -> mod",
                "mod.Out -> mod.helper",
                "mod.helper -> mod",
            },
            id="nested-class-body-references-fold-into-outer-class",
        ),
    ],
)
def test_top_level_only(build_decl_graph, assert_edges, src, expected_edges):
    graph = build_decl_graph({"mod.py": src})
    assert_edges(graph, expected_edges)
