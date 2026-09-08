"""Module-value extraction on recursive functions.

``file_to_nodes`` records, for every top-level function, the modules its
``return`` expressions denote (so ``get_config().NAME`` can be followed).
A ``return f(...)`` inside ``f`` walks back into ``f``; without a cycle
guard the walk fanned out ``2^depth`` for a function with two such
returns and never finished. These pin that the build terminates and the
graph is the same one a non-recursive body would give.
"""

from __future__ import annotations


def test_self_recursive_returns_terminate(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def f(x, k=0):
                if k == 0:
                    return f(x, k=1)
                if k == 1:
                    return f(x, k=2)
                if k == 2:
                    return f(x, k=3)
                return x
            """
        }
    )
    assert_edges(graph, {"mod.f -> mod"})


def test_mutually_recursive_returns_terminate(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def f(x):
                return g(x)

            def g(x):
                if x:
                    return f(x)
                return g(x - 1)
            """
        }
    )
    assert_edges(
        graph,
        {"mod.f -> mod", "mod.f -> mod.g", "mod.g -> mod", "mod.g -> mod.f"},
    )


def test_cross_file_recursive_returns_terminate(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "a.py": """
            import b

            def f(x):
                if x:
                    return b.g(x)
                return b.g(x - 1)
            """,
            "b.py": """
            import a

            def g(x):
                if x:
                    return a.f(x)
                return a.f(x - 1)
            """,
        }
    )
    assert_edges(
        graph,
        {
            "a.b -> a",
            "a.b -> b",
            "a.f -> a",
            "a.f -> a.b",
            "a.f -> b",
            "a.f -> b.g",
            "b.a -> a",
            "b.a -> b",
            "b.g -> a",
            "b.g -> a.f",
            "b.g -> b",
            "b.g -> b.a",
        },
    )
