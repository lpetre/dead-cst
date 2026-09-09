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


def test_local_rebound_to_call_on_itself_terminates(build_decl_graph, assert_edges):
    """A local rebound to a call on itself in several branches. Every
    ``cursor`` read has four reachable definitions, three of which lead
    straight back to ``cursor`` with one more step, so a guard keyed on
    ``(definition, steps)`` never sees a repeat and fans out ``4^depth``
    paths (gigabytes on this one method). The definition-keyed guard
    cuts each re-entry; ``cursor`` denotes no module either way."""
    graph = build_decl_graph(
        {
            "mod.py": """
            def find(collection, sort, offset, limit):
                cursor = collection.find()
                if sort:
                    cursor = cursor.sort(sort)
                if offset:
                    cursor = cursor.skip(offset)
                if limit:
                    cursor = cursor.limit(limit)
                return cursor.to_list()
            """
        }
    )
    assert_edges(graph, {"mod.find -> mod"})


def test_module_variable_rebound_to_call_on_itself_terminates(build_decl_graph, assert_edges):
    """The same shape at module level, where the variable has a node."""
    graph = build_decl_graph(
        {
            "mod.py": """
            cursor = make()
            if SORT:
                cursor = cursor.sort(SORT)
            if OFFSET:
                cursor = cursor.skip(OFFSET)
            if LIMIT:
                cursor = cursor.limit(LIMIT)
            RESULT = cursor.to_list()
            """
        }
    )
    assert_edges(
        graph,
        {
            "mod.RESULT -> mod",
            "mod.RESULT -> mod.cursor",
            "mod.cursor -> mod",
            "mod.cursor -> mod.cursor",
        },
    )


def test_self_rebind_keeps_module_reached_through_other_definition(build_decl_graph, assert_edges):
    """Cutting a definition's re-entry must not lose the module reached
    through the name's *other* definition: ``m = m.sub`` re-reads ``m``,
    whose ``m = pkg`` binding still makes ``m.NAME`` a use of
    ``pkg.NAME`` and, through the rebind, of ``pkg.sub.NAME``."""
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "NAME = 'root'\n",
            "pkg/sub.py": "NAME = 'sub'\n",
            "use.py": """
            import pkg

            def who(deep):
                m = pkg
                if deep:
                    m = m.sub
                return m.NAME
            """,
        }
    )
    assert_edges(
        graph,
        {
            "pkg.NAME -> pkg",
            "pkg.sub -> pkg",
            "pkg.sub.NAME -> pkg.sub",
            "use.pkg -> pkg",
            "use.pkg -> use",
            "use.who -> pkg",
            "use.who -> pkg.NAME",
            "use.who -> pkg.sub",
            "use.who -> pkg.sub.NAME",
            "use.who -> use",
            "use.who -> use.pkg",
        },
    )
