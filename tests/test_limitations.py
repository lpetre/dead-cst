"""Negative tests that document known gaps in the analysis.

Each case asserts the *current* edge set and includes a comment about
the ideal behaviour. When the analyser is improved these tests will
start producing the commented-out edges and will begin to fail -- that
is the signal to promote them into ``test_declarations`` or
``test_imports``.
"""

import pytest


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        # ------------------------------------------------------------------
        # Assignment patterns the visitor cannot fully unpack
        # ------------------------------------------------------------------
        pytest.param(
            {"mod.py": "type T = int\n"},
            # PEP 695 ``type`` statements are ignored, so the alias
            # never appears in the graph.
            set(),
            id="pep-695-type-statement-not-captured",
        ),
        pytest.param(
            {
                "mod.py": """
                def a(): pass
                def b(): pass
                __all__ = ['a', 'b']
                """,
            },
            # ``__all__`` strings are not followed, so on their own they
            # keep nothing alive. ``ModuleDundersPlugin`` keeps the
            # ``__all__`` variable itself alive but does not chase the
            # listed names.
            {
                "mod.__all__ -> mod",
                "mod.a -> mod",
                "mod.b -> mod",
            },
            id="dunder-all-string-literals-not-followed",
        ),
        # ------------------------------------------------------------------
        # Dynamic / runtime features
        # ------------------------------------------------------------------
        pytest.param(
            {
                "other.py": "def g(): pass\n",
                "mod.py": """
                from other import *
                def a(): g()
                """,
            },
            # ``import *`` is deliberately skipped. ``mod.a`` should
            # point at ``other.g`` but the reference cannot be resolved.
            {
                "mod.a -> mod",
                "other.g -> other",
            },
            id="star-import-not-resolved",
        ),
        pytest.param(
            {
                "mod.py": """
                def a(): pass
                def b(): getattr(__import__('mod'), 'a')()
                """,
            },
            # ``getattr`` / dynamic attribute access is invisible to
            # static analysis. Ideally ``mod.b`` would point at
            # ``mod.a``.
            {
                "mod.a -> mod",
                "mod.b -> mod",
            },
            id="getattr-dynamic-access-produces-no-edge",
        ),
        pytest.param(
            {
                "mod.py": """
                x = 1
                del x
                """,
            },
            # ``del x`` removes the binding at runtime, so ``x`` is
            # effectively dead. The visitor does not model ``del`` and
            # keeps the declaration in the graph.
            {
                "mod -> mod.x",
                "mod.x -> mod",
            },
            id="del-does-not-remove-declaration",
        ),
    ],
)
def test_limitation(build_decl_graph, assert_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)
