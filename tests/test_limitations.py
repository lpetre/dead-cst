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
            # keep nothing alive. The CLI has a separate
            # ``--preserve-dunder-all`` flag that handles this.
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
        # ------------------------------------------------------------------
        # Cross-kind redeclarations keep the shadowed symbol alive
        # ------------------------------------------------------------------
        pytest.param(
            {
                "other.py": "def f(): pass\n",
                "mod.py": """
                from other import f
                def f(): pass
                f()
                """,
            },
            # ``def f`` shadows the earlier ``from other import f``, so
            # at runtime the module-level ``f()`` always calls the local
            # function. Ideally no edges would point at ``other`` -- the
            # import is dead. Today the shadowed import node is kept in
            # the graph and its edges reach ``other.f``, keeping ``other``
            # alive as a false positive.
            {
                "mod -> mod.f",
                "mod -> other",
                "mod -> other.f",
                "mod.f -> mod",
                "mod.f -> other",
                "mod.f -> other.f",
                "other.f -> other",
            },
            id="import-shadowed-by-function-keeps-import-alive",
        ),
        pytest.param(
            {
                "other.py": "def f(): pass\n",
                "mod.py": """
                def f(): pass
                from other import f
                f()
                """,
            },
            # Symmetric case: the import shadows the prior ``def`` but
            # both declarations survive in the graph with the same fqname.
            {
                "mod -> mod.f",
                "mod -> other",
                "mod -> other.f",
                "mod.f -> mod",
                "mod.f -> other",
                "mod.f -> other.f",
                "other.f -> other",
            },
            id="function-shadowed-by-import-keeps-function-alive",
        ),
        pytest.param(
            {
                "other.py": "def f(): pass\n",
                "mod.py": """
                from other import f
                f = 1
                print(f)
                """,
            },
            # ``f = 1`` clobbers the import, so ``print(f)`` never
            # reaches ``other``. Ideally ``other`` would be unreachable,
            # but the shadowed import node is kept alive and its upstream
            # edges drag ``other`` / ``other.f`` into the graph.
            {
                "mod -> mod.f",
                "mod -> other",
                "mod -> other.f",
                "mod.f -> mod",
                "mod.f -> other",
                "mod.f -> other.f",
                "other.f -> other",
            },
            id="import-rebound-to-constant-keeps-import-alive",
        ),
        pytest.param(
            {
                "a.py": "def x(): pass\n",
                "b.py": "def x(): pass\n",
                "mod.py": """
                from a import x
                from b import x
                x()
                """,
            },
            # The second import shadows the first, so only ``b.x`` is
            # reachable at runtime. Today both imports produce nodes and
            # both upstream modules stay alive.
            {
                "a.x -> a",
                "b.x -> b",
                "mod -> a",
                "mod -> a.x",
                "mod -> b",
                "mod -> b.x",
                "mod -> mod.x",
                "mod.x -> a",
                "mod.x -> a.x",
                "mod.x -> b",
                "mod.x -> b.x",
                "mod.x -> mod",
            },
            id="two-imports-same-alias-both-kept-alive",
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
