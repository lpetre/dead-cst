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


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        # ------------------------------------------------------------------
        # Cross-kind shadowing: both the shadowed and shadowing decl
        # survive as distinct nodes at the same fqname, so the shadowed
        # node's upstream edges keep its (dead) import source alive.
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
            # The import at col 18 and the ``def`` at col 0 are distinct
            # nodes. The shadowed import still has outgoing edges to
            # ``other`` / ``other.f``, and the module-level call reaches
            # both, so ``other`` stays alive.
            {
                "mod -> mod.f@1:18",
                "mod -> mod.f@2:0",
                "mod -> other",
                "mod -> other.f@1:0",
                "mod.f@1:18 -> other",
                "mod.f@1:18 -> other.f@1:0",
                "mod.f@2:0 -> mod",
                "other.f@1:0 -> other",
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
            {
                "mod -> mod.f@1:0",
                "mod -> mod.f@2:18",
                "mod -> other",
                "mod -> other.f@1:0",
                "mod.f@2:18 -> mod",
                "mod.f@2:18 -> other",
                "mod.f@2:18 -> other.f@1:0",
                "other.f@1:0 -> other",
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
            {
                "mod -> mod.f@1:18",
                "mod -> mod.f@2:0",
                "mod -> other",
                "mod -> other.f@1:0",
                "mod.f@1:18 -> other",
                "mod.f@1:18 -> other.f@1:0",
                "mod.f@2:0 -> mod",
                "other.f@1:0 -> other",
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
            # Two distinct ``mod.x`` import nodes, one per source module.
            # Both stay alive, dragging ``a`` and ``b`` along.
            {
                "a.x@1:0 -> a",
                "b.x@1:0 -> b",
                "mod -> a",
                "mod -> a.x@1:0",
                "mod -> b",
                "mod -> b.x@1:0",
                "mod -> mod.x@1:14",
                "mod -> mod.x@2:14",
                "mod.x@1:14 -> a",
                "mod.x@1:14 -> a.x@1:0",
                "mod.x@2:14 -> b",
                "mod.x@2:14 -> b.x@1:0",
                "mod.x@2:14 -> mod",
            },
            id="two-imports-same-alias-both-kept-alive",
        ),
        # ------------------------------------------------------------------
        # Same-kind redeclaration: node identity is now per-position, so
        # each ``def f`` is its own node. The shadowed node still has
        # outgoing body edges which keep their targets alive -- phase 2
        # (flow-sensitive referent filtering) should drop the edge
        # ``mod -> <shadowed>`` at the call site so the dead body
        # becomes unreachable and collects dead_helper along with it.
        # ------------------------------------------------------------------
        pytest.param(
            {
                "mod.py": """
                def dead_helper(): pass
                def f(): dead_helper()
                def f(): pass
                f()
                """,
            },
            # ``mod.f@2:0`` is the shadowed decl; it has the body edge
            # to ``dead_helper`` and no parent edge (trie only keeps the
            # last). ``f()`` reaches both f nodes, so the dead body is
            # kept alive -- false positive.
            {
                "mod -> mod.f@2:0",
                "mod -> mod.f@3:0",
                "mod.dead_helper@1:0 -> mod",
                "mod.f@2:0 -> mod.dead_helper@1:0",
                "mod.f@3:0 -> mod",
            },
            id="dead-function-body-kept-alive",
        ),
        pytest.param(
            {
                "mod.py": """
                def dead_helper(): pass
                class C:
                    def m(self): dead_helper()
                class C: pass
                C()
                """,
            },
            {
                "mod -> mod.C@2:0",
                "mod -> mod.C@4:0",
                "mod.C@2:0 -> mod.dead_helper@1:0",
                "mod.C@4:0 -> mod",
                "mod.dead_helper@1:0 -> mod",
            },
            id="dead-method-body-kept-alive",
        ),
        pytest.param(
            {
                "mod.py": """
                def a(): pass
                def b(): pass
                def f(): a()
                def f(): b()
                def f(): pass
                f()
                """,
            },
            # Three ``def f`` nodes. The call site reaches all three;
            # only the last has a parent edge. The body edges to ``a``
            # and ``b`` live on the shadowed nodes at lines 3 and 4.
            {
                "mod -> mod.f@3:0",
                "mod -> mod.f@4:0",
                "mod -> mod.f@5:0",
                "mod.a@1:0 -> mod",
                "mod.b@2:0 -> mod",
                "mod.f@3:0 -> mod.a@1:0",
                "mod.f@4:0 -> mod.b@2:0",
                "mod.f@5:0 -> mod",
            },
            id="chain-of-shadowed-functions-keeps-all-bodies-alive",
        ),
    ],
)
def test_redeclaration_limitation(build_decl_graph, assert_positional_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_positional_edges(graph, expected_edges)
