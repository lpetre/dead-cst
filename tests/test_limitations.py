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
        # ------------------------------------------------------------------
        # Same-kind redeclarations collapse to one graph node, so any
        # references made from the body of a shadowed (and therefore
        # unreachable) declaration are folded into the surviving node
        # and keep their targets alive.
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
            # The first ``def f`` is shadowed before it's ever called,
            # so its body never runs -- ``dead_helper`` is dead. Because
            # ``SymbolNode`` hashes on (fqname, type, path) both ``def f``
            # nodes collapse into one, and the edge from the dead body
            # keeps ``dead_helper`` alive.
            {
                "mod -> mod.f",
                "mod.dead_helper -> mod",
                "mod.f -> mod",
                "mod.f -> mod.dead_helper",
            },
            id="dead-function-body-kept-alive-by-node-collapse",
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
            # The first ``C`` is shadowed, so its method ``m`` is never
            # reachable and ``dead_helper`` is dead. The two ``C`` class
            # nodes collapse into one and the method-body edge survives.
            {
                "mod -> mod.C",
                "mod.C -> mod",
                "mod.C -> mod.dead_helper",
                "mod.dead_helper -> mod",
            },
            id="dead-method-body-kept-alive-by-node-collapse",
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
            # All but the last ``def f`` are dead, so both ``a`` and
            # ``b`` are only referenced from unreachable bodies. Node
            # collapse keeps both alive.
            {
                "mod -> mod.f",
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.f -> mod",
                "mod.f -> mod.a",
                "mod.f -> mod.b",
            },
            id="chain-of-shadowed-functions-keeps-all-bodies-alive",
        ),
    ],
)
def test_limitation(build_decl_graph, assert_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)
