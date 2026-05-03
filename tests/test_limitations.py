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
                def src(): return 1
                if (Y := src()): pass
                def use(): return Y
                """,
            },
            # PEP 572 walrus expressions at module scope bind a name
            # in the enclosing (module) scope, but the visitor does not
            # record that name as a top-level declaration. ``use``
            # therefore gets no ``use -> mod.Y`` edge -- ideally the
            # graph would contain ``mod.Y -> mod`` and
            # ``mod.use -> mod.Y``.
            {
                "mod -> mod.src",
                "mod.src -> mod",
                "mod.use -> mod",
            },
            id="walrus-toplevel-binding-not-captured",
        ),
        pytest.param(
            {
                "mod.py": """
                nums = [1, 2, 3]
                result = [last := n for n in nums]
                def use(): return last
                """,
            },
            # A walrus inside a comprehension leaks its binding to the
            # enclosing scope (here, the module). The visitor doesn't
            # capture that leak either, so ``use`` references a name
            # the graph never models. Ideally ``mod.last`` would exist
            # with ``mod.use -> mod.last``.
            {
                "mod.nums -> mod",
                "mod.result -> mod",
                "mod.result -> mod.nums",
                "mod.use -> mod",
            },
            id="walrus-comprehension-toplevel-leak-not-captured",
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
            # ``from other import *`` is fanned out at the module level,
            # so ``mod`` points at every top-level decl in ``other``.
            # Per-access resolution is still missing: ideally
            # ``mod.a -> other.g`` would also be present, but
            # ScopeProvider cannot bind the bare ``g`` reference back to
            # the star import.
            {
                "mod -> other",
                "mod -> other.g",
                "mod.a -> mod",
                "other.g -> other",
            },
            id="star-import-fans-out-but-misses-per-access-edge",
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
