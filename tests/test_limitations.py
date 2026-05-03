"""Negative tests that document known gaps in the analysis.

Each case asserts the *current* edge set and includes a comment about
the ideal behaviour. When the analyser is improved these tests will
start producing the commented-out edges and will begin to fail -- that
is the signal to promote them into ``test_declarations`` or
``test_imports``.
"""

import libcst
import pytest


@pytest.mark.parametrize(
    "files, expected_edges",
    [
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


def test_pep750_tstring_unparseable(build_decl_graph):
    """PEP 750 template strings (3.14) crash the analyser.

    The pinned ``libcst`` cannot parse ``t"..."`` literals, so any file
    containing one aborts ``build_symbol_graph`` with a
    ``ParserSyntaxError`` before the symbol graph is built. Ideally the
    visitor would either resolve interpolated names (yielding
    ``mod.greet -> mod.NAME`` here) or at minimum skip the file. When
    libcst gains t-string support this test will start to fail -- that
    is the signal to add positive coverage in ``test_declarations``.
    """
    with pytest.raises(libcst.ParserSyntaxError):
        build_decl_graph(
            {
                "mod.py": """
                NAME = "world"
                def greet(): return t"hello {NAME}"
                """,
            }
        )
