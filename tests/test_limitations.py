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


def test_pep750_tstring_unparseable(build_decl_graph, assert_edges):
    """PEP 750 template strings (3.14) bypass the visitor.

    The pinned ``libcst`` cannot parse ``t"..."`` literals. Rather than
    aborting the whole run, the analyser emits an ``[unparseable]
    <module>`` synthetic node flagged ``ENTRYPOINT`` and edged at the
    real module node, so the file stays alive in reachability and
    importers can still target the module. Decls inside the file are
    invisible -- there are none in this graph -- so ideally
    ``mod.greet -> mod.NAME`` would also be present once libcst gains
    t-string support. That is the signal to promote this case into
    ``test_declarations``.
    """
    graph = build_decl_graph(
        {
            "mod.py": """
            NAME = "world"
            def greet(): return t"hello {NAME}"
            """,
        }
    )
    assert_edges(graph, {"[unparseable] mod -> mod"})
