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
            {
                "mod.py": """
                def f(): pass
                b = c = f
                """,
            },
            # Chained assignment: ideally both ``b`` and ``c`` would
            # point at ``f``. Today only the last-processed target gets
            # the edge (``mod.c -> mod.f`` is missing).
            {
                "mod.b -> mod",
                "mod.b -> mod.f",
                "mod.c -> mod",
                "mod.f -> mod",
            },
            id="chained-assignment-only-one-target-linked",
        ),
        pytest.param(
            {
                "mod.py": """
                def f(): return 1, 2
                a, b = f()
                """,
            },
            # Tuple unpacking from a single call expression: ideally
            # both ``a`` and ``b`` would point at ``f``. Today only the
            # first iterated target gets the edge.
            {
                "mod.a -> mod",
                "mod.a -> mod.f",
                "mod.b -> mod",
                "mod.f -> mod",
            },
            id="tuple-unpack-from-call-only-first-target-linked",
        ),
        pytest.param(
            {"mod.py": "[a, b] = 1, 2\n"},
            # ``[a, b] = ...`` is a valid assignment target pattern but
            # the visitor does not recognise list targets at all, so no
            # declarations are produced for ``a`` or ``b``.
            set(),
            id="list-target-pattern-produces-no-decls",
        ),
        pytest.param(
            {"mod.py": "(a, (b, c)) = 1, (2, 3)\n"},
            # Nested-tuple unpacking: ideally all of ``a``, ``b``, and
            # ``c`` would be tracked. Today only the outermost level is
            # descended into, so ``b`` and ``c`` never appear.
            {"mod.a -> mod"},
            id="nested-tuple-target-misses-inner-names",
        ),
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
    ],
)
def test_limitation(build_decl_graph, assert_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason=(
        "AnnAssign without a value (e.g. `x: T`) triggers a stack-balance"
        " assertion in SymbolVisitor._add_variable. Once fixed, promote"
        " this into the positive declaration suite."
    ),
)
def test_ann_assign_without_value_crashes(build_decl_graph):
    build_decl_graph({"mod.py": "T = int\nx: T\n"})
