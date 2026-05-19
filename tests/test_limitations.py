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
        pytest.param(
            {
                "mod.py": """
                nums = [1, 2, 3]
                result = [last := n for n in nums]
                def use(): return last
                """,
            },
            # Per PEP 572, a walrus inside a comprehension binds its
            # target in the *containing* scope -- ``mod.last`` should
            # surface as a top-level decl and ``use``'s reference to
            # ``last`` should route to it.
            #
            # ty has a ``// TODO walrus in comprehensions is implicitly
            # nonlocal`` at
            # ``vendor/ruff/crates/ty_python_core/src/builder.rs:3605``,
            # so the walrus's ``DefinitionKind::NamedExpression``
            # currently lives in the comprehension scope rather than
            # the enclosing module scope. Our ``ingest_decls`` loop
            # iterates the module's global scope and so doesn't see
            # the leaked binding -- ``mod.last`` is never minted, the
            # ``use``-site reference goes unresolved, and reachability
            # treats ``last`` as if it were never written.
            {
                "mod.nums -> mod",
                "mod.result -> mod",
                "mod.result -> mod.nums",
                "mod.use -> mod",
            },
            id="comprehension-walrus-doesnt-leak-to-enclosing-scope",
        ),
    ],
)
def test_limitation(build_decl_graph, assert_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)
