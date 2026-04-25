"""End-to-end tests for synthetic ``unreachable`` graph nodes.

When the visitor sees a statically-dead ``if`` / ``while`` suite (per
:mod:`dead_cst._branches`) it creates a synthetic ``SymbolNode`` with
``type="synthetic"`` and a fqname prefixed with ``<unreachable ``.
Every reference made from inside that suite gets a parallel edge
``synthetic -> referent`` -- the original ``enclosing-decl -> referent``
edge is left in place. The synthetic node is an orphan (no incoming
edges) so reachability never visits it; the parallel edges are purely
for surfacing.

These tests exercise the integration end-to-end through
``build_symbol_graph``: visitor creation of nodes, attribution of
references inside dead suites, and behavior across nested suites and
import references.
"""

from __future__ import annotations

import networkx as nx

from dead_cst._branches import is_unreachable_node


def _unreachable_nodes(graph: nx.DiGraph) -> list:
    return sorted(
        (n for n in graph.nodes if is_unreachable_node(n)),
        key=lambda n: (str(n.path), n.position.start.line, n.position.start.column),
    )


def test_if_false_creates_synthetic_node(build_decl_graph):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    branches = _unreachable_nodes(graph)
    assert len(branches) == 1
    suite = branches[0]
    assert suite.type == "synthetic"
    # Fqname is opaque; identification goes through is_unreachable_node.
    assert is_unreachable_node(suite)
    # libcst positions an ``IndentedBlock`` at its first statement, not
    # at the ``if`` keyword. Pin both line and column to lock down the
    # convention surfacing relies on.
    assert suite.position.start.line == 3
    assert suite.position.start.column == 4


def test_if_true_marks_else_as_unreachable(build_decl_graph):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if True:
                helper()
            else:
                helper()
            """
        }
    )
    assert len(_unreachable_nodes(graph)) == 1


def test_unknown_condition_creates_no_synthetic_node(build_decl_graph):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if cond:
                helper()
            else:
                helper()
            """
        }
    )
    assert _unreachable_nodes(graph) == []


def test_unreachable_node_has_edge_to_internal_referent(
    build_decl_graph, assert_unreachable_edges
):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    # Position is the body's first statement.
    assert_unreachable_edges(graph, {"<3:4> -> mod.helper"})


def test_unreachable_node_has_edge_to_import_referent(
    build_decl_graph, assert_unreachable_edges
):
    # Reference to an imported name from inside a dead suite produces
    # the same edge fan-out as a reference from a live decl: the local
    # import decl, the upstream module, and the resolved target.
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def helper(): pass",
            "pkg/b.py": """
            from pkg.a import helper
            if False:
                helper()
            """,
        }
    )
    assert_unreachable_edges(
        graph,
        {
            "<3:4> -> pkg.b.helper",
            "<3:4> -> pkg.a",
            "<3:4> -> pkg.a.helper",
        },
    )


def test_original_edges_are_preserved_when_branch_is_dead(
    build_decl_graph, assert_edges
):
    # The "real symbol graph" still contains the enclosing-module ->
    # helper edge. The unreachable-edge filtering in ``assert_edges``
    # ensures this assertion is unaffected by the synthetic node.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    assert_edges(
        graph,
        {
            "mod.helper -> mod",
            "mod -> mod.helper",
        },
    )


def test_nested_dead_suites_create_separate_synthetic_nodes(build_decl_graph):
    graph = build_decl_graph(
        {
            "mod.py": """
            def a(): pass
            def b(): pass
            if False:
                a()
                if False:
                    b()
            """
        }
    )
    branches = _unreachable_nodes(graph)
    assert len(branches) == 2


def test_nested_dead_suite_attributes_to_innermost(
    build_decl_graph, assert_unreachable_edges
):
    # ``a()`` is inside the outer dead suite only; ``b()`` is inside
    # both, but the innermost suite owns it.
    graph = build_decl_graph(
        {
            "mod.py": """
            def a(): pass
            def b(): pass
            if False:
                a()
                if False:
                    b()
            """
        }
    )
    assert_unreachable_edges(
        graph,
        {
            "<4:4> -> mod.a",
            "<6:8> -> mod.b",
        },
    )


def test_while_false_creates_synthetic_node(build_decl_graph, assert_unreachable_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            while False:
                helper()
            """
        }
    )
    assert_unreachable_edges(graph, {"<3:4> -> mod.helper"})


def test_while_true_else_is_unreachable(build_decl_graph, assert_unreachable_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            while True:
                pass
            else:
                helper()
            """
        }
    )
    assert_unreachable_edges(graph, {"<5:4> -> mod.helper"})


def test_synthetic_nodes_are_unreachable_from_entrypoints(build_decl_graph):
    # No incoming edges, so reachability skips them. Critical so the
    # parallel edges do not accidentally keep symbols alive.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    suite = _unreachable_nodes(graph)[0]
    assert graph.in_degree(suite) == 0


def test_decls_inside_dead_branch_remain_in_graph(build_decl_graph, assert_edges):
    # Per the design decision: top-level decls defined inside a dead
    # branch keep their normal node + edges. The synthetic node coexists
    # with them; pruning is a follow-up.
    graph = build_decl_graph(
        {
            "mod.py": """
            if False:
                def foo(): pass
            """
        }
    )
    fqnames = {n.fqname for n in graph.nodes if not is_unreachable_node(n)}
    assert "mod.foo" in fqnames
    assert_edges(
        graph,
        {
            "mod.foo -> mod",
        },
    )


def test_elif_false_in_chain_creates_synthetic_for_only_dead_branch(
    build_decl_graph, assert_unreachable_edges
):
    graph = build_decl_graph(
        {
            "mod.py": """
            def a(): pass
            def b(): pass
            def c(): pass
            if cond:
                a()
            elif False:
                b()
            else:
                c()
            """
        }
    )
    # Only the ``elif False:`` body is unreachable.
    assert_unreachable_edges(graph, {"<7:4> -> mod.b"})
