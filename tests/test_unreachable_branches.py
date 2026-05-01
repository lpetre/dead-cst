"""End-to-end tests for ``EdgeFlags.DEAD_BRANCH`` on graph edges.

When the visitor sees a statically-dead ``if`` / ``while`` suite (per
:mod:`dead_cst._branches`) it records the suite's position. The
analyzer's apply step then flags every reference whose access position
falls inside any recorded dead suite with
:data:`dead_cst.EdgeFlags.DEAD_BRANCH` -- a single tagged edge per
reference, no parallel synthetic source node.

By default :func:`find_reachable` does not filter on this flag, so
dead-code references still propagate liveness through the enclosing
decl. :func:`find_kept_alive_by_dead_branches` returns the strict
diff: symbols that would become unreachable if every dead suite were
severed.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from dead_cst import find_kept_alive_by_dead_branches, find_reachable


def _dead_suite_positions(graph: nx.MultiDiGraph, file: Path) -> tuple:
    return graph.graph.get("dead_suites", {}).get(file, ())


def test_if_false_records_dead_suite(build_decl_graph, tmp_path):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    suites = _dead_suite_positions(graph, tmp_path / "mod.py")
    assert len(suites) == 1
    pos = suites[0]
    # libcst positions an ``IndentedBlock`` at its first statement, not
    # at the ``if`` keyword. Pin both line and column to lock down the
    # convention surfacing relies on.
    assert pos.start.line == 3
    assert pos.start.column == 4


def test_if_true_marks_else_as_unreachable(build_decl_graph, tmp_path):
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
    assert len(_dead_suite_positions(graph, tmp_path / "mod.py")) == 1


def test_unknown_condition_records_no_dead_suite(build_decl_graph, tmp_path):
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
    assert _dead_suite_positions(graph, tmp_path / "mod.py") == ()


def test_dead_branch_internal_ref_is_flagged(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_dead_branch_import_ref_is_flagged(build_decl_graph, assert_dead_branch_edges):
    # Reference to an imported name from inside a dead suite produces
    # the same edge fan-out as a reference from a live decl: the local
    # import decl, the upstream module, and the resolved target. All
    # of them get the ``DEAD_BRANCH`` flag.
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
    assert_dead_branch_edges(
        graph,
        {
            "pkg.b -> pkg.b.helper",
            "pkg.b -> pkg.a",
            "pkg.b -> pkg.a.helper",
        },
    )


def test_default_find_reachable_traverses_dead_branch_edges(build_decl_graph):
    """Default reachability still keeps ``helper`` alive via the dead-branch ref.

    Today's behavior preservation: even when the only reference to
    ``helper`` is ``if False: helper()``, ``find_reachable`` from the
    module-as-entrypoint still reaches it. The edge is flagged but
    not skipped.
    """
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    # Mark the module as an entrypoint manually; the build_decl_graph
    # fixture doesn't run plugins.
    module = next(n for n in graph.nodes if n.fqname == "mod")
    graph.nodes[module]["entrypoint"] = True
    helper = next(n for n in graph.nodes if n.fqname == "mod.helper")
    assert helper in find_reachable(graph)


def test_find_kept_alive_by_dead_branches_returns_strict_diff(build_decl_graph):
    """Strict pruning surfaces ``helper`` as kept-alive-only-by-dead-code."""
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if False:
                helper()
            """
        }
    )
    module = next(n for n in graph.nodes if n.fqname == "mod")
    graph.nodes[module]["entrypoint"] = True
    helper = next(n for n in graph.nodes if n.fqname == "mod.helper")
    blast = find_kept_alive_by_dead_branches(graph)
    assert helper in blast


def test_nested_dead_suites_record_separate_positions(build_decl_graph, tmp_path):
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
    assert len(_dead_suite_positions(graph, tmp_path / "mod.py")) == 2


def test_nested_dead_suites_flag_both_refs(build_decl_graph, assert_dead_branch_edges):
    # Both refs are inside at least one dead suite; both are flagged.
    # The previous synthetic-node model attributed each ref to the
    # innermost suite -- that fidelity intentionally goes away here.
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
    assert_dead_branch_edges(graph, {"mod -> mod.a", "mod -> mod.b"})


def test_while_false_flags_internal_ref(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            while False:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_while_true_else_flags_internal_ref(build_decl_graph, assert_dead_branch_edges):
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
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_decls_inside_dead_branch_remain_in_graph(build_decl_graph, assert_edges):
    # Per the design decision: top-level decls defined inside a dead
    # branch keep their normal node + edges. The flagged-edge model
    # only marks references made from dead code; the decl itself is
    # not flagged.
    graph = build_decl_graph(
        {
            "mod.py": """
            if False:
                def foo(): pass
            """
        }
    )
    fqnames = {n.fqname for n in graph.nodes}
    assert "mod.foo" in fqnames
    assert_edges(
        graph,
        {
            "mod.foo -> mod",
        },
    )


def test_elif_false_in_chain_flags_only_dead_branch(build_decl_graph, assert_dead_branch_edges):
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
    # Only the ``elif False:`` body is dead, so only ``b`` is flagged.
    assert_dead_branch_edges(graph, {"mod -> mod.b"})
