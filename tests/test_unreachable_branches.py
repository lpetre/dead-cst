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


def test_custom_detector_folds_constants(tmp_path, write_files, assert_dead_branch_edges):
    """End-to-end: a custom detector can mark non-literal branches dead.

    Demonstrates the user-facing extension story: a company-specific
    detector that knows ``settings.IS_PROD`` is always ``True`` walks
    the file and returns the dead else-branch's :class:`CodeRange`.
    The analyzer feeds those positions through the same flag-derivation
    path as the literal-only default, so internal references inside
    the dead branch land tagged with ``EdgeFlags.DEAD_BRANCH``.
    """
    from dataclasses import dataclass

    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider

    from dead_cst import build_symbol_graph

    write_files(
        {
            "mod.py": """
            from settings import IS_PROD
            def prod_only(): pass
            def dev_only(): pass
            if IS_PROD:
                prod_only()
            else:
                dev_only()
            """,
            "settings.py": "IS_PROD = True\n",
        }
    )

    @dataclass(frozen=True)
    class IsProdDetector:
        name: str = "is_prod"
        version: int = 1

        def find_regions(self, wrapper: MetadataWrapper):
            positions = wrapper.resolve(PositionProvider)
            out = []

            class _V(cst.CSTVisitor):
                def visit_If(self, node: cst.If) -> None:
                    test = node.test
                    # Hard-coded "settings.IS_PROD is always True" --
                    # the default literal detector returns ``None`` here.
                    if isinstance(test, cst.Name) and test.value == "IS_PROD":
                        if isinstance(node.orelse, cst.Else):
                            pos = positions.get(node.orelse.body)
                            if pos is not None:
                                out.append(pos)

            wrapper.module.visit(_V())
            return out

    graph = build_symbol_graph({tmp_path: []}, unreachable_detector=IsProdDetector())
    assert_dead_branch_edges(graph, {"mod -> mod.dev_only"})


def test_default_detector_does_not_flag_named_condition(tmp_path, write_files):
    """Sanity check: the default detector leaves ``if NAME`` branches alone.

    Counterpart to :func:`test_custom_detector_folds_constants` --
    without a custom detector, ``if IS_PROD:`` is unknown and no
    suite is recorded as dead.
    """
    from dead_cst import build_symbol_graph

    write_files(
        {
            "mod.py": """
            from settings import IS_PROD
            def f(): pass
            if IS_PROD:
                f()
            else:
                f()
            """,
            "settings.py": "IS_PROD = True\n",
        }
    )
    graph = build_symbol_graph({tmp_path: []})
    assert _dead_suite_positions(graph, tmp_path / "mod.py") == ()


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


# ----------------------------------------------------------------------
# Default detector + constant-folding pass: ``DEBUG = False; if DEBUG:``
# patterns are caught out of the box, including chains the user vision
# called out (``foo = False; bar = foo or False; if bar: ...``).
# ----------------------------------------------------------------------


def test_default_detector_folds_module_constant(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            DEBUG = False
            def helper(): pass
            if DEBUG:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_default_detector_folds_chained_constants(build_decl_graph, assert_dead_branch_edges):
    # The user-visioned case: ``bar`` resolves through ``foo`` only
    # after the second pass of the constant-folding fixpoint loop.
    graph = build_decl_graph(
        {
            "mod.py": """
            foo = False
            bar = foo or False
            def helper(): pass
            if bar:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_default_detector_folds_long_chain(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            a = True
            b = a
            c = not b
            d = c or False
            def helper(): pass
            if d:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_default_detector_folds_annotated_constant(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            DEBUG: bool = False
            def helper(): pass
            if DEBUG:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_default_detector_folds_constant_in_function(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                FLAG = False
                if FLAG:
                    helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_default_detector_marks_else_dead_when_constant_is_truthy(
    build_decl_graph, assert_dead_branch_edges
):
    graph = build_decl_graph(
        {
            "mod.py": """
            ENABLED = True
            def helper(): pass
            if ENABLED:
                pass
            else:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_default_detector_does_not_fold_conditional_binding(
    build_decl_graph, assert_dead_branch_edges
):
    # Both bindings reach the access; their values disagree, so the
    # fold pass refuses to commit and the suite stays live.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if cond:
                cfg = True
            else:
                cfg = False
            if cfg:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, set())


def test_default_detector_does_not_fold_non_literal_rhs(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            cfg = compute()
            def helper(): pass
            if cfg:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, set())


def test_default_detector_does_not_fold_imported_name(build_decl_graph, assert_dead_branch_edges):
    # Imports go through ``ImportAssignment``, whose binding node's
    # parent is ``ImportAlias`` (not ``AssignTarget``/``AnnAssign``),
    # so the fold pass returns no RHS and the access stays unknown.
    graph = build_decl_graph(
        {
            "settings.py": "DEBUG = False\n",
            "mod.py": """
            from settings import DEBUG
            def helper(): pass
            if DEBUG:
                helper()
            """,
        }
    )
    assert_dead_branch_edges(graph, set())
