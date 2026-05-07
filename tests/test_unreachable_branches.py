"""End-to-end tests for ``EdgeFlags.DEAD_BRANCH`` on graph edges.

When the visitor sees a statically-dead ``if`` / ``while`` suite (per
:mod:`dead_cst.branches`) it records the suite's position. The
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

from dead_cst.analyze import (
    _find_kept_alive_by_dead_branches as find_kept_alive_by_dead_branches,
    _find_reachable as find_reachable,
)


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


def test_custom_detector_folds_constants(make_analysis, write_files, assert_dead_branch_edges):
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

    graph = make_analysis(unreachable_detector=IsProdDetector()).materialize_all()
    assert_dead_branch_edges(graph, {"mod -> mod.dev_only"})


def test_default_detector_does_not_flag_named_condition(tmp_path, make_analysis, write_files):
    """Sanity check: the default detector leaves ``if NAME`` branches alone.

    Counterpart to :func:`test_custom_detector_folds_constants` --
    without a custom detector, ``if IS_PROD:`` is unknown and no
    suite is recorded as dead.
    """

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
    graph = make_analysis().materialize_all()
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


def test_default_detector_folds_walrus_binding(build_decl_graph, assert_dead_branch_edges):
    """Walrus bindings fold the same as ``Assign`` / ``AnnAssign``.

    ``_constant_assignment_rhs`` recognises ``NamedExpr`` parents, so a
    walrus whose RHS is a plain literal enters the fold table just
    like ``DEBUG = False`` would. The if-test reads the bound name and
    the body is flagged dead.
    """
    graph = build_decl_graph(
        {
            "mod.py": """
            (DEBUG := False)
            def helper(): pass
            if DEBUG:
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


def test_default_detector_folds_walrus_in_if_test(build_decl_graph, assert_dead_branch_edges):
    """Walrus *expression* truthiness folds through ``NamedExpr``.

    ``evaluate_truthiness`` unwraps ``NamedExpr`` to its value, so when
    the walrus's RHS is a literal the whole expression's truthiness is
    statically known and the if-body is flagged dead.
    """
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            if (DEBUG := False):
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


# ----------------------------------------------------------------------
# Default detector: post-terminator unreachable region. Statements
# after an unconditional ``return`` / ``raise`` / ``break`` /
# ``continue`` / ``assert <statically-falsy>`` in the same suite are
# unreachable. The check is purely suite-relative -- a ``raise`` in
# a try body kills the rest of *that* suite, not the ``except``
# handler that runs on its own path.
# ----------------------------------------------------------------------


def test_return_kills_following_statements(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                return
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_raise_kills_following_statements(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                raise RuntimeError()
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_assert_false_kills_following_statements(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                assert False
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_assert_folded_constant_kills_following_statements(
    build_decl_graph, assert_dead_branch_edges
):
    # ``assert NEVER`` where ``NEVER = False`` -- the constant-folding
    # pass resolves the test to False, then the terminator scan picks
    # it up. Two passes composing on the same suite.
    graph = build_decl_graph(
        {
            "mod.py": """
            NEVER = False
            def helper(): pass
            def caller():
                assert NEVER
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_assert_truthy_does_not_kill_following_statements(
    build_decl_graph, assert_dead_branch_edges
):
    # ``assert True`` (or any truthy literal) is a no-op at runtime --
    # following statements are reachable.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                assert True
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, set())


def test_break_kills_following_statements_in_loop(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                for _ in range(10):
                    break
                    helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_continue_kills_following_statements_in_loop(build_decl_graph, assert_dead_branch_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                while True:
                    continue
                    helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_raise_in_try_body_does_not_kill_except_handler(build_decl_graph, assert_dead_branch_edges):
    # The user-asked case: ``raise`` in a try body is suite-relative.
    # ``func()`` after the raise in the try body is dead; ``other()``
    # in the except handler runs on its own path and stays live.
    graph = build_decl_graph(
        {
            "mod.py": """
            def func(): pass
            def other(): pass
            def caller():
                try:
                    raise Exception()
                    func()
                except Exception:
                    other()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.func"})


def test_raise_in_except_body_kills_following_statements(
    build_decl_graph, assert_dead_branch_edges
):
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                try:
                    pass
                except Exception:
                    raise
                    helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod.caller -> mod.helper"})


def test_terminator_as_last_statement_marks_nothing(build_decl_graph, assert_dead_branch_edges):
    # No tail to mark -- the terminator is the last statement.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller():
                helper()
                return
            """
        }
    )
    assert_dead_branch_edges(graph, set())


def test_terminator_inside_if_does_not_kill_outside(build_decl_graph, assert_dead_branch_edges):
    # The ``return`` lives in the if-body suite. Statements at the
    # outer (function-body) suite are reachable: when ``cond`` is
    # falsy the if body is skipped and execution falls through.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            def caller(cond):
                if cond:
                    return
                helper()
            """
        }
    )
    assert_dead_branch_edges(graph, set())


def test_terminator_chained_with_constant_fold(build_decl_graph, assert_dead_branch_edges):
    # A function-call chain: ``if FLAG: return`` where ``FLAG`` is a
    # folded constant. ``unreachable_suites`` already recognized this
    # but the test guards against passes interfering with each other.
    graph = build_decl_graph(
        {
            "mod.py": """
            FLAG = False
            def helper(): pass
            def caller():
                if FLAG:
                    return
                helper()
            """
        }
    )
    # The if-body itself is dead (FLAG=False), but no terminator at
    # function-body scope, so ``helper()`` is reachable.
    assert_dead_branch_edges(graph, set())


def test_module_level_terminator_kills_following_statements(
    build_decl_graph, assert_dead_branch_edges
):
    # Module-scope: ``raise`` at top level prevents any subsequent
    # module statement from running, so the trailing ``helper()`` is
    # dead.
    graph = build_decl_graph(
        {
            "mod.py": """
            def helper(): pass
            raise SystemExit()
            helper()
            """
        }
    )
    assert_dead_branch_edges(graph, {"mod -> mod.helper"})


# ----------------------------------------------------------------------
# Custom detector subclass overrides ``resolve`` to fold non-Name
# expressions (function calls, attribute access). The user's example:
# ``check_flag("migration-abc")`` is treated as a known constant.
# ----------------------------------------------------------------------


def test_custom_detector_override_folds_call_in_if(make_analysis, write_files):
    """A subclass that knows ``check_flag(name)`` is constant.

    The override answers for the ``Call`` node directly; the detector
    threads the answer through both ``unreachable_suites`` (so the
    ``if`` body is marked dead) and ``fold_constants`` (so a chain
    like ``flag = check_flag(...); if flag:`` would also resolve, see
    the next test).
    """
    from dataclasses import dataclass

    import libcst as cst

    from dead_cst import EdgeFlags
    from dead_cst.branches import DefaultUnreachableRegionDetector

    write_files(
        {
            "mod.py": """
            def prod_only(): pass
            def dev_only(): pass
            if check_flag("migration-abc"):
                prod_only()
            else:
                dev_only()
            """,
        }
    )

    @dataclass(frozen=True)
    class FlagAwareDetector(DefaultUnreachableRegionDetector):
        name: str = "flag_aware"
        version: int = 1

        def resolve(self, expr):
            if (
                isinstance(expr, cst.Call)
                and isinstance(expr.func, cst.Name)
                and expr.func.value == "check_flag"
            ):
                return True
            return None

    graph = make_analysis(unreachable_detector=FlagAwareDetector()).materialize_all()
    dead = {
        f"{src.fqname} -> {dst.fqname}"
        for src, dst, attrs in graph.edges(data=True)
        if attrs.get("flags", EdgeFlags.NONE) & EdgeFlags.DEAD_BRANCH
    }
    # ``check_flag(...)`` resolves to True, so the else branch is dead.
    assert dead == {"mod -> mod.dev_only"}


def test_custom_detector_override_folds_through_assignment(make_analysis, write_files):
    """Override answer composes with the fixpoint fold pass.

    ``flag = check_flag(...)`` propagates the override's answer
    through the assignment: by the time ``if flag:`` is evaluated,
    the fold table already has ``id(flag) -> False``, so the if body
    is recognized as dead.
    """
    from dataclasses import dataclass

    import libcst as cst

    from dead_cst import EdgeFlags
    from dead_cst.branches import DefaultUnreachableRegionDetector

    write_files(
        {
            "mod.py": """
            def helper(): pass
            flag = check_flag("migration-abc")
            if flag:
                helper()
            """,
        }
    )

    @dataclass(frozen=True)
    class FlagAwareDetector(DefaultUnreachableRegionDetector):
        name: str = "flag_aware"
        version: int = 1

        def resolve(self, expr):
            if (
                isinstance(expr, cst.Call)
                and isinstance(expr.func, cst.Name)
                and expr.func.value == "check_flag"
            ):
                return False
            return None

    graph = make_analysis(unreachable_detector=FlagAwareDetector()).materialize_all()
    dead = {
        f"{src.fqname} -> {dst.fqname}"
        for src, dst, attrs in graph.edges(data=True)
        if attrs.get("flags", EdgeFlags.NONE) & EdgeFlags.DEAD_BRANCH
    }
    assert dead == {"mod -> mod.helper"}


def test_default_resolve_returns_none() -> None:
    # The base detector's ``resolve`` is a no-op hook. Verifying the
    # default explicitly keeps the contract for subclasses clear.
    import libcst as cst

    from dead_cst.branches import DefaultUnreachableRegionDetector

    detector = DefaultUnreachableRegionDetector()
    assert detector.resolve(cst.Name("anything")) is None
    assert detector.resolve(cst.Integer("0")) is None
