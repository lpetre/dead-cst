"""Unit tests for the goal-directed truthiness resolver.

The resolver layers flow-sensitive ``Name`` lookup over the literal-only
:func:`~dead_cst.branches.evaluate_truthiness`, propagating simple
``Name = literal`` (and ``Name: T = literal``) bindings through
chained forms like ``a = False; b = a or False``. Cases here cover the
value pipeline -- direct literals, chains, boolean operators,
conditional bindings -- without going through the symbol graph; the
end-to-end behaviour is in ``test_unreachable_branches.py``.
"""

from __future__ import annotations

import textwrap

import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider

from dead_cst.branches import TruthinessResolver


def _resolve_lookup(src: str) -> dict[str, list[bool | None]]:
    """Return ``{name -> [truthiness, ...]}`` for every Name access.

    Each entry's list has one bool per *access* of that name (binding
    LHS occurrences are excluded), in document order; accesses that
    don't fold appear as ``None`` so callers can assert both presence
    and absence.
    """
    module = cst.parse_module(textwrap.dedent(src).strip())
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    resolver = TruthinessResolver(wrapper)
    scopes = wrapper.resolve(ScopeProvider)

    access_nodes: set[int] = set()
    for scope in set(scopes.values()):
        for access in scope.accesses:
            if isinstance(access.node, cst.Name):
                access_nodes.add(id(access.node))

    result: dict[str, list] = {}

    class _Collect(cst.CSTVisitor):
        def visit_Name(self, node: cst.Name) -> None:
            if id(node) in access_nodes:
                result.setdefault(node.value, []).append(resolver.evaluate(node))

    wrapper.module.visit(_Collect())
    return result


def test_direct_literal_assignment_folds() -> None:
    out = _resolve_lookup(
        """
        x = False
        if x:
            pass
        """
    )
    # Two ``x`` accesses: the LHS isn't an access (it's a binding) so
    # we only see the if-test occurrence.
    assert out["x"] == [False]


def test_truthy_literal_folds_to_true() -> None:
    out = _resolve_lookup(
        """
        x = 1
        if x:
            pass
        """
    )
    assert out["x"] == [True]


def test_chained_constants_resolve_through_recursion() -> None:
    # ``b``'s RHS references ``a``; the resolver recurses into ``a``
    # and memoizes the result before answering for ``b``.
    out = _resolve_lookup(
        """
        a = False
        b = a or False
        if b:
            pass
        """
    )
    # ``a`` appears once on the RHS of ``b``'s assignment and ``b``
    # once in the if-test. Both fold to False.
    assert out["a"] == [False]
    assert out["b"] == [False]


def test_long_chain_resolves() -> None:
    out = _resolve_lookup(
        """
        a = True
        b = a
        c = not b
        d = c or False
        if d:
            pass
        """
    )
    assert out["a"] == [True]
    assert out["b"] == [True]
    assert out["c"] == [False]
    assert out["d"] == [False]


def test_conditional_binding_does_not_fold() -> None:
    # Both bindings are live at the access; their values disagree, so
    # the resolver returns nothing for the if-test access.
    out = _resolve_lookup(
        """
        if cond:
            x = True
        else:
            x = False
        if x:
            pass
        """
    )
    # The two LHS bindings aren't accesses; the only ``x`` access is
    # in the if-test, and it stays unknown.
    assert out["x"] == [None]


def test_reassignment_keeps_only_last_binding_value() -> None:
    out = _resolve_lookup(
        """
        x = True
        x = False
        if x:
            pass
        """
    )
    # Flow analysis kills the line-1 binding, so the access folds to
    # the live (line-2) value.
    assert out["x"] == [False]


def test_non_literal_rhs_blocks_fold() -> None:
    out = _resolve_lookup(
        """
        x = compute()
        if x:
            pass
        """
    )
    assert out["x"] == [None]


def test_cyclic_self_reference_stays_unknown() -> None:
    # ``a`` and ``b`` reference each other; the cycle sentinel makes
    # the resolver bottom out at ``None`` instead of recursing forever.
    out = _resolve_lookup(
        """
        a = b
        b = a
        if a:
            pass
        """
    )
    # ``a`` is accessed twice (RHS of ``b`` + if-test), ``b`` once
    # (RHS of ``a``); none of the accesses fold.
    assert out["a"] == [None, None]
    assert out["b"] == [None]


def test_tuple_unpacking_does_not_fold() -> None:
    # ``_constant_assignment_rhs`` only handles single-Name LHS; tuple
    # unpacking goes through ``AssignTarget`` whose ``.target`` is a
    # ``Tuple``, and we bail.
    out = _resolve_lookup(
        """
        a, b = (False, False)
        if a:
            pass
        """
    )
    assert out["a"] == [None]


def test_walrus_binding_folds() -> None:
    # ``_constant_assignment_rhs`` recognises ``NamedExpr`` bindings the
    # same way it recognises ``Assign`` / ``AnnAssign`` -- the walrus
    # target's value is unambiguously the RHS, so a literal RHS folds.
    out = _resolve_lookup(
        """
        (x := False)
        if x:
            pass
        """
    )
    assert out["x"] == [False]


def test_walrus_in_if_test_folds() -> None:
    # ``evaluate_truthiness`` unwraps ``NamedExpr`` to its value, so the
    # if-test's truthiness is statically known even when the test is a
    # bare walrus expression. There's no Name access here -- the walrus
    # target is a binding, and the if-test's NamedExpr is an expression
    # node, not a Name lookup -- so nothing shows up in ``out``.
    out = _resolve_lookup(
        """
        if (x := False):
            pass
        """
    )
    assert "x" not in out


def test_attribute_target_does_not_fold() -> None:
    out = _resolve_lookup(
        """
        obj.flag = False
        if flag:
            pass
        """
    )
    # ``obj.flag = ...`` doesn't bind a Name at all -- there's no
    # binding for ``flag`` in this scope, so the if-test access has
    # no referents.
    assert out["flag"] == [None]


def test_annotated_assignment_folds() -> None:
    out = _resolve_lookup(
        """
        x: bool = False
        if x:
            pass
        """
    )
    assert out["x"] == [False]


def test_annotated_assignment_without_value_does_not_fold() -> None:
    # ``x: bool`` declares without binding a value; nothing to fold.
    out = _resolve_lookup(
        """
        x: bool
        if x:
            pass
        """
    )
    assert out["x"] == [None]


def test_function_scope_constant_folds() -> None:
    out = _resolve_lookup(
        """
        def f():
            FLAG = False
            if FLAG:
                pass
        """
    )
    assert out["FLAG"] == [False]


def test_lambda_scope_does_not_crash() -> None:
    # FunctionScope.node may be a Lambda; _scope_body must skip
    # gracefully (lambdas can't host if/while anyway).
    out = _resolve_lookup(
        """
        f = lambda x: x or 0
        FLAG = False
        if FLAG:
            pass
        """
    )
    assert out["FLAG"] == [False]


def test_keyword_names_resolve_to_language_truthiness() -> None:
    # ``True`` / ``False`` / ``None`` are language keywords, handled
    # by ``evaluate_truthiness`` directly. The resolver returns their
    # language-defined truthiness and never offers them to the external
    # ``resolve_expr`` (a custom resolver must not be able to redefine
    # ``True is True``).
    seen: list[cst.BaseExpression] = []

    def external(expr):
        seen.append(expr)
        return None

    out = _resolve_with_external(
        """
        if True:
            x = 1
        if False:
            y = 2
        """,
        external,
    )
    assert out["True"] == [True]
    assert out["False"] == [False]
    # External resolver never sees True/False/None Name nodes.
    assert not any(isinstance(e, cst.Name) and e.value in {"True", "False", "None"} for e in seen)


# ----------------------------------------------------------------------
# TruthinessResolver accepts an optional ``resolve_expr`` so a custom
# detector's ``resolve`` method can answer for non-Name expressions
# (Calls, Attributes) that the literal-only path would skip.
# ----------------------------------------------------------------------


def _resolve_with_external(src: str, resolve_expr) -> dict[str, list[bool | None]]:
    """Same as :func:`_resolve_lookup` but plumbs an external resolver."""
    module = cst.parse_module(textwrap.dedent(src).strip())
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    resolver = TruthinessResolver(wrapper, resolve_expr=resolve_expr)
    scopes = wrapper.resolve(ScopeProvider)

    access_nodes: set[int] = set()
    for scope in set(scopes.values()):
        for access in scope.accesses:
            if isinstance(access.node, cst.Name):
                access_nodes.add(id(access.node))

    result: dict[str, list] = {}

    class _Collect(cst.CSTVisitor):
        def visit_Name(self, node: cst.Name) -> None:
            if id(node) in access_nodes:
                result.setdefault(node.value, []).append(resolver.evaluate(node))

    wrapper.module.visit(_Collect())
    return result


def test_external_resolver_folds_call_through_assignment() -> None:
    # The user's example: ``flag = check_flag("x"); if flag:``. The
    # external resolver answers for the Call expression on the RHS;
    # the resolver propagates that into the access for ``flag``.
    def resolver(expr):
        if (
            isinstance(expr, cst.Call)
            and isinstance(expr.func, cst.Name)
            and expr.func.value == "check_flag"
        ):
            return True
        return None

    out = _resolve_with_external(
        """
        flag = check_flag("migration-abc")
        if flag:
            pass
        """,
        resolver,
    )
    assert out["flag"] == [True]


def test_external_resolver_chains_through_boolean_op() -> None:
    # Compose with literal handling: ``check_flag(...) or False``
    # resolves because evaluate_truthiness recurses into the boolean
    # op and the resolver answers for the call.
    def resolver(expr):
        if (
            isinstance(expr, cst.Call)
            and isinstance(expr.func, cst.Name)
            and expr.func.value == "check_flag"
        ):
            return False
        return None

    out = _resolve_with_external(
        """
        flag = check_flag("x") or False
        if flag:
            pass
        """,
        resolver,
    )
    assert out["flag"] == [False]


def test_external_resolver_returning_none_does_not_block_literals() -> None:
    # A resolver that returns None for everything must not regress the
    # literal-only fold path.
    out = _resolve_with_external(
        """
        x = False
        if x:
            pass
        """,
        lambda _: None,
    )
    assert out["x"] == [False]
