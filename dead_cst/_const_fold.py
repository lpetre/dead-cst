"""Fixpoint constant folding for ``Name`` accesses.

A separate pre-pass for the unreachable-region detector. Walks every
``Name`` access in a module, looks up its binding(s) via
:class:`~libcst.metadata.ScopeProvider`, and -- when every live binding
ties back to a simple ``Name = expr`` (or ``Name: T = expr``)
assignment whose RHS is statically truthy/falsy -- records the
access's known truthiness in a map.

Iteration is the point. A direct one-shot resolver only sees literal
RHSes, so ``a = False; b = a or False; if b:`` doesn't fold. Each
pass over the access list propagates one more level of indirection;
once a pass produces no new entries the table is at fixpoint and the
caller (``DefaultUnreachableRegionDetector``) can build a closure
over the table to feed back into :func:`unreachable_suites`.

The optional ``resolve_expr`` parameter plumbs custom truthiness
through the same loop: when a subclassed detector overrides
:meth:`DefaultUnreachableRegionDetector.resolve` to know that
``check_flag("migration-abc")`` is ``True``, that knowledge composes
with literal folding -- ``flag = check_flag("migration-abc"); if
flag:`` resolves correctly because the RHS evaluation consults the
external resolver before falling back to the table.

Only single-target ``Name`` LHS shapes participate. Tuple unpacking,
attribute / subscript targets, walrus, augmented assign, parameter
defaults, and import bindings all return ``None`` from the RHS lookup
helper, which the caller treats as "unknown" -- the safe default.
Bindings whose live values disagree (e.g. ``if cond: x = True else:
x = False``) likewise stay unknown. Cycles (``a = b; b = a``) never
escape the unknown bucket because neither ever resolves.
"""

from __future__ import annotations

from typing import Mapping

import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider, ParentNodeProvider
from libcst.metadata.scope_provider import (
    Assignment,
    ClassScope,
    FunctionScope,
    GlobalScope,
)

from ._branches import ResolveExpr, evaluate_truthiness
from ._flow import live_referents


def fold_constants(
    wrapper: MetadataWrapper,
    resolve_expr: ResolveExpr | None = None,
) -> dict[int, bool]:
    """Map ``id(name_access_node) -> truthiness`` for every foldable access.

    Iteratively propagates known constants through simple
    ``Name = literal`` (or ``Name: T = literal``) assignments until no
    new entries are produced. The returned dict is keyed by Python
    ``id()`` of the access ``cst.Name`` node so the caller can build a
    flow-sensitive resolver: ``lambda expr: truthy.get(id(expr)) if
    isinstance(expr, cst.Name) else None``.

    ``resolve_expr``, when supplied, gets first crack at any
    expression encountered while evaluating an assignment's RHS.
    Returning a ``bool`` short-circuits literal handling for that
    node; returning ``None`` defers. This is how subclasses of
    :class:`DefaultUnreachableRegionDetector` thread custom domain
    knowledge through the loop -- e.g. ``flag = check_flag("x")``
    folds when the override answers for the call.

    Names that don't fold -- mixed-value bindings, non-literal RHS,
    unsupported assignment shape, cyclic references -- are simply
    absent from the map.
    """
    scopes = wrapper.resolve(ScopeProvider)
    parent_map = wrapper.resolve(ParentNodeProvider)
    module = wrapper.module

    accesses: list = []
    for scope in set(scopes.values()):
        for access in scope.accesses:
            if isinstance(access.node, cst.Name):
                accesses.append(access)

    rhs_cache: dict[int, cst.BaseExpression | None] = {}

    def _rhs_for(binding_node: cst.CSTNode) -> cst.BaseExpression | None:
        key = id(binding_node)
        if key not in rhs_cache:
            rhs_cache[key] = _constant_assignment_rhs(binding_node, parent_map)
        return rhs_cache[key]

    truthy: dict[int, bool] = {}

    def resolve(expr: cst.BaseExpression) -> bool | None:
        # External (custom) resolver gets first try -- it may know
        # things the fold table doesn't (e.g. a feature-flag call).
        if resolve_expr is not None:
            v = resolve_expr(expr)
            if v is not None:
                return v
        # Fall back to the fold table; only Name accesses live here.
        if isinstance(expr, cst.Name):
            return truthy.get(id(expr))
        return None

    while True:
        progressed = False
        for access in accesses:
            if id(access.node) in truthy:
                continue
            referents = [
                r
                for r in access.referents
                if isinstance(r, Assignment) and isinstance(r.node, cst.Name)
            ]
            if not referents:
                continue
            # Use the *binding* scope's body, not the access's. A
            # function-scope access reading a module-level constant
            # has a referent whose scope is the module; ``live_referents``
            # needs to walk the body that contains the binding(s), not
            # the body that contains the access. ScopeProvider's
            # contract is that all referents of one access share one
            # scope, so any referent's scope is fine -- match the
            # existing visitor pattern of taking ``referents[0].scope``.
            body = _scope_body(referents[0].scope, module)
            if body is None:
                continue
            # Filter to bindings live on at least one path to the
            # access. ``live_referents`` runs the same flow model the
            # visitor uses for shadowed-decl resolution, so a later
            # rebinding correctly hides an earlier one.
            live_ids = {
                id(n) for n in live_referents(body, access.node, [r.node for r in referents])
            }
            live = [r for r in referents if id(r.node) in live_ids]
            if not live:
                continue
            values: set[bool] = set()
            unknown = False
            for ref in live:
                rhs = _rhs_for(ref.node)
                if rhs is None:
                    unknown = True
                    break
                v = evaluate_truthiness(rhs, resolve)
                if v is None:
                    unknown = True
                    break
                values.add(v)
            if unknown or len(values) != 1:
                continue
            truthy[id(access.node)] = next(iter(values))
            progressed = True
        if not progressed:
            return truthy


def _scope_body(scope, module: cst.Module) -> list | None:
    """Statement list for ``scope``, or ``None`` for unsupported kinds.

    Mirrors :meth:`SymbolVisitor._scope_body` but additionally guards
    against ``FunctionScope`` over a ``Lambda`` (whose ``body`` is a
    ``BaseExpression``, not a ``BaseSuite``). Comprehension scopes and
    lambdas return ``None`` so the caller skips them rather than
    guessing -- they can't host ``if`` / ``while`` statements anyway.
    """
    if isinstance(scope, GlobalScope):
        return list(module.body)
    if isinstance(scope, ClassScope):
        return list(scope.node.body.body)
    if isinstance(scope, FunctionScope) and isinstance(scope.node, cst.FunctionDef):
        return list(scope.node.body.body)
    return None


def _constant_assignment_rhs(
    binding_node: cst.CSTNode,
    parent_map: Mapping[cst.CSTNode, cst.CSTNode],
) -> cst.BaseExpression | None:
    """RHS of a simple ``name = expr`` (or ``name: T = expr``) binding.

    Returns ``None`` for any shape we don't fold: tuple/list unpacking,
    attribute or subscript targets, walrus, augmented assign,
    parameter defaults, import bindings, etc. Multi-target chained
    assignment (``a = b = expr``) is supported because all targets
    share one RHS.
    """
    parent = parent_map.get(binding_node)
    if isinstance(parent, cst.AssignTarget):
        grandparent = parent_map.get(parent)
        if not isinstance(grandparent, cst.Assign):
            return None
        if not isinstance(parent.target, cst.Name):
            return None
        if parent.target is not binding_node:
            return None
        return grandparent.value
    if isinstance(parent, cst.AnnAssign):
        if not isinstance(parent.target, cst.Name):
            return None
        if parent.target is not binding_node:
            return None
        return parent.value
    return None
