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
caller (``DefaultUnreachableRegionDetector``) can hand it to
:func:`~dead_cst._branches.unreachable_suites` as a ``resolve_name``
callback.

Only single-target ``Name`` LHS shapes participate. Tuple unpacking,
attribute / subscript targets, walrus, augmented assign, parameter
defaults, and import bindings all return ``None`` from the RHS lookup
helper, which the caller treats as "unknown" -- the safe default.
Bindings whose live values disagree (e.g. ``if cond: x = True else:
x = False``) likewise stay unknown. Cycles (``a = b; b = a``) never
escape the unknown bucket because neither ever resolves.
"""

from __future__ import annotations

import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider, ParentNodeProvider
from libcst.metadata.scope_provider import (
    Assignment,
    ClassScope,
    FunctionScope,
    GlobalScope,
)

from ._branches import evaluate_truthiness
from ._flow import live_referents


def fold_constants(wrapper: MetadataWrapper) -> dict[int, bool]:
    """Map ``id(name_access_node) -> truthiness`` for every foldable access.

    Iteratively propagates known constants through simple
    ``Name = literal`` (or ``Name: T = literal``) assignments until no
    new entries are produced. The returned dict is keyed by Python
    ``id()`` of the access ``cst.Name`` node so the caller can build a
    flow-sensitive ``resolve_name`` closure: ``lambda n: truthy.get(id(n))``.

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

    def resolve(name_node: cst.Name) -> bool | None:
        return truthy.get(id(name_node))

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
    parent_map: dict[cst.CSTNode, cst.CSTNode],
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
