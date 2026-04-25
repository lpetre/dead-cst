"""Flow-sensitive referent filter.

Given a single access and a set of candidate referents that all share
one lexical scope (which is what ``ScopeProvider`` guarantees -- see
``tests/test_scope_provider_contract.py``), return the subset that is
actually live at the access under a control-flow-aware model.

Rules applied during a forward pass over the scope's statement list:

* Sequential: a later binding in the same block kills earlier bindings.
* ``if`` / ``elif`` / ``else``: each branch sees the pre-``if`` state.
  The post-``if`` state is the union of each branch's end-state. A
  missing ``else`` counts as a pass-through branch.
* ``try`` / ``except`` / ``else`` / ``finally``: body and each handler
  both see the pre-``try`` state (pessimistic -- an exception can
  happen before any binding in the body). The post-``try`` state is
  the union of the body's end-state and each handler's end-state; the
  ``else`` clause runs after the body on the no-exception path;
  ``finally`` runs afterwards on all paths.
* ``for`` / ``while``: the body may execute zero or more times, so the
  post-loop state is the pre-loop state unioned with the body's
  end-state. ``else`` runs if the loop exits normally.

Not yet modelled: walrus expressions, ``global`` / ``nonlocal``
rebindings, exception re-raise edges, ``match`` statements, ``del``.
Wired into edge resolution via ``SymbolVisitor``; see
``tests/test_flow_sensitive_filter.py`` for the filter's own cases
and ``tests/test_declarations.py::test_shadowed_declarations`` for
the end-to-end behaviour.
"""

from __future__ import annotations

from typing import Callable, Sequence

import libcst as cst


def _descendant_ids(node: cst.CSTNode, cache: dict[int, frozenset[int]]) -> frozenset[int]:
    key = id(node)
    if key in cache:
        return cache[key]
    ids = {key}
    for child in node.children:
        ids |= _descendant_ids(child, cache)
    frozen = frozenset(ids)
    cache[key] = frozen
    return frozen


def _walk_flow(
    stmts: Sequence[cst.BaseStatement],
    incoming: set[cst.CSTNode],
    referent_set: set[cst.CSTNode],
    cache: dict[int, frozenset[int]],
    observe: Callable[[cst.BaseStatement, set[cst.CSTNode]], None] | None,
) -> set[cst.CSTNode]:
    """Forward-walk ``stmts`` evolving the live referent set.

    ``observe`` (when supplied) is invoked once per statement with the
    pre-statement state -- callers use it to capture the state seen by
    a specific access node nested inside ``stmt``.
    """

    def _referents_in(node: cst.CSTNode) -> set[cst.CSTNode]:
        ids = _descendant_ids(node, cache)
        return {r for r in referent_set if id(r) in ids}

    def _else_flow(
        orelse: cst.Else | cst.If | None, incoming: set[cst.CSTNode]
    ) -> set[cst.CSTNode]:
        if orelse is None:
            return set(incoming)
        if isinstance(orelse, cst.Else):
            return _walk_flow(orelse.body.body, incoming, referent_set, cache, observe)
        # ``elif`` is an ``If`` in the orelse slot; run it as a nested If.
        return _walk_flow([orelse], incoming, referent_set, cache, observe)

    state = set(incoming)
    for stmt in stmts:
        # Record the state the access sees *before* any bindings in the
        # same statement take effect. RHS-of-assignment and
        # expression-statement uses both match that convention.
        if observe is not None:
            observe(stmt, state)

        if isinstance(stmt, cst.If):
            body_end = _walk_flow(stmt.body.body, state, referent_set, cache, observe)
            orelse_end = _else_flow(stmt.orelse, state)
            state = body_end | orelse_end
            continue

        if isinstance(stmt, cst.Try):
            # Body and each handler both see the pre-try state.
            body_end = _walk_flow(stmt.body.body, state, referent_set, cache, observe)
            post = set(body_end)
            for handler in stmt.handlers:
                post |= _walk_flow(handler.body.body, state, referent_set, cache, observe)
            if stmt.orelse is not None:
                # else runs only on the no-exception path, after body.
                post = _walk_flow(stmt.orelse.body.body, body_end, referent_set, cache, observe) | (
                    post - body_end
                )
            if stmt.finalbody is not None:
                post = _walk_flow(stmt.finalbody.body, post, referent_set, cache, observe)
            state = post
            continue

        if isinstance(stmt, (cst.For, cst.While)):
            body_end = _walk_flow(stmt.body.body, state, referent_set, cache, observe)
            post = state | body_end
            if stmt.orelse is not None:
                post = _walk_flow(stmt.orelse.body.body, post, referent_set, cache, observe)
            state = post
            continue

        bindings_here = _referents_in(stmt)
        if bindings_here:
            state = bindings_here

    return state


def live_referents(
    scope_body: Sequence[cst.BaseStatement],
    access_node: cst.CSTNode,
    referent_nodes: Sequence[cst.CSTNode],
) -> set[cst.CSTNode]:
    """Filter ``referent_nodes`` to those live at ``access_node``.

    ``scope_body`` is the ordered statement list of the lexical scope
    containing the access (``module.body`` for module-level accesses,
    ``functiondef.body.body`` for a function scope, etc.). The access
    and every referent must be reachable from ``scope_body``.
    """
    cache: dict[int, frozenset[int]] = {}
    referent_set = set(referent_nodes)
    access_id = id(access_node)
    observed: list[set[cst.CSTNode]] = []

    def _observe(stmt: cst.BaseStatement, state: set[cst.CSTNode]) -> None:
        if access_id in _descendant_ids(stmt, cache):
            observed.append(set(state))

    _walk_flow(scope_body, set(), referent_set, cache, _observe)

    result: set[cst.CSTNode] = set()
    for s in observed:
        result |= s
    return result


def live_at_exit(
    scope_body: Sequence[cst.BaseStatement],
    referent_nodes: Sequence[cst.CSTNode],
) -> set[cst.CSTNode]:
    """Return the subset of ``referent_nodes`` live after ``scope_body`` runs.

    Same flow model as :func:`live_referents` but observes the state
    *after* the last statement, so callers can see which bindings
    survive to the end of the scope -- e.g. which top-level decls a
    module exports across all reachable control-flow paths.
    """
    cache: dict[int, frozenset[int]] = {}
    return _walk_flow(scope_body, set(), set(referent_nodes), cache, None)
