"""Prototype flow-sensitive referent filter (phase 2).

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
This is a prototype -- it is *not* wired into the edge resolution
pass. See ``tests/test_flow_sensitive_filter.py`` for the cases it
currently handles.
"""

from __future__ import annotations

from typing import Sequence

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

    def _referents_in(node: cst.CSTNode) -> set[cst.CSTNode]:
        ids = _descendant_ids(node, cache)
        return {r for r in referent_set if id(r) in ids}

    def _contains_access(node: cst.CSTNode) -> bool:
        return access_id in _descendant_ids(node, cache)

    def _flow(
        stmts: Sequence[cst.BaseStatement], incoming: set[cst.CSTNode]
    ) -> set[cst.CSTNode]:
        state = set(incoming)
        for stmt in stmts:
            # Record the state the access sees *before* any bindings
            # in the same statement take effect. RHS-of-assignment and
            # expression-statement uses both match that convention.
            if _contains_access(stmt):
                observed.append(set(state))

            if isinstance(stmt, cst.If):
                body_end = _flow(stmt.body.body, state)
                orelse_end = _else_flow(stmt.orelse, state)
                state = body_end | orelse_end
                continue

            if isinstance(stmt, cst.Try):
                # Body and each handler both see the pre-try state.
                body_end = _flow(stmt.body.body, state)
                post = set(body_end)
                for handler in stmt.handlers:
                    post |= _flow(handler.body.body, state)
                if stmt.orelse is not None:
                    # else runs only on the no-exception path, after body.
                    post = _flow(stmt.orelse.body.body, body_end) | (post - body_end)
                if stmt.finalbody is not None:
                    post = _flow(stmt.finalbody.body, post)
                state = post
                continue

            if isinstance(stmt, (cst.For, cst.While)):
                body_end = _flow(stmt.body.body, state)
                post = state | body_end
                if stmt.orelse is not None:
                    post = _flow(stmt.orelse.body.body, post)
                state = post
                continue

            bindings_here = _referents_in(stmt)
            if bindings_here:
                state = bindings_here

        return state

    def _else_flow(
        orelse: cst.Else | cst.If | None, incoming: set[cst.CSTNode]
    ) -> set[cst.CSTNode]:
        if orelse is None:
            return set(incoming)
        if isinstance(orelse, cst.Else):
            return _flow(orelse.body.body, incoming)
        # ``elif`` is an ``If`` in the orelse slot; run it as a nested If.
        return _flow([orelse], incoming)

    _flow(scope_body, set())

    result: set[cst.CSTNode] = set()
    for s in observed:
        result |= s
    return result
