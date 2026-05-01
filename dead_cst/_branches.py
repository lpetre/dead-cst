"""Static evaluator for conditional truthiness.

Determines, when possible, whether the truthiness of an expression is
statically known. Used to identify dead branches of ``if`` / ``while``
statements so callers can mark references that live inside them with
:data:`dead_cst._symbols.EdgeFlags.DEAD_BRANCH`.

Only handles a small whitelist of literal forms: the ``True`` /
``False`` / ``None`` keywords, integer / string literals,
empty-vs-non-empty collection literals, and ``not`` / ``and`` / ``or``
over those. Anything involving a name lookup (other than the three
keywords), attribute access, function call, comparison, or other
dynamic operation returns ``None`` (unknown). Returning ``None`` is
always the safe default: callers must treat the branch as live.
"""

from __future__ import annotations

from typing import Sequence

import libcst as cst


_KEYWORDS: dict[str, bool] = {
    "True": True,
    "False": False,
    "None": False,
}


def evaluate_truthiness(node: cst.BaseExpression) -> bool | None:
    """Best-effort static truthiness for ``node``.

    Returns ``True`` / ``False`` if the value's truthiness is statically
    determinable, ``None`` otherwise. Never raises.
    """
    if isinstance(node, cst.Name):
        return _KEYWORDS.get(node.value)

    if isinstance(node, cst.Integer):
        try:
            return int(node.value, 0) != 0
        except ValueError:
            return None

    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        try:
            value = node.evaluated_value
        except Exception:
            return None
        if value is None:
            return None
        return bool(value)

    if isinstance(node, (cst.Tuple, cst.List, cst.Set)):
        for elt in node.elements:
            if isinstance(elt, cst.StarredElement):
                return None
        return len(node.elements) > 0

    if isinstance(node, cst.Dict):
        for elt in node.elements:
            if isinstance(elt, cst.StarredDictElement):
                return None
        return len(node.elements) > 0

    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Not):
        inner = evaluate_truthiness(node.expression)
        return None if inner is None else not inner

    if isinstance(node, cst.BooleanOperation):
        left = evaluate_truthiness(node.left)
        if isinstance(node.operator, cst.And):
            if left is False:
                return False
            right = evaluate_truthiness(node.right)
            if right is False:
                return False
            if left is True and right is True:
                return True
            return None
        if isinstance(node.operator, cst.Or):
            if left is True:
                return True
            right = evaluate_truthiness(node.right)
            if right is True:
                return True
            if left is False and right is False:
                return False
            return None

    return None


def unreachable_suites(stmt: cst.BaseStatement) -> list[cst.BaseSuite]:
    """Return every dead suite inside ``stmt``.

    Supports ``cst.If`` (including ``elif`` / ``else`` chains) and
    ``cst.While``. Returns ``[]`` for any other statement type or when
    no branch can be shown to be unreachable.

    Returns the suite nodes themselves so callers that need source
    positions (e.g. the visitor) can read them off the node.
    """
    if isinstance(stmt, cst.If):
        return _unreachable_in_if(stmt, branch_taken=False)
    if isinstance(stmt, cst.While):
        truth = evaluate_truthiness(stmt.test)
        if truth is False:
            return [stmt.body]
        # ``while True:`` exits only via break / return / exception, so
        # the ``else`` clause (which fires on normal exit) never runs.
        if truth is True and stmt.orelse is not None:
            return [stmt.orelse.body]
        return []
    return []


def unreachable_bodies(stmt: cst.BaseStatement) -> list[Sequence[cst.CSTNode]]:
    """Return the ``.body`` of every dead suite inside ``stmt``.

    Thin wrapper over :func:`unreachable_suites` for callers that only
    need the statement list, not the enclosing suite. Each entry is
    typed as ``Sequence[cst.CSTNode]`` because libcst's
    ``BaseSuite.body`` may be ``Sequence[BaseStatement]`` (an indented
    block) or ``Sequence[BaseSmallStatement]`` (a one-line suite like
    ``if False: x = 1``).
    """
    return [suite.body for suite in unreachable_suites(stmt)]


def _unreachable_in_if(node: cst.If, branch_taken: bool) -> list[cst.BaseSuite]:
    """Walk an ``if`` / ``elif`` / ``else`` chain collecting dead suites.

    ``branch_taken`` is ``True`` when an earlier branch in the chain is
    known to fire; everything from this point on is then unreachable.
    """
    dead: list[cst.BaseSuite] = []
    truth = None if branch_taken else evaluate_truthiness(node.test)

    if branch_taken or truth is False:
        dead.append(node.body)

    next_taken = branch_taken or truth is True

    orelse = node.orelse
    if orelse is None:
        return dead
    if isinstance(orelse, cst.Else):
        if next_taken:
            dead.append(orelse.body)
        return dead
    dead.extend(_unreachable_in_if(orelse, next_taken))
    return dead
