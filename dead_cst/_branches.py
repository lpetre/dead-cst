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

Module-level unreachable-region detection is exposed as a swappable
callable (:data:`UnreachableRegionDetector`) so downstream consumers
can fold in domain knowledge -- e.g. config flags whose values are
fixed in production -- without forking the analyzer. The default
detector, :func:`default_unreachable_regions`, walks every ``if`` and
``while`` in the module and runs the literal-only evaluator above on
the test expression.
"""

from __future__ import annotations

from typing import Callable, Sequence

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider


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


# Type alias for the unreachable-region detector callable. The
# analyzer invokes one detector per module after the visitor has
# walked it; the returned list of :class:`CodeRange` determines which
# declarations and references are flagged with
# :data:`dead_cst._symbols.EdgeFlags.DEAD_BRANCH` and which positions
# are surfaced as "unreachable code at line X" reports.
#
# Pass a custom detector to :func:`dead_cst.build_symbol_graph` to
# layer on company-specific constant folding (e.g. "``settings.IS_PROD``
# is always ``True``"). The default,
# :func:`default_unreachable_regions`, runs literal-only truthiness on
# every ``if`` / ``while`` test.
#
# Optional ``name`` and ``version`` attributes on the callable feed
# the per-file cache fingerprint, so swapping detectors invalidates
# stale ``VisitorPayload`` blobs automatically. Bare functions fall
# back to ``__name__`` and ``0``; set / bump these explicitly when
# the detector's logic changes.
UnreachableRegionDetector = Callable[[MetadataWrapper], list[CodeRange]]


def default_unreachable_regions(wrapper: MetadataWrapper) -> list[CodeRange]:
    """Default detector: literal-truthiness analysis on ``if`` / ``while``.

    Walks every ``cst.If`` and ``cst.While`` in the module, runs
    :func:`unreachable_suites` on each, and returns the
    :class:`CodeRange` of every dead suite. The order of the result
    follows the document order of the suites.
    """
    positions = wrapper.resolve(PositionProvider)
    found: list[CodeRange] = []

    class _Collector(cst.CSTVisitor):
        def visit_If(self, node: cst.If) -> None:
            self._collect(node)

        def visit_While(self, node: cst.While) -> None:
            self._collect(node)

        def _collect(self, stmt: cst.BaseStatement) -> None:
            for suite in unreachable_suites(stmt):
                pos = positions.get(suite)
                if pos is not None:
                    found.append(pos)

    wrapper.module.visit(_Collector())
    return found


# Cache-fingerprint metadata for the default detector. Custom detectors
# that want stable cache reuse should set their own ``name`` / ``version``
# (a Unix-epoch int by convention, matching the ``EdgePlugin`` story).
default_unreachable_regions.name = "default"  # type: ignore[attr-defined]
default_unreachable_regions.version = 1  # type: ignore[attr-defined]


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
