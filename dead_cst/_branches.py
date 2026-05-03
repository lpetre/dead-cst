"""Static evaluator for conditional truthiness.

Determines, when possible, whether the truthiness of an expression is
statically known. Used to identify dead branches of ``if`` / ``while``
statements so callers can mark references that live inside them with
:data:`dead_cst._symbols.EdgeFlags.DEAD_BRANCH`.

Only handles a small whitelist of literal forms: the ``True`` /
``False`` / ``None`` keywords, integer / string literals,
empty-vs-non-empty collection literals, and ``not`` / ``and`` / ``or``
over those. Anything involving an attribute access, function call,
comparison, or other dynamic operation returns ``None`` (unknown).
Bare ``Name`` nodes outside the three keywords are also unknown by
default; callers that have scope information (notably
:class:`DefaultUnreachableRegionDetector` via
:func:`~dead_cst._const_fold.fold_constants`) pass a ``resolve_name``
callback to fold names whose binding resolves to a known constant.
Returning ``None`` is always the safe default: callers must treat the
branch as live.

Module-level unreachable-region detection is exposed as the
:class:`UnreachableRegionDetector` protocol so downstream consumers
can fold in domain knowledge -- e.g. config flags whose values are
fixed in production -- without forking the analyzer.
:class:`DefaultUnreachableRegionDetector` ships the literal-only
behavior above plus a fixpoint constant-folding pre-pass so simple
``DEBUG = False; if DEBUG:`` patterns are caught out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

from ._cacheable import Cacheable


_KEYWORDS: dict[str, bool] = {
    "True": True,
    "False": False,
    "None": False,
}


# Resolver returns the statically-known truthiness of the constant a
# ``Name`` is bound to, or ``None`` when the binding can't be folded.
ResolveName = Callable[[cst.Name], bool | None]


def evaluate_truthiness(
    node: cst.BaseExpression,
    resolve_name: ResolveName | None = None,
) -> bool | None:
    """Best-effort static truthiness for ``node``.

    Returns ``True`` / ``False`` if the value's truthiness is statically
    determinable, ``None`` otherwise. Never raises.

    ``resolve_name``, when supplied, is consulted for any ``cst.Name``
    that isn't one of the ``True`` / ``False`` / ``None`` keywords.
    """
    if isinstance(node, cst.Name):
        if node.value in _KEYWORDS:
            return _KEYWORDS[node.value]
        if resolve_name is not None:
            return resolve_name(node)
        return None

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
        inner = evaluate_truthiness(node.expression, resolve_name)
        return None if inner is None else not inner

    if isinstance(node, cst.BooleanOperation):
        left = evaluate_truthiness(node.left, resolve_name)
        if isinstance(node.operator, cst.And):
            if left is False:
                return False
            right = evaluate_truthiness(node.right, resolve_name)
            if right is False:
                return False
            if left is True and right is True:
                return True
            return None
        if isinstance(node.operator, cst.Or):
            if left is True:
                return True
            right = evaluate_truthiness(node.right, resolve_name)
            if right is True:
                return True
            if left is False and right is False:
                return False
            return None

    return None


def unreachable_suites(
    stmt: cst.BaseStatement,
    resolve_name: ResolveName | None = None,
) -> list[cst.BaseSuite]:
    """Return every dead suite inside ``stmt``.

    Supports ``cst.If`` (including ``elif`` / ``else`` chains) and
    ``cst.While``. Returns ``[]`` for any other statement type or when
    no branch can be shown to be unreachable.

    Returns the suite nodes themselves so callers that need source
    positions (e.g. the visitor) can read them off the node.
    """
    if isinstance(stmt, cst.If):
        return _unreachable_in_if(stmt, branch_taken=False, resolve_name=resolve_name)
    if isinstance(stmt, cst.While):
        truth = evaluate_truthiness(stmt.test, resolve_name)
        if truth is False:
            return [stmt.body]
        # ``while True:`` exits only via break / return / exception, so
        # the ``else`` clause (which fires on normal exit) never runs.
        if truth is True and stmt.orelse is not None:
            return [stmt.orelse.body]
        return []
    return []


def unreachable_bodies(
    stmt: cst.BaseStatement,
    resolve_name: ResolveName | None = None,
) -> list[Sequence[cst.CSTNode]]:
    """Return the ``.body`` of every dead suite inside ``stmt``.

    Thin wrapper over :func:`unreachable_suites` for callers that only
    need the statement list, not the enclosing suite. Each entry is
    typed as ``Sequence[cst.CSTNode]`` because libcst's
    ``BaseSuite.body`` may be ``Sequence[BaseStatement]`` (an indented
    block) or ``Sequence[BaseSmallStatement]`` (a one-line suite like
    ``if False: x = 1``).
    """
    return [suite.body for suite in unreachable_suites(stmt, resolve_name)]


@runtime_checkable
class UnreachableRegionDetector(Cacheable, Protocol):
    """Finds statically-unreachable source regions in a parsed module.

    Detectors run once per file after the visitor walk; the returned
    list of :class:`CodeRange` determines which references land
    flagged with :data:`dead_cst._symbols.EdgeFlags.DEAD_BRANCH` and
    which positions are surfaced as "unreachable code at line X"
    reports.

    Pass a custom detector to :func:`dead_cst.build_symbol_graph` to
    layer on company-specific constant folding (e.g.
    ``settings.IS_PROD`` is always ``True``). The default,
    :class:`DefaultUnreachableRegionDetector`, runs literal-only
    truthiness on every ``if`` / ``while`` test, augmented by a
    fixpoint constant-folding pre-pass over simple ``Name = literal``
    assignments.

    Inherits the ``(name, version)`` contract from :class:`Cacheable`
    so swapping detectors invalidates stale ``VisitorPayload`` blobs
    automatically.
    """

    def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]: ...


@dataclass(frozen=True)
class DefaultUnreachableRegionDetector:
    """Built-in :class:`UnreachableRegionDetector`.

    Runs two passes per file:

    1. :func:`~dead_cst._const_fold.fold_constants` -- a fixpoint
       constant-folding pass that propagates simple
       ``Name = literal`` (and ``Name: T = literal``) assignments
       through their access points, including chained forms like
       ``a = False; b = a or False; if b:``.
    2. A walk over every ``cst.If`` / ``cst.While``, calling
       :func:`unreachable_suites` with a ``resolve_name`` closure
       that consults the folded table.

    The folding pass is keyed by ``id`` of the access node, so it
    stays flow-sensitive: a later rebinding shadows an earlier one,
    and conditional bindings whose live values disagree refuse to
    fold (the safe default).
    """

    name: str = "default"
    version: int = 2

    def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]:
        # Local import to avoid a top-level cycle: ``_const_fold``
        # depends on :func:`evaluate_truthiness` from this module.
        from ._const_fold import fold_constants

        positions = wrapper.resolve(PositionProvider)
        truthy = fold_constants(wrapper)

        def resolve(name: cst.Name) -> bool | None:
            return truthy.get(id(name))

        found: list[CodeRange] = []

        class _Collector(cst.CSTVisitor):
            def visit_If(self, node: cst.If) -> None:
                self._collect(node)

            def visit_While(self, node: cst.While) -> None:
                self._collect(node)

            def _collect(self, stmt: cst.BaseStatement) -> None:
                for suite in unreachable_suites(stmt, resolve):
                    pos = positions.get(suite)
                    if pos is not None:
                        found.append(pos)

        wrapper.module.visit(_Collector())
        return found


def _unreachable_in_if(
    node: cst.If,
    branch_taken: bool,
    resolve_name: ResolveName | None = None,
) -> list[cst.BaseSuite]:
    """Walk an ``if`` / ``elif`` / ``else`` chain collecting dead suites.

    ``branch_taken`` is ``True`` when an earlier branch in the chain is
    known to fire; everything from this point on is then unreachable.
    """
    dead: list[cst.BaseSuite] = []
    truth = None if branch_taken else evaluate_truthiness(node.test, resolve_name)

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
    dead.extend(_unreachable_in_if(orelse, next_taken, resolve_name))
    return dead
