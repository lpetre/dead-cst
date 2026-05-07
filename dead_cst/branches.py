"""Static evaluator for conditional truthiness.

Determines, when possible, whether the truthiness of an expression is
statically known. Used to identify dead branches of ``if`` / ``while``
statements so callers can mark references that live inside them with
:data:`dead_cst.graph.EdgeFlags.DEAD_BRANCH`.

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


# Resolver returns the statically-known truthiness of an arbitrary
# expression, or ``None`` to defer to ``evaluate_truthiness``'s
# built-in literal handling. Custom detectors override
# :meth:`DefaultUnreachableRegionDetector.resolve` to plug in
# domain-specific knowledge -- e.g. ``check_flag("migration-abc")``
# is always ``True`` in production.
ResolveExpr = Callable[[cst.BaseExpression], bool | None]


def evaluate_truthiness(
    node: cst.BaseExpression,
    resolve_expr: ResolveExpr | None = None,
) -> bool | None:
    """Best-effort static truthiness for ``node``.

    Returns ``True`` / ``False`` if the value's truthiness is statically
    determinable, ``None`` otherwise. Never raises.

    ``resolve_expr``, when supplied, gets first crack at every
    non-keyword expression. Returning a ``bool`` short-circuits the
    built-in handling for that node; returning ``None`` falls through
    to the literal cases below. The ``True`` / ``False`` / ``None``
    keywords always resolve to their language-defined truthiness and
    are never passed to the resolver.
    """
    # Keywords are language semantics, not user-overridable.
    if isinstance(node, cst.Name) and node.value in _KEYWORDS:
        return _KEYWORDS[node.value]

    # User-supplied resolver gets first crack at any expression --
    # ``Name`` lookups, ``Call`` patterns the user knows about, etc.
    if resolve_expr is not None:
        v = resolve_expr(node)
        if v is not None:
            return v

    if isinstance(node, cst.Name):
        # Bare non-keyword Name with no resolver answer -> unknown.
        return None

    if isinstance(node, cst.Integer):
        try:
            return node.evaluated_value != 0
        except ValueError:
            return None

    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        try:
            value = node.evaluated_value
        except (SyntaxError, UnicodeDecodeError):
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

    if isinstance(node, cst.NamedExpr):
        # ``(x := V)`` evaluates to ``V``; the assignment side-effect
        # doesn't change the expression's runtime value.
        return evaluate_truthiness(node.value, resolve_expr)

    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Not):
        inner = evaluate_truthiness(node.expression, resolve_expr)
        return None if inner is None else not inner

    if isinstance(node, cst.BooleanOperation):
        left = evaluate_truthiness(node.left, resolve_expr)
        if isinstance(node.operator, cst.And):
            if left is False:
                return False
            right = evaluate_truthiness(node.right, resolve_expr)
            if right is False:
                return False
            if left is True and right is True:
                return True
            return None
        if isinstance(node.operator, cst.Or):
            if left is True:
                return True
            right = evaluate_truthiness(node.right, resolve_expr)
            if right is True:
                return True
            if left is False and right is False:
                return False
            return None

    return None


def unreachable_suites(
    stmt: cst.BaseStatement,
    resolve_expr: ResolveExpr | None = None,
) -> list[cst.BaseSuite]:
    """Return every dead suite inside ``stmt``.

    Supports ``cst.If`` (including ``elif`` / ``else`` chains) and
    ``cst.While``. Returns ``[]`` for any other statement type or when
    no branch can be shown to be unreachable.

    Returns the suite nodes themselves so callers that need source
    positions (e.g. the visitor) can read them off the node.
    """
    if isinstance(stmt, cst.If):
        return _unreachable_in_if(stmt, branch_taken=False, resolve_expr=resolve_expr)
    if isinstance(stmt, cst.While):
        truth = evaluate_truthiness(stmt.test, resolve_expr)
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
    resolve_expr: ResolveExpr | None = None,
) -> list[Sequence[cst.CSTNode]]:
    """Return the ``.body`` of every dead suite inside ``stmt``.

    Thin wrapper over :func:`unreachable_suites` for callers that only
    need the statement list, not the enclosing suite. Each entry is
    typed as ``Sequence[cst.CSTNode]`` because libcst's
    ``BaseSuite.body`` may be ``Sequence[BaseStatement]`` (an indented
    block) or ``Sequence[BaseSmallStatement]`` (a one-line suite like
    ``if False: x = 1``).
    """
    return [suite.body for suite in unreachable_suites(stmt, resolve_expr)]


@runtime_checkable
class UnreachableRegionDetector(Cacheable, Protocol):
    """Finds statically-unreachable source regions in a parsed module.

    Detectors run once per file after the visitor walk; the returned
    list of :class:`CodeRange` determines which references land
    flagged with :data:`dead_cst.graph.EdgeFlags.DEAD_BRANCH` and
    which positions are surfaced as "unreachable code at line X"
    reports.

    The shipped :class:`DefaultUnreachableRegionDetector` covers
    literal-only truthiness on ``if`` / ``while`` tests, fixpoint
    constant-folding over simple ``Name = literal`` assignments, and
    post-terminator regions inside every suite. Custom detectors
    typically subclass it and override
    :meth:`DefaultUnreachableRegionDetector.resolve` to layer on
    domain knowledge -- e.g. ``check_flag("migration-abc")`` is
    always ``True`` in production.

    Inherits the ``(name, version)`` contract from :class:`Cacheable`
    so swapping detectors invalidates stale ``VisitorPayload`` blobs
    automatically.
    """

    def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]: ...


@dataclass(frozen=True)
class DefaultUnreachableRegionDetector:
    """Built-in :class:`UnreachableRegionDetector`.

    Runs three passes per file:

    1. :func:`~dead_cst._const_fold.fold_constants` -- a fixpoint
       constant-folding pass that propagates simple
       ``Name = literal`` (and ``Name: T = literal``) assignments
       through their access points, including chained forms like
       ``a = False; b = a or False; if b:``.
    2. A walk over every ``cst.If`` / ``cst.While``, calling
       :func:`unreachable_suites` with the truthiness resolver below
       so folded constants influence branch reachability.
    3. A walk over every statement-bearing suite (module body and
       every ``IndentedBlock``) marking the trailing region after an
       unconditional terminator as unreachable. Terminators are
       ``return`` / ``raise`` / ``break`` / ``continue`` and
       ``assert <statically-falsy>``. The check is purely
       suite-relative, so a ``raise`` inside a ``try`` body still
       kills the rest of the try body even though the surrounding
       ``except`` runs on its own path.

    Subclasses extend the analysis by overriding :meth:`resolve` to
    return ``True`` / ``False`` for expressions whose truthiness is
    fixed in a particular environment (e.g.
    ``check_flag("migration-abc")`` is always ``True`` in production).
    The override gets first crack at every non-keyword expression in
    every ``if`` / ``while`` test, every ``assert`` test, and every
    foldable assignment RHS; returning ``None`` (the default) defers
    to the built-in literal handling. Constants resolved this way
    flow through the same fixpoint loop as ``Name = literal``
    bindings, so a single high-level decision (``check_flag(...) ==
    True``) propagates through chains and into ``if`` / ``assert``
    branches automatically.

    The folding pass is keyed by ``id`` of the access node, so it
    stays flow-sensitive: a later rebinding shadows an earlier one,
    and conditional bindings whose live values disagree refuse to
    fold (the safe default).
    """

    name: str = "default"
    version: int = 1777800597

    def resolve(self, expr: cst.BaseExpression) -> bool | None:
        """Hook for domain-specific constant folding. Default: defer.

        Override in a subclass to return ``True`` / ``False`` for any
        expression whose truthiness is fixed in your environment.
        Returning ``None`` falls through to ``evaluate_truthiness``'s
        built-in literal handling. The override is consulted recursively
        for every subexpression of an ``if`` / ``while`` / ``assert``
        test and every foldable assignment RHS, so a check like
        ``isinstance(expr, cst.Call) and ...`` runs on every node;
        keep it cheap with an early-return on the wrong type.
        """
        return None

    def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]:
        # Local import to avoid a top-level cycle: ``_const_fold``
        # depends on :func:`evaluate_truthiness` from this module.
        from ._const_fold import fold_constants

        positions = wrapper.resolve(PositionProvider)
        # Run fold_constants with the subclass hook plumbed in so the
        # fixpoint loop can resolve names whose RHS depends on a
        # custom-folded expression (``flag = check_flag("x"); if flag:``).
        truthy = fold_constants(wrapper, resolve_expr=self.resolve)

        def resolve(expr: cst.BaseExpression) -> bool | None:
            # Subclass hook gets first try -- a custom detector may
            # know things the fold table doesn't.
            v = self.resolve(expr)
            if v is not None:
                return v
            # Fall back to the fold table: keyed by Name access id.
            if isinstance(expr, cst.Name):
                return truthy.get(id(expr))
            return None

        found: list[CodeRange] = []

        def is_terminator(stmt: cst.CSTNode) -> bool:
            """``True`` iff ``stmt`` unconditionally exits its enclosing suite.

            Recognized: ``return`` / ``raise`` / ``break`` / ``continue``
            and ``assert <statically-falsy>``. A ``SimpleStatementLine``
            is treated as a terminator if any of its small statements is
            one -- ``x = 1; raise; y = 2`` ends control at ``raise``,
            and anything after it on a later line is dead too.
            """
            if isinstance(stmt, cst.SimpleStatementLine):
                for sm in stmt.body:
                    if isinstance(sm, (cst.Return, cst.Raise, cst.Break, cst.Continue)):
                        return True
                    if isinstance(sm, cst.Assert):
                        if evaluate_truthiness(sm.test, resolve) is False:
                            return True
            return False

        def scan_suite(stmts) -> None:
            """Emit one dead region for the tail after the first terminator."""
            for i, stmt in enumerate(stmts):
                if not is_terminator(stmt):
                    continue
                tail = stmts[i + 1 :]
                if not tail:
                    return
                first_pos = positions.get(tail[0])
                last_pos = positions.get(tail[-1])
                if first_pos is None or last_pos is None:
                    return
                found.append(CodeRange(start=first_pos.start, end=last_pos.end))
                return

        class _Collector(cst.CSTVisitor):
            def visit_Module(self, node: cst.Module) -> None:
                # Module body isn't an IndentedBlock; scan it directly.
                scan_suite(list(node.body))

            def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
                # Every nested suite (function body, class body,
                # if/while/for/try/with bodies, ``else`` / ``finally``
                # clauses, ``except`` handlers) lands here.
                scan_suite(list(node.body))

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
    resolve_expr: ResolveExpr | None = None,
) -> list[cst.BaseSuite]:
    """Walk an ``if`` / ``elif`` / ``else`` chain collecting dead suites.

    ``branch_taken`` is ``True`` when an earlier branch in the chain is
    known to fire; everything from this point on is then unreachable.
    """
    dead: list[cst.BaseSuite] = []
    current: cst.If | None = node
    while current is not None:
        truth = None if branch_taken else evaluate_truthiness(current.test, resolve_expr)
        if branch_taken or truth is False:
            dead.append(current.body)
        branch_taken = branch_taken or truth is True

        orelse = current.orelse
        if orelse is None:
            return dead
        if isinstance(orelse, cst.Else):
            if branch_taken:
                dead.append(orelse.body)
            return dead
        current = orelse
    return dead


# Re-exported so detector authors who write a full
# :meth:`UnreachableRegionDetector.find_regions` (rather than subclassing
# :class:`DefaultUnreachableRegionDetector`) get the fixpoint
# constant-folding pass alongside the rest of the detector helpers.
from ._const_fold import fold_constants  # noqa: E402

__all__ = [
    "DefaultUnreachableRegionDetector",
    "ResolveExpr",
    "UnreachableRegionDetector",
    "evaluate_truthiness",
    "fold_constants",
    "unreachable_bodies",
    "unreachable_suites",
]
