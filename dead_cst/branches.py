"""Static evaluator for conditional truthiness.

Determines, when possible, whether the truthiness of an expression is
statically known. Used to identify dead branches of ``if`` / ``while``
statements so callers can mark references that live inside them with
:data:`dead_cst.graph.EdgeFlags.DEAD_BRANCH`.

Three layers, in increasing power and scope:

* :func:`evaluate_truthiness` -- single-expression, literal-only. The
  ``True`` / ``False`` / ``None`` keywords, integer / string literals,
  empty-vs-non-empty collection literals, and ``not`` / ``and`` / ``or``
  over those. Anything involving an attribute access, function call,
  comparison, or other dynamic operation returns ``None`` (unknown).
  An optional ``resolve_expr`` callback gets first crack at every
  non-keyword expression so callers can layer in their own knowledge.
* :class:`TruthinessResolver` -- file-scoped, name-aware. Wraps a
  parsed module; ``resolver.evaluate(expr)`` adds flow-sensitive
  ``Name`` lookup on top of literal handling, memoizes by access node,
  and only walks each ``live_referents`` slice it actually needs.
  Goal-directed: the file's full constant table is never built up
  front, so files with few ``if``/``while``/``assert`` tests pay
  proportionally less. Sister method ``resolver.resolve_constant(expr)``
  returns the underlying ``str`` / ``int`` / ``bool`` / ``None`` value
  (wrapped in :class:`Const`) over the same flow walk -- intended for
  custom :meth:`DefaultUnreachableRegionDetector.resolve` overrides
  that need to pattern-match against a flag *name* (e.g.
  ``check_flag(FEATURE_A)`` where ``FEATURE_A = "feature_a"``).
* :class:`UnreachableRegionDetector` -- module-level dead-region
  finder, the protocol the analyzer plugs in. The shipped
  :class:`DefaultUnreachableRegionDetector` builds one
  :class:`TruthinessResolver` per file and consults it from a single
  CST visit that collects every conditional / suite-bearing site.

Returning ``None`` is always the safe default: callers must treat the
branch as live.

Module-level unreachable-region detection is exposed as the
:class:`UnreachableRegionDetector` protocol so downstream consumers
can fold in domain knowledge -- e.g. config flags whose values are
fixed in production -- without forking the analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, ParentNodeProvider, PositionProvider
from libcst.metadata.scope_provider import (
    Access,
    Assignment,
    Scope,
    ScopeProvider,
)

from ._cacheable import Cacheable
from ._flow import live_referents, scope_body


_KEYWORDS: dict[str, bool] = {
    "True": True,
    "False": False,
    "None": False,
}


# Literal values :meth:`TruthinessResolver.resolve_constant` will
# fold. Keep this conservative -- arbitrary container literals
# (tuples, sets, frozensets) compose with mutation in ways the static
# walk doesn't track, and ``bytes`` / ``float`` rarely show up as flag
# names. Callers that need richer folding can layer it on top via
# their own ``resolve`` override.
ConstValue = str | int | bool | None


@dataclass(frozen=True, slots=True)
class Const:
    """Statically-resolved literal value.

    Wraps the result of :meth:`TruthinessResolver.resolve_constant`
    so ``Const(None)`` (the ``None`` literal was proved) stays
    distinct from ``None`` (resolution failed / value unknown). The
    resolver only folds :data:`ConstValue` -- ``str`` / ``int`` /
    ``bool`` / ``None`` -- so that's the type ``value`` may take.
    """

    value: ConstValue


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


# ---------------------------------------------------------------------------
# Goal-directed truthiness resolver
# ---------------------------------------------------------------------------


# Sentinel for "evaluation in progress" so cyclic ``Name`` references
# (``a = b; b = a``) bottom out at ``None`` instead of recursing
# forever. ``_MISSING`` is the cache-miss default for ``dict.get``,
# distinct from ``None`` (a real "couldn't determine" answer) so the
# cache stays well-defined.
_PENDING: object = object()
_MISSING: object = object()


def _literal_constant(node: cst.BaseExpression) -> Const | None:
    """Direct literal value of ``node`` as a :class:`Const`, or ``None``.

    Folds the ``True`` / ``False`` / ``None`` keywords, integer
    literals, and string literals (including concatenated strings).
    Anything else returns ``None`` (unknown); the caller layers Name
    resolution on top.
    """
    if isinstance(node, cst.Name):
        if node.value == "True":
            return Const(True)
        if node.value == "False":
            return Const(False)
        if node.value == "None":
            return Const(None)
        return None
    if isinstance(node, cst.Integer):
        try:
            return Const(node.evaluated_value)
        except ValueError:
            return None
    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        try:
            value = node.evaluated_value
        except (SyntaxError, UnicodeDecodeError):
            return None
        if not isinstance(value, str):
            # ``b"..."`` concatenations land here -- not in
            # :data:`ConstValue` so we report unknown.
            return None
        return Const(value)
    return None


def _constant_assignment_rhs(
    binding_node: cst.CSTNode,
    parent_map: Mapping[cst.CSTNode, cst.CSTNode],
) -> cst.BaseExpression | None:
    """RHS of a simple ``name = expr`` (or ``name: T = expr``) binding.

    Returns ``None`` for any shape we don't fold: tuple/list unpacking,
    attribute or subscript targets, augmented assign, parameter
    defaults, import bindings, etc. Multi-target chained assignment
    (``a = b = expr``) is supported because all targets share one RHS.
    Walrus (``name := expr``) is also supported -- the surrounding
    expression context doesn't matter for fold purposes since the
    binding's value is unambiguously ``expr``.
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
    if isinstance(parent, cst.NamedExpr):
        if not isinstance(parent.target, cst.Name):
            return None
        if parent.target is not binding_node:
            return None
        return parent.value
    return None


class TruthinessResolver:
    """File-scoped, flow-sensitive truthiness for one parsed module.

    Goal-directed: nothing is computed up front. ``evaluate(expr)``
    walks just the slice it needs, memoizes by node id, and only
    consults :func:`live_referents` for the names that actually feed
    into a query. Files with no conditional tests pay only the cost of
    construction (which itself is lazy in resolving libcst metadata).

    Composes with subclass / detector overrides via the optional
    ``resolve_expr`` callback: it gets first crack at every non-keyword
    expression. Returning a ``bool`` short-circuits; returning ``None``
    falls through to the built-in literal + name-resolution handling.

    From-scratch :class:`UnreachableRegionDetector` implementations can
    instantiate one resolver per file and pass ``resolver.evaluate`` as
    the ``resolve_expr`` argument to :func:`unreachable_suites` /
    :func:`evaluate_truthiness`. The resolver is the supported way to
    get name-aware truthiness without re-implementing the flow walk.
    """

    __slots__ = (
        "_wrapper",
        "_module",
        "_resolve_expr",
        "_scopes",
        "_parent_map",
        "_access_index",
        "_eval_name_cache",
        "_eval_name_const_cache",
        "_live_cache",
        "_rhs_cache",
        "_descendant_cache",
    )

    def __init__(
        self,
        wrapper: MetadataWrapper,
        resolve_expr: ResolveExpr | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._module = wrapper.module
        self._resolve_expr = resolve_expr
        # Resolved lazily on first ``evaluate`` -- a detector that's
        # constructed but never queried (file with no conditionals)
        # never pays for ScopeProvider / ParentNodeProvider.
        self._scopes: Mapping[cst.CSTNode, Scope] | None = None
        self._parent_map: Mapping[cst.CSTNode, cst.CSTNode] | None = None
        # ``id(scope) -> {id(access.node) -> Access}`` built on demand.
        self._access_index: dict[int, dict[int, Access]] = {}
        # ``id(name_access) -> bool | None | _PENDING``. ``_PENDING``
        # is the cycle sentinel; once evaluation finishes the entry is
        # overwritten with the final answer.
        self._eval_name_cache: dict[int, object] = {}
        # Same shape as ``_eval_name_cache`` but for the
        # value-folding path: ``id(name_access) -> Const | None | _PENDING``.
        self._eval_name_const_cache: dict[int, object] = {}
        # ``id(name_access) -> set[live referent CSTNode]``.
        self._live_cache: dict[int, set[cst.CSTNode]] = {}
        # ``id(binding_node) -> rhs_expr | None`` (caches the
        # ``_constant_assignment_rhs`` lookup).
        self._rhs_cache: dict[int, cst.BaseExpression | None] = {}
        # Hoisted ``_descendant_ids`` cache shared across every
        # ``live_referents`` call routed through this resolver.
        self._descendant_cache: dict[int, set[int]] = {}

    def evaluate(self, expr: cst.BaseExpression) -> bool | None:
        """Best-effort statically-known truthiness of ``expr``.

        Same return contract as :func:`evaluate_truthiness`: ``True`` /
        ``False`` when the truthiness is determined, ``None`` when
        unknown. Composes literal handling with flow-sensitive ``Name``
        lookup; intermediate ``Name`` results are memoized.
        """
        return evaluate_truthiness(expr, resolve_expr=self._compose)

    def _compose(self, expr: cst.BaseExpression) -> bool | None:
        """Resolver chain: external ``resolve_expr`` → name lookup → defer.

        Only invoked from ``evaluate_truthiness`` after its keyword
        short-circuit, so any ``Name`` reaching here is non-keyword.
        """
        if self._resolve_expr is not None:
            v = self._resolve_expr(expr)
            if v is not None:
                return v
        if isinstance(expr, cst.Name):
            return self._evaluate_name(expr)
        return None

    def _evaluate_name(self, name: cst.Name) -> bool | None:
        key = id(name)
        cached = self._eval_name_cache.get(key, _MISSING)
        if cached is _PENDING:
            # Cyclic dependency (``a = b; b = a``). Bottom out unknown;
            # the surrounding evaluation will fold it the same way it
            # folds any other ``None``.
            return None
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        self._eval_name_cache[key] = _PENDING
        result = self._evaluate_name_uncached(name)
        self._eval_name_cache[key] = result
        return result

    def _evaluate_name_uncached(self, name: cst.Name) -> bool | None:
        scopes = self._scopes
        if scopes is None:
            scopes = self._wrapper.resolve(ScopeProvider)
            self._scopes = scopes
        scope = scopes.get(name)
        if scope is None:
            return None
        access = self._access_for(name, scope)
        if access is None:
            return None
        referents = [
            r
            for r in access.referents
            if isinstance(r, Assignment) and isinstance(r.node, cst.Name)
        ]
        if not referents:
            return None
        body = scope_body(referents[0].scope, self._module)
        if body is None:
            return None
        live_set = self._live_cache.get(id(name))
        if live_set is None:
            live_set = live_referents(
                body,
                name,
                [r.node for r in referents],
                cache=self._descendant_cache,
            )
            self._live_cache[id(name)] = live_set
        if not live_set:
            return None
        values: set[bool] = set()
        for live_node in live_set:
            rhs = self._rhs_for(live_node)
            if rhs is None:
                return None
            v = self.evaluate(rhs)
            if v is None:
                return None
            values.add(v)
        if len(values) != 1:
            return None
        return next(iter(values))

    def resolve_constant(self, expr: cst.BaseExpression) -> Const | None:
        """Best-effort statically-known *value* of ``expr``.

        Returns a :class:`Const` wrapping the literal value (one of
        :data:`ConstValue` -- ``str`` / ``int`` / ``bool`` / ``None``)
        when statically determinable; ``None`` when the value cannot
        be folded. ``Const(None)`` (the ``None`` literal was proved)
        stays distinct from a bare ``None`` return ("unknown").

        Folds direct literals plus flow-sensitive ``Name`` lookup over
        bindings to those literals -- the same flow walk
        :meth:`evaluate` uses for truthiness, but returning the value
        instead of its boolean projection. Multi-target assignment
        (``A = B = "feature_a"``), :class:`cst.AnnAssign`, and walrus
        share one RHS so all targets fold the same way. Cyclic
        bindings (``a = b; b = a``) bottom out at ``None`` via the
        same ``_PENDING`` sentinel as :meth:`evaluate`.

        Intended for custom :meth:`DefaultUnreachableRegionDetector.resolve`
        overrides that need the actual value of an argument -- e.g.
        pattern-matching ``check_flag(FEATURE_A)`` where
        ``FEATURE_A = "feature_a"``::

            def resolve(self, expr):
                if (
                    isinstance(expr, cst.Call)
                    and is_name(expr.func, "check_flag")
                    and len(expr.args) == 1
                ):
                    flag = self._truthiness.resolve_constant(expr.args[0].value)
                    if flag is not None and flag.value in self._on_flags:
                        return True
                return None

        The caller passes ``self._truthiness = TruthinessResolver(...)``
        through ``resolve_expr=`` so the same instance is reused
        across the file -- caching is per resolver instance.

        Calls and other dynamic operations are *not* folded -- the
        resolver knows nothing about your runtime. Override
        :meth:`DefaultUnreachableRegionDetector.resolve` to layer on
        domain knowledge for those.
        """
        lit = _literal_constant(expr)
        if lit is not None:
            return lit
        if isinstance(expr, cst.Name):
            return self._evaluate_name_const(expr)
        return None

    def _evaluate_name_const(self, name: cst.Name) -> Const | None:
        key = id(name)
        cached = self._eval_name_const_cache.get(key, _MISSING)
        if cached is _PENDING:
            return None
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        self._eval_name_const_cache[key] = _PENDING
        result = self._evaluate_name_const_uncached(name)
        self._eval_name_const_cache[key] = result
        return result

    def _evaluate_name_const_uncached(self, name: cst.Name) -> Const | None:
        scopes = self._scopes
        if scopes is None:
            scopes = self._wrapper.resolve(ScopeProvider)
            self._scopes = scopes
        scope = scopes.get(name)
        if scope is None:
            return None
        access = self._access_for(name, scope)
        if access is None:
            return None
        referents = [
            r
            for r in access.referents
            if isinstance(r, Assignment) and isinstance(r.node, cst.Name)
        ]
        if not referents:
            return None
        body = scope_body(referents[0].scope, self._module)
        if body is None:
            return None
        live_set = self._live_cache.get(id(name))
        if live_set is None:
            live_set = live_referents(
                body,
                name,
                [r.node for r in referents],
                cache=self._descendant_cache,
            )
            self._live_cache[id(name)] = live_set
        if not live_set:
            return None
        values: set[Const] = set()
        for live_node in live_set:
            rhs = self._rhs_for(live_node)
            if rhs is None:
                return None
            v = self.resolve_constant(rhs)
            if v is None:
                return None
            values.add(v)
        if len(values) != 1:
            return None
        return next(iter(values))

    def _access_for(self, name: cst.Name, scope: Scope) -> Access | None:
        """Look up the ``Access`` record for ``name`` in ``scope``.

        ``scope.accesses`` is iterated once per scope on first hit and
        indexed by access-node id; subsequent lookups in the same scope
        are O(1). Names that aren't accesses (LHS bindings, etc.)
        return ``None``.
        """
        scope_key = id(scope)
        index = self._access_index.get(scope_key)
        if index is None:
            index = {id(a.node): a for a in scope.accesses}
            self._access_index[scope_key] = index
        return index.get(id(name))

    def _rhs_for(self, binding_node: cst.CSTNode) -> cst.BaseExpression | None:
        key = id(binding_node)
        cached = self._rhs_cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        parent_map = self._parent_map
        if parent_map is None:
            parent_map = self._wrapper.resolve(ParentNodeProvider)
            self._parent_map = parent_map
        rhs = _constant_assignment_rhs(binding_node, parent_map)
        self._rhs_cache[key] = rhs
        return rhs


@runtime_checkable
class UnreachableRegionDetector(Cacheable, Protocol):
    """Finds statically-unreachable source regions in a parsed module.

    Detectors run once per file after the visitor walk; the returned
    list of :class:`CodeRange` determines which references land
    flagged with :data:`dead_cst.graph.EdgeFlags.DEAD_BRANCH` and
    which positions are surfaced as "unreachable code at line X"
    reports.

    The shipped :class:`DefaultUnreachableRegionDetector` covers
    literal-only truthiness on ``if`` / ``while`` tests, name-aware
    truthiness over simple ``Name = literal`` chains, and
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

    Two passes per file:

    1. A single :class:`cst.CSTVisitor` walk collects every
       ``cst.If`` / ``cst.While`` and every statement-bearing suite
       (module body and every ``IndentedBlock``).
    2. For each collected site, a :class:`TruthinessResolver` answers
       the truthiness queries on demand: ``unreachable_suites`` for
       conditional branches, and a per-suite scan for the trailing
       region after an unconditional terminator (``return`` /
       ``raise`` / ``break`` / ``continue`` / ``assert <falsy>``). The
       check is purely suite-relative, so a ``raise`` inside a ``try``
       body still kills the rest of the try body even though the
       surrounding ``except`` runs on its own path.

    The goal-directed resolver replaces the previous up-front
    fixpoint table: only the names that actually feed an
    ``if``/``while``/``assert`` test pay for ``live_referents`` and
    ``ScopeProvider`` resolution.

    Subclasses extend the analysis by overriding :meth:`resolve` to
    return ``True`` / ``False`` for expressions whose truthiness is
    fixed in a particular environment (e.g.
    ``check_flag("migration-abc")`` is always ``True`` in production).
    The override gets first crack at every non-keyword expression
    routed through the resolver chain; returning ``None`` (the default)
    defers to the built-in literal handling and name lookup. Constants
    resolved this way compose with name resolution: a single high-level
    decision (``check_flag(...)`` is ``True``) propagates through
    chains and into ``if`` / ``assert`` branches automatically.

    The override receives the active :class:`TruthinessResolver` as
    its second argument so it can call
    :meth:`TruthinessResolver.resolve_constant` to fold a flag's
    *value* (``check_flag(FEATURE_A)`` where ``FEATURE_A = "feature_a"``)
    before pattern-matching.
    """

    name: str = "default"
    version: int = 1778281000

    def resolve(
        self,
        expr: cst.BaseExpression,
        resolver: TruthinessResolver,
    ) -> bool | None:
        """Hook for domain-specific constant folding. Default: defer.

        Override in a subclass to return ``True`` / ``False`` for any
        expression whose truthiness is fixed in your environment.
        Returning ``None`` falls through to the built-in literal
        handling and name lookup. The override is consulted recursively
        for every subexpression of an ``if`` / ``while`` / ``assert``
        test and every foldable assignment RHS, so a check like
        ``isinstance(expr, cst.Call) and ...`` runs on every node;
        keep it cheap with an early-return on the wrong type.

        ``resolver`` is the active :class:`TruthinessResolver` for the
        current file, supplied by :meth:`find_regions`. Use it to fold
        a flag-name argument before matching::

            def resolve(self, expr, resolver):
                if (
                    isinstance(expr, cst.Call)
                    and isinstance(expr.func, cst.Name)
                    and expr.func.value == "check_flag"
                    and len(expr.args) == 1
                ):
                    flag = resolver.resolve_constant(expr.args[0].value)
                    if flag is not None and flag.value in self._on_flags:
                        return True
                return None
        """
        return None

    def find_regions(self, wrapper: MetadataWrapper) -> list[CodeRange]:
        positions = wrapper.resolve(PositionProvider)
        # The resolver-with-resolver bridge: ``resolve_expr`` is a
        # one-arg callable by contract, so we wrap ``self.resolve``
        # in a closure that injects the active resolver as the second
        # argument. The list-based late binding breaks the
        # chicken-and-egg between the two.
        resolver_holder: list[TruthinessResolver] = []

        def hook(expr: cst.BaseExpression) -> bool | None:
            return self.resolve(expr, resolver_holder[0])

        resolver = TruthinessResolver(wrapper, resolve_expr=hook)
        resolver_holder.append(resolver)

        sites = _SiteCollector()
        wrapper.module.visit(sites)

        found: list[CodeRange] = []

        for stmt in sites.if_while:
            for suite in unreachable_suites(stmt, resolver.evaluate):
                pos = positions.get(suite)
                if pos is not None:
                    found.append(pos)

        for stmts in sites.suites:
            self._scan_terminators(stmts, resolver, positions, found)

        return found

    @staticmethod
    def _scan_terminators(
        stmts: list[cst.CSTNode],
        resolver: TruthinessResolver,
        positions,
        found: list[CodeRange],
    ) -> None:
        """Emit one dead region for the tail after the first terminator in ``stmts``."""
        for i, stmt in enumerate(stmts):
            if not _is_terminator(stmt, resolver):
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


def _is_terminator(stmt: cst.CSTNode, resolver: TruthinessResolver) -> bool:
    """``True`` iff ``stmt`` unconditionally exits its enclosing suite.

    Recognized: ``return`` / ``raise`` / ``break`` / ``continue`` and
    ``assert <statically-falsy>``. A ``SimpleStatementLine`` is treated
    as a terminator if any of its small statements is one --
    ``x = 1; raise; y = 2`` ends control at ``raise``, and anything
    after it on a later line is dead too.
    """
    if isinstance(stmt, cst.SimpleStatementLine):
        for sm in stmt.body:
            if isinstance(sm, (cst.Return, cst.Raise, cst.Break, cst.Continue)):
                return True
            if isinstance(sm, cst.Assert):
                if resolver.evaluate(sm.test) is False:
                    return True
    return False


class _SiteCollector(cst.CSTVisitor):
    """Pass-1 walk: gather every site the detector may need to query.

    Output is two lists: ``if_while`` for the conditional-suite check
    and ``suites`` for the per-suite terminator scan. The walk is a
    single pass with no metadata dependency, so the cost is just the
    raw CST traversal (the file's expensive ``ScopeProvider`` resolve
    only fires later, when :class:`TruthinessResolver` actually has to
    answer a ``Name`` query).
    """

    def __init__(self) -> None:
        super().__init__()
        self.if_while: list[cst.If | cst.While] = []
        self.suites: list[list[cst.CSTNode]] = []

    def visit_Module(self, node: cst.Module) -> None:
        # Module body isn't an IndentedBlock; collect it directly.
        self.suites.append(list(node.body))

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.suites.append(list(node.body))

    def visit_If(self, node: cst.If) -> None:
        self.if_while.append(node)

    def visit_While(self, node: cst.While) -> None:
        self.if_while.append(node)


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


__all__ = [
    "Const",
    "ConstValue",
    "DefaultUnreachableRegionDetector",
    "ResolveExpr",
    "TruthinessResolver",
    "UnreachableRegionDetector",
    "evaluate_truthiness",
    "unreachable_bodies",
    "unreachable_suites",
]
