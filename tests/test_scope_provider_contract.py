"""Unit tests for the LibCST metadata contract the visitor relies on.

The redeclaration false positives in ``test_limitations.py`` all have
the same root cause: ``ScopeProvider`` returns every in-scope binding a
name could refer to, flow-insensitively, and the visitor currently
edges to all of them. These tests pin that contract directly, without
going through ``build_symbol_graph``, so it's obvious when phase 2
(flow-sensitive referent filtering) changes either our own behaviour
or LibCST's.
"""

from __future__ import annotations

import textwrap

import libcst as cst
import pytest
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider


def _access_referents(src: str, name: str) -> list[tuple[int, list[tuple[str, int]]]]:
    """Return ``[(access_line, [(referent_type, referent_line), ...]), ...]``
    for every access to ``name`` in ``src``, sorted by access line.

    Referent lists are sorted so tests don't depend on LibCST's
    internal iteration order.
    """
    wrap = MetadataWrapper(cst.parse_module(textwrap.dedent(src).strip()))
    scopes = wrap.resolve(ScopeProvider)
    positions = wrap.resolve(PositionProvider)

    seen: set[int] = set()
    results: list[tuple[int, list[tuple[str, int]]]] = []
    for scope in set(scopes.values()):
        for access in scope.accesses:
            if id(access) in seen:
                continue
            seen.add(id(access))
            if getattr(access.node, "value", None) != name:
                continue
            access_line = positions[access.node].start.line
            refs = sorted(
                (type(r).__name__, positions[r.node].start.line) for r in access.referents
            )
            results.append((access_line, refs))
    return sorted(results)


@pytest.mark.parametrize(
    "src, name, expected",
    [
        pytest.param(
            """
            def f(): pass
            f()
            """,
            "f",
            [(2, [("Assignment", 1)])],
            id="single-binding-yields-one-referent",
        ),
        pytest.param(
            """
            def f(): pass
            def f(): pass
            f()
            """,
            "f",
            [(3, [("Assignment", 1), ("Assignment", 2)])],
            # Mirrors ``dead-function-body-kept-alive`` in test_limitations.
            # ScopeProvider hands the call both defs even though only
            # line 2 is reachable at runtime.
            id="redeclared-def-yields-all-bindings",
        ),
        pytest.param(
            """
            from other import f
            def f(): pass
            f()
            """,
            "f",
            [(3, [("Assignment", 2), ("ImportAssignment", 1)])],
            # Mirrors ``import-shadowed-by-function-keeps-import-alive``.
            id="import-shadowed-by-def-yields-both",
        ),
        pytest.param(
            """
            def f(): pass
            from other import f
            f()
            """,
            "f",
            [(3, [("Assignment", 1), ("ImportAssignment", 2)])],
            # Mirrors ``function-shadowed-by-import-keeps-function-alive``.
            id="def-shadowed-by-import-yields-both",
        ),
        pytest.param(
            """
            from a import x
            from b import x
            x()
            """,
            "x",
            [(3, [("ImportAssignment", 1), ("ImportAssignment", 2)])],
            # Mirrors ``two-imports-same-alias-both-kept-alive``.
            id="same-alias-from-two-modules-yields-both",
        ),
        pytest.param(
            """
            from other import f
            f = 1
            print(f)
            """,
            "f",
            [(3, [("Assignment", 2), ("ImportAssignment", 1)])],
            # Mirrors ``import-rebound-to-constant-keeps-import-alive``.
            id="import-rebound-to-constant-yields-both",
        ),
        pytest.param(
            """
            def a(): pass
            def b(): pass
            def f(): a()
            def f(): b()
            def f(): pass
            f()
            """,
            "f",
            [(6, [("Assignment", 3), ("Assignment", 4), ("Assignment", 5)])],
            # Mirrors ``chain-of-shadowed-functions-keeps-all-bodies-alive``.
            # Demonstrates this scales past two: every prior binding is
            # a valid referent regardless of how far back it was.
            id="chain-of-redeclared-defs-yields-all",
        ),
    ],
)
def test_scope_provider_returns_all_in_scope_bindings(
    src: str, name: str, expected: list[tuple[int, list[tuple[str, int]]]]
) -> None:
    """ScopeProvider is flow-insensitive.

    Every binding of ``name`` reachable in the access's scope appears in
    ``access.referents``, even those that have been shadowed by a later
    rebinding. Phase-2 filtering in the visitor is what's responsible
    for discarding the dead ones.
    """
    assert _access_referents(src, name) == expected


@pytest.mark.parametrize(
    "src, name, expected",
    [
        pytest.param(
            """
            x = 1
            def foo():
                x = 2
            print(x)
            """,
            "x",
            [(4, [("Assignment", 1)])],
            id="rebind-in-nested-function-does-not-leak-to-outer",
        ),
        pytest.param(
            """
            x = 1
            def foo():
                print(x)
            """,
            "x",
            [(3, [("Assignment", 1)])],
            id="outer-binding-visible-from-inner-scope",
        ),
        pytest.param(
            """
            def f(): pass
            def outer():
                def f(): pass
                f()
            """,
            "f",
            [(4, [("Assignment", 3)])],
            id="inner-def-shadows-outer-for-inner-uses",
        ),
        pytest.param(
            """
            x = 1
            [x for x in range(3)]
            print(x)
            """,
            "x",
            [
                (2, [("Assignment", 2)]),
                (3, [("Assignment", 1)]),
            ],
            # Comprehensions get their own scope in Py3. The outer
            # ``print(x)`` correctly resolves to ``x = 1`` even though
            # a comprehension rebound ``x`` between them.
            id="comprehension-variable-does-not-leak",
        ),
        pytest.param(
            """
            if cond:
                x = 1
            else:
                x = 2
            print(x)
            """,
            "x",
            [(5, [("Assignment", 2), ("Assignment", 4)])],
            # ``if`` is not a new scope and neither branch dominates
            # the other -- both bindings are valid at the access.
            # Any phase-2 filter that drops either edge here is wrong.
            id="if-else-branches-both-correct",
        ),
        pytest.param(
            """
            try:
                x = 1
            except Exception:
                x = 2
            print(x)
            """,
            "x",
            [(5, [("Assignment", 2), ("Assignment", 4)])],
            # Same structural story as if/else, and the pattern most
            # likely to appear in real code (optional-import fallbacks).
            id="try-except-branches-both-correct",
        ),
        pytest.param(
            """
            def foo():
                x = 1
                x = 2
                print(x)
            """,
            "x",
            [(4, [("Assignment", 2), ("Assignment", 3)])],
            # Proves the shadowing case occurs inside function scopes
            # too, not just at module level. Phase 2 must drop the
            # edge to line 2.
            id="same-function-scope-shadowing",
        ),
    ],
)
def test_scope_provider_respects_nesting(
    src: str, name: str, expected: list[tuple[int, list[tuple[str, int]]]]
) -> None:
    """ScopeProvider already filters by lexical scope.

    Nesting boundaries (functions, comprehensions) are respected:
    rebindings in a nested scope never appear as referents for an
    access in a containing scope, and shadowed outer bindings do not
    appear for an access in a nested scope. Conversely, ``if`` /
    ``try`` are NOT new scopes, so their branch bindings both surface
    -- and when neither branch dominates the access, both are the
    correct answer. This means phase-2 filtering is a dominator
    question within a single scope, not a cross-scope one.
    """
    assert _access_referents(src, name) == expected
