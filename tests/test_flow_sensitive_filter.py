"""Unit tests for the phase-2 flow-sensitive filter prototype.

Each case parses a module, asks ``ScopeProvider`` for the access and
its (multi-)referent list -- the same data the visitor gets today --
and runs ``live_referents`` over the module body. Cases are organised
to mirror ``test_limitations.py`` entries: straight-line shadowing,
branch coexistence, loop bodies, nested scope isolation.

The filter is *not* wired into edge resolution yet; this file only
demonstrates the rule.
"""

from __future__ import annotations

import textwrap

import libcst as cst
import pytest
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider

from dead_cst._flow import live_referents


def _resolve_access(
    src: str, name: str, access_line: int
) -> tuple[cst.Module, cst.CSTNode, list[cst.CSTNode], list[int]]:
    """Find the access to ``name`` on ``access_line`` and return the
    module, the access node, the referent nodes, and the list of
    referent source-line numbers (for readable assertions)."""
    module = cst.parse_module(textwrap.dedent(src).strip())
    wrap = MetadataWrapper(module, unsafe_skip_copy=True)
    scopes = wrap.resolve(ScopeProvider)
    positions = wrap.resolve(PositionProvider)

    seen: set[int] = set()
    for scope in set(scopes.values()):
        for access in scope.accesses:
            if id(access) in seen:
                continue
            seen.add(id(access))
            if getattr(access.node, "value", None) != name:
                continue
            if positions[access.node].start.line != access_line:
                continue
            ref_nodes = [r.node for r in access.referents]
            ref_lines = [positions[n].start.line for n in ref_nodes]
            return wrap.module, access.node, ref_nodes, ref_lines
    raise AssertionError(f"no access {name!r} on line {access_line}")


def _live_lines(
    src: str, name: str, access_line: int, *, scope_body: str | None = None
) -> list[int]:
    """Run the filter and return the sorted source lines of live referents.

    ``scope_body`` selects which enclosing scope's body to pass in:
    ``"module"`` (default) for module-level accesses, or
    ``"function:<fnname>"`` to use that function def's body.
    """
    module, access, referents, _ = _resolve_access(src, name, access_line)

    body: list[cst.BaseStatement]
    if scope_body is None or scope_body == "module":
        body = list(module.body)
    elif scope_body.startswith("function:"):
        fn_name = scope_body.split(":", 1)[1]
        fn = _find_function(module, fn_name)
        body = list(fn.body.body)
    else:  # pragma: no cover - defensive
        raise ValueError(scope_body)

    wrap = MetadataWrapper(module, unsafe_skip_copy=True)
    positions = wrap.resolve(PositionProvider)
    live = live_referents(body, access, referents)
    return sorted(positions[n].start.line for n in live)


def _find_function(module: cst.Module, name: str) -> cst.FunctionDef:
    for stmt in module.body:
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == name:
            return stmt
    raise AssertionError(f"function {name!r} not found")


# ----------------------------------------------------------------------
# Straight-line shadowing: the phase-2 target. Each case here
# corresponds to a false positive documented in test_limitations.py.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, name, access_line, expected",
    [
        pytest.param(
            """
            def f(): pass
            def f(): pass
            f()
            """,
            "f",
            3,
            [2],
            # Was 2 referents; filter drops the shadowed line-1 def.
            id="redeclared-def-keeps-only-last",
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
            6,
            [5],
            id="three-way-chain-keeps-only-last",
        ),
        pytest.param(
            """
            from other import f
            def f(): pass
            f()
            """,
            "f",
            3,
            [2],
            id="import-shadowed-by-def-drops-import",
        ),
        pytest.param(
            """
            def f(): pass
            from other import f
            f()
            """,
            "f",
            3,
            [2],
            id="def-shadowed-by-import-drops-def",
        ),
        pytest.param(
            """
            from other import f
            f = 1
            print(f)
            """,
            "f",
            3,
            [2],
            id="import-rebound-to-constant-drops-import",
        ),
        pytest.param(
            """
            from a import x
            from b import x
            x()
            """,
            "x",
            3,
            [2],
            # Even though both imports resolve to different modules,
            # the second statically kills the first at this access.
            id="same-alias-two-imports-keeps-last",
        ),
    ],
)
def test_straight_line_shadowing(
    src: str, name: str, access_line: int, expected: list[int]
) -> None:
    assert _live_lines(src, name, access_line) == expected


# ----------------------------------------------------------------------
# Branch coexistence: phase 2 MUST NOT filter these down. Each case
# asserts the filter preserves all in-scope referents because neither
# branch dominates the access.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, name, access_line, expected",
    [
        pytest.param(
            """
            if cond:
                x = 1
            else:
                x = 2
            print(x)
            """,
            "x",
            5,
            [2, 4],
            id="if-else-keeps-both-branches",
        ),
        pytest.param(
            """
            if cond:
                x = 1
            elif other:
                x = 2
            else:
                x = 3
            print(x)
            """,
            "x",
            7,
            [2, 4, 6],
            id="elif-chain-keeps-all-branches",
        ),
        pytest.param(
            """
            x = 0
            if cond:
                x = 1
            print(x)
            """,
            "x",
            4,
            [1, 3],
            # No else clause: the pre-if binding survives on the
            # not-taken path.
            id="if-without-else-keeps-fallthrough",
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
            5,
            [2, 4],
            # The optional-import fallback pattern. Both branches
            # must stay alive.
            id="try-except-keeps-both-branches",
        ),
        pytest.param(
            """
            x = 0
            for i in range(3):
                x = i
            print(x)
            """,
            "x",
            4,
            [1, 3],
            # Loop body may execute zero times, so the pre-loop
            # binding survives.
            id="for-loop-keeps-preloop-and-body",
        ),
    ],
)
def test_branches_preserved(
    src: str, name: str, access_line: int, expected: list[int]
) -> None:
    assert _live_lines(src, name, access_line) == expected


# ----------------------------------------------------------------------
# Sequential kills across / within branches.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, name, access_line, expected",
    [
        pytest.param(
            """
            if cond:
                x = 1
            else:
                x = 2
            x = 3
            print(x)
            """,
            "x",
            6,
            [5],
            # A post-if rebinding dominates the access and kills both
            # branch bindings.
            id="post-if-rebinding-kills-branches",
        ),
        pytest.param(
            """
            if cond:
                x = 1
                x = 2
            else:
                x = 3
            print(x)
            """,
            "x",
            6,
            [3, 5],
            # Inside the true branch, the second binding kills the
            # first; the false branch is unaffected.
            id="within-branch-shadowing",
        ),
    ],
)
def test_sequential_and_branch_interaction(
    src: str, name: str, access_line: int, expected: list[int]
) -> None:
    assert _live_lines(src, name, access_line) == expected


# ----------------------------------------------------------------------
# Nested-scope independence. ScopeProvider already filters these down
# to a single referent, so the filter has nothing to do, but these
# pin that the prototype is a no-op in those cases.
# ----------------------------------------------------------------------


def test_nested_function_rebinding_does_not_reach_outer_access() -> None:
    src = """\
    x = 1
    def foo():
        x = 2
    print(x)
    """
    assert _live_lines(src, "x", 4) == [1]


def test_shadowed_inner_def_from_inner_use() -> None:
    # The inner access has one referent already; passing the inner
    # function's body as the scope keeps that invariant.
    src = """\
    def f(): pass
    def outer():
        def f(): pass
        f()
    """
    assert _live_lines(src, "f", 4, scope_body="function:outer") == [3]
