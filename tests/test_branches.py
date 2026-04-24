"""Unit tests for the conditional truthiness evaluator.

The evaluator is intentionally narrow: it only handles literal forms
that can be decided without any name lookup or runtime state. Tests
are organised into three groups:

* ``evaluate_truthiness`` on individual expressions, covering the
  whitelist (``True`` / ``False`` / ``None``, ints, strings, empty vs
  non-empty collections, ``not`` / ``and`` / ``or``).
* ``evaluate_truthiness`` on expressions that must return ``None``
  (any name lookup, attribute, call, comparison, f-string, etc.).
* ``unreachable_bodies`` on ``if`` / ``while`` statements, including
  ``elif`` chains and the asymmetric handling of ``while True:``.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import libcst as cst
import pytest

from dead_cst._branches import evaluate_truthiness, unreachable_bodies


def _expr(src: str) -> cst.BaseExpression:
    """Parse ``src`` as a single expression statement and return its value."""
    module = cst.parse_module(src)
    stmt = module.body[0]
    assert isinstance(stmt, cst.SimpleStatementLine)
    expr = stmt.body[0]
    assert isinstance(expr, cst.Expr)
    return expr.value


def _stmt(src: str) -> cst.BaseStatement:
    """Parse ``src`` and return its first top-level statement."""
    return cst.parse_module(textwrap.dedent(src).strip()).body[0]


def _bodies_as_code(bodies: list[Sequence[cst.CSTNode]]) -> list[str]:
    """Stringify each dead body as concatenated source for assertions."""
    module = cst.Module([])
    return [
        "".join(module.code_for_node(s) for s in body).strip() for body in bodies
    ]


# ----------------------------------------------------------------------
# evaluate_truthiness: positive cases (statically determinable).
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, expected",
    [
        ("True", True),
        ("False", False),
        ("None", False),
        ("0", False),
        ("1", True),
        ("42", True),
        ("0x0", False),
        ("0xff", True),
        ("0b0", False),
        ("0b1", True),
        ('""', False),
        ('"x"', True),
        ('b""', False),
        ('b"x"', True),
        ('"a" "b"', True),
        ('"" ""', False),
        ("()", False),
        ("(1,)", True),
        ("[]", False),
        ("[1]", True),
        ("{}", False),
        ("{1: 2}", True),
        ("{1, 2}", True),
        ("not True", False),
        ("not False", True),
        ("not 0", True),
        ("not not 1", True),
        ("True and True", True),
        ("True and False", False),
        ("False and unknown", False),
        ("True or False", True),
        ("False or False", False),
        ("True or unknown", True),
        ("not (True and False)", True),
    ],
)
def test_evaluate_truthiness_known(src: str, expected: bool) -> None:
    assert evaluate_truthiness(_expr(src)) is expected


# ----------------------------------------------------------------------
# evaluate_truthiness: anything outside the whitelist must return None.
# Returning None means callers treat the branch as live -- the safe
# default. Each entry here would be unsound to fold.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "x",
        "TYPE_CHECKING",
        "obj.attr",
        "f()",
        "1 == 1",
        "1 < 2",
        "1 is 1",
        "1.0",
        "0.0",
        'f"x"',
        "[*xs]",
        "{**d}",
        "(1, *xs)",
        "x and y",
        "x or y",
        "not x",
        "True and x",
        "False or x",
    ],
)
def test_evaluate_truthiness_unknown(src: str) -> None:
    assert evaluate_truthiness(_expr(src)) is None


# ----------------------------------------------------------------------
# unreachable_bodies: simple ``if`` / ``while``.
# ----------------------------------------------------------------------


def test_if_false_marks_body_dead() -> None:
    stmt = _stmt(
        """
        if False:
            x = 1
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["x = 1"]


def test_if_true_marks_else_dead() -> None:
    stmt = _stmt(
        """
        if True:
            x = 1
        else:
            x = 2
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["x = 2"]


def test_if_true_without_else_has_no_dead_body() -> None:
    stmt = _stmt(
        """
        if True:
            x = 1
        """
    )
    assert unreachable_bodies(stmt) == []


def test_if_unknown_returns_no_dead_bodies() -> None:
    stmt = _stmt(
        """
        if cond:
            x = 1
        else:
            x = 2
        """
    )
    assert unreachable_bodies(stmt) == []


def test_if_false_with_else_marks_only_body() -> None:
    stmt = _stmt(
        """
        if False:
            x = 1
        else:
            x = 2
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["x = 1"]


# ----------------------------------------------------------------------
# unreachable_bodies: elif chains.
# ----------------------------------------------------------------------


def test_elif_after_true_branch_all_dead() -> None:
    stmt = _stmt(
        """
        if True:
            a = 1
        elif other:
            b = 2
        else:
            c = 3
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["b = 2", "c = 3"]


def test_elif_with_false_first_branch_only_first_dead() -> None:
    stmt = _stmt(
        """
        if False:
            a = 1
        elif cond:
            b = 2
        else:
            c = 3
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["a = 1"]


def test_elif_false_in_middle_marks_only_that_branch() -> None:
    stmt = _stmt(
        """
        if cond:
            a = 1
        elif False:
            b = 2
        else:
            c = 3
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["b = 2"]


def test_elif_true_kills_following_branches_even_when_outer_unknown() -> None:
    # When the head ``if`` is dynamic but a later elif is statically
    # True, the True branch is reachable (when no earlier test fires)
    # and any branches after it can never run.
    stmt = _stmt(
        """
        if cond:
            a = 1
        elif True:
            b = 2
        elif other:
            c = 3
        else:
            d = 4
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["c = 3", "d = 4"]


def test_all_branches_false_keeps_else_live() -> None:
    stmt = _stmt(
        """
        if False:
            a = 1
        elif False:
            b = 2
        else:
            c = 3
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["a = 1", "b = 2"]


# ----------------------------------------------------------------------
# unreachable_bodies: ``while``.
# ----------------------------------------------------------------------


def test_while_false_marks_body_dead() -> None:
    stmt = _stmt(
        """
        while False:
            x = 1
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["x = 1"]


def test_while_true_with_else_marks_else_dead() -> None:
    # ``else`` on a loop runs only on normal exit; ``while True:`` never
    # exits normally, so the else clause is unreachable.
    stmt = _stmt(
        """
        while True:
            x = 1
        else:
            y = 2
        """
    )
    assert _bodies_as_code(unreachable_bodies(stmt)) == ["y = 2"]


def test_while_true_without_else_has_no_dead_body() -> None:
    stmt = _stmt(
        """
        while True:
            x = 1
        """
    )
    assert unreachable_bodies(stmt) == []


def test_while_unknown_returns_no_dead_bodies() -> None:
    stmt = _stmt(
        """
        while cond:
            x = 1
        """
    )
    assert unreachable_bodies(stmt) == []


# ----------------------------------------------------------------------
# unreachable_bodies: non-conditional statements are passed through.
# ----------------------------------------------------------------------


def test_unsupported_statement_returns_empty() -> None:
    stmt = _stmt("x = 1")
    assert unreachable_bodies(stmt) == []
