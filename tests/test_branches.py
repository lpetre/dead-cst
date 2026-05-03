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

from libcst.metadata import MetadataWrapper

from dead_cst._branches import (
    DefaultUnreachableRegionDetector,
    evaluate_truthiness,
    unreachable_bodies,
)


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
    return ["".join(module.code_for_node(s) for s in body).strip() for body in bodies]


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
# evaluate_truthiness: ``resolve_name`` callback. Callers with scope
# info (notably the constant-folding pass) pass a resolver so names
# bound to a known constant fold the same as a literal would.
# ----------------------------------------------------------------------


def _name_resolver(values: dict[str, bool | None]):
    return lambda n: values.get(n.value)


def test_resolver_folds_name_to_constant() -> None:
    assert evaluate_truthiness(_expr("foo"), _name_resolver({"foo": False})) is False
    assert evaluate_truthiness(_expr("foo"), _name_resolver({"foo": True})) is True


def test_resolver_keywords_take_precedence_over_resolver() -> None:
    # ``True`` / ``False`` / ``None`` are language keywords; the
    # resolver should never see them even if the caller supplies an
    # entry under those names.
    called: list[str] = []

    def resolver(n):
        called.append(n.value)
        return True

    assert evaluate_truthiness(_expr("True"), resolver) is True
    assert evaluate_truthiness(_expr("False"), resolver) is False
    assert evaluate_truthiness(_expr("None"), resolver) is False
    assert called == []


def test_resolver_threads_through_not_and_or() -> None:
    res = _name_resolver({"foo": False, "bar": True})
    assert evaluate_truthiness(_expr("not foo"), res) is True
    assert evaluate_truthiness(_expr("foo and bar"), res) is False
    assert evaluate_truthiness(_expr("foo or bar"), res) is True
    assert evaluate_truthiness(_expr("foo or unknown"), res) is None


def test_resolver_returning_none_stays_unknown() -> None:
    # Resolver returning None means "I don't know" -- same as no
    # resolver at all.
    assert evaluate_truthiness(_expr("foo"), lambda n: None) is None


def test_unreachable_bodies_with_resolver() -> None:
    # The resolver flips a name from "unknown" to "False", so the if
    # body becomes statically dead.
    stmt = _stmt(
        """
        if foo:
            x = 1
        """
    )
    assert unreachable_bodies(stmt) == []
    assert _bodies_as_code(unreachable_bodies(stmt, _name_resolver({"foo": False}))) == ["x = 1"]


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


# ----------------------------------------------------------------------
# DefaultUnreachableRegionDetector: module-level CodeRange detector.
# ----------------------------------------------------------------------


def _wrapper(src: str) -> MetadataWrapper:
    return MetadataWrapper(cst.parse_module(textwrap.dedent(src).strip() + "\n"))


def test_default_detector_returns_empty_for_no_dead_code() -> None:
    regions = DefaultUnreachableRegionDetector().find_regions(
        _wrapper(
            """
            def f(): pass
            """
        )
    )
    assert regions == []


def test_default_detector_finds_if_false_body() -> None:
    regions = DefaultUnreachableRegionDetector().find_regions(
        _wrapper(
            """
            if False:
                x = 1
            """
        )
    )
    assert len(regions) == 1
    assert regions[0].start.line == 2


def test_default_detector_finds_multiple_dead_suites() -> None:
    regions = DefaultUnreachableRegionDetector().find_regions(
        _wrapper(
            """
            if False:
                a = 1

            if True:
                b = 2
            else:
                c = 3
            """
        )
    )
    # First the ``if False`` body, then the ``else`` of ``if True``.
    starts = sorted(r.start.line for r in regions)
    assert starts == [2, 7]


def test_default_detector_handles_while() -> None:
    regions = DefaultUnreachableRegionDetector().find_regions(
        _wrapper(
            """
            while False:
                x = 1
            """
        )
    )
    assert len(regions) == 1


def test_default_detector_carries_fingerprint_metadata() -> None:
    """``name`` / ``version`` feed the cache fingerprint via ``Cacheable``."""
    detector = DefaultUnreachableRegionDetector()
    assert detector.name == "default"
    assert isinstance(detector.version, int)
