"""Plugin: keep stdlib ``unittest`` test classes and lifecycle hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst

from .._symbols import NodeFlags
from ._core import (
    SYNTHETIC_POSITION,
    GraphOp,
    ObserveContext,
    PluginContext,
    _asname_value,
    make_payload,
    is_from_module,
    is_name,
    simple_name,
    synthetic_node,
)

if TYPE_CHECKING:
    from .._visitor import VisitorPayload

UNITTEST_PREFIX = "<unittest>:"

# Module-level functions ``unittest`` discovers by name.
_MODULE_HOOKS: frozenset[str] = frozenset({"setUpModule", "tearDownModule", "load_tests"})

# Classes from ``unittest`` whose subclasses are auto-discovered.
_TEST_BASE_CLASSES: frozenset[str] = frozenset({"TestCase", "IsolatedAsyncioTestCase"})


@dataclass
class UnittestPlugin:
    """Mark stdlib ``unittest`` discoveries as entrypoints.

    For every module that imports ``unittest``:

    * Top-level classes whose base list mentions ``unittest.TestCase``
      (or an aliased / ``from``-imported equivalent, including
      ``IsolatedAsyncioTestCase``) are marked alive.
    * Top-level ``setUpModule`` / ``tearDownModule`` / ``load_tests``
      functions are marked alive (the unittest discovery protocol).

    Pure per-file work: the file's ``payload.imports`` provides the
    ``unittest``-import prefilter, the file's CST yields the class
    bases and module-level hook function names, and the contribution
    is appended to the cached payload.

    Limitations:

    * Only direct subclasses of a ``unittest`` base class are detected.
      A test class that inherits from a project-local mixin which in
      turn extends ``TestCase`` won't be picked up by this plugin --
      users can keep it alive with an explicit ``-e`` entrypoint, or
      via the pytest plugin's filename heuristics if the file is named
      ``test_*.py`` / ``*_test.py``.
    * ``from unittest import *`` is invisible to the prefilter (the
      resolver doesn't emit a graph node for stdlib star imports), so
      such files are skipped. Use ``from unittest import TestCase``.
    """

    name: str = "unittest"
    version: str = "1"

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        if not _file_imports_unittest(ctx.payload):
            return None

        module_node = next((n for n in ctx.payload.nodes if n.type == "module"), None)
        if module_node is None:
            return None

        module_aliases, base_aliases = _collect_unittest_imports(ctx.module)
        if not module_aliases and not base_aliases:
            return None

        wanted = _find_testcase_subclasses(
            ctx.module, module_aliases, base_aliases
        ) | _find_module_hooks(ctx.module)
        if not wanted:
            return None

        targets = [
            n
            for n in ctx.payload.nodes
            if n.type in ("function", "class") and simple_name(n.fqname) in wanted
        ]
        if not targets:
            return None

        synth = synthetic_node(
            f"{UNITTEST_PREFIX}{module_node.fqname}",
            ctx.path,
            flags=NodeFlags.ENTRYPOINT,
        )
        edges = [(synth, t, SYNTHETIC_POSITION) for t in targets]
        return make_payload(nodes=[synth], edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()


def _file_imports_unittest(payload) -> bool:
    """True iff any non-star import in the file targets ``unittest``.

    Star imports are deliberately excluded: ``from unittest import *``
    binds the test base classes anonymously, but we can't reliably tie
    them to a graph node without re-parsing every file. The
    pre-refactor plugin had the same limitation by virtue of relying
    on ``ctx.importers`` (which doesn't surface stdlib star imports).
    The escape hatch is the same: switch to ``from unittest import
    TestCase``.
    """
    for _src, imp, _pos in payload.imports:
        if imp.star:
            continue
        if imp.module == "unittest" or imp.module.startswith("unittest."):
            return True
    return False


def _collect_unittest_imports(module: cst.Module) -> tuple[set[str], set[str]]:
    """Return ``(module_aliases, base_class_aliases)``.

    ``module_aliases`` are local names bound to the ``unittest`` module
    (so ``unittest.TestCase`` / ``ut.TestCase`` can be matched).
    ``base_class_aliases`` are local names bound directly to ``TestCase``
    / ``IsolatedAsyncioTestCase`` (so a bare ``TestCase`` reference can be
    matched).
    """
    module_aliases: set[str] = set()
    base_aliases: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.Import):
                for alias in small.names:
                    if not is_name(alias.name, "unittest"):
                        continue
                    module_aliases.add(_asname_value(alias) or "unittest")
            elif isinstance(small, cst.ImportFrom):
                if not is_from_module(small, "unittest"):
                    continue
                if isinstance(small.names, cst.ImportStar):
                    base_aliases.update(_TEST_BASE_CLASSES)
                    continue
                for alias in small.names:
                    target = alias.name.value if isinstance(alias.name, cst.Name) else None
                    if target not in _TEST_BASE_CLASSES:
                        continue
                    base_aliases.add(_asname_value(alias) or target)
    return module_aliases, base_aliases


def _find_testcase_subclasses(
    module: cst.Module, module_aliases: set[str], base_aliases: set[str]
) -> set[str]:
    found: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.ClassDef):
            continue
        for base in stmt.bases:
            if _is_testcase_base(base.value, module_aliases, base_aliases):
                found.add(stmt.name.value)
                break
    return found


def _is_testcase_base(
    expr: cst.BaseExpression, module_aliases: set[str], base_aliases: set[str]
) -> bool:
    if isinstance(expr, cst.Name):
        return expr.value in base_aliases
    if isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
        return expr.value.value in module_aliases and expr.attr.value in _TEST_BASE_CLASSES
    return False


def _find_module_hooks(module: cst.Module) -> set[str]:
    return {
        stmt.name.value
        for stmt in module.body
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value in _MODULE_HOOKS
    }
