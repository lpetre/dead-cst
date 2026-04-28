"""Plugin: keep stdlib ``unittest`` test classes and lifecycle hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst

from .._symbols import SymbolNode
from ._core import (
    GraphOp,
    PluginContext,
    _asname_value,
    is_from_module,
    is_name,
    mark_entrypoints,
    simple_name,
)

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

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # ``unittest`` is stdlib, so the resolver doesn't emit a synthetic
        # node for it (``ctx.importers`` can't prefilter). Walk the graph's
        # import declarations directly; same idea, just keyed off the
        # ``Import.module`` field instead of an ``[external dist]`` marker.
        candidate_paths: set[Path] = set()
        decls_by_path: dict[Path, list[SymbolNode]] = {}
        for node in ctx.base_nodes():
            if node.type in ("function", "class"):
                decls_by_path.setdefault(node.path, []).append(node)
            elif (
                node.type == "import"
                and node.imports is not None
                and (
                    node.imports.module == "unittest" or node.imports.module.startswith("unittest.")
                )
            ):
                candidate_paths.add(node.path)
        if not candidate_paths:
            return

        for path, module_node in ctx.base_modules():
            if path not in candidate_paths:
                continue
            module = ctx.parse(path)
            if module is None:
                continue

            module_aliases, base_aliases = _collect_unittest_imports(module)
            if not module_aliases and not base_aliases:
                # ``unittest`` is reachable through an import chain but the
                # module itself doesn't bind any of the names we care about.
                continue
            test_class_names = _find_testcase_subclasses(module, module_aliases, base_aliases)
            hook_names = _find_module_hooks(module)

            wanted = test_class_names | hook_names
            if not wanted:
                continue

            module_decls = decls_by_path.get(path, [])
            targets = [d for d in module_decls if simple_name(d.fqname) in wanted]
            if not targets:
                continue

            yield from mark_entrypoints(f"{UNITTEST_PREFIX}{module_node.fqname}", path, targets)


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
                    # ``from unittest import *`` brings every public name in,
                    # including the test base classes.
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
