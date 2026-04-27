"""Plugin: keep pytest-discovered tests, conftest decls, and ``@pytest.fixture``
functions alive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst

from .._symbols import SymbolNode
from ._core import GraphOp, PluginContext, mark_entrypoints


@dataclass
class PytestPlugin:
    """Mark pytest-discovered symbols as entrypoints.

    pytest auto-discovers tests, fixtures, and plugin hooks by filename and
    decorator conventions; without this plugin those symbols look unused to
    a static analyzer. Specifically:

    * ``conftest.py`` modules: every top-level function, class, and
      variable is marked alive (fixtures, ``pytest_*`` hooks, module-level
      ``pytest_plugins`` / ``collect_ignore``, helper functions used by
      tests).
    * Modules matching ``test_*.py`` / ``*_test.py``: every top-level
      ``test_*`` function and ``Test*`` class is marked alive.
    * Top-level functions decorated with ``@pytest.fixture`` (or the bare
      ``@fixture`` form, with or without a call) anywhere in the project
      -- fixtures are looked up by parameter name and have no static
      caller edge.

    Marker decorators (``@pytest.mark.*``) are not interpreted: they only
    affect collection, never reachability.
    """

    name: str = "pytest"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        decls_by_path: dict[Path, list[SymbolNode]] = {}
        for node in ctx.graph.nodes:
            if node.type in ("function", "class", "variable") and node.path.is_relative_to(
                ctx.base
            ):
                decls_by_path.setdefault(node.path, []).append(node)

        # Fixture-branch prefilter: ``@pytest.fixture`` / ``@fixture``
        # require importing pytest somewhere in the file. Free graph
        # query, no source scan.
        fixture_candidates = ctx.importers("pytest")

        for path, module_node in ctx.base_modules():
            module_decls = decls_by_path.get(path, [])
            filename = path.name

            if filename == "conftest.py":
                yield from mark_entrypoints(
                    f"<pytest:conftest>:{module_node.fqname}", path, module_decls
                )
            elif _is_test_filename(filename):
                test_decls = [d for d in module_decls if _is_test_decl(d)]
                yield from mark_entrypoints(
                    f"<pytest:tests>:{module_node.fqname}", path, test_decls
                )

            if path not in fixture_candidates:
                continue
            module = ctx.parse(path)
            if module is None:
                continue
            fixture_names = _find_fixture_names(module)
            if not fixture_names:
                continue
            fixture_decls = [
                d
                for d in module_decls
                if d.type == "function" and d.fqname.rsplit(".", 1)[-1] in fixture_names
            ]
            yield from mark_entrypoints(
                f"<pytest:fixtures>:{module_node.fqname}", path, fixture_decls
            )


def _is_test_filename(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")


def _is_test_decl(node: SymbolNode) -> bool:
    simple = node.fqname.rsplit(".", 1)[-1]
    if node.type == "function" and simple.startswith("test_"):
        return True
    if node.type == "class" and simple.startswith("Test"):
        return True
    return False


def _find_fixture_names(module: cst.Module) -> set[str]:
    names: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        if any(_is_pytest_fixture_decorator(dec.decorator) for dec in stmt.decorators):
            names.add(stmt.name.value)
    return names


def _is_pytest_fixture_decorator(expr: cst.BaseExpression) -> bool:
    """Match ``@pytest.fixture``, ``@pytest.fixture(...)``, ``@fixture``, ``@fixture(...)``."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if isinstance(expr, cst.Attribute):
        return (
            isinstance(expr.value, cst.Name)
            and expr.value.value == "pytest"
            and expr.attr.value == "fixture"
        )
    if isinstance(expr, cst.Name):
        return expr.value == "fixture"
    return False
