"""Plugin: keep pytest-discovered tests, conftest decls, and ``@pytest.fixture``
functions alive."""

from __future__ import annotations

from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst

from ..graph import NodeFlags, SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    GraphOp,
    ObserveContext,
    PluginContext,
    make_payload,
    module_node,
    payload_imports_module,
    simple_name,
    synthetic_node,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload

PYTEST_CONFTEST_PREFIX = "<pytest:conftest>:"
PYTEST_TESTS_PREFIX = "<pytest:tests>:"
PYTEST_FIXTURES_PREFIX = "<pytest:fixtures>:"


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

    Pure per-file work: filename + payload decls for the test/conftest
    branches, CST + payload imports for the fixture branch.
    """

    name: str = "pytest"
    version: int = 1777760307

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        module = module_node(ctx.payload)
        if module is None:
            return None

        module_decls = [n for n in ctx.payload.nodes if n.type in ("function", "class", "variable")]
        filename = ctx.path.name
        new_nodes: list[SymbolNode] = []
        new_edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        if filename == "conftest.py":
            _emit_seed(
                new_nodes,
                new_edges,
                f"{PYTEST_CONFTEST_PREFIX}{module.fqname}",
                ctx.path,
                module_decls,
            )
        elif _is_test_filename(filename):
            test_decls = [d for d in module_decls if _is_test_decl(d)]
            _emit_seed(
                new_nodes,
                new_edges,
                f"{PYTEST_TESTS_PREFIX}{module.fqname}",
                ctx.path,
                test_decls,
            )

        if payload_imports_module(ctx.payload, "pytest"):
            fixture_names = _find_fixture_names(ctx.module)
            if fixture_names:
                fixture_decls = [
                    d
                    for d in module_decls
                    if d.type == "function" and simple_name(d.fqname) in fixture_names
                ]
                _emit_seed(
                    new_nodes,
                    new_edges,
                    f"{PYTEST_FIXTURES_PREFIX}{module.fqname}",
                    ctx.path,
                    fixture_decls,
                )

        if not new_nodes:
            return None
        return make_payload(nodes=new_nodes, edges=new_edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()


def _emit_seed(nodes, edges, fqname, path, targets):
    if not targets:
        return
    synth = synthetic_node(fqname, path, flags=NodeFlags.ENTRYPOINT)
    nodes.append(synth)
    edges.extend((synth, t, SYNTHETIC_POSITION) for t in targets)


def _is_test_filename(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")


def _is_test_decl(node: SymbolNode) -> bool:
    simple = simple_name(node.fqname)
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
