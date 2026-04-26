"""Plugin: keep FastAPI route handlers and lifecycle hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
from libcst.metadata import FullRepoManager

from .._symbols import SymbolNode
from ._core import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node

# Attribute names used by FastAPI / APIRouter to register a callable. Matched
# against the rightmost attribute of the decorator expression, so any binding
# (``app``, ``router``, ``v1_router``, ``self.app`` ...) is accepted.
_ROUTE_DECORATORS: frozenset[str] = frozenset(
    {
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "options",
        "head",
        "trace",
        "api_route",
        "websocket",
        "websocket_route",
        "middleware",
        "exception_handler",
        "on_event",
    }
)


@dataclass
class FastAPIPlugin:
    """Mark FastAPI route handlers and lifecycle hooks as entrypoints.

    FastAPI registers callables via decorators -- ``@app.get("/")``,
    ``@router.post(...)``, ``@app.websocket(...)``, ``@app.middleware(...)``,
    ``@app.exception_handler(...)``, ``@app.on_event(...)`` -- and the
    framework, not user code, invokes them at runtime. To a static analyzer
    they look unused.

    For every top-level function whose decorator's rightmost attribute matches
    a known FastAPI registration name (``get``/``post``/.../``api_route``/
    ``websocket``/``middleware``/``exception_handler``/``on_event``), this
    plugin emits a synthetic entrypoint with an edge to the function. The
    analyzer's regular reference edges then keep dependencies (Pydantic
    models, ``Depends(...)`` callables, helpers) reachable from there.

    Because the decorator's owner type is not known statically, matching is
    by attribute name only. ``@app.get(...)`` and ``@router.get(...)`` are
    both accepted; so is ``@self.app.get(...)``. Bare names without an
    attribute (``@get(...)``) are not matched -- FastAPI does not expose
    these as module-level decorators, and matching them would collide with
    unrelated code (e.g. ``@get`` from another library).

    Application construction (``FastAPI(lifespan=...)``,
    ``app.include_router(router)``, ``app.add_api_route(..., endpoint=fn)``,
    ``Depends(fn)``) is plain reference passing and is already tracked by
    the regular analyzer; this plugin only fills the decorator-only gap.

    This is a :class:`CSTAwareEdgePlugin` because decorator detection needs
    the original CST.
    """

    name: str = "fastapi"
    cst_aware: bool = True

    def contribute(
        self, ctx: PluginContext, managers: dict[Path, FullRepoManager]
    ) -> Iterable[GraphOp]:
        modules_by_path: dict[Path, SymbolNode] = {}
        decls_by_path: dict[Path, dict[str, list[SymbolNode]]] = {}
        for node in ctx.graph.nodes:
            if node.type == "module":
                modules_by_path[node.path] = node
            elif node.type == "function":
                simple = node.fqname.rsplit(".", 1)[-1]
                decls_by_path.setdefault(node.path, {}).setdefault(simple, []).append(node)

        for path, module_node in modules_by_path.items():
            wrapper = _wrapper_for(path, managers)
            if wrapper is None:
                continue
            handler_names = _find_handler_names(wrapper.module)
            if not handler_names:
                continue
            by_name = decls_by_path.get(path, {})
            handler_decls: list[SymbolNode] = []
            for name in handler_names:
                handler_decls.extend(by_name.get(name, []))
            if not handler_decls:
                continue
            synth = synthetic_node(
                fqname=f"<fastapi:routes>:{module_node.fqname}",
                path=path,
            )
            yield AddNode(synth, entrypoint=True)
            for decl in handler_decls:
                yield AddEdge(synth, decl)


def _wrapper_for(path: Path, managers: dict[Path, FullRepoManager]):
    for base, mgr in managers.items():
        if not path.is_relative_to(base):
            continue
        try:
            return mgr.get_metadata_wrapper_for_path(path)
        except Exception:
            return None
    return None


def _find_handler_names(module: cst.Module) -> set[str]:
    names: set[str] = set()
    for stmt in module.body:
        func = _as_function_def(stmt)
        if func is None:
            continue
        if any(_is_fastapi_decorator(dec.decorator) for dec in func.decorators):
            names.add(func.name.value)
    return names


def _as_function_def(stmt: cst.BaseStatement) -> cst.FunctionDef | None:
    """Unwrap a top-level function definition, including ``async def``."""
    if isinstance(stmt, cst.FunctionDef):
        return stmt
    return None


def _is_fastapi_decorator(expr: cst.BaseExpression) -> bool:
    """Match ``<x>.<route_name>(...)`` -- with or without trailing call.

    FastAPI decorators are conventionally called (``@app.get("/")``), but we
    also accept the uncalled form for robustness; matching the attribute name
    is what distinguishes a registration from an unrelated decorator.
    """
    if isinstance(expr, cst.Call):
        expr = expr.func
    if not isinstance(expr, cst.Attribute):
        return False
    return expr.attr.value in _ROUTE_DECORATORS
