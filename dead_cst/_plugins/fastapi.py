"""Plugin: keep FastAPI route handlers and lifecycle hooks alive.

Strategy: instead of marking each handler as its own entrypoint, find the
top-level ``FastAPI()`` / ``APIRouter()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.<method>(...)``
decorator. ``FastAPI()`` instances are then seeded as entrypoints; routers
stay pass-through so an unused ``APIRouter`` still surfaces as dead code.

This routes ``why-alive`` chains through the app variable users actually
recognize ("alive because it's a route on ``app``") and lets the existing
``app.include_router(router)`` reference flow keep sub-routers reachable
without any special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst

from ._core import (
    AddEdge,
    AddNode,
    GraphOp,
    PluginContext,
    collect_module_imports,
    find_handlers,
    single_target_assignment,
)

# Attribute names FastAPI / APIRouter use to register a callable. Matched as
# the rightmost attribute of ``@<instance>.<name>(...)``.
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

# Classes whose instances we treat specially. Value records whether the
# instance should be seeded as an entrypoint.
_INSTANCE_KINDS: dict[str, bool] = {
    "FastAPI": True,  # uvicorn loads ``module:app`` -- always an entrypoint
    "APIRouter": False,  # only alive if reached via ``include_router``
}


@dataclass
class FastAPIPlugin:
    """Mark FastAPI apps as entrypoints and wire route handlers through them.

    For each module the plugin:

    1. Inspects ``from fastapi import ...`` / ``import fastapi`` to learn
       which local names refer to ``FastAPI`` and ``APIRouter``.
    2. Finds top-level assignments ``X = FastAPI(...)`` / ``X = APIRouter(...)``
       (including ``AnnAssign`` and aliased / module-prefixed forms) and
       records ``X`` as an instance of that kind.
    3. For every top-level function decorated ``@X.<route_name>(...)``,
       emits an edge ``X -> handler`` so the handler is reachable whenever
       ``X`` is.
    4. Seeds every ``FastAPI`` instance as an entrypoint. Routers are not
       seeded -- a router that is never ``include_router``\\ed has no path
       from any entrypoint and stays dead, which is the correct signal.

    Application wiring (``app.include_router(router)``,
    ``FastAPI(lifespan=fn)``, ``app.add_api_route(..., endpoint=fn)``,
    ``Depends(fn)``) is plain reference passing already tracked by the
    regular analyzer.

    Limitations: only top-level ``X = FastAPI(...)`` / ``X = APIRouter(...)``
    assignments with a single ``Name`` target are detected. Factory-style
    apps (``def create_app(): return FastAPI()``) and class-attribute apps
    (``self.app = FastAPI()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints.
    """

    name: str = "fastapi"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Prefilter via the import graph: only files that actually import
        # ``fastapi`` can declare an app or router. The analyzer's resolver
        # already added ``[external dist] fastapi`` predecessors for them.
        candidate_paths = ctx.importers("fastapi")
        if not candidate_paths:
            return

        for path, module_node in ctx.base_modules():
            if path not in candidate_paths:
                continue
            module = ctx.parse(path)
            if module is None:
                continue
            fastapi_imports = collect_module_imports(module, "fastapi", _INSTANCE_KINDS)
            if not fastapi_imports:
                continue
            instances = _find_instances(module, fastapi_imports)
            if not instances:
                continue
            handlers = find_handlers(module, set(instances), _ROUTE_DECORATORS)

            module_fqname = module_node.fqname
            for var_name, kind in instances.items():
                instance_decls = ctx.find_declarations(f"{module_fqname}.{var_name}")
                if not instance_decls:
                    continue
                for instance_decl in instance_decls:
                    if _INSTANCE_KINDS[kind]:
                        yield AddNode(instance_decl, entrypoint=True)
                    for handler_name in handlers.get(var_name, ()):
                        for handler_decl in ctx.find_declarations(
                            f"{module_fqname}.{handler_name}"
                        ):
                            yield AddEdge(instance_decl, handler_decl)


def _find_instances(module: cst.Module, fastapi_imports: dict[str, str]) -> dict[str, str]:
    """Return ``{var_name: 'FastAPI' | 'APIRouter'}`` for top-level instances."""
    instances: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target_name, value = single_target_assignment(small)
            if target_name is None or not isinstance(value, cst.Call):
                continue
            kind = _classify_call(value.func, fastapi_imports)
            if kind is not None:
                instances[target_name] = kind
    return instances


def _classify_call(func: cst.BaseExpression, fastapi_imports: dict[str, str]) -> str | None:
    """Return ``"FastAPI"`` / ``"APIRouter"`` if ``func`` calls one, else ``None``."""
    if isinstance(func, cst.Name):
        target = fastapi_imports.get(func.value)
        if target in _INSTANCE_KINDS:
            return target
    elif isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
        # ``fastapi.FastAPI(...)`` / ``fa.APIRouter(...)``
        if fastapi_imports.get(func.value.value) == "<module>":
            attr = func.attr.value
            if attr in _INSTANCE_KINDS:
                return attr
    return None
