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

from ._core import (
    AddEdge,
    AddNode,
    GraphOp,
    PluginContext,
    collect_module_imports,
    find_call_assignments,
    find_handlers,
)

# Attribute names FastAPI / APIRouter use to register a callable. Matched as
# the rightmost attribute of ``@<instance>.<name>(...)``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset(
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
            instances = find_call_assignments(module, fastapi_imports, _INSTANCE_KINDS)
            if not instances:
                continue
            handlers = find_handlers(module, set(instances), _REGISTRATION_DECORATORS)

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
