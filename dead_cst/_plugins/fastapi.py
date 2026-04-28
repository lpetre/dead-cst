"""Plugin: keep FastAPI route handlers and lifecycle hooks alive.

Strategy: every FastAPI / APIRouter instance we want to wire up is a
top-level variable that the analyzer has already linked back to the
``fastapi`` import -- whether the assignment is the literal
``X = FastAPI()``, the aliased ``X = F()`` after ``from fastapi import
FastAPI as F``, or the factory form ``X = create_app()`` whose body
returns ``FastAPI(...)``. The plugin reuses those reference edges:

1. Direct shape (``X = FastAPI(...)`` / ``X = APIRouter(...)``,
   ``X = fastapi.FastAPI(...)``, etc.) is recognized syntactically via
   ``find_call_assignments`` -- this is unambiguous, so it gives the
   kind directly even for ``import fastapi`` forms where the graph
   alone can't distinguish the two classes.
2. Indirect shape (any variable decorated by ``@X.<route_verb>(...)``)
   is detected via the route-decorator scan; kind is read off the
   import nodes encountered while walking forward from ``X`` in the
   graph, which transparently handles factory wrappers because the
   factory's body is already linked to the right import.

For each detected instance the plugin emits ``X -> handler`` edges for
its decorated handlers and seeds ``FastAPI`` instances as entrypoints
(uvicorn loads ``module:app``); routers are not seeded, so an
``APIRouter`` that nothing ``include_router``s stays dead -- the right
signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst
import networkx as nx

from .._symbols import SymbolNode
from ._core import (
    AddEdge,
    AddNode,
    GraphOp,
    PluginContext,
    collect_module_imports,
    decorator_owner,
    find_call_assignments,
    require_resolved_dep,
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

    1. Finds direct ``X = FastAPI(...)`` / ``X = APIRouter(...)``
       assignments by inspecting the file's ``fastapi`` imports and
       matching call sites. This is the unambiguous path and handles
       module-prefixed forms like ``import fastapi; X = fastapi.FastAPI()``
       that pure graph reachability cannot distinguish (the variable's
       edge goes straight to the ``fastapi`` synthetic with no
       intermediate ``FastAPI`` vs ``APIRouter`` import to discriminate).
    2. Collects every ``@<X>.<verb>(...)`` decorator (``verb`` from
       FastAPI's registration set). For each owner ``X`` not already
       resolved by step 1, walks forward from ``X`` in the symbol graph;
       if the walk hits an ``import`` node bound to ``FastAPI`` or
       ``APIRouter``, ``X`` is a factory-produced instance of that kind.
       Variables that reach ``fastapi`` only through unrelated symbols
       (e.g. ``HTTPException``) are not classified, so they don't get
       marked as apps.
    3. For each detected instance, emits ``X -> handler`` edges for its
       decorated handlers. ``FastAPI`` instances are seeded as
       entrypoints; routers are not, so an ``APIRouter`` that nothing
       ``include_router``s stays dead.

    Application wiring (``app.include_router(router)``,
    ``FastAPI(lifespan=fn)``, ``app.add_api_route(..., endpoint=fn)``,
    ``Depends(fn)``) is plain reference passing already tracked by the
    regular analyzer.

    Limitations: only top-level decls with a single ``Name`` decorator
    target ``@X.<verb>(...)`` are handled. Class-attribute apps
    (``self.app = FastAPI(); @self.app.get(...)``) and decorators that
    chain through extra calls aren't recognized.
    """

    name: str = "fastapi"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        fastapi_node = require_resolved_dep(ctx, "fastapi", "FastAPI")
        if fastapi_node is None:
            return

        for path, module_node in ctx.base_modules():
            module = ctx.parse(path)
            if module is None:
                continue

            fastapi_imports = collect_module_imports(module, "fastapi", _INSTANCE_KINDS)
            direct: dict[str, str] = (
                find_call_assignments(module, fastapi_imports, _INSTANCE_KINDS)
                if fastapi_imports
                else {}
            )
            decorated = _route_decorator_candidates(module)
            if not direct and not decorated:
                continue

            module_fqname = module_node.fqname
            for var_name in set(direct) | set(decorated):
                for instance_decl in ctx.find_declarations(f"{module_fqname}.{var_name}"):
                    kind = direct.get(var_name) or _walk_to_fastapi_kind(
                        ctx.graph, instance_decl, fastapi_node
                    )
                    if kind is None:
                        continue
                    if _INSTANCE_KINDS[kind]:
                        yield AddNode(instance_decl, entrypoint=True)
                    for handler_name in decorated.get(var_name, ()):
                        for handler_decl in ctx.find_declarations(
                            f"{module_fqname}.{handler_name}"
                        ):
                            yield AddEdge(instance_decl, handler_decl)


def _route_decorator_candidates(module: cst.Module) -> dict[str, list[str]]:
    """Collect every top-level ``@<X>.<route_verb>(...)``-decorated function.

    Returns ``{owner_var_name: [handler_func_name, ...]}``. Owner is read
    via :func:`decorator_owner` so bare-name (``@get``) and attribute-on-
    non-name (``@self.app.get``) forms are skipped.
    """
    handlers: dict[str, list[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            owner = decorator_owner(dec.decorator, _REGISTRATION_DECORATORS)
            if owner is None:
                continue
            handlers.setdefault(owner, []).append(stmt.name.value)
            break
    return handlers


def _walk_to_fastapi_kind(
    graph: nx.DiGraph, instance_decl: SymbolNode, fastapi_node: SymbolNode
) -> str | None:
    """Return ``"FastAPI"`` / ``"APIRouter"`` reached from ``instance_decl``, else ``None``.

    Walks forward through reference edges until hitting an ``import``
    node bound to one of the two classes. The factory case
    (``X = create_app()``) drops out because the factory's body
    references ``FastAPI`` and the analyzer already recorded that
    edge. Returns ``None`` when the chain doesn't reach a
    discriminating import -- callers treat that as "not a FastAPI
    instance" rather than guessing a kind.
    """
    seen: set[SymbolNode] = set()
    stack: list[SymbolNode] = [instance_decl]
    while stack:
        node = stack.pop()
        if node in seen or node is fastapi_node:
            continue
        seen.add(node)
        kind = _import_kind(node)
        if kind is not None:
            return kind
        stack.extend(graph.successors(node))
    return None


def _import_kind(node: SymbolNode) -> str | None:
    """Return ``"FastAPI"`` / ``"APIRouter"`` if ``node`` imports either, else ``None``.

    The plugin requires ``fastapi`` to be a resolved distribution
    (:func:`require_resolved_dep`), so the visitor always encodes
    ``from fastapi import FastAPI`` as ``Import.module="fastapi"`` +
    ``Import.decl="FastAPI"``.
    """
    if node.type != "import" or node.imports is None:
        return None
    if node.imports.module != "fastapi":
        return None
    return node.imports.decl if node.imports.decl in _INSTANCE_KINDS else None
