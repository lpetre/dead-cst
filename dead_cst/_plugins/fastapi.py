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
   alone can't distinguish the two classes. Resolution happens in
   :meth:`observe`.
2. Indirect shape (any variable decorated by ``@X.<route_verb>(...)``)
   is detected via the route-decorator scan. Per-file ``observe``
   emits a ``<fastapi-pending>:<X.fqname>`` marker plus the
   ``X -> handler`` edges; the per-base ``finalize`` pass walks the
   graph forward from each pending marker, classifies via
   ``walk_to_instance_kind``, and promotes ``FastAPI`` instances to
   entrypoints.
"""

from __future__ import annotations

from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .._symbols import NodeFlags, SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    AddNode,
    GraphOp,
    ObserveContext,
    PluginContext,
    _payload_from,
    collect_module_imports,
    find_call_assignments,
    find_handlers,
    require_resolved_dep,
    simple_name,
    synthetic_node,
    walk_to_instance_kind,
)

if TYPE_CHECKING:
    from .._visitor import VisitorPayload

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

FASTAPI_APP_PREFIX = "<fastapi-app>:"
FASTAPI_PENDING_PREFIX = "<fastapi-pending>:"


@dataclass
class FastAPIPlugin:
    """Mark FastAPI apps as entrypoints and wire route handlers through them.

    For each module the plugin observes:

    1. Direct ``X = FastAPI(...)`` / ``X = APIRouter(...)`` assignments
       via ``find_call_assignments``. This is the unambiguous path and
       handles module-prefixed forms like
       ``import fastapi; X = fastapi.FastAPI()`` that pure graph
       reachability cannot distinguish (the variable's edge goes
       straight to the ``fastapi`` synthetic with no intermediate
       ``FastAPI`` vs ``APIRouter`` import to discriminate). Direct
       FastAPI hits get a :data:`FASTAPI_APP_PREFIX` synthetic
       entrypoint plus an edge pointing at the variable; routers do
       not.
    2. ``@<X>.<verb>(...)`` decorators via ``find_handlers``. Each
       ``X -> handler`` edge is emitted unconditionally; whether ``X``
       is reachable depends on the next step.
    3. Variables decorated but not directly classified get a
       :data:`FASTAPI_PENDING_PREFIX` marker synthetic with an edge
       to the variable. The :meth:`finalize` pass picks these up.

    Finalize walks forward from each pending variable, hits the
    ``fastapi`` external-dist synthetic if any, classifies the variable
    via ``walk_to_instance_kind``, and -- for ``FastAPI`` instances --
    emits a ``<fastapi-app>:`` synthetic entrypoint plus an edge to
    the variable. Routers and unclassified variables stay as-is, so
    an ``APIRouter`` that nothing ``include_router``s remains dead.
    """

    name: str = "fastapi"
    version: str = "1"

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        fastapi_imports = collect_module_imports(ctx.module, "fastapi", _INSTANCE_KINDS)
        direct = find_call_assignments(ctx.module, fastapi_imports, _INSTANCE_KINDS)
        decorated = find_handlers(ctx.module, None, _REGISTRATION_DECORATORS)
        if not direct and not decorated:
            return None

        decls_by_name = _decls_by_simple_name(ctx.payload.nodes)
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        # Every variable observed (direct or just decorated) gets either
        # an app entrypoint, a router classification, or a pending marker
        # for finalize to resolve.
        for var_name in direct.keys() | decorated.keys():
            var_decls = decls_by_name.get(var_name, [])
            kind = direct.get(var_name)
            for var_decl in var_decls:
                if kind is None:
                    pending = synthetic_node(f"{FASTAPI_PENDING_PREFIX}{var_decl.fqname}", ctx.path)
                    nodes.append(pending)
                    edges.append((pending, var_decl, SYNTHETIC_POSITION))
                elif _INSTANCE_KINDS[kind]:
                    seed = synthetic_node(
                        f"{FASTAPI_APP_PREFIX}{var_decl.fqname}",
                        ctx.path,
                        flags=NodeFlags.ENTRYPOINT,
                    )
                    nodes.append(seed)
                    edges.append((seed, var_decl, SYNTHETIC_POSITION))
                # APIRouter direct hits get only handler edges below.

                for handler_name in decorated.get(var_name, ()):
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((var_decl, handler_decl, SYNTHETIC_POSITION))

        if not nodes and not edges:
            return None
        return _payload_from(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        fastapi_node = require_resolved_dep(ctx, "fastapi")
        if fastapi_node is None:
            return

        for synth in list(ctx.base_nodes()):
            if synth.type != "synthetic" or not synth.fqname.startswith(FASTAPI_PENDING_PREFIX):
                continue
            successors = list(ctx.graph.successors(synth))
            if not successors:
                continue
            for var in successors:
                kind = walk_to_instance_kind(
                    ctx.graph, var, fastapi_node, "fastapi", _INSTANCE_KINDS
                )
                if kind is None or not _INSTANCE_KINDS[kind]:
                    continue
                seed = synthetic_node(f"{FASTAPI_APP_PREFIX}{var.fqname}", var.path)
                yield AddNode(seed, entrypoint=True)
                yield AddEdge(seed, var)


def _decls_by_simple_name(nodes) -> dict[str, list[SymbolNode]]:
    out: dict[str, list[SymbolNode]] = {}
    for n in nodes:
        if n.type in ("class", "function", "variable", "import"):
            out.setdefault(simple_name(n.fqname), []).append(n)
    return out
