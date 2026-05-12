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
   ``X -> handler`` edges; the per-package ``finalize`` pass walks the
   graph forward from each pending marker and promotes
   ``FastAPI`` instances to entrypoints.
3. Factory functions / classes whose body constructs a FastAPI /
   APIRouter instance are tagged at ``observe`` time with a
   ``<fastapi-factory>:<kind>:<decl.fqname>`` marker. This lets a
   cross-package consumer's pending-variable walk hit a discriminator
   even when the factory uses the ``import fastapi; fastapi.FastAPI()``
   form -- the attribute access lands as a bare
   ``[external dist] fastapi`` edge after :func:`resolve_edges` drops
   the ``decl='FastAPI'`` half, so the graph alone can't tell FastAPI
   from APIRouter on the downstream walk.
"""

from __future__ import annotations

import libcst as cst
from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Container, Iterable

import networkx as nx

from ..graph import NodeFlags, SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    AddNode,
    GraphOp,
    ObserveContext,
    PluginContext,
    collect_module_imports,
    decls_by_simple_name,
    find_call_assignments,
    find_handlers,
    make_payload,
    matched_attr_call,
    require_resolved_dep,
    synthetic_node,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload

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
# Synthetic emitted from ``observe`` for any top-level function / class whose
# body constructs a ``FastAPI()`` / ``APIRouter()`` instance. Lets
# :meth:`FastAPIPlugin.finalize` classify cross-package factory chains where
# the framework class is reached via ``import fastapi; fastapi.FastAPI()`` --
# the attribute-form access lands as a bare ``[external dist] fastapi`` edge
# (the resolver drops ``decl='FastAPI'`` for external classifications), so the
# graph alone can't distinguish FastAPI from APIRouter on the downstream walk.
# Format: ``<fastapi-factory>:<FastAPI|APIRouter>:<owner.fqname>``.
FASTAPI_FACTORY_PREFIX = "<fastapi-factory>:"


class _ConstructorFinder(cst.CSTVisitor):
    """Collect ``FastAPI``/``APIRouter`` construction kinds inside a decl body."""

    def __init__(self, imports: dict[str, str], valid_targets: Container[str]) -> None:
        super().__init__()
        self._imports = imports
        self._valid_targets = valid_targets
        self.kinds: set[str] = set()

    def visit_Call(self, node: cst.Call) -> bool | None:
        kind = matched_attr_call(node.func, self._imports, self._valid_targets, unwrap_call=False)
        if kind is not None:
            self.kinds.add(kind)
        return None


def _walk_factory_kind(
    graph: nx.DiGraph,
    start: SymbolNode,
    terminal: SymbolNode,
) -> str | None:
    """Forward walk from ``start`` to a FastAPI / APIRouter discriminator.

    Like :func:`walk_to_instance_kind` but also recognizes the
    :data:`FASTAPI_FACTORY_PREFIX` markers ``observe`` emits for factory
    functions that construct via ``fastapi.FastAPI(...)`` -- the
    attribute-form whose ``decl`` info is dropped by the external-edge
    classifier, so the import-node check alone misses it across files.
    """
    seen: set[SymbolNode] = set()
    stack: list[SymbolNode] = [start]
    while stack:
        node = stack.pop()
        if node in seen or node is terminal:
            continue
        seen.add(node)
        if node.type == "import" and node.imports is not None:
            decl = node.imports.decl
            if decl is not None and node.imports.module == "fastapi" and decl in _INSTANCE_KINDS:
                return decl
        if node.type == "synthetic" and node.fqname.startswith(FASTAPI_FACTORY_PREFIX):
            kind = node.fqname[len(FASTAPI_FACTORY_PREFIX) :].split(":", 1)[0]
            if kind in _INSTANCE_KINDS:
                return kind
        stack.extend(graph.successors(node))
    return None


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
    4. Top-level decls whose body constructs a FastAPI / APIRouter
       instance get a :data:`FASTAPI_FACTORY_PREFIX` marker. This
       discriminator survives cross-package walks where the external
       edge would otherwise lose ``decl='FastAPI'`` info.

    Finalize walks forward from each pending variable until it hits
    a discriminator (a ``from fastapi import FastAPI``-style import
    node or a factory marker), classifies the variable, and -- for
    ``FastAPI`` instances -- emits a ``<fastapi-app>:`` synthetic
    entrypoint plus an edge to the variable. Routers and unclassified
    variables stay as-is, so an ``APIRouter`` that nothing
    ``include_router``s remains dead.
    """

    name: str = "fastapi"
    version: int = 1778973600

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        fastapi_imports = collect_module_imports(ctx.module, "fastapi", _INSTANCE_KINDS)
        direct = find_call_assignments(ctx.module, fastapi_imports, _INSTANCE_KINDS)
        decorated = find_handlers(ctx.module, None, _REGISTRATION_DECORATORS)
        # Top-level decls whose body constructs a FastAPI / APIRouter instance.
        # ``fastapi_imports`` doubles as the binding map for ``matched_attr_call``
        # so both ``FastAPI()`` (named import) and ``fastapi.FastAPI()`` (module
        # import) are recognized.
        factory_kinds = _find_factory_decls(ctx.module, fastapi_imports)
        if not direct and not decorated and not factory_kinds:
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
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

        # Factory markers: decl bodies that build a FastAPI / APIRouter
        # instance. Anchored to the constructing decl so finalize's forward
        # walk hits them regardless of which file the consumer lives in.
        for decl_name, kinds in factory_kinds.items():
            for decl in decls_by_name.get(decl_name, []):
                for kind in kinds:
                    marker = synthetic_node(
                        f"{FASTAPI_FACTORY_PREFIX}{kind}:{decl.fqname}", ctx.path
                    )
                    nodes.append(marker)
                    edges.append((decl, marker, SYNTHETIC_POSITION))

        if not nodes and not edges:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        fastapi_node = require_resolved_dep(ctx, "fastapi")
        if fastapi_node is None:
            return

        for synth in list(ctx.package_nodes()):
            if synth.type != "synthetic" or not synth.fqname.startswith(FASTAPI_PENDING_PREFIX):
                continue
            successors = list(ctx.graph.successors(synth))
            if not successors:
                continue
            for var in successors:
                kind = _walk_factory_kind(ctx.graph, var, fastapi_node)
                if kind is None or not _INSTANCE_KINDS[kind]:
                    continue
                seed = synthetic_node(f"{FASTAPI_APP_PREFIX}{var.fqname}", var.path)
                yield AddNode(seed, entrypoint=True)
                yield AddEdge(seed, var)


def _find_factory_decls(module: cst.Module, fastapi_imports: dict[str, str]) -> dict[str, set[str]]:
    """Return ``{decl_name: {kind, ...}}`` for top-level decls whose body
    constructs a FastAPI / APIRouter instance.

    Scans every top-level ``def`` / ``class`` body for ``FastAPI(...)`` /
    ``APIRouter(...)`` call shapes (named or module-prefixed). Skips files
    that don't import ``fastapi`` at all -- ``fastapi_imports`` would be
    empty, and ``matched_attr_call`` would reject every candidate anyway.
    """
    if not fastapi_imports:
        return {}
    out: dict[str, set[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
            continue
        finder = _ConstructorFinder(fastapi_imports, _INSTANCE_KINDS)
        stmt.body.visit(finder)
        if finder.kinds:
            out[stmt.name.value] = finder.kinds
    return out
