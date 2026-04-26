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
from pathlib import Path
from typing import Iterable

import libcst as cst
from libcst.metadata import FullRepoManager

from .._symbols import SymbolNode
from ._core import AddEdge, AddNode, GraphOp, PluginContext

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

    This is a :class:`CSTAwareEdgePlugin` because detection needs the
    original CST.
    """

    name: str = "fastapi"
    cst_aware: bool = True

    def contribute(
        self, ctx: PluginContext, managers: dict[Path, FullRepoManager]
    ) -> Iterable[GraphOp]:
        modules_by_path: dict[Path, SymbolNode] = {}
        for node in ctx.graph.nodes:
            if node.type == "module":
                modules_by_path[node.path] = node

        # Cheap prefilter: importing FastAPI / APIRouter requires the literal
        # ``fastapi`` substring somewhere in the file. Modules that lack it
        # can't be candidates and are skipped without a CST walk.
        candidate_paths = set(ctx.grep("fastapi", paths=modules_by_path.keys()))

        for path, module_node in modules_by_path.items():
            if path not in candidate_paths:
                continue
            wrapper = _wrapper_for(path, managers)
            if wrapper is None:
                continue
            fastapi_imports = _collect_fastapi_imports(wrapper.module)
            if not fastapi_imports:
                continue
            instances = _find_instances(wrapper.module, fastapi_imports)
            if not instances:
                continue
            handlers = _find_handlers(wrapper.module, set(instances))

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


def _wrapper_for(path: Path, managers: dict[Path, FullRepoManager]):
    for base, mgr in managers.items():
        if not path.is_relative_to(base):
            continue
        try:
            return mgr.get_metadata_wrapper_for_path(path)
        except Exception:
            return None
    return None


def _collect_fastapi_imports(module: cst.Module) -> dict[str, str]:
    """Return ``{local_name: target}`` for names imported from ``fastapi``.

    ``target`` is one of ``"FastAPI"``, ``"APIRouter"``, or ``"<module>"``
    (the whole ``fastapi`` package, for ``import fastapi``).
    """
    bindings: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                if not _is_fastapi_module(small):
                    continue
                if isinstance(small.names, cst.ImportStar):
                    continue
                for alias in small.names:
                    target = alias.name.value if isinstance(alias.name, cst.Name) else None
                    if target not in _INSTANCE_KINDS:
                        continue
                    local = alias.asname.name.value if alias.asname else target
                    if isinstance(local, str):
                        bindings[local] = target
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    if not _is_name(alias.name, "fastapi"):
                        continue
                    local = alias.asname.name.value if alias.asname else "fastapi"
                    if isinstance(local, str):
                        bindings[local] = "<module>"
    return bindings


def _is_fastapi_module(node: cst.ImportFrom) -> bool:
    if node.relative:
        return False
    return _is_name(node.module, "fastapi")


def _is_name(node: cst.CSTNode | None, value: str) -> bool:
    return isinstance(node, cst.Name) and node.value == value


def _find_instances(module: cst.Module, fastapi_imports: dict[str, str]) -> dict[str, str]:
    """Return ``{var_name: 'FastAPI' | 'APIRouter'}`` for top-level instances."""
    instances: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target_name, value = _single_target_assignment(small)
            if target_name is None or not isinstance(value, cst.Call):
                continue
            kind = _classify_call(value.func, fastapi_imports)
            if kind is not None:
                instances[target_name] = kind
    return instances


def _single_target_assignment(
    stmt: cst.BaseSmallStatement,
) -> tuple[str | None, cst.BaseExpression | None]:
    """Extract ``(name, rhs)`` for ``X = ...`` / ``X: T = ...``; else ``(None, None)``."""
    if isinstance(stmt, cst.Assign):
        if len(stmt.targets) != 1:
            return None, None
        target = stmt.targets[0].target
        if isinstance(target, cst.Name):
            return target.value, stmt.value
    elif isinstance(stmt, cst.AnnAssign):
        if isinstance(stmt.target, cst.Name) and stmt.value is not None:
            return stmt.target.value, stmt.value
    return None, None


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


def _find_handlers(module: cst.Module, instance_vars: set[str]) -> dict[str, list[str]]:
    """Return ``{instance_var: [handler_func_name, ...]}`` for decorated handlers."""
    handlers: dict[str, list[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            owner = _route_decorator_owner(dec.decorator)
            if owner is None or owner not in instance_vars:
                continue
            handlers.setdefault(owner, []).append(stmt.name.value)
            break
    return handlers


def _route_decorator_owner(expr: cst.BaseExpression) -> str | None:
    """For ``@X.get(...)`` / ``@X.get`` return ``"X"`` (only when the rightmost
    attribute is a known FastAPI registration name and ``X`` is a bare
    ``Name``). Returns ``None`` otherwise."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if not isinstance(expr, cst.Attribute):
        return None
    if expr.attr.value not in _ROUTE_DECORATORS:
        return None
    if not isinstance(expr.value, cst.Name):
        return None
    return expr.value.value
