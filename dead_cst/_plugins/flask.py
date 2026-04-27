"""Plugin: keep Flask route handlers and lifecycle hooks alive.

Strategy: instead of marking each handler as its own entrypoint, find the
top-level ``Flask()`` / ``Blueprint()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.<method>(...)``
decorator. ``Flask()`` instances are then seeded as entrypoints; blueprints
stay pass-through so an unused ``Blueprint`` still surfaces as dead code.

This routes ``why-alive`` chains through the app variable users actually
recognize ("alive because it's a route on ``app``") and lets the existing
``app.register_blueprint(bp)`` reference flow keep blueprints reachable
without any special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst

from ._core import AddEdge, AddNode, GraphOp, PluginContext

# Attribute names Flask / Blueprint use to register a callable. Matched as
# the rightmost attribute of ``@<instance>.<name>(...)``. Includes the HTTP
# verb shortcuts (``app.get`` / ``app.post`` / ...), request lifecycle
# hooks, error handlers, template / context helpers, URL processors, and
# the ``app_*`` variants Blueprints use to register app-scoped callbacks.
_ROUTE_DECORATORS: frozenset[str] = frozenset(
    {
        "route",
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "before_request",
        "after_request",
        "teardown_request",
        "teardown_appcontext",
        "before_first_request",
        "before_app_request",
        "after_app_request",
        "teardown_app_request",
        "before_app_first_request",
        "errorhandler",
        "app_errorhandler",
        "context_processor",
        "app_context_processor",
        "template_filter",
        "app_template_filter",
        "template_test",
        "app_template_test",
        "template_global",
        "app_template_global",
        "url_value_preprocessor",
        "app_url_value_preprocessor",
        "url_defaults",
        "app_url_defaults",
        "shell_context_processor",
        "record",
        "record_once",
    }
)

# Classes whose instances we treat specially. Value records whether the
# instance should be seeded as an entrypoint.
_INSTANCE_KINDS: dict[str, bool] = {
    "Flask": True,  # WSGI servers load ``module:app`` -- always an entrypoint
    "Blueprint": False,  # only alive if reached via ``register_blueprint``
}


@dataclass
class FlaskPlugin:
    """Mark Flask apps as entrypoints and wire route handlers through them.

    For each module the plugin:

    1. Inspects ``from flask import ...`` / ``import flask`` to learn
       which local names refer to ``Flask`` and ``Blueprint``.
    2. Finds top-level assignments ``X = Flask(...)`` / ``X = Blueprint(...)``
       (including ``AnnAssign`` and aliased / module-prefixed forms) and
       records ``X`` as an instance of that kind.
    3. For every top-level function decorated ``@X.<route_name>(...)``,
       emits an edge ``X -> handler`` so the handler is reachable whenever
       ``X`` is.
    4. Seeds every ``Flask`` instance as an entrypoint. Blueprints are not
       seeded -- a blueprint that is never ``register_blueprint``\\ed has
       no path from any entrypoint and stays dead, which is the correct
       signal.

    Application wiring (``app.register_blueprint(bp)``,
    ``app.add_url_rule(..., view_func=fn)``, ``app.errorhandler(404)(fn)``)
    is plain reference passing already tracked by the regular analyzer.

    Limitations: only top-level ``X = Flask(...)`` / ``X = Blueprint(...)``
    assignments with a single ``Name`` target are detected. Factory-style
    apps (``def create_app(): return Flask(__name__)``) and class-attribute
    apps (``self.app = Flask(__name__)``) are not handled; users can still
    keep those alive with explicit ``-e`` entrypoints.
    """

    name: str = "flask"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Prefilter via the import graph: only files that actually import
        # ``flask`` can declare an app or blueprint. The analyzer's resolver
        # already added ``[external dist] flask`` predecessors for them.
        candidate_paths = ctx.importers("flask")
        if not candidate_paths:
            return

        for path, module_node in ctx.base_modules():
            if path not in candidate_paths:
                continue
            module = ctx.parse(path)
            if module is None:
                continue
            flask_imports = _collect_flask_imports(module)
            if not flask_imports:
                continue
            instances = _find_instances(module, flask_imports)
            if not instances:
                continue
            handlers = _find_handlers(module, set(instances))

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


def _collect_flask_imports(module: cst.Module) -> dict[str, str]:
    """Return ``{local_name: target}`` for names imported from ``flask``.

    ``target`` is one of ``"Flask"``, ``"Blueprint"``, or ``"<module>"``
    (the whole ``flask`` package, for ``import flask``).
    """
    bindings: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                if not _is_flask_module(small):
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
                    if not _is_name(alias.name, "flask"):
                        continue
                    local = alias.asname.name.value if alias.asname else "flask"
                    if isinstance(local, str):
                        bindings[local] = "<module>"
    return bindings


def _is_flask_module(node: cst.ImportFrom) -> bool:
    if node.relative:
        return False
    return _is_name(node.module, "flask")


def _is_name(node: cst.CSTNode | None, value: str) -> bool:
    return isinstance(node, cst.Name) and node.value == value


def _find_instances(module: cst.Module, flask_imports: dict[str, str]) -> dict[str, str]:
    """Return ``{var_name: 'Flask' | 'Blueprint'}`` for top-level instances."""
    instances: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target_name, value = _single_target_assignment(small)
            if target_name is None or not isinstance(value, cst.Call):
                continue
            kind = _classify_call(value.func, flask_imports)
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


def _classify_call(func: cst.BaseExpression, flask_imports: dict[str, str]) -> str | None:
    """Return ``"Flask"`` / ``"Blueprint"`` if ``func`` calls one, else ``None``."""
    if isinstance(func, cst.Name):
        target = flask_imports.get(func.value)
        if target in _INSTANCE_KINDS:
            return target
    elif isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
        # ``flask.Flask(...)`` / ``fl.Blueprint(...)``
        if flask_imports.get(func.value.value) == "<module>":
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
    """For ``@X.route(...)`` / ``@X.route`` return ``"X"`` (only when the
    rightmost attribute is a known Flask registration name and ``X`` is a
    bare ``Name``). Returns ``None`` otherwise."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if not isinstance(expr, cst.Attribute):
        return None
    if expr.attr.value not in _ROUTE_DECORATORS:
        return None
    if not isinstance(expr.value, cst.Name):
        return None
    return expr.value.value
