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

from ._core import (
    AddEdge,
    AddNode,
    GraphOp,
    PluginContext,
    collect_module_imports,
    find_call_assignments,
    find_handlers,
)

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
            flask_imports = collect_module_imports(module, "flask", _INSTANCE_KINDS)
            if not flask_imports:
                continue
            instances = find_call_assignments(module, flask_imports, _INSTANCE_KINDS)
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
