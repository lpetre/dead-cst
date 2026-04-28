"""Plugin: keep Flask route handlers and lifecycle hooks alive.

Strategy: every Flask / Blueprint instance we want to wire up is a
top-level variable that the analyzer has already linked back to the
``flask`` import -- whether the assignment is the literal
``X = Flask(__name__)``, the aliased ``X = F(__name__)`` after
``from flask import Flask as F``, or the factory form
``X = create_app()`` whose body returns ``Flask(...)``. The plugin
reuses those reference edges:

1. Direct shape (``X = Flask(...)`` / ``X = Blueprint(...)``,
   ``X = flask.Flask(...)``, etc.) is recognized syntactically via
   ``find_call_assignments`` -- this is unambiguous, so it gives the
   kind directly even for ``import flask`` forms where the graph
   alone can't distinguish the two classes.
2. Indirect shape (any variable decorated by ``@X.<route_verb>(...)``)
   is detected via the route-decorator scan; kind is read off the
   import nodes encountered while walking forward from ``X`` in the
   graph, which transparently handles the canonical Flask
   ``create_app()`` factory because the factory's body is already
   linked to the right import.

For each detected instance the plugin emits ``X -> handler`` edges for
its decorated handlers and seeds ``Flask`` instances as entrypoints
(WSGI servers load ``module:app``); blueprints are not seeded, so a
``Blueprint`` that nothing ``register_blueprint``s stays dead -- the
right signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from ._core import (
    AddEdge,
    AddNode,
    GraphOp,
    PluginContext,
    collect_module_imports,
    find_call_assignments,
    find_handlers,
    require_resolved_dep,
    walk_to_instance_kind,
)

# Attribute names Flask / Blueprint use to register a callable. Matched as
# the rightmost attribute of ``@<instance>.<name>(...)``. Includes the HTTP
# verb shortcuts (``app.get`` / ``app.post`` / ...), request lifecycle
# hooks, error handlers, template / context helpers, URL processors, and
# the ``app_*`` variants Blueprints use to register app-scoped callbacks.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset(
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

    1. Finds direct ``X = Flask(...)`` / ``X = Blueprint(...)``
       assignments by inspecting the file's ``flask`` imports and
       matching call sites. This is the unambiguous path and handles
       module-prefixed forms like ``import flask; X = flask.Flask()``
       that pure graph reachability cannot distinguish (the variable's
       edge goes straight to the ``flask`` synthetic with no
       intermediate ``Flask`` vs ``Blueprint`` import to discriminate).
    2. Collects every ``@<X>.<verb>(...)`` decorator (``verb`` from
       Flask's registration set). For each owner ``X`` not already
       resolved by step 1, walks forward from ``X`` in the symbol graph;
       if the walk hits an ``import`` node bound to ``Flask`` or
       ``Blueprint``, ``X`` is a factory-produced instance of that kind.
       Variables that reach ``flask`` only through unrelated symbols
       (e.g. ``request``, ``url_for``) are not classified, so they
       don't get marked as apps.
    3. For each detected instance, emits ``X -> handler`` edges for its
       decorated handlers. ``Flask`` instances are seeded as entrypoints;
       blueprints are not, so a ``Blueprint`` that nothing
       ``register_blueprint``s stays dead.

    Application wiring (``app.register_blueprint(bp)``,
    ``app.add_url_rule(..., view_func=fn)``, ``app.errorhandler(404)(fn)``)
    is plain reference passing already tracked by the regular analyzer.

    Limitations: only top-level decls with a single ``Name`` decorator
    target ``@X.<verb>(...)`` are handled. Class-attribute apps
    (``self.app = Flask(__name__); @self.app.route(...)``) and
    decorators that chain through extra calls aren't recognized.
    """

    name: str = "flask"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        flask_node = require_resolved_dep(ctx, "flask")
        if flask_node is None:
            return
        reaches_flask = nx.ancestors(ctx.graph, flask_node)

        for path, module_node in ctx.base_modules():
            module = ctx.parse(path)
            if module is None:
                continue

            flask_imports = collect_module_imports(module, "flask", _INSTANCE_KINDS)
            direct = find_call_assignments(module, flask_imports, _INSTANCE_KINDS)
            decorated = find_handlers(module, None, _REGISTRATION_DECORATORS)
            if not direct and not decorated:
                continue

            module_fqname = module_node.fqname
            for var_name in direct.keys() | decorated.keys():
                for instance_decl in ctx.find_declarations(f"{module_fqname}.{var_name}"):
                    kind = direct.get(var_name)
                    if kind is None:
                        if instance_decl not in reaches_flask:
                            continue
                        kind = walk_to_instance_kind(
                            ctx.graph, instance_decl, flask_node, "flask", _INSTANCE_KINDS
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
