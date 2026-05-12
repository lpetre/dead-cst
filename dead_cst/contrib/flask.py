"""Plugin: keep Flask route handlers and lifecycle hooks alive.

Mirrors :class:`FastAPIPlugin`: direct ``X = Flask(...)`` /
``X = Blueprint(...)`` assignments are classified per-file in
:meth:`observe`, while factory-style apps (``X = create_app()``) are
deferred to :meth:`finalize` via ``<flask-pending>:`` markers that the
graph walk resolves once cross-file edges are in place. Factory
functions / classes whose body constructs a Flask / Blueprint instance
also get a ``<flask-factory>:<kind>:<owner.fqname>`` marker so the
cross-package walk has a discriminator even when the factory uses the
``import flask; flask.Flask()`` attribute form -- in that shape
:func:`resolve_edges` drops the ``decl='Flask'`` half of the external
classification and the import-node check alone misses the case.
"""

from __future__ import annotations

from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

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
    find_factory_decls,
    find_handlers,
    make_payload,
    require_resolved_dep,
    synthetic_node,
    walk_to_instance_kind,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload

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

FLASK_APP_PREFIX = "<flask-app>:"
FLASK_PENDING_PREFIX = "<flask-pending>:"
# See ``FASTAPI_FACTORY_PREFIX`` for the rationale. Same shape, same
# motivation: cross-package factory chains lose the ``decl='Flask'``
# half of an ``import flask; flask.Flask()`` access through the
# external-edge classifier, so we anchor a marker on the constructing
# decl that finalize's walk can recognize regardless of file.
# Format: ``<flask-factory>:<Flask|Blueprint>:<owner.fqname>``.
FLASK_FACTORY_PREFIX = "<flask-factory>:"


@dataclass
class FlaskPlugin:
    """Mark Flask apps as entrypoints and wire route handlers through them.

    Two-phase shape mirrors :class:`FastAPIPlugin`:

    * :meth:`observe` (per-file) classifies direct
      ``X = Flask(...)`` / ``X = Blueprint(...)`` assignments and emits
      ``X -> handler`` edges for every ``@<X>.<verb>(...)`` decorator.
      Direct ``Flask`` hits get a :data:`FLASK_APP_PREFIX` synthetic
      entrypoint pointed at the variable. Variables that have handlers
      but no direct kind get a :data:`FLASK_PENDING_PREFIX` marker
      synthetic linked to the variable, deferred to :meth:`finalize`.
      Top-level decls whose body constructs a Flask / Blueprint
      instance get a :data:`FLASK_FACTORY_PREFIX` marker that survives
      cross-package walks.
    * :meth:`finalize` (per-package) walks each pending marker forward
      through the graph, classifies via :func:`walk_to_instance_kind`
      (with :data:`FLASK_FACTORY_PREFIX` so factory markers count as
      discriminators), and promotes ``Flask`` factories to entrypoints.
      Blueprints stay pass-through.
    """

    name: str = "flask"
    version: int = 1778973600

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        flask_imports = collect_module_imports(ctx.module, "flask", _INSTANCE_KINDS)
        direct = find_call_assignments(ctx.module, flask_imports, _INSTANCE_KINDS)
        decorated = find_handlers(ctx.module, None, _REGISTRATION_DECORATORS)
        # Top-level decls whose body constructs a Flask / Blueprint instance.
        # Recognizes both ``Flask()`` (named import) and ``flask.Flask()``
        # (module import) shapes via ``matched_attr_call``.
        factory_kinds = find_factory_decls(ctx.module, flask_imports, _INSTANCE_KINDS)
        if not direct and not decorated and not factory_kinds:
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        for var_name in direct.keys() | decorated.keys():
            var_decls = decls_by_name.get(var_name, [])
            kind = direct.get(var_name)
            for var_decl in var_decls:
                if kind is None:
                    pending = synthetic_node(f"{FLASK_PENDING_PREFIX}{var_decl.fqname}", ctx.path)
                    nodes.append(pending)
                    edges.append((pending, var_decl, SYNTHETIC_POSITION))
                elif _INSTANCE_KINDS[kind]:
                    seed = synthetic_node(
                        f"{FLASK_APP_PREFIX}{var_decl.fqname}",
                        ctx.path,
                        flags=NodeFlags.ENTRYPOINT,
                    )
                    nodes.append(seed)
                    edges.append((seed, var_decl, SYNTHETIC_POSITION))

                for handler_name in decorated.get(var_name, ()):
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((var_decl, handler_decl, SYNTHETIC_POSITION))

        # Factory markers: see fastapi.py for the rationale.
        for decl_name, kinds in factory_kinds.items():
            for decl in decls_by_name.get(decl_name, []):
                for kind in kinds:
                    marker = synthetic_node(f"{FLASK_FACTORY_PREFIX}{kind}:{decl.fqname}", ctx.path)
                    nodes.append(marker)
                    edges.append((decl, marker, SYNTHETIC_POSITION))

        if not nodes and not edges:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        flask_node = require_resolved_dep(ctx, "flask")
        if flask_node is None:
            return

        for synth in list(ctx.package_nodes()):
            if synth.type != "synthetic" or not synth.fqname.startswith(FLASK_PENDING_PREFIX):
                continue
            for var in list(ctx.graph.successors(synth)):
                kind = walk_to_instance_kind(
                    ctx.graph,
                    var,
                    flask_node,
                    "flask",
                    _INSTANCE_KINDS,
                    factory_marker_prefix=FLASK_FACTORY_PREFIX,
                )
                if kind is None or not _INSTANCE_KINDS[kind]:
                    continue
                seed = synthetic_node(f"{FLASK_APP_PREFIX}{var.fqname}", var.path)
                yield AddNode(seed, entrypoint=True)
                yield AddEdge(seed, var)
