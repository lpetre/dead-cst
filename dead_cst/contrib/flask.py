"""Plugin: keep Flask route handlers and lifecycle hooks alive.

Mirrors :class:`FastAPIPlugin`: direct ``X = Flask(...)`` /
``X = Blueprint(...)`` assignments are classified per-file in
:meth:`DispatchAppPlugin.observe`, while factory-style apps
(``X = create_app()``) are deferred to :meth:`DispatchAppPlugin.finalize`
via ``<flask-pending>:`` markers that the graph walk resolves once
cross-file edges are in place. Factory functions / classes whose body
constructs a Flask / Blueprint instance also get a
``<flask-factory>:<kind>:<owner.fqname>`` marker so the cross-package
walk has a discriminator even when the factory uses the
``import flask; flask.Flask()`` attribute form -- in that shape
:func:`resolve_edges` drops the ``decl='Flask'`` half of the external
classification and the import-node check alone misses the case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..plugins.decl_shapes import DispatchAppPlugin

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
_INSTANCE_KINDS: Mapping[str, bool] = {
    "Flask": True,  # WSGI servers load ``module:app`` -- always an entrypoint
    "Blueprint": False,  # only alive if reached via ``register_blueprint``
}


@dataclass
class FlaskPlugin(DispatchAppPlugin):
    """Mark Flask apps as entrypoints and wire route handlers through them.

    Concrete configuration of the factory-aware
    :class:`DispatchAppPlugin` shape:

    * Direct ``X = Flask(...)`` / ``X = Blueprint(...)`` assignments are
      classified per-file. Direct ``Flask`` hits get a ``<flask-app>:``
      synthetic entrypoint pointed at the variable.
    * ``@<X>.<verb>(...)`` decorators emit ``X -> handler`` edges.
      Variables that have handlers but no direct kind get a
      ``<flask-pending>:`` marker the cross-package finalize pass
      resolves, supporting the ``X = create_app()`` factory shape.
    * Top-level decls whose body constructs a Flask / Blueprint
      instance get a ``<flask-factory>:<kind>:<owner.fqname>`` marker so
      the pending walk can classify cross-package factory chains where
      the framework class is reached via the attribute-form
      ``flask.Flask()`` and the external-edge classifier drops the
      ``decl='Flask'`` half.
    """

    name: str = "flask"
    version: int = 1778973600
    app_module: str = "flask"
    registration_decorators: frozenset[str] = _REGISTRATION_DECORATORS
    instance_kinds: Mapping[str, bool] = field(default_factory=lambda: dict(_INSTANCE_KINDS))
