"""Plugin: keep Flask route handlers and lifecycle hooks alive."""

from __future__ import annotations

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


def flask_plugin() -> DispatchAppPlugin:
    """Mark Flask apps as entrypoints and wire route handlers through them.

    Handles direct (``X = Flask(...)``), aliased
    (``from flask import Flask as F; X = F(...)``), module-prefixed
    (``import flask; X = flask.Flask(...)``), and factory-style
    (``X = create_app()``) construction. ``Blueprint`` instances are
    wired but not seeded as entrypoints, so a blueprint that nothing
    ``register_blueprint``s stays dead.
    """
    return DispatchAppPlugin(
        marker_prefix="flask",
        app_classes=("flask.Flask",),
        registration_decorators=_REGISTRATION_DECORATORS,
    )
