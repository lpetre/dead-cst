"""Plugin: keep FastAPI route handlers and lifecycle hooks alive."""

from __future__ import annotations

from ..plugins.decl_shapes import DispatchAppPlugin

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


def fastapi_plugin() -> DispatchAppPlugin:
    """Mark FastAPI apps as entrypoints and wire route handlers through them.

    Handles direct (``X = FastAPI(...)``), aliased
    (``from fastapi import FastAPI as F; X = F(...)``), module-prefixed
    (``import fastapi; X = fastapi.FastAPI(...)``), and factory-style
    (``X = create_app()``) construction. ``APIRouter`` instances are
    wired but not seeded as entrypoints, so a router that nothing
    ``include_router``s stays dead.
    """
    return DispatchAppPlugin(
        marker_prefix="fastapi",
        app_classes=("fastapi.FastAPI",),
        registration_decorators=_REGISTRATION_DECORATORS,
    )
