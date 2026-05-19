"""Plugin: keep FastMCP servers, tool handlers, and lifecycle hooks alive."""

from __future__ import annotations

from ..plugins.decl_shapes import DispatchAppPlugin

# Attribute names FastMCP uses to register a callable. Matched as the
# rightmost attribute of ``@<instance>.<name>(...)``. Both bare-decorator
# (``@mcp.tool``) and call-decorator (``@mcp.tool()``) forms are picked
# up by ``find_handlers``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset(
    {
        "tool",
        "resource",
        "prompt",
        "completion",
    }
)


def fastmcp_plugin() -> DispatchAppPlugin:
    """Mark FastMCP servers as entrypoints and wire handlers through them.

    Handles direct (``mcp = FastMCP(...)``), aliased
    (``from fastmcp import FastMCP as M; mcp = M(...)``),
    module-prefixed (``import fastmcp; mcp = fastmcp.FastMCP(...)``),
    and factory-style (``mcp = create_server()``) construction.

    Only the ``fastmcp`` import path is recognized. Users on the
    Anthropic MCP SDK's compatibility layer
    (``from mcp.server.fastmcp import FastMCP``) can keep their handlers
    alive with explicit ``-e`` entrypoints or a project-local plugin.
    """
    return DispatchAppPlugin(
        marker_prefix="fastmcp",
        app_classes=("fastmcp.FastMCP",),
        registration_decorators=_REGISTRATION_DECORATORS,
    )
