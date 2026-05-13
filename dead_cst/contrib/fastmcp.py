"""Plugin: keep FastMCP servers, tool handlers, and lifecycle hooks alive.

Strategy: every FastMCP server instance we want to wire up is a
top-level variable that the analyzer has already linked back to the
``fastmcp`` import -- whether the assignment is the literal
``mcp = FastMCP()``, the aliased ``mcp = M()`` after
``from fastmcp import FastMCP as M``, or the factory form
``mcp = create_server()`` whose body returns ``FastMCP(...)``. The
plugin reuses those reference edges via the factory-aware
:class:`DispatchAppPlugin` base:

1. Direct shape (``mcp = FastMCP(...)``, ``mcp = fastmcp.FastMCP(...)``,
   etc.) is recognized syntactically. Each direct hit is seeded as an
   entrypoint -- the ``fastmcp`` CLI loads ``module:mcp`` by import
   path the same way ``uvicorn`` loads ``module:app``, so a FastMCP
   server is the framework-visible entrypoint.
2. Indirect shape (any variable decorated by ``@mcp.tool(...)`` /
   ``@mcp.resource(...)`` / etc.) produces a ``<fastmcp-pending>:``
   marker plus the ``mcp -> handler`` edges. The per-package finalize
   pass walks the graph forward from each pending marker and promotes
   FastMCP instances to entrypoints once the discriminating import
   node (or factory marker) is reached.
3. Factory functions / classes whose body constructs a ``FastMCP``
   instance are tagged with a ``<fastmcp-factory>:FastMCP:<decl.fqname>``
   marker so a cross-package consumer's pending-variable walk hits a
   discriminator even when the factory uses the
   ``import fastmcp; fastmcp.FastMCP()`` form -- the attribute access
   lands as a bare ``[external dist] fastmcp`` edge after
   :func:`resolve_edges` drops the ``decl='FastMCP'`` half.

This mirrors :class:`FastAPIPlugin`'s shape; the only difference is
that FastMCP has a single kind today, so ``instance_kinds`` is a
one-entry map rather than the FastAPI / Flask two-kind map (no
``Router`` / ``Blueprint`` peer to disambiguate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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

# Classes whose instances we treat as auto-entrypoints. FastMCP has a
# single server class today; we keep the mapping for parity with the
# FastAPI / Flask shape so adding a future "Router"-style peer is just
# a dict entry.
_INSTANCE_KINDS: Mapping[str, bool] = {
    "FastMCP": True,  # ``fastmcp run module:mcp`` -- always an entrypoint
}


@dataclass
class FastMCPPlugin(DispatchAppPlugin):
    """Mark FastMCP servers as entrypoints and wire handlers through them.

    Concrete configuration of the factory-aware
    :class:`DispatchAppPlugin` shape:

    * Direct ``X = FastMCP(...)`` assignments get a ``<fastmcp-app>:``
      synthetic entrypoint plus an edge pointing at the variable.
    * ``@<X>.tool(...)`` / ``@<X>.resource(...)`` / ``@<X>.prompt(...)``
      / ``@<X>.completion(...)`` decorators produce ``X -> handler``
      edges unconditionally; whether ``X`` is reachable depends on
      classification.
    * Variables decorated but not directly classified get a
      ``<fastmcp-pending>:`` marker. Finalize walks forward from each
      pending variable until it hits a discriminator (a
      ``from fastmcp import FastMCP``-style import node or a factory
      marker), classifies the variable, and emits a ``<fastmcp-app>:``
      synthetic entrypoint plus an edge to the variable.
    * Top-level decls whose body constructs a ``FastMCP`` instance get
      a ``<fastmcp-factory>:`` marker. This discriminator survives
      cross-package walks where the external edge would otherwise lose
      ``decl='FastMCP'`` info.

    Only the ``fastmcp`` import path is recognized. Users on the
    Anthropic MCP SDK's compatibility layer
    (``from mcp.server.fastmcp import FastMCP``) can keep their handlers
    alive with explicit ``-e`` entrypoints or a project-local plugin.
    """

    name: str = "fastmcp"
    version: int = 1778671880
    app_module: str = "fastmcp"
    registration_decorators: frozenset[str] = _REGISTRATION_DECORATORS
    instance_kinds: Mapping[str, bool] = field(default_factory=lambda: dict(_INSTANCE_KINDS))
