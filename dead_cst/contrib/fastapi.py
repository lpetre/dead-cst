"""Plugin: keep FastAPI route handlers and lifecycle hooks alive.

Strategy: every FastAPI / APIRouter instance we want to wire up is a
top-level variable that the analyzer has already linked back to the
``fastapi`` import -- whether the assignment is the literal
``X = FastAPI()``, the aliased ``X = F()`` after ``from fastapi import
FastAPI as F``, or the factory form ``X = create_app()`` whose body
returns ``FastAPI(...)``. The plugin reuses those reference edges via
the factory-aware :class:`DispatchAppPlugin` base:

1. Direct shape (``X = FastAPI(...)`` / ``X = APIRouter(...)``,
   ``X = fastapi.FastAPI(...)``, etc.) is recognized syntactically.
   This is unambiguous, so it gives the kind directly even for
   ``import fastapi`` forms where the graph alone can't distinguish
   the two classes.
2. Indirect shape (any variable decorated by ``@X.<route_verb>(...)``)
   produces a ``<fastapi-pending>:<X.fqname>`` marker plus the
   ``X -> handler`` edges. The per-package finalize pass walks the
   graph forward from each pending marker and promotes ``FastAPI``
   instances to entrypoints.
3. Factory functions / classes whose body constructs a FastAPI /
   APIRouter instance are tagged with a
   ``<fastapi-factory>:<kind>:<decl.fqname>`` marker. This lets a
   cross-package consumer's pending-variable walk hit a discriminator
   even when the factory uses the ``import fastapi; fastapi.FastAPI()``
   form -- the attribute access lands as a bare
   ``[external dist] fastapi`` edge after :func:`resolve_edges` drops
   the ``decl='FastAPI'`` half, so the graph alone can't tell FastAPI
   from APIRouter on the downstream walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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

# Classes whose instances we treat specially. Value records whether the
# instance should be seeded as an entrypoint.
_INSTANCE_KINDS: Mapping[str, bool] = {
    "FastAPI": True,  # uvicorn loads ``module:app`` -- always an entrypoint
    "APIRouter": False,  # only alive if reached via ``include_router``
}


@dataclass
class FastAPIPlugin(DispatchAppPlugin):
    """Mark FastAPI apps as entrypoints and wire route handlers through them.

    Concrete configuration of the factory-aware
    :class:`DispatchAppPlugin` shape:

    * Direct ``X = FastAPI(...)`` / ``X = APIRouter(...)`` assignments
      handle module-prefixed forms like
      ``import fastapi; X = fastapi.FastAPI()`` that pure graph
      reachability cannot distinguish (the variable's edge goes
      straight to the ``fastapi`` synthetic with no intermediate
      ``FastAPI`` vs ``APIRouter`` import to discriminate). Direct
      FastAPI hits get a ``<fastapi-app>:`` synthetic entrypoint plus
      an edge pointing at the variable; routers do not.
    * ``@<X>.<verb>(...)`` decorators produce ``X -> handler`` edges
      unconditionally; whether ``X`` is reachable depends on
      classification.
    * Variables decorated but not directly classified get a
      ``<fastapi-pending>:`` marker. Finalize walks forward from each
      pending variable until it hits a discriminator (a
      ``from fastapi import FastAPI``-style import node or a factory
      marker), classifies the variable, and -- for ``FastAPI``
      instances -- emits a ``<fastapi-app>:`` synthetic entrypoint plus
      an edge to the variable. Routers and unclassified variables stay
      as-is, so an ``APIRouter`` that nothing ``include_router``s
      remains dead.
    * Top-level decls whose body constructs a FastAPI / APIRouter
      instance get a ``<fastapi-factory>:`` marker. This discriminator
      survives cross-package walks where the external edge would
      otherwise lose ``decl='FastAPI'`` info.
    """

    name: str = "fastapi"
    version: int = 1778973600
    app_module: str = "fastapi"
    registration_decorators: frozenset[str] = _REGISTRATION_DECORATORS
    instance_kinds: Mapping[str, bool] = field(default_factory=lambda: dict(_INSTANCE_KINDS))
