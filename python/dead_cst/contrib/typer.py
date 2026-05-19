"""Plugin: keep Typer command and callback handlers alive.

Strategy: find top-level ``Typer()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.command(...)``
and ``@<instance>.callback(...)`` decorator. Typer instances are *not*
seeded as entrypoints (``seed_as_entrypoint=False``); reachability is
expected to flow through ``[project.scripts]`` (handled by
:class:`ProjectScriptsPlugin`) or an ``if __name__ == "__main__":
app()`` block (handled by :class:`MainBlockPlugin`). Sub-typers
attached via ``app.add_typer(sub)`` are kept alive through the
ordinary reference tracked on that call.

This routes ``why-alive`` chains through the Typer app variable users
recognize ("alive because it's a command on ``app``") and lets a
sub-typer that's never ``add_typer``'d surface as dead code, mirroring
the behavior of :func:`fastapi_plugin` for ``APIRouter``.
"""

from __future__ import annotations

from ..plugins.decl_shapes import DispatchAppPlugin


def typer_plugin() -> DispatchAppPlugin:
    """Wire Typer command and callback handlers through their app instance.

    Limitations: only top-level ``X = Typer(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return Typer()``) and class-attribute apps
    (``self.app = Typer()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints.
    """
    return DispatchAppPlugin(
        marker_prefix="typer",
        app_classes=("typer.Typer",),
        registration_decorators=frozenset({"command", "callback"}),
        seed_as_entrypoint=False,
    )
