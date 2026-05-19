"""Plugin: keep Typer command and callback handlers alive."""

from __future__ import annotations

from ..plugins.decl_shapes import DispatchAppPlugin


def typer_plugin() -> DispatchAppPlugin:
    """Wire Typer command and callback handlers through their app instance.

    Typer apps are not seeded as entrypoints; reachability flows
    through ``[project.scripts]`` (:class:`ProjectScriptsPlugin`) or an
    ``if __name__ == "__main__": app()`` block
    (:class:`MainBlockPlugin`). Sub-typers attached via
    ``app.add_typer(sub)`` are kept alive through the ordinary
    reference on that call.

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
