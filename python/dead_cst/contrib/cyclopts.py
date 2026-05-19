"""Plugin: keep cyclopts command and default handlers alive."""

from __future__ import annotations

from ..plugins.decl_shapes import DispatchAppPlugin


def cyclopts_plugin() -> DispatchAppPlugin:
    """Wire cyclopts command and default handlers through their app instance.

    Cyclopts apps are not seeded as entrypoints; reachability flows
    through ``[project.scripts]`` (:class:`ProjectScriptsPlugin`) or an
    ``if __name__ == "__main__": app()`` block
    (:class:`MainBlockPlugin`). Sub-apps attached via
    ``app.command(sub)`` are kept alive through the ordinary reference
    on that call.

    Limitations: only top-level ``X = App(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return App()``) and class-attribute apps
    (``self.app = App()``) are not handled. Nested-attribute decorators
    such as ``@app.meta.default`` are not recognized -- ``app.meta`` is
    not itself a tracked instance.
    """
    return DispatchAppPlugin(
        marker_prefix="cyclopts",
        app_classes=("cyclopts.App",),
        registration_decorators=frozenset({"command", "default"}),
        seed_as_entrypoint=False,
    )
