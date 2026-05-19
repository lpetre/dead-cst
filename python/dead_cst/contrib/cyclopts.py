"""Plugin: keep cyclopts command and default handlers alive.

Strategy: find top-level ``App()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.command(...)``
and ``@<instance>.default(...)`` decorator. Cyclopts apps are *not*
seeded as entrypoints (``seed_as_entrypoint=False``); reachability is
expected to flow through ``[project.scripts]`` (handled by
:class:`ProjectScriptsPlugin`) or an ``if __name__ == "__main__":
app()`` block (handled by :class:`MainBlockPlugin`). Sub-apps attached
via ``app.command(sub)`` are kept alive through the ordinary reference
tracked on that call.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..plugins.decl_shapes import DispatchAppPlugin


@dataclass
class CycloptsPlugin(DispatchAppPlugin):
    """Wire cyclopts command and default handlers through their app instance.

    Limitations: only top-level ``X = App(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return App()``) and class-attribute apps
    (``self.app = App()``) are not handled. Nested-attribute decorators
    such as ``@app.meta.default`` are not recognized -- ``app.meta`` is
    not itself a tracked instance.
    """

    name: str = "cyclopts"
    version: int = 1778020575
    app_classes: tuple[str, ...] = ("cyclopts.App",)
    registration_decorators: frozenset[str] = frozenset({"command", "default"})
    seed_as_entrypoint: bool = False
