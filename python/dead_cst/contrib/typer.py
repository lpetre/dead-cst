"""Plugin: keep Typer command and callback handlers alive.

Strategy: find top-level ``Typer()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.command(...)``
and ``@<instance>.callback(...)`` decorator. Typer instances are *not*
seeded as entrypoints; reachability is expected to flow through
``[project.scripts]`` (handled by :class:`ProjectScriptsPlugin`) or an
``if __name__ == "__main__": app()`` block (handled by
:class:`MainBlockPlugin`). Sub-typers attached via ``app.add_typer(sub)``
are kept alive through the ordinary reference tracked on that call.

This routes ``why-alive`` chains through the Typer app variable users
recognize ("alive because it's a command on ``app``") and lets a
sub-typer that's never ``add_typer``'d surface as dead code, mirroring
the behavior of :class:`FastAPIPlugin` for ``APIRouter``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..plugins.decl_shapes import DispatchAppPlugin


@dataclass
class TyperPlugin(DispatchAppPlugin):
    """Wire Typer command and callback handlers through their app instance.

    Limitations: only top-level ``X = Typer(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return Typer()``) and class-attribute apps
    (``self.app = Typer()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints.
    """

    name: str = "typer"
    version: int = 1777760307
    app_module: str = "typer"
    constructor_targets: frozenset[str] = frozenset({"Typer"})
    registration_decorators: frozenset[str] = frozenset({"command", "callback"})
