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

from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..graph import SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    GraphOp,
    ObserveContext,
    PluginContext,
    collect_module_imports,
    decls_by_simple_name,
    find_call_assignments,
    find_handlers,
    make_payload,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload

# Attribute names ``Typer`` uses to register a callable. Matched as the
# rightmost attribute of ``@<instance>.<name>(...)``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"command", "callback"})

_TYPER_TARGETS: frozenset[str] = frozenset({"Typer"})


@dataclass
class TyperPlugin:
    """Wire Typer command and callback handlers through their app instance.

    For each module the plugin:

    1. Inspects ``from typer import ...`` / ``import typer`` to learn
       which local names refer to ``Typer``.
    2. Finds top-level assignments ``X = Typer(...)`` (including
       ``AnnAssign`` and aliased / module-prefixed forms) and records
       ``X`` as a Typer instance.
    3. For every top-level function decorated ``@X.command(...)`` or
       ``@X.callback(...)``, emits an edge ``X -> handler`` so the
       handler is reachable whenever ``X`` is.

    Pure per-file work: instance detection and decorator scanning are
    both file-local CST passes; the corresponding ``SymbolNode`` decls
    are looked up in this file's :class:`VisitorPayload`. Typer
    instances are not auto-marked as entrypoints; reachability flows
    through ``[project.scripts]`` or a ``__main__`` block as in the
    pre-refactor implementation.

    Limitations: only top-level ``X = Typer(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return Typer()``) and class-attribute apps
    (``self.app = Typer()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints.
    """

    name: str = "typer"
    version: int = 1777760307

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        typer_imports = collect_module_imports(ctx.module, "typer", _TYPER_TARGETS)
        if not typer_imports:
            return None
        instances = set(find_call_assignments(ctx.module, typer_imports, _TYPER_TARGETS))
        if not instances:
            return None
        handlers = find_handlers(ctx.module, instances, _REGISTRATION_DECORATORS)
        if not handlers:
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []
        for var_name, handler_names in handlers.items():
            for instance_decl in decls_by_name.get(var_name, []):
                for handler_name in handler_names:
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((instance_decl, handler_decl, SYNTHETIC_POSITION))
        if not edges:
            return None
        return make_payload(edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()
