"""Plugin: keep cyclopts command and default handlers alive.

Strategy: find top-level ``App()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.command(...)``
and ``@<instance>.default(...)`` decorator. Cyclopts apps are *not*
seeded as entrypoints; reachability is expected to flow through
``[project.scripts]`` (handled by :class:`ProjectScriptsPlugin`) or an
``if __name__ == "__main__": app()`` block (handled by
:class:`MainBlockPlugin`). Sub-apps attached via ``app.command(sub)``
are kept alive through the ordinary reference tracked on that call.

This routes ``why-alive`` chains through the cyclopts app variable
users recognize ("alive because it's a command on ``app``") and lets a
sub-app that's never registered surface as dead code, mirroring the
behavior of :class:`TyperPlugin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from libcst.metadata import CodeRange

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

# Attribute names a cyclopts ``App`` uses to register a callable.
# Matched as the rightmost attribute of ``@<instance>.<name>(...)``.
# ``command`` registers a subcommand; ``default`` registers the
# no-subcommand handler.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"command", "default"})

_CYCLOPTS_TARGETS: frozenset[str] = frozenset({"App"})


@dataclass
class CycloptsPlugin:
    """Wire cyclopts command and default handlers through their app instance.

    For each module the plugin:

    1. Inspects ``from cyclopts import ...`` / ``import cyclopts`` to
       learn which local names refer to ``App``.
    2. Finds top-level assignments ``X = App(...)`` (including
       ``AnnAssign`` and aliased / module-prefixed forms) and records
       ``X`` as a cyclopts instance.
    3. For every top-level function decorated ``@X.command(...)`` or
       ``@X.default(...)``, emits an edge ``X -> handler`` so the
       handler is reachable whenever ``X`` is.

    Pure per-file work: instance detection and decorator scanning are
    both file-local CST passes; the corresponding ``SymbolNode`` decls
    are looked up in this file's :class:`VisitorPayload`. Cyclopts
    instances are not auto-marked as entrypoints; reachability flows
    through ``[project.scripts]`` or a ``__main__`` block as in the
    Typer plugin.

    Limitations: only top-level ``X = App(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return App()``) and class-attribute apps
    (``self.app = App()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints. Nested-attribute decorators
    such as ``@app.meta.default`` are not recognized -- ``app.meta`` is
    not itself a tracked instance.
    """

    name: str = "cyclopts"
    version: int = 1778020575

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        cyclopts_imports = collect_module_imports(ctx.module, "cyclopts", _CYCLOPTS_TARGETS)
        if not cyclopts_imports:
            return None
        instances = set(find_call_assignments(ctx.module, cyclopts_imports, _CYCLOPTS_TARGETS))
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
