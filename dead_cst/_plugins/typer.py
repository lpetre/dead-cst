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
from typing import Iterable

from ._core import (
    AddEdge,
    GraphOp,
    PluginContext,
    collect_module_imports,
    find_call_assignments,
    find_handlers,
)

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

    Typer instances are not auto-marked as entrypoints. The expected
    wiring is ``[project.scripts]`` (``ProjectScriptsPlugin`` seeds
    ``module:app``) or an ``if __name__ == "__main__": app()`` block
    (``MainBlockPlugin``); both paths make the instance reachable, after
    which this plugin's edges keep its commands alive. Sub-typers added
    via ``app.add_typer(sub)`` are reached through the regular reference
    tracking on that call.

    Limitations: only top-level ``X = Typer(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return Typer()``) and class-attribute apps
    (``self.app = Typer()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints.
    """

    name: str = "typer"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Prefilter via the import graph: only files that actually import
        # ``typer`` can declare an app. Free because the resolver already
        # added ``[external dist] typer`` predecessors for them.
        candidate_paths = ctx.importers("typer")
        if not candidate_paths:
            return

        for path, module_node in ctx.base_modules():
            if path not in candidate_paths:
                continue
            module = ctx.parse(path)
            if module is None:
                continue
            typer_imports = collect_module_imports(module, "typer", _TYPER_TARGETS)
            if not typer_imports:
                continue
            instances = set(find_call_assignments(module, typer_imports, _TYPER_TARGETS))
            if not instances:
                continue
            handlers = find_handlers(module, instances, _REGISTRATION_DECORATORS)

            module_fqname = module_node.fqname
            for var_name in instances:
                instance_decls = ctx.find_declarations(f"{module_fqname}.{var_name}")
                if not instance_decls:
                    continue
                for instance_decl in instance_decls:
                    for handler_name in handlers.get(var_name, ()):
                        for handler_decl in ctx.find_declarations(
                            f"{module_fqname}.{handler_name}"
                        ):
                            yield AddEdge(instance_decl, handler_decl)
