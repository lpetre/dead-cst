"""Plugin: keep Click command and sub-group handlers alive.

Strategy: find top-level Click ``Group`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.command(...)``,
``@<instance>.group(...)``, and ``@<instance>.result_callback(...)``
decorator. Click groups are *not* seeded as entrypoints; reachability is
expected to flow through ``[project.scripts]`` (handled by
:class:`ProjectScriptsPlugin`) or an ``if __name__ == "__main__": cli()``
block (handled by :class:`MainBlockPlugin`). Sub-groups attached via
``cli.add_command(sub)`` are kept alive through the ordinary reference
tracked on that call.

Two top-level forms produce a Click ``Group``:

* ``@click.group()`` / ``@click.group`` decorating a function -- the
  function's name is rebound to the resulting ``Group`` (the common
  Click idiom).
* ``X = click.Group(...)`` -- an explicit constructor call, mirroring
  the Typer / FastAPI pattern.

A standalone ``@click.command()`` decorator on a top-level function does
not by itself create reachability; such commands are expected to be
listed in ``[project.scripts]``, invoked from a ``__main__`` block, or
attached to a group via ``add_command``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import libcst as cst
from libcst.metadata import CodeRange

from .._symbols import SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    ObserveContext,
    collect_module_imports,
    decls_by_simple_name,
    decorator_owner,
    find_handlers,
    make_payload,
)
from .decl_shapes import DecoratedDeclPlugin

if TYPE_CHECKING:
    from .._visitor import VisitorPayload

# Attribute names a Click ``Group`` uses to register a callable. Matched
# as the rightmost attribute of ``@<instance>.<name>(...)``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"command", "group", "result_callback"})

_SUBGROUP_DECORATOR: frozenset[str] = frozenset({"group"})

# Names (as imported from ``click``) that produce a ``Group`` when used
# as a decorator on a function. ``click.group`` is the function form;
# ``click.Group`` is the class -- the latter can also be used as a
# decorator (``@click.Group()``) but is much rarer.
_GROUP_DECORATOR_NAMES: frozenset[str] = frozenset({"group", "Group"})

# Names (as imported from ``click``) that produce a ``Group`` when called
# as a constructor: ``X = click.Group(...)``. ``click.group(...)`` outside
# of a decorator context also returns a ``Group``, but the analyzer can't
# tell ``X = click.group(...)`` apart from ``X = some_helper.group(...)``
# without flow analysis, so we limit the constructor form to the class.
_GROUP_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"Group"})


@dataclass
class ClickPlugin(DecoratedDeclPlugin):
    """Wire Click command and sub-group handlers through their owning group.

    Inherits the ``@click.group()`` / ``X = click.Group(...)`` discovery
    from :class:`DecoratedDeclPlugin` (configured for the ``click``
    module + the two name sets above). Adds:

    * a fixpoint pass over :meth:`_find_names` so ``@<known_group>.group(...)``
      registrations register their inner functions as new groups too;
    * an :meth:`observe` override that emits ``instance -> handler``
      edges for every top-level ``@<group>.command(...)`` /
      ``@<group>.group(...)`` / ``@<group>.result_callback(...)`` instead
      of seeding entrypoint synthetics. Click groups are reached through
      ``[project.scripts]`` / ``__main__`` / ``add_command``, not through
      the discovery itself.

    Limitations: only top-level definitions / assignments with a single
    ``Name`` target are detected. Factory-style groups
    (``def make_cli(): ...; cli = make_cli()``) and class-attribute
    groups (``self.cli = click.Group()``) are not handled; users can
    still keep those alive with explicit ``-e`` entrypoints.
    """

    name: str = "click"
    version: str = "1"
    decorator_module: str = "click"
    decorator_names: frozenset[str] = _GROUP_DECORATOR_NAMES
    constructor_names: frozenset[str] = _GROUP_CONSTRUCTOR_NAMES

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        click_imports = collect_module_imports(
            ctx.module, self.decorator_module, self.decorator_names
        )
        if not click_imports:
            return None
        instances = self._find_names(ctx.module, click_imports)
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

    def _find_names(self, module: cst.Module, imports: dict[str, str]) -> set[str]:
        """Find Click groups, including nested ``@<known_group>.group(...)``.

        First pass: the base's decorator + constructor scan. Second
        pass (fixpoint): ``@<known_group>.group(...)`` produces a new
        group, which can then own further sub-groups. Iterate until
        stable so a chain of ``@cli.group() -> @admin.group() -> ...``
        is fully discovered regardless of source order.
        """
        instances = super()._find_names(module, imports)
        changed = True
        while changed:
            changed = False
            for stmt in module.body:
                if not isinstance(stmt, cst.FunctionDef):
                    continue
                if stmt.name.value in instances:
                    continue
                for dec in stmt.decorators:
                    owner = decorator_owner(dec.decorator, _SUBGROUP_DECORATOR)
                    if owner is not None and owner in instances:
                        instances.add(stmt.name.value)
                        changed = True
                        break
        return instances
