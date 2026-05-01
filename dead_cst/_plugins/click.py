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

from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst

from .._symbols import SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    GraphOp,
    ObserveContext,
    PluginContext,
    _payload_from,
    collect_module_imports,
    decorator_owner,
    find_handlers,
    matched_attr_call,
    simple_name,
    single_target_assignment,
)

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
class ClickPlugin:
    """Wire Click command and sub-group handlers through their owning group.

    For each module the plugin:

    1. Inspects ``from click import ...`` / ``import click`` to learn
       which local names refer to ``click.group`` (function), ``click.Group``
       (class), and the ``click`` module itself.
    2. Records every top-level function decorated with ``@click.group(...)``
       / ``@click.Group(...)`` (and aliased / module-prefixed forms) as a
       Click group instance whose name is the function's name.
    3. Also records every top-level assignment ``X = click.Group(...)``
       (including ``AnnAssign`` and aliased forms) as a Click group
       instance bound to ``X``.
    4. For every top-level function decorated ``@X.command(...)``,
       ``@X.group(...)``, or ``@X.result_callback(...)``, emits an edge
       ``X -> handler`` so the handler is reachable whenever ``X`` is.

    Pure per-file work: the ``@<known_group>.group(...)`` fixpoint
    operates on the file's CST only -- a chain of nested groups in
    one module converges in one observe pass.

    Limitations: only top-level definitions / assignments with a single
    ``Name`` target are detected. Factory-style groups
    (``def make_cli(): ...; cli = make_cli()``) and class-attribute
    groups (``self.cli = click.Group()``) are not handled; users can
    still keep those alive with explicit ``-e`` entrypoints.
    """

    name: str = "click"
    version: str = "1"

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        click_imports = collect_module_imports(ctx.module, "click", _GROUP_DECORATOR_NAMES)
        if not click_imports:
            return None
        instances = _find_instances(ctx.module, click_imports)
        if not instances:
            return None
        handlers = find_handlers(ctx.module, instances, _REGISTRATION_DECORATORS)
        if not handlers:
            return None

        decls_by_name = _decls_by_simple_name(ctx.payload.nodes)
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []
        for var_name, handler_names in handlers.items():
            for instance_decl in decls_by_name.get(var_name, []):
                for handler_name in handler_names:
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((instance_decl, handler_decl, SYNTHETIC_POSITION))
        if not edges:
            return None
        return _payload_from(edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()


def _find_instances(module: cst.Module, click_imports: dict[str, str]) -> set[str]:
    """Return the set of top-level names bound to a Click ``Group``.

    Detects three forms:

    * Functions decorated with ``@click.group(...)`` / ``@click.Group(...)``
      (and aliased / module-prefixed forms) -- the function's name.
    * Assignments ``X = click.Group(...)`` (and aliased / module-prefixed
      forms) -- the assignment target.
    * Functions decorated with ``@<known_group>.group(...)`` -- nested
      groups registered inline; resolved via fixpoint so a chain of
      ``@cli.group() -> @admin.group() -> ...`` is fully discovered
      regardless of source order.
    """
    instances: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, cst.FunctionDef):
            for dec in stmt.decorators:
                if matched_attr_call(dec.decorator, click_imports, _GROUP_DECORATOR_NAMES):
                    instances.add(stmt.name.value)
                    break
        elif isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                target_name, value = single_target_assignment(small)
                if target_name is None or not isinstance(value, cst.Call):
                    continue
                if matched_attr_call(
                    value.func, click_imports, _GROUP_CONSTRUCTOR_NAMES, unwrap_call=False
                ):
                    instances.add(target_name)

    # Fixpoint: ``@<known_group>.group(...)`` produces a new group; that new
    # group can then own further sub-groups. Iterate until stable.
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


def _decls_by_simple_name(nodes) -> dict[str, list[SymbolNode]]:
    out: dict[str, list[SymbolNode]] = {}
    for n in nodes:
        if n.type in ("class", "function", "variable", "import"):
            out.setdefault(simple_name(n.fqname), []).append(n)
    return out
