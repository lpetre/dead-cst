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
from typing import Iterable

import libcst as cst

from ._core import (
    AddEdge,
    GraphOp,
    PluginContext,
    collect_module_imports,
    decorator_owner,
    find_handlers,
    require_resolved_dep,
    matched_attr_call,
    single_target_assignment,
)

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

    Click groups are not auto-marked as entrypoints. The expected wiring
    is ``[project.scripts]`` (``ProjectScriptsPlugin`` seeds
    ``module:cli``) or an ``if __name__ == "__main__": cli()`` block
    (``MainBlockPlugin``); both paths make the group reachable, after
    which this plugin's edges keep its commands alive. Sub-groups added
    via ``cli.add_command(sub)`` are reached through the regular
    reference tracking on that call.

    Limitations: only top-level definitions / assignments with a single
    ``Name`` target are detected. Factory-style groups
    (``def make_cli(): ...; cli = make_cli()``) and class-attribute
    groups (``self.cli = click.Group()``) are not handled; users can
    still keep those alive with explicit ``-e`` entrypoints.
    """

    name: str = "click"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        if require_resolved_dep(ctx, "click") is None:
            return
        candidate_paths = ctx.importers("click")

        for path, module_node in ctx.base_modules():
            if path not in candidate_paths:
                continue
            module = ctx.parse(path)
            if module is None:
                continue
            click_imports = collect_module_imports(module, "click", _GROUP_DECORATOR_NAMES)
            if not click_imports:
                continue
            instances = _find_instances(module, click_imports)
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
