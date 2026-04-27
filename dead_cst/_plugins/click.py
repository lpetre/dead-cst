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

from ._core import AddEdge, GraphOp, PluginContext

# Attribute names a Click ``Group`` uses to register a callable. Matched
# as the rightmost attribute of ``@<instance>.<name>(...)``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"command", "group", "result_callback"})

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
        # Prefilter via the import graph: only files that actually import
        # ``click`` can declare a group. Free because the resolver already
        # added ``[external dist] click`` predecessors for them.
        candidate_paths = ctx.importers("click")
        if not candidate_paths:
            return

        for path, module_node in ctx.base_modules():
            if path not in candidate_paths:
                continue
            module = ctx.parse(path)
            if module is None:
                continue
            click_imports = _collect_click_imports(module)
            if not click_imports:
                continue
            instances = _find_instances(module, click_imports)
            if not instances:
                continue
            handlers = _find_handlers(module, instances)

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


def _collect_click_imports(module: cst.Module) -> dict[str, str]:
    """Return ``{local_name: target}`` for names imported from ``click``.

    ``target`` is one of ``"group"``, ``"Group"``, or ``"<module>"`` (the
    whole ``click`` package, for ``import click``).
    """
    bindings: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                if not _is_click_module(small):
                    continue
                if isinstance(small.names, cst.ImportStar):
                    continue
                for alias in small.names:
                    target = alias.name.value if isinstance(alias.name, cst.Name) else None
                    if target not in _GROUP_DECORATOR_NAMES:
                        continue
                    local = alias.asname.name.value if alias.asname else target
                    if isinstance(local, str):
                        bindings[local] = target
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    if not _is_name(alias.name, "click"):
                        continue
                    local = alias.asname.name.value if alias.asname else "click"
                    if isinstance(local, str):
                        bindings[local] = "<module>"
    return bindings


def _is_click_module(node: cst.ImportFrom) -> bool:
    if node.relative:
        return False
    return _is_name(node.module, "click")


def _is_name(node: cst.CSTNode | None, value: str) -> bool:
    return isinstance(node, cst.Name) and node.value == value


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
                if _is_group_decorator(dec.decorator, click_imports):
                    instances.add(stmt.name.value)
                    break
        elif isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                target_name, value = _single_target_assignment(small)
                if target_name is None or not isinstance(value, cst.Call):
                    continue
                if _is_group_constructor(value.func, click_imports):
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
                owner = _registration_decorator_owner(dec.decorator, attr="group")
                if owner is not None and owner in instances:
                    instances.add(stmt.name.value)
                    changed = True
                    break
    return instances


def _is_group_decorator(expr: cst.BaseExpression, click_imports: dict[str, str]) -> bool:
    """Return ``True`` if ``expr`` is ``click.group`` / ``click.Group`` (or
    aliased / module-prefixed) used as a decorator (with or without a call)."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if isinstance(expr, cst.Name):
        return click_imports.get(expr.value) in _GROUP_DECORATOR_NAMES
    if isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
        if click_imports.get(expr.value.value) == "<module>":
            return expr.attr.value in _GROUP_DECORATOR_NAMES
    return False


def _is_group_constructor(func: cst.BaseExpression, click_imports: dict[str, str]) -> bool:
    """Return ``True`` if ``func`` denotes a Click ``Group`` constructor.

    Limited to the class form (``click.Group``); the function form
    (``click.group``) is recognized only as a decorator, since
    ``X = click.group(...)`` typed on a single line is rare and the class
    form is the unambiguous spelling for an explicit constructor call.
    """
    if isinstance(func, cst.Name):
        return click_imports.get(func.value) in _GROUP_CONSTRUCTOR_NAMES
    if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
        if click_imports.get(func.value.value) == "<module>":
            return func.attr.value in _GROUP_CONSTRUCTOR_NAMES
    return False


def _single_target_assignment(
    stmt: cst.BaseSmallStatement,
) -> tuple[str | None, cst.BaseExpression | None]:
    """Extract ``(name, rhs)`` for ``X = ...`` / ``X: T = ...``; else ``(None, None)``."""
    if isinstance(stmt, cst.Assign):
        if len(stmt.targets) != 1:
            return None, None
        target = stmt.targets[0].target
        if isinstance(target, cst.Name):
            return target.value, stmt.value
    elif isinstance(stmt, cst.AnnAssign):
        if isinstance(stmt.target, cst.Name) and stmt.value is not None:
            return stmt.target.value, stmt.value
    return None, None


def _find_handlers(module: cst.Module, instance_vars: set[str]) -> dict[str, list[str]]:
    """Return ``{instance_var: [handler_func_name, ...]}`` for decorated handlers."""
    handlers: dict[str, list[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            owner = _registration_decorator_owner(dec.decorator)
            if owner is None or owner not in instance_vars:
                continue
            handlers.setdefault(owner, []).append(stmt.name.value)
            break
    return handlers


def _registration_decorator_owner(
    expr: cst.BaseExpression, *, attr: str | None = None
) -> str | None:
    """For ``@X.command(...)`` / ``@X.command`` return ``"X"`` (only when the
    rightmost attribute is a known Click registration name and ``X`` is a
    bare ``Name``). Returns ``None`` otherwise.

    When ``attr`` is given, only that specific attribute name is matched
    (e.g. ``attr="group"`` for sub-group discovery)."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if not isinstance(expr, cst.Attribute):
        return None
    if attr is None:
        if expr.attr.value not in _REGISTRATION_DECORATORS:
            return None
    elif expr.attr.value != attr:
        return None
    if not isinstance(expr.value, cst.Name):
        return None
    return expr.value.value
