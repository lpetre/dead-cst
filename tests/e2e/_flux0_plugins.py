"""Project-specific plugin prototypes for the flux0 e2e tests.

These plugins close flux0's two ``importlib``-driven blind spots
using only the public ``dead_cst`` API plus a couple of CST helpers
that live in ``dead_cst._plugins._core`` today (and are flagged
inline as candidates for promotion to the public surface).

Both plugins parse the analyzed source's CST to learn which modules
to revive -- they do not re-execute ``importlib`` /
``pkgutil.iter_modules`` at analysis time. Discovery happens by
inspecting the CST that dead-cst's visitor already parsed, then
matching on the assembled symbol graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst

from dead_cst import AddEdge, AddNode, GraphOp, PluginContext
from dead_cst._plugins import synthetic_node
from dead_cst._plugins._core import (
    collect_module_imports,
    matched_attr_call,
    single_target_assignment,
)
from dead_cst._symbols import SymbolNode

# ``click.group`` (function) and ``click.Group`` (class) both produce
# a Group when used as a decorator; only the class produces one when
# used as a constructor (``X = click.Group(...)``).
_CLICK_GROUP_DECORATORS: frozenset[str] = frozenset({"group", "Group"})
_CLICK_GROUP_CONSTRUCTORS: frozenset[str] = frozenset({"Group"})


def _find_click_group_names(module: cst.Module) -> set[str]:
    """Return top-level names bound to a Click ``Group`` in ``module``.

    Mirrors ``flux0_cli.main.register_commands``' runtime check
    ``isinstance(cmd, click.Group)`` by recognising the two static
    forms that produce a Group:

    * ``@click.group()`` / ``@click.Group()`` decorating a function --
      the function's name is rebound to the Group.
    * ``X = click.Group(...)`` -- ``X`` holds a Group.

    Reuses the same primitives the builtin ``ClickPlugin`` uses.
    Those helpers live in ``dead_cst._plugins._core`` and are not
    exported today; reaching into ``_core`` from a user plugin is a
    documented gap.
    """
    click_imports = collect_module_imports(module, "click", _CLICK_GROUP_DECORATORS)
    if not click_imports:
        return set()

    names: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, cst.FunctionDef):
            for dec in stmt.decorators:
                if matched_attr_call(dec.decorator, click_imports, _CLICK_GROUP_DECORATORS):
                    names.add(stmt.name.value)
                    break
        elif isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                target, value = single_target_assignment(small)
                if target is None or not isinstance(value, cst.Call):
                    continue
                if matched_attr_call(
                    value.func, click_imports, _CLICK_GROUP_CONSTRUCTORS, unwrap_call=False
                ):
                    names.add(target)
    return names


def _module_surface(ctx: PluginContext, fqname: str) -> list[SymbolNode]:
    """Return ``fqname``'s module + every node under it.

    Models ``importlib.import_module(fqname)``: the module's whole
    top-level surface (decls, imports, side-effecting assignments)
    runs at import time, plus every transitively nested submodule
    that the package's own imports bring in.
    """
    mod = ctx.find_module(fqname)
    if mod is None:
        return []
    prefix = fqname + "."
    return [n for n in ctx.base_nodes() if n.fqname == fqname or n.fqname.startswith(prefix)]


def _read_string_list(module: cst.Module, var_name: str) -> list[str]:
    """Extract ``<var_name> = [<str-literal>, ...]`` (or tuple) at module level.

    Returns the string literals in declaration order, or ``[]`` if no
    such assignment exists or the RHS isn't a static list/tuple of
    strings. Non-literal entries are dropped silently -- the runtime
    consumer would just fail on them; we just don't pretend to know
    what they resolve to.
    """
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target, value = single_target_assignment(small)
            if target != var_name:
                continue
            if not isinstance(value, (cst.List, cst.Tuple)):
                return []
            out: list[str] = []
            for elt in value.elements:
                if not isinstance(elt, cst.Element):
                    continue
                inner = elt.value
                if isinstance(inner, cst.SimpleString):
                    out.append(inner.evaluated_value)
            return out
    return []


@dataclass
class Flux0CliCommandsPlugin:
    """Mirror ``flux0_cli/main.py:register_commands``.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/cli/src/flux0_cli/main.py#L61

    The runtime loop iterates every module under ``flux0_cli.cmds``
    and registers each top-level attribute that ``isinstance``-checks
    as a ``click.Group``. We mirror that: per submodule, parse the
    CST, find every top-level Click Group, and seed the analyzer's
    reachability walk from it. The builtin ``ClickPlugin`` (when
    composed with this one) then carries reachability through any
    ``@<group>.command(...)`` handlers; flux0's custom-decorator
    handlers (``@get_options(group, ...)``) intentionally stay
    unrecognised -- that's a separate static-analysis blind spot.
    """

    package: str = "flux0_cli.cmds"
    name: str = "flux0_cli_cmds"
    version: str = "1"

    def observe(self, ctx) -> None:
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        prefix = self.package + "."
        targets: list[SymbolNode] = []
        for mod_node in [
            n for n in ctx.base_nodes() if n.type == "module" and n.fqname.startswith(prefix)
        ]:
            cst_module = ctx.parse(mod_node.path)
            if cst_module is None:
                continue
            for group_name in _find_click_group_names(cst_module):
                targets.extend(ctx.find_declarations(f"{mod_node.fqname}.{group_name}"))

        if not targets:
            return
        synth = synthetic_node(f"<{self.name}>", ctx.base)
        yield AddNode(synth, entrypoint=True)
        for t in targets:
            yield AddEdge(synth, t)


@dataclass
class Flux0InternalModulesPlugin:
    """Mirror ``flux0_server.main.INTERNAL_MODULES``.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/server/src/flux0_server/main.py#L51

    flux0_server lists modules to import by dotted name in
    ``INTERNAL_MODULES``, then ``importlib``-loads each one and calls
    its ``init_module`` / ``shutdown_module``. The plugin reads the
    list literal from the CST (so an upstream rename or addition
    automatically flows through), resolves each fqname to a module,
    and treats the whole module surface as alive.
    """

    module_fqname: str = "flux0_server.main"
    variable_name: str = "INTERNAL_MODULES"
    name: str = "flux0_internal_modules"
    version: str = "1"

    def observe(self, ctx) -> None:
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        owner = ctx.find_module(self.module_fqname)
        if owner is None:
            return
        cst_module = ctx.parse(owner.path)
        if cst_module is None:
            return
        fqnames = _read_string_list(cst_module, self.variable_name)
        if not fqnames:
            return

        targets: list[SymbolNode] = []
        for fqname in fqnames:
            targets.extend(_module_surface(ctx, fqname))
        if not targets:
            return

        synth = synthetic_node(f"<{self.name}>", owner.path)
        yield AddNode(synth, entrypoint=True)
        for t in targets:
            yield AddEdge(synth, t)
