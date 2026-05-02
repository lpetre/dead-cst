"""Project-specific plugin prototypes for the flux0 e2e tests.

This module also sketches two abstract plugin shapes that generalize
the patterns flux0 happens to need; the project-specific subclasses
are 2-3 lines of configuration each. Both bases use only public
``dead_cst`` helpers + a couple of CST primitives that live in
``dead_cst._plugins._core`` (and are flagged inline as candidates
for promotion to the public surface).

  DecoratedDeclPlugin
    "Find decorated decls in files matching a search path."
    Pure observe-time. Per-file CST inspection turns directly into
    file-local entrypoint synthetics via ``entrypoint_payload``.

  LiteralListPlugin
    "Read a specific symbol's literal value and treat each fqname
    inside it as alive."
    Observe captures the literal once, in the file that owns the
    symbol; finalize resolves the fqnames against the assembled
    graph (because they point at other files). The two phases are
    necessary because cross-file targets aren't constructible until
    after the visitor has built every base's nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst

from dead_cst import AddEdge, GraphOp, NodeFlags, PluginContext
from dead_cst._plugins import (
    SYNTHETIC_POSITION,
    ObserveContext,
    entrypoint_payload,
    synthetic_node,
)
from dead_cst._plugins._core import (
    _payload_from,
    collect_module_imports,
    matched_attr_call,
    simple_name,
    single_target_assignment,
)
from dead_cst._symbols import SymbolNode
from dead_cst._visitor import VisitorPayload


# ---------------------------------------------------------------------------
# Generic shapes
# ---------------------------------------------------------------------------


@dataclass
class DecoratedDeclPlugin:
    """Mark decls as entrypoints when their file matches a search path
    and they're decorated with one of ``decorator_names`` (or assigned
    via ``constructor_names``) sourced from ``decorator_module``.

    Two top-level forms are recognised, mirroring how the builtin
    ``ClickPlugin`` detects Click groups:

    * ``@<module>.<name>(...)`` decorating a function -- the function's
      name becomes the target.
    * ``X = <module>.<ctor>(...)`` -- ``X`` becomes the target.

    Aliased / module-prefixed forms (``import click as c`` -> ``@c.group``)
    flow through ``collect_module_imports`` + ``matched_attr_call`` and
    are handled identically.

    Configuration:

    * ``package_prefix`` -- restrict to files whose module fqname is
      ``<prefix>`` or under ``<prefix>.``. Empty matches every file.
    * ``decorator_module`` -- the module the decorators are imported
      from (``"click"`` for Click groups).
    * ``decorator_names`` -- names produced by *decorator* form.
    * ``constructor_names`` -- names produced by *constructor* form.

    Override :meth:`in_scope` for predicates the prefix can't express
    (e.g. exclude ``test_*`` files).

    Pure observe: every match becomes a file-local entrypoint synthetic
    via :func:`entrypoint_payload`, so no finalize work is needed.
    """

    package_prefix: str = ""
    decorator_module: str = ""
    decorator_names: frozenset[str] = frozenset()
    constructor_names: frozenset[str] = frozenset()
    name: str = "decorated_decl"
    version: str = "1"

    def in_scope(self, ctx: ObserveContext) -> bool:
        if not self.package_prefix:
            return True
        module_node = next((n for n in ctx.payload.nodes if n.type == "module"), None)
        if module_node is None:
            return False
        fqname = module_node.fqname
        return fqname == self.package_prefix or fqname.startswith(self.package_prefix + ".")

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        if not self.in_scope(ctx):
            return None
        if not self.decorator_module:
            return None

        imports = collect_module_imports(
            ctx.module, self.decorator_module, self.decorator_names | self.constructor_names
        )
        if not imports:
            return None

        names = self._find_names(ctx.module, imports)
        if not names:
            return None

        targets = [n for n in ctx.payload.nodes if simple_name(n.fqname) in names]
        return entrypoint_payload(f"<{self.name}>:{ctx.path.name}", ctx.path, targets)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()

    def _find_names(self, module: cst.Module, imports: dict[str, str]) -> set[str]:
        out: set[str] = set()
        for stmt in module.body:
            if isinstance(stmt, cst.FunctionDef):
                for dec in stmt.decorators:
                    if matched_attr_call(dec.decorator, imports, self.decorator_names):
                        out.add(stmt.name.value)
                        break
            elif isinstance(stmt, cst.SimpleStatementLine):
                for small in stmt.body:
                    target, value = single_target_assignment(small)
                    if target is None or not isinstance(value, cst.Call):
                        continue
                    if matched_attr_call(
                        value.func, imports, self.constructor_names, unwrap_call=False
                    ):
                        out.add(target)
        return out


@dataclass
class LiteralListPlugin:
    """Read ``<owner_fqname>.<variable_name>`` (a top-level list/tuple of
    string literals) and treat each entry as a fqname to keep alive.

    Each entry is resolved against the assembled graph -- as a module
    fqname (whole module surface kept alive, mirroring
    ``importlib.import_module``) or, falling back, as a single decl
    fqname.

    Cache strategy: every CST inspection happens in :meth:`observe`,
    so it goes through dead-cst's per-file payload cache and only
    re-runs when the owner file actually changes. The captured
    fqnames are encoded as synthetic ENTRYPOINT-flagged decls (one
    per entry) parented at the owner file. Each synth carries a
    file-local edge to the variable's decl, so ``why-alive
    <owner_fqname>.<variable_name>`` shows our synthetics as
    predecessors -- the chain visualises "this list keeps these
    fqnames alive."

    :meth:`finalize` only does graph walks: find our synthetics by
    fqname prefix, decode the captured target fqname out, resolve
    against the assembled graph, emit the cross-file edge. No CST
    parsing, no per-file state -- safe to serve observe from cache
    on every warm run.
    """

    owner_fqname: str = ""
    variable_name: str = ""
    name: str = "literal_list"
    version: str = "1"

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        if not self.owner_fqname or not self.variable_name:
            return None
        module_node = next((n for n in ctx.payload.nodes if n.type == "module"), None)
        if module_node is None or module_node.fqname != self.owner_fqname:
            return None
        fqnames = _read_string_list(ctx.module, self.variable_name)
        if not fqnames:
            return None

        # Locate the variable's decl in this file's payload so the synth
        # can hang off it. Edge ``synth -> variable_decl`` keeps the
        # variable alive (in case nothing else references it) and gives
        # ``why-alive`` a readable chain.
        variable_decl = next(
            (
                n
                for n in ctx.payload.nodes
                if n.type == "variable" and simple_name(n.fqname) == self.variable_name
            ),
            None,
        )

        synth_prefix = self._synth_prefix()
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, object]] = []
        for fqname in fqnames:
            synth = synthetic_node(f"{synth_prefix}{fqname}", ctx.path, flags=NodeFlags.ENTRYPOINT)
            nodes.append(synth)
            if variable_decl is not None:
                edges.append((synth, variable_decl, SYNTHETIC_POSITION))
        return _payload_from(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        synth_prefix = self._synth_prefix()
        for node in ctx.base_nodes():
            if node.type != "synthetic" or not node.fqname.startswith(synth_prefix):
                continue
            captured = node.fqname[len(synth_prefix) :]
            for target in self._resolve(ctx, captured):
                yield AddEdge(node, target)

    def _synth_prefix(self) -> str:
        return f"<{self.name}>:"

    def _resolve(self, ctx: PluginContext, fqname: str) -> list[SymbolNode]:
        mod = ctx.find_module(fqname)
        if mod is not None:
            prefix = fqname + "."
            return [
                n for n in ctx.base_nodes() if n.fqname == fqname or n.fqname.startswith(prefix)
            ]
        return list(ctx.find_declarations(fqname))


def _read_string_list(module: cst.Module, var_name: str) -> list[str]:
    """Return ``<var_name> = [<str>, ...]`` (or tuple) at module level.

    Empty list when no such assignment exists or the RHS isn't a
    static list/tuple of string literals. Non-literal entries are
    dropped silently -- the runtime would just fail on them and we
    don't pretend to resolve them.
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


# ---------------------------------------------------------------------------
# Project-specific subclasses
# ---------------------------------------------------------------------------


@dataclass
class Flux0CliCommandsPlugin(DecoratedDeclPlugin):
    """Mirror ``flux0_cli/main.py:register_commands``.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/cli/src/flux0_cli/main.py#L61

    The runtime check is ``isinstance(cmd, click.Group)`` after
    ``importlib``-loading every module under ``flux0_cli.cmds``. We
    reproduce that statically: per submodule, find every top-level
    ``@click.group()`` function and ``X = click.Group(...)`` assignment.
    """

    package_prefix: str = "flux0_cli.cmds"
    decorator_module: str = "click"
    decorator_names: frozenset[str] = frozenset({"group", "Group"})
    constructor_names: frozenset[str] = frozenset({"Group"})
    name: str = "flux0_cli_cmds"


@dataclass
class Flux0InternalModulesPlugin(LiteralListPlugin):
    """Mirror ``flux0_server.main.INTERNAL_MODULES``.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/server/src/flux0_server/main.py#L51
    """

    owner_fqname: str = "flux0_server.main"
    variable_name: str = "INTERNAL_MODULES"
    name: str = "flux0_internal_modules"
