"""Reusable plugin shapes for dynamic-import discovery patterns.

Two abstract bases that target idioms framework-flavoured codebases
keep stumbling into:

  :class:`DecoratedDeclPlugin`
    "Find decorated decls in files matching a search path." Pure
    observe-time. The same shape Click / FastAPI / Flask / Typer use
    to detect their framework instances; subclass and configure with
    the decorator-source module + names. ``ClickPlugin`` is itself a
    consumer.

  :class:`LiteralListPlugin`
    "Read a top-level ``X = [\"...\", \"...\"]`` literal and treat each
    fqname inside it as alive." Captures the parsed entries during
    observe (so the work rides the per-file payload cache and does not
    re-run on warm runs); finalize only does graph lookups.

Both bases use only the public plugin-helpers re-exported from
:mod:`dead_cst.plugins`; user subclasses don't need to reach into
``_core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, cast

import libcst as cst
from libcst.metadata import CodeRange, PositionProvider

from ..graph import NodeFlags, SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    GraphOp,
    ObserveContext,
    PluginContext,
    collect_module_imports,
    decls_by_simple_name,
    find_call_assignments,
    find_handlers,
    make_payload,
    matched_attr_call,
    module_node,
    simple_name,
    single_target_assignment,
    synthetic_node,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload


@dataclass(kw_only=True)
class DecoratedDeclPlugin:
    """Mark decls as entrypoints when their file is in scope and they
    bind to a name imported from ``decorator_module`` either via
    decorator (``@<module>.<name>(...)`` on a function) or constructor
    (``X = <module>.<ctor>(...)``).

    Configuration:

    * ``package_prefix`` -- restrict to files whose module fqname is
      ``<prefix>`` or under ``<prefix>.``. Empty matches every file.
    * ``decorator_module`` -- the dotted module name the decorators /
      constructors are imported from (e.g. ``"click"``).
    * ``decorator_names`` -- names produced by decorator form.
    * ``constructor_names`` -- names produced by constructor form.

    Aliased / module-prefixed forms (``import click as c`` ->
    ``@c.group``) flow through :func:`collect_module_imports` and
    :func:`matched_attr_call`, so subclasses don't have to model them.

    Override :meth:`in_scope` for predicates the prefix can't express.
    Pure observe: matches turn directly into a per-file entrypoint
    payload, so finalize is a no-op.

    Abstract base: subclasses must set ``name`` and ``version``
    themselves. The cache fingerprint is ``(name, version)``, so
    every concrete plugin needs its own ``name`` to avoid aliasing
    other instances' cached observe payloads. Bump ``version`` to a
    fresh epoch any time the subclass's config changes.
    """

    name: str
    version: int
    package_prefix: str = ""
    decorator_module: str = ""
    decorator_names: frozenset[str] = frozenset()
    constructor_names: frozenset[str] = frozenset()

    def in_scope(self, ctx: ObserveContext) -> bool:
        """Return True if this file should be inspected.

        Default uses ``package_prefix`` against the file's module
        fqname. Subclasses can override for more complex predicates.
        """
        if not self.package_prefix:
            return True
        module = module_node(ctx.payload)
        if module is None:
            return False
        fqname = module.fqname
        return fqname == self.package_prefix or fqname.startswith(self.package_prefix + ".")

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        # Cheap config-static gate before any payload scan.
        if not self.decorator_module:
            return None
        if not (self.decorator_names or self.constructor_names):
            return None
        if not self.in_scope(ctx):
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
        if not targets:
            return None

        synth = synthetic_node(
            f"<{self.name}>:{ctx.path.name}", ctx.path, flags=NodeFlags.ENTRYPOINT
        )
        return make_payload(
            nodes=[synth],
            edges=[(synth, t, SYNTHETIC_POSITION) for t in targets],
        )

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()

    def _find_names(self, module: cst.Module, imports: dict[str, str]) -> set[str]:
        """Return top-level names bound to a configured decorator/constructor.

        Subclasses can override to extend the search (e.g. Click's
        nested-group fixpoint that picks up ``@<known_group>.group(...)``
        forms after the initial pass).
        """
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


@dataclass(kw_only=True)
class LiteralListPlugin:
    """Read ``<owner_fqname>.<variable_name>`` (a top-level list/tuple of
    string literals) and treat each entry as a fqname to keep alive.

    Each entry resolves against the assembled graph as either a module
    fqname (whole module surface revived, mirroring
    ``importlib.import_module``) or a single decl fqname.

    Cache strategy: every CST inspection happens in :meth:`observe`,
    so it goes through dead-cst's per-file payload cache and only
    re-runs when the owner file actually changes. Captured fqnames
    are encoded as ENTRYPOINT-flagged synthetic decls (one per entry,
    positioned at the literal's line). Each synth carries an edge to
    the variable's decl, so ``why-alive <owner>.<var>`` shows the
    list as the entrypoint chain. :meth:`finalize` only walks the
    assembled graph -- no CST parsing, no per-file plugin state.

    Abstract base: subclasses must set ``name`` and ``version``
    themselves. The cache fingerprint is ``(name, version)``, so
    every concrete plugin needs its own ``name`` to avoid aliasing
    other instances' cached observe payloads. Bump ``version`` to a
    fresh epoch any time the subclass's config changes.
    """

    name: str
    version: int
    owner_fqname: str = ""
    variable_name: str = ""

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        if not self.owner_fqname or not self.variable_name:
            return None
        module = module_node(ctx.payload)
        if module is None or module.fqname != self.owner_fqname:
            return None
        captured = _read_string_list_with_positions(ctx.module, self.variable_name)
        if not captured:
            return None

        variable_decl = next(
            (
                n
                for n in ctx.payload.nodes
                if n.type == "variable" and simple_name(n.fqname) == self.variable_name
            ),
            None,
        )

        prefix = self._synth_prefix()
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []
        for fqname, position in captured:
            synth = synthetic_node(
                f"{prefix}{fqname}", ctx.path, flags=NodeFlags.ENTRYPOINT, position=position
            )
            nodes.append(synth)
            if variable_decl is not None:
                edges.append((synth, variable_decl, position))
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        prefix = self._synth_prefix()
        for node in ctx.package_nodes():
            if node.type != "synthetic" or not node.fqname.startswith(prefix):
                continue
            captured = node.fqname[len(prefix) :]
            for target in self._resolve(ctx, captured):
                yield AddEdge(node, target)

    def _synth_prefix(self) -> str:
        return f"<{self.name}>:"

    def _resolve(self, ctx: PluginContext, fqname: str) -> list[SymbolNode]:
        """Resolve a captured fqname to a module surface or a single decl."""
        surface = ctx.module_surface(fqname)
        if surface:
            return surface
        return list(ctx.find_declarations(fqname))


@dataclass(kw_only=True)
class DispatchAppPlugin:
    """Wire ``@<instance>.<reg_decorator>(...)`` handlers to their app instance.

    Generic shape behind :class:`~dead_cst.contrib.TyperPlugin` and
    :class:`~dead_cst.contrib.CycloptsPlugin`: find top-level
    ``X = <Ctor>(...)`` assignments where ``Ctor`` is one of
    ``constructor_targets`` imported from ``app_module``, then for every
    top-level function decorated ``@X.<name>(...)`` where ``name`` is in
    ``registration_decorators`` emit an edge ``X -> handler``.

    Pure observe: instance detection and decorator scanning are
    file-local CST passes; the corresponding ``SymbolNode`` decls are
    looked up in this file's :class:`VisitorPayload`. App instances
    are not auto-marked as entrypoints; reachability flows through
    ``[project.scripts]`` or a ``__main__`` block.

    Abstract base: subclasses must set ``name`` and ``version``. The
    cache fingerprint is ``(name, version)``, so every concrete plugin
    needs its own ``name``.
    """

    name: str
    version: int
    app_module: str = ""
    constructor_targets: frozenset[str] = frozenset()
    registration_decorators: frozenset[str] = frozenset()

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        if not (self.app_module and self.constructor_targets and self.registration_decorators):
            return None
        imports = collect_module_imports(ctx.module, self.app_module, self.constructor_targets)
        if not imports:
            return None
        instances = set(find_call_assignments(ctx.module, imports, self.constructor_targets))
        if not instances:
            return None
        handlers = find_handlers(ctx.module, instances, self.registration_decorators)
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


def _read_string_list_with_positions(
    module: cst.Module, var_name: str
) -> list[tuple[str, CodeRange]]:
    """Return ``[(value, position), ...]`` for ``<var_name> = [<str>, ...]``.

    Empty list when no such assignment exists or the RHS isn't a
    static list/tuple of string literals. Non-literal entries drop
    silently. The metadata wrapper only resolves once we've located
    the matching assignment, so files that don't contain the variable
    pay only a body scan.
    """
    target_value: cst.BaseExpression | None = None
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target, value = single_target_assignment(small)
            if target == var_name:
                target_value = value
                break
        if target_value is not None:
            break

    if not isinstance(target_value, (cst.List, cst.Tuple)):
        return []

    positions = cst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(PositionProvider)
    out: list[tuple[str, CodeRange]] = []
    for elt in target_value.elements:
        if not isinstance(elt, cst.Element):
            continue
        inner = elt.value
        if not isinstance(inner, cst.SimpleString):
            continue
        # ``evaluated_value`` is ``str | bytes`` because b"..." is also a
        # ``SimpleString``; bytes literals can't be valid module fqnames.
        value = inner.evaluated_value
        if not isinstance(value, str):
            continue
        out.append((value, cast(CodeRange, positions[inner])))
    return out
