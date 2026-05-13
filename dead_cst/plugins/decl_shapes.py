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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, cast

import libcst as cst
from libcst.metadata import CodeRange, PositionProvider

from ..graph import NodeFlags, SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    AddNode,
    GraphOp,
    ObserveContext,
    PluginContext,
    collect_module_imports,
    decls_by_simple_name,
    find_call_assignments,
    find_factory_decls,
    find_handlers,
    make_payload,
    matched_attr_call,
    module_node,
    require_resolved_dep,
    simple_name,
    single_target_assignment,
    synthetic_node,
    walk_to_instance_kind,
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
        for node in ctx.package_nodes:
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

    Generic shape behind :class:`~dead_cst.contrib.TyperPlugin`,
    :class:`~dead_cst.contrib.CycloptsPlugin`,
    :class:`~dead_cst.contrib.FlaskPlugin`,
    :class:`~dead_cst.contrib.FastAPIPlugin`, and
    :class:`~dead_cst.contrib.CeleryPlugin`: find top-level
    ``X = <Ctor>(...)`` assignments where ``Ctor`` is recognized as an
    app constructor imported from ``app_module``, then for every
    top-level function decorated ``@X.<name>(...)`` where ``name`` is in
    ``registration_decorators`` emit an edge ``X -> handler``.

    The plugin has two modes:

    * **Pure dispatch** (``instance_kinds`` empty -- Typer / Cyclopts):
      only top-level ``X = Ctor(...)`` assignments are tracked, and
      every ``X -> handler`` edge requires ``X`` to be one of those
      direct constructions. App instances are *not* auto-marked as
      entrypoints; reachability flows through ``[project.scripts]`` or
      a ``__main__`` block. Pure observe-time; :meth:`finalize` is a
      no-op.
    * **Factory-aware** (``instance_kinds`` non-empty -- Flask /
      FastAPI / Celery): each kind in the mapping is recognized as a
      constructor, and the boolean value tells us whether direct
      ``X = Kind(...)`` hits should be promoted to entrypoints (e.g.
      ``Flask=True``, ``Blueprint=False``). Factory functions / classes
      whose body constructs one of the kinds get a
      ``<{name}-factory>:<kind>:<owner.fqname>`` marker. Variables
      decorated by handler decorators but not directly classified get a
      ``<{name}-pending>:<var.fqname>`` marker that :meth:`finalize`
      resolves by walking forward through the graph until it hits a
      classifying import node or factory marker.

    Configuration:

    * ``app_module`` -- dotted module name the constructors are
      imported from (``"flask"``, ``"fastapi"``, ``"celery"``,
      ``"typer"``, ...).
    * ``constructor_targets`` -- legacy field used only when
      ``instance_kinds`` is empty. Names of constructors to recognize.
    * ``registration_decorators`` -- attribute names on the instance
      that register a handler (``"route"`` / ``"get"`` / ``"task"`` /
      ``"command"`` / ...).
    * ``instance_kinds`` -- mapping ``{kind_name: auto_entrypoint}``.
      Setting any entry opts the plugin into factory-aware mode. The
      auto-entrypoint flag controls whether a direct hit on that kind
      gets a ``<{name}-app>:`` entrypoint synthetic (the framework's
      "this is always alive" classification, e.g. WSGI / ASGI / Celery
      worker autoloads).

    Aliased / module-prefixed forms (``import flask as f`` ->
    ``f.Flask()``) flow through :func:`collect_module_imports` and
    :func:`matched_attr_call`, so subclasses don't have to model them.

    Subclasses inject framework-specific behaviour (e.g. Celery's
    ``@shared_task``) by overriding :meth:`observe` to call
    ``super().observe(ctx)`` and splice extra nodes / edges into the
    returned payload.

    Abstract base: subclasses must set ``name`` and ``version``. The
    cache fingerprint is ``(name, version)``, so every concrete plugin
    needs its own ``name``.
    """

    name: str
    version: int
    app_module: str = ""
    constructor_targets: frozenset[str] = frozenset()
    registration_decorators: frozenset[str] = frozenset()
    instance_kinds: Mapping[str, bool] = field(default_factory=dict)

    @property
    def _factory_aware(self) -> bool:
        return bool(self.instance_kinds)

    @property
    def _targets(self) -> Mapping[str, bool] | frozenset[str]:
        # Factory-aware plugins drive recognition off the kinds map;
        # pure-dispatch plugins keep the legacy flat-set config.
        return self.instance_kinds if self._factory_aware else self.constructor_targets

    def _prefix(self, kind: str) -> str:
        """Return the synthetic prefix ``<{name}-{kind}>:`` for ``app`` / ``pending`` / ``factory``."""
        return f"<{self.name}-{kind}>:"

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        if not (self.app_module and self.registration_decorators):
            return None
        targets = self._targets
        if not targets:
            return None
        imports = collect_module_imports(ctx.module, self.app_module, targets)

        # ``find_call_assignments`` / ``find_factory_decls`` need imports to
        # recognize the constructors; ``find_handlers`` does not (it scans
        # ``@<owner>.<attr>`` decorator shapes directly). Factory-aware mode
        # depends on that: ``app = create_app()`` in a file that imports the
        # factory (not Flask itself) still needs a pending marker emitted
        # from the decorator scan alone.
        direct = find_call_assignments(ctx.module, imports, targets) if imports else {}
        if self._factory_aware:
            decorated = find_handlers(ctx.module, None, self.registration_decorators)
            factory_kinds = find_factory_decls(ctx.module, imports, targets) if imports else {}
            if not direct and not decorated and not factory_kinds:
                return None
        else:
            # Pure-dispatch: only emit edges for handlers owned by a
            # direct construction, so the imports gate is meaningful.
            if not direct:
                return None
            decorated = find_handlers(ctx.module, set(direct), self.registration_decorators)
            factory_kinds = {}
            if not decorated:
                return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        app_prefix = self._prefix("app")
        pending_prefix = self._prefix("pending")
        factory_prefix = self._prefix("factory")
        for var_name in direct.keys() | decorated.keys():
            var_decls = decls_by_name.get(var_name, [])
            kind = direct.get(var_name)
            for var_decl in var_decls:
                if self._factory_aware:
                    if kind is None:
                        pending = synthetic_node(f"{pending_prefix}{var_decl.fqname}", ctx.path)
                        nodes.append(pending)
                        edges.append((pending, var_decl, SYNTHETIC_POSITION))
                    elif self.instance_kinds[kind]:
                        seed = synthetic_node(
                            f"{app_prefix}{var_decl.fqname}",
                            ctx.path,
                            flags=NodeFlags.ENTRYPOINT,
                        )
                        nodes.append(seed)
                        edges.append((seed, var_decl, SYNTHETIC_POSITION))
                    # Non-entrypoint direct kinds (e.g. Blueprint / APIRouter)
                    # get only handler edges below.
                for handler_name in decorated.get(var_name, ()):
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((var_decl, handler_decl, SYNTHETIC_POSITION))

        # Factory markers: anchored on the constructing decl so finalize's
        # forward walk hits them regardless of which file the consumer lives
        # in. Required because ``import <mod>; <mod>.<Cls>()`` flows through
        # an external-edge that loses ``decl='<Cls>'`` after
        # :func:`resolve_edges`, so the import-node check alone can't tell
        # the kinds apart on the downstream walk.
        for decl_name, kinds in factory_kinds.items():
            for decl in decls_by_name.get(decl_name, []):
                for kind in kinds:
                    marker = synthetic_node(f"{factory_prefix}{kind}:{decl.fqname}", ctx.path)
                    nodes.append(marker)
                    edges.append((decl, marker, SYNTHETIC_POSITION))

        if not nodes and not edges:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        if not self._factory_aware:
            return
        app_node = require_resolved_dep(ctx, self.app_module)
        if app_node is None:
            return
        app_prefix = self._prefix("app")
        pending_prefix = self._prefix("pending")
        factory_prefix = self._prefix("factory")
        for synth in ctx.package_nodes:
            if synth.type != "synthetic" or not synth.fqname.startswith(pending_prefix):
                continue
            for var in list(ctx.graph.successors(synth)):
                kind = walk_to_instance_kind(
                    ctx.graph,
                    var,
                    app_node,
                    self.app_module,
                    self.instance_kinds,
                    factory_marker_prefix=factory_prefix,
                )
                if kind is None or not self.instance_kinds[kind]:
                    continue
                seed = synthetic_node(
                    f"{app_prefix}{var.fqname}", var.path, flags=NodeFlags.ENTRYPOINT
                )
                yield AddNode(seed)
                yield AddEdge(seed, var)


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
