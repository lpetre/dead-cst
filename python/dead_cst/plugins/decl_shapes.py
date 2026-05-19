"""Reusable plugin shapes for dynamic-import discovery patterns.

Two abstract bases that target idioms framework-flavoured codebases
keep stumbling into:

  :class:`DecoratedDeclPlugin`
    "Find decorated decls in files matching a search path." The same
    shape Click uses to detect its framework instances.

  :class:`LiteralListPlugin`
    "Read a top-level ``X = [\"...\", \"...\"]`` literal and treat each
    fqname inside it as alive."

  :class:`DispatchAppPlugin`
    "Find top-level ``X = Ctor(...)`` apps and wire decorated handlers
    through them." Concrete subclasses configure the framework module
    name (``flask`` / ``fastapi`` / ...), the constructor names, and
    the per-instance registration decorators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from ..graph import NodeFlags

if TYPE_CHECKING:
    from dead_cst import _native as native


@dataclass(kw_only=True)
class DecoratedDeclPlugin:
    """Mark decls as entrypoints when they bind to a name imported from
    ``decorator_module`` either via decorator (``@<module>.<name>(...)``
    on a function) or constructor (``X = <module>.<ctor>(...)``).

    Subclasses must set ``name`` and ``version``. The cache fingerprint
    is ``(name, version)`` — every concrete plugin needs a unique
    ``name``.
    """

    name: str
    version: int
    package_prefix: str = ""
    decorator_module: str = ""
    decorator_names: frozenset[str] = frozenset()
    constructor_names: frozenset[str] = frozenset()

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        if not self.decorator_module:
            return
        if not (self.decorator_names or self.constructor_names):
            return
        names = sorted(self.decorator_names | self.constructor_names)

        prefix = self.package_prefix

        def in_scope(path: str) -> bool:
            if not prefix:
                return True
            module = ctx.module_for(path)
            if module is None:
                return False
            return module.fqname == prefix or module.fqname.startswith(prefix + ".")

        seeds_by_path: dict[str, list[native.NativeNode]] = {}
        for dec_ref in (
            native.query(ctx).decorators().where_module(self.decorator_module).where_name(names)
        ):
            if in_scope(dec_ref.path):
                seeds_by_path.setdefault(dec_ref.path, []).append(dec_ref.decorated)
        for cons_ref in (
            native.query(ctx).constructions().where_module(self.decorator_module).where_name(names)
        ):
            if in_scope(cons_ref.path):
                seeds_by_path.setdefault(cons_ref.path, []).append(cons_ref.var)

        for path, targets in seeds_by_path.items():
            yield native.AddNode(
                fqname=f"<{self.name}>:{Path(path).name}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to=targets,
            )


@dataclass(kw_only=True)
class DispatchAppPlugin:
    """Wire ``@<instance>.<reg_decorator>(...)`` handlers to their app instance.

    Two modes:

    * **Pure dispatch** (``instance_kinds`` empty -- Typer / Cyclopts /
      Click): only direct ``X = Ctor(...)`` constructions are tracked;
      handlers register through them. App instances are not auto-marked
      as entrypoints.
    * **Factory-aware** (``instance_kinds`` non-empty -- Flask /
      FastAPI / FastMCP / Celery): each kind in the mapping is
      recognized as a constructor, and the boolean value tells us
      whether direct hits get promoted to entrypoints. Variables that
      receive decorators but aren't directly constructed get a factory
      walk through the assembled graph.
    """

    name: str
    version: int
    app_modules: tuple[str, ...] = ()
    constructor_targets: frozenset[str] = frozenset()
    registration_decorators: frozenset[str] = frozenset()
    instance_kinds: Mapping[str, bool] = field(default_factory=dict)

    @property
    def _factory_aware(self) -> bool:
        return bool(self.instance_kinds)

    @property
    def _targets(self) -> Mapping[str, bool] | frozenset[str]:
        return self.instance_kinds if self._factory_aware else self.constructor_targets

    def _prefix(self, kind: str) -> str:
        return f"<{self.name}-{kind}>:"

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        if not (self.app_modules and self.registration_decorators):
            return
        targets = self._targets
        if not targets:
            return
        target_names = list(targets)
        decorator_attrs = list(self.registration_decorators)

        direct: list = []
        for module in self.app_modules:
            direct.extend(
                native.query(ctx).constructions().where_module(module).where_name(target_names)
            )
        handlers = list(native.query(ctx).decorators().where_owner_attr(decorator_attrs))
        factory_decls: list = []
        if self._factory_aware:
            for module in self.app_modules:
                factory_decls.extend(
                    native.query(ctx)
                    .factories()
                    .of_module(module)
                    .where_name(target_names)
                    .collect()
                )

        direct_by_owner: dict[tuple[str, str], list[tuple["native.NativeNode", str]]] = {}
        for ref in direct:
            simple = ref.var.fqname.rsplit(".", 1)[-1]
            direct_by_owner.setdefault((ref.var.path, simple), []).append((ref.var, ref.class_name))

        vars_by_file: dict[tuple[str, str], native.NativeNode] = {}
        for n in ctx.nodes():
            if n.kind != "variable":
                continue
            simple = n.fqname.rsplit(".", 1)[-1]
            vars_by_file.setdefault((n.path, simple), n)

        app_prefix = self._prefix("app")
        factory_prefix = self._prefix("factory")

        if self._factory_aware:
            for ref in direct:
                if self.instance_kinds.get(ref.class_name):
                    yield native.AddNode(
                        fqname=f"{app_prefix}{ref.var.fqname}",
                        path=ref.var.path,
                        flags=int(NodeFlags.ENTRYPOINT),
                        edges_to=[ref.var],
                    )
            for fref in factory_decls:
                for kind in fref.kinds:
                    yield native.AddNode(
                        fqname=f"{factory_prefix}{kind}:{fref.decl.fqname}",
                        path=fref.decl.path,
                        edges_from=[fref.decl],
                    )

        if self._factory_aware:
            for h in handlers:
                var = vars_by_file.get((h.decorated.path, h.decorator_owner or ""))
                if var is not None:
                    yield native.AddEdge(var, h.decorated)
        else:
            for h in handlers:
                for var_node, _kind in direct_by_owner.get(
                    (h.decorated.path, h.decorator_owner or ""), []
                ):
                    yield native.AddEdge(var_node, h.decorated)

        if self._factory_aware:
            classified: set[tuple[str, str]] = set()
            for h in handlers:
                key = (h.decorated.path, h.decorator_owner or "")
                if key in direct_by_owner or key in classified:
                    continue
                var = vars_by_file.get(key)
                if var is None:
                    continue
                for desc in ctx.descendants(var):
                    if desc.kind != "synthetic":
                        continue
                    if not desc.fqname.startswith(factory_prefix):
                        continue
                    kind = desc.fqname[len(factory_prefix) :].split(":", 1)[0]
                    if not self.instance_kinds.get(kind):
                        continue
                    classified.add(key)
                    yield native.AddNode(
                        fqname=f"{app_prefix}{var.fqname}",
                        path=var.path,
                        flags=int(NodeFlags.ENTRYPOINT),
                        edges_to=[var],
                    )
                    break


@dataclass(kw_only=True)
class LiteralListPlugin:
    """Read ``<owner_fqname>.<variable_name>`` (a top-level list/tuple of
    string literals) and treat each entry as a fqname to keep alive.

    Each entry resolves against the assembled graph as either a module
    fqname (whole module surface revived, mirroring
    ``importlib.import_module``) or a single decl fqname.

    Subclasses must set ``name`` and ``version``. The cache fingerprint
    is ``(name, version)`` — every concrete plugin needs a unique ``name``.
    """

    name: str
    version: int
    owner_fqname: str = ""
    variable_name: str = ""

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        if not self.owner_fqname or not self.variable_name:
            return
        var_fqname = f"{self.owner_fqname}.{self.variable_name}"
        # Use a calls-style approach via reading the source isn't possible
        # — instead resolve the literal list by reading the dunder all
        # exports machinery (if used). Simpler: look up declarations and
        # walk successors via the visitor's ``X = ['...']`` edges.
        decls = native.query(ctx).declarations(var_fqname)
        if not decls:
            return

        # Walk forward from each variable decl: the visitor wires
        # ``var -> referent`` edges for every string-literal entry that
        # named a project decl (via __all__-style emission), so the
        # successors of the variable node are the targets.
        prefix = f"<{self.name}>:"
        for decl in decls:
            for desc in ctx.descendants(decl):
                if desc is decl:
                    continue
                if desc.kind in ("module", "function", "class", "variable"):
                    yield native.AddNode(
                        fqname=f"{prefix}{desc.fqname}",
                        path=decl.path,
                        flags=int(NodeFlags.ENTRYPOINT),
                        edges_to=[desc],
                    )
