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
    through them." Concrete subclasses configure the fully-qualified
    app-class names (``flask.Flask`` / ``fastapi.FastAPI`` / ...) and
    the per-instance registration decorators. Discovery is transitive
    over subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..graph import NodeFlags
from ._base import Plugin, native


@dataclass(kw_only=True)
class DecoratedDeclPlugin(Plugin):
    """Mark decls as entrypoints when they bind to a name imported from
    ``decorator_module`` either via decorator (``@<module>.<name>(...)``
    on a function) or constructor (``X = <module>.<ctor>(...)``).

    ``marker_prefix`` controls the synthetic node fqname emitted per
    file — each concrete plugin should pick a unique short string so
    its markers don't collide with other plugins'.
    """

    marker_prefix: str
    package_prefix: str = ""
    decorator_module: str = ""
    decorator_names: frozenset[str] = frozenset()
    constructor_names: frozenset[str] = frozenset()

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        if not self.decorator_module:
            return
        if not (self.decorator_names or self.constructor_names):
            return
        # Cheap import-presence guard. Querying decorator/construction
        # types forces ty to resolve ``decorator_module`` out of the
        # venv (parse, build SemanticIndex, walk type hierarchy) —
        # ~100-400ms per framework on a typical project. If nothing
        # imports the module, no decorated decl can exist, so skip
        # the entire query path.
        if not native.query(ctx).imports().of(self.decorator_module).exists():
            return
        names = sorted(self.decorator_names | self.constructor_names)

        prefix = self.package_prefix

        def in_scope(path: str) -> bool:
            if not prefix:
                return True
            module_idx = native.query(ctx).modules().with_path(path).first_idx()
            if module_idx is None:
                return False
            (_kind, _path, fqname, _flags) = ctx.node_attrs([module_idx])[0]
            return fqname == prefix or fqname.startswith(prefix + ".")

        seeds_by_path: dict[str, list[int]] = {}
        for dec_row in (
            native.query(ctx)
            .decorators()
            .where_module(self.decorator_module)
            .where_name(names)
            .collect()
        ):
            if in_scope(dec_row.path):
                seeds_by_path.setdefault(dec_row.path, []).append(dec_row.decorated_idx)
        for cons_row in (
            native.query(ctx)
            .constructions()
            .where_module(self.decorator_module)
            .where_name(names)
            .collect()
        ):
            if in_scope(cons_row.path):
                seeds_by_path.setdefault(cons_row.path, []).append(cons_row.var_idx)

        for path, target_idxs in seeds_by_path.items():
            yield native.AddNodeByIdx(
                fqname=f"<{self.marker_prefix}>:{Path(path).name}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=target_idxs,
            )


@dataclass(frozen=True)
class DispatchAppSpec:
    """Pure-data description of what a :class:`DispatchAppPlugin` needs
    the walker to gather. Has no behavior or policy hooks — those live
    on the plugin's :meth:`DispatchAppPlugin.policy` method.

    Fields mirror the plugin's class attributes: same semantics, just
    repackaged so the batched walker can take a list of specs and run
    one fused pass instead of one pass per plugin. Frozen so it's
    hashable / cacheable.
    """

    marker_prefix: str
    app_classes: tuple[str, ...]
    registration_decorators: frozenset[str]
    seed_as_entrypoint: bool


@dataclass
class DispatchAppGather:
    """Pre-walked data, scoped to one plugin's spec.

    Both the standalone single-plugin path and the batched fused-walk
    path produce one of these per plugin and hand it to
    :meth:`DispatchAppPlugin.policy`. ``vars_by_file`` is shared
    across plugins in the batched path (it's plugin-agnostic).
    """

    spec: DispatchAppSpec
    direct: list[native.ConstructionIdxRef]
    factory_decls: list[tuple[native.FactoryIdxRef, str]]
    handlers: list[native.DecoratorIdxRef]
    vars_by_file: dict[tuple[str, str], int]


@dataclass(kw_only=True)
class DispatchAppPlugin(Plugin):
    """Wire ``@<instance>.<reg_decorator>(...)`` handlers to their app instance.

    Subclasses configure ``app_classes`` -- the fully-qualified class
    names that the plugin treats as "app constructors". Discovery is
    transitive: any subclass of a listed class also counts, so
    ``("flask.Flask",)`` covers ``flask.Flask`` AND any project-level
    ``CustomFlask(Flask)`` AND third-party Flask extensions for free.

    Handler wiring (``@instance.deco(...) -> instance``) always fires
    for owners whose name matches a top-level variable in the same
    file. Entrypoint promotion is gated on ``seed_as_entrypoint``:

    * ``seed_as_entrypoint=True`` (default, factory-aware frameworks
      like Flask / FastAPI / Celery / FastMCP):
      every direct ``X = <app_class>(...)`` becomes an entrypoint, and
      a factory walk promotes vars assigned from a function that
      returns an ``app_class`` instance.
    * ``seed_as_entrypoint=False`` (pure-dispatch frameworks like
      Cyclopts / Typer / Click): apps are NOT entrypoint-promoted;
      reachability is expected to flow through ``[project.scripts]``
      / ``if __name__ == "__main__": app()`` / explicit ``-e``. Lets
      unused sub-apps surface as dead code.

    ``marker_prefix`` is the short label used in the ``<{marker}-app>:``
    and ``<{marker}-factory>:`` synthetic fqnames the plugin emits.

    Result-level dedup: if the same construction site is reachable
    from two ``app_classes`` entries (e.g. one is a base of the
    other), it only emits one entrypoint marker.

    **Spec / policy split.** Subclasses customize behavior by
    overriding :meth:`policy` — emission is decoupled from the
    fetch (which is described by :attr:`spec` and performed by the
    shared walker). The batched driver
    :class:`BatchDispatchAppPlugin` honors policy overrides for every
    wrapped plugin by calling each instance's :meth:`policy`.
    """

    marker_prefix: str
    app_classes: tuple[str, ...] = ()
    registration_decorators: frozenset[str] = frozenset()
    seed_as_entrypoint: bool = True

    @property
    def spec(self) -> DispatchAppSpec:
        """Frozen-dataclass view of this plugin's gather config.

        Override-friendly: subclasses change what's gathered by
        overriding the underlying class attributes (or this property
        if a dynamic spec is needed). The batched walker consumes the
        spec; it never inspects per-plugin fields directly.
        """
        return DispatchAppSpec(
            marker_prefix=self.marker_prefix,
            app_classes=self.app_classes,
            registration_decorators=self.registration_decorators,
            seed_as_entrypoint=self.seed_as_entrypoint,
        )

    def _prefix(self, kind: str) -> str:
        return f"<{self.marker_prefix}-{kind}>:"

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        if not self._is_active(ctx):
            return
        gathered = _gather_one(ctx, self.spec)
        if gathered is None:
            return
        yield from self.policy(ctx, gathered)

    def _is_active(self, ctx: native.ProjectContext) -> bool:
        """Cheap import-presence + config-completeness guard. Skips the
        ~100-400ms ``subclasses().of_fqn(...)`` walk when no file in the
        project imports the framework's root package.
        """
        if not (self.app_classes and self.registration_decorators):
            return False
        app_modules = {fqn.rpartition(".")[0] for fqn in self.app_classes if "." in fqn}
        return any(native.query(ctx).imports().of(m).exists() for m in app_modules if m)

    def policy(
        self, ctx: native.ProjectContext, gathered: DispatchAppGather
    ) -> Iterable[native.GraphOp]:
        """Emit ops from pre-gathered data. The override point for
        subclasses that want to extend or customize emission without
        changing what gets gathered.

        Subclasses that want to add framework-specific extras should
        ``yield from super().policy(ctx, gathered)`` first, then yield
        their additional ops. The standalone :meth:`run` path and the
        batched :class:`BatchDispatchAppPlugin` path both call this
        method, so any override is honored uniformly.
        """
        spec = gathered.spec
        direct = gathered.direct
        factory_decls = gathered.factory_decls
        handlers = gathered.handlers
        vars_by_file = gathered.vars_by_file

        # direct_by_owner: (path, simple_name) -> [var_idx, ...]
        direct_by_owner: dict[tuple[str, str], list[int]] = {}
        direct_attrs: list[tuple[native.NodeKind, str, str, int]] = []
        if direct:
            direct_attrs = ctx.node_attrs([r.var_idx for r in direct])
            for ref, (_kind, _path, fqname, _flags) in zip(direct, direct_attrs, strict=True):
                simple = fqname.rsplit(".", 1)[-1]
                direct_by_owner.setdefault((ref.path, simple), []).append(ref.var_idx)

        app_prefix = f"<{spec.marker_prefix}-app>:"
        factory_prefix = f"<{spec.marker_prefix}-factory>:"

        # 3. Entrypoint-promote every direct construction (when enabled).
        if spec.seed_as_entrypoint and direct:
            # Reuse the direct_attrs computed above.
            for ref, (_kind, _path, fqname, _flags) in zip(direct, direct_attrs, strict=True):
                yield native.AddNodeByIdx(
                    fqname=f"{app_prefix}{fqname}",
                    path=ref.path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to_idx=[ref.var_idx],
                )

        # 4. Emit factory markers so the descendant walk in step 6 can
        # find them.
        if factory_decls:
            factory_attrs = ctx.node_attrs([fref.decl_idx for fref, _kind in factory_decls])
            for (fref, kind), (_k, _p, decl_fqname, _f) in zip(
                factory_decls, factory_attrs, strict=True
            ):
                yield native.AddNodeByIdx(
                    fqname=f"{factory_prefix}{kind}:{decl_fqname}",
                    path=fref.path,
                    edges_from_idx=[fref.decl_idx],
                )

        # 5. Wire decorator handlers to their owner var.
        # When seed_as_entrypoint=False (pure dispatch), only wire to
        # vars that came from a direct construction -- this is what
        # makes ``from cyclopts import *; app = App()`` stay invisible
        # to the plugin (the star import never resolves the App name,
        # so direct_by_owner has no entry for ``app``).
        # When seed_as_entrypoint=True (factory-aware), use vars_by_file
        # so ``app = create_app()`` factory chains also pick up
        # handler edges.
        for h in handlers:
            key = (h.path, h.decorator_owner or "")
            if spec.seed_as_entrypoint:
                var_idx = vars_by_file.get(key)
                if var_idx is not None:
                    yield native.AddEdgeByIdx(var_idx, h.decorated_idx)
            else:
                for var_idx in direct_by_owner.get(key, []):
                    yield native.AddEdgeByIdx(var_idx, h.decorated_idx)

        # 6. Factory walk: vars whose *direct* successor is one of the
        # factory decls get entrypoint-promoted. Skipped under
        # seed_as_entrypoint=False.
        #
        # We invert the question: rather than walk each var's
        # descendants looking for a factory marker, we collect the
        # direct predecessors of every factory decl into a single set
        # and check ``var in factory_reachers``. This rules out the
        # over-promotion case ``wrapper = app; app = create_app()`` —
        # ``wrapper -> app -> create_app`` is two hops, so ``wrapper``
        # doesn't reach the factory directly and stays unclassified.
        # Only ``app`` (whose direct successor *is* ``create_app``)
        # promotes.
        if spec.seed_as_entrypoint and factory_decls:
            factory_reachers: set[int] = set()
            for fref, _kind in factory_decls:
                factory_reachers.update(
                    native.query(ctx).from_idx(fref.decl_idx).direct_predecessors()
                )

            classified: set[tuple[str, str]] = set()
            for h in handlers:
                key = (h.path, h.decorator_owner or "")
                if key in direct_by_owner or key in classified:
                    continue
                var_idx = vars_by_file.get(key)
                if var_idx is None or var_idx not in factory_reachers:
                    continue
                classified.add(key)
                (_kind, var_path, var_fqname, _flags) = ctx.node_attrs([var_idx])[0]
                yield native.AddNodeByIdx(
                    fqname=f"{app_prefix}{var_fqname}",
                    path=var_path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to_idx=[var_idx],
                )


# ---------------------------------------------------------------------------
# Walker: spec → gather
#
# The "gather" half of the spec / policy split. Both the standalone
# ``DispatchAppPlugin.run`` path and the batched
# ``BatchDispatchAppPlugin.run`` path flow through these helpers; only
# the fan-out shape differs.
# ---------------------------------------------------------------------------


def _module_to_names(
    ctx: native.ProjectContext,
    app_classes: tuple[str, ...],
    subclass_cache: dict[str, list[tuple[str, str]]] | None = None,
) -> dict[str, set[str]]:
    """Expand each ``app_class`` fqn into ``{module: {simple_name, ...}}``,
    walking subclasses transitively. ``subclass_cache`` (optional) lets
    the batched walker memoise across plugins that share an app class.
    """
    out: dict[str, set[str]] = {}
    for fqn in app_classes:
        module, _, name = fqn.rpartition(".")
        if not module or not name:
            continue
        out.setdefault(module, set()).add(name)
    if not app_classes:
        return out
    for fqn in app_classes:
        if subclass_cache is not None and fqn in subclass_cache:
            pairs = subclass_cache[fqn]
        else:
            sub_idxs = native.query(ctx).subclasses().of_fqn(fqn).indices()
            pairs = []
            if sub_idxs:
                for _kind, _path, sub_fqname, _flags in ctx.node_attrs(sub_idxs):
                    sub_module, _, sub_name = sub_fqname.rpartition(".")
                    if sub_module and sub_name:
                        pairs.append((sub_module, sub_name))
            if subclass_cache is not None:
                subclass_cache[fqn] = pairs
        for sub_module, sub_name in pairs:
            out.setdefault(sub_module, set()).add(sub_name)
    return out


def _fetch_direct(
    ctx: native.ProjectContext, module_to_names: dict[str, set[str]]
) -> list[native.ConstructionIdxRef]:
    """Run a construction query per distinct module (with the union of
    requested names) and dedup by ``var_idx``."""
    direct_seen: set[int] = set()
    direct: list[native.ConstructionIdxRef] = []
    for module, names in module_to_names.items():
        for ref in (
            native.query(ctx)
            .constructions()
            .where_module(module)
            .where_name(sorted(names))
            .collect()
        ):
            if ref.var_idx in direct_seen:
                continue
            direct_seen.add(ref.var_idx)
            direct.append(ref)
    return direct


def _fetch_factory_decls(
    ctx: native.ProjectContext, module_to_names: dict[str, set[str]]
) -> list[tuple[native.FactoryIdxRef, str]]:
    """Run a factory query per distinct module and dedup by
    ``(decl_idx, kind)``."""
    factory_seen: set[tuple[int, str]] = set()
    factory_decls: list[tuple[native.FactoryIdxRef, str]] = []
    for module, names in module_to_names.items():
        for fref in (
            native.query(ctx).factories().of_module(module).where_name(sorted(names)).collect()
        ):
            for kind in fref.kinds:
                key = (fref.decl_idx, kind)
                if key in factory_seen:
                    continue
                factory_seen.add(key)
                factory_decls.append((fref, kind))
    return factory_decls


def _fetch_handlers(
    ctx: native.ProjectContext, registration_decorators: frozenset[str]
) -> list[native.DecoratorIdxRef]:
    return list(
        native.query(ctx).decorators().where_owner_attr(list(registration_decorators)).collect()
    )


def _build_vars_by_file(ctx: native.ProjectContext) -> dict[tuple[str, str], int]:
    """One project-wide variable scan + one ``node_attrs`` batch fetch.
    Result is plugin-agnostic, so the batched walker builds it once
    and hands the same dict to every plugin's gather."""
    vars_by_file: dict[tuple[str, str], int] = {}
    var_idxs = native.query(ctx).decls().with_kind("variable").indices()
    if not var_idxs:
        return vars_by_file
    var_attrs = ctx.node_attrs(var_idxs)
    for idx, (_kind, path, fqname, _flags) in zip(var_idxs, var_attrs, strict=True):
        simple = fqname.rsplit(".", 1)[-1]
        vars_by_file.setdefault((path, simple), idx)
    return vars_by_file


def _gather_one(ctx: native.ProjectContext, spec: DispatchAppSpec) -> DispatchAppGather | None:
    """Standalone single-plugin gather. Returns ``None`` when the spec's
    ``app_classes`` resolve to no usable ``{module: names}`` map —
    matches the legacy short-circuit ``DispatchAppPlugin.run`` did."""
    module_to_names = _module_to_names(ctx, spec.app_classes)
    if not module_to_names:
        return None
    return DispatchAppGather(
        spec=spec,
        direct=_fetch_direct(ctx, module_to_names),
        factory_decls=_fetch_factory_decls(ctx, module_to_names) if spec.seed_as_entrypoint else [],
        handlers=_fetch_handlers(ctx, spec.registration_decorators),
        vars_by_file=_build_vars_by_file(ctx),
    )


def _gather_batched(
    ctx: native.ProjectContext, specs: list[DispatchAppSpec]
) -> list[DispatchAppGather | None]:
    """Fused walker for a batch of specs. Returns one
    ``DispatchAppGather`` per input spec (in order), or ``None`` for
    any spec whose ``app_classes`` resolved empty.

    Fusion strategy:

    * Shared subclass-walk cache — each ``app_class`` fqn's
      transitive lookup runs at most once across all specs.
    * Per-distinct-module construction & factory queries — two specs
      both targeting ``flask.Flask`` share one query; rows route back
      to specs by matching ``class_name`` against each spec's map.
    * Single project-wide ``vars_by_file`` scan shared across specs.
    * Per-spec handler query (no fusion — the ref API doesn't carry
      which attr matched, so a fused result can't route to the right
      spec without further work).
    """
    if not specs:
        return []
    subclass_cache: dict[str, list[tuple[str, str]]] = {}
    module_to_names_per_spec: list[dict[str, set[str]]] = [
        _module_to_names(ctx, spec.app_classes, subclass_cache=subclass_cache) for spec in specs
    ]

    # Result slot per input spec; ``None`` for empty-map specs.
    gathered: list[DispatchAppGather | None] = [None] * len(specs)
    live_indices = [i for i, mtn in enumerate(module_to_names_per_spec) if mtn]
    if not live_indices:
        return gathered

    union_modules: dict[str, set[str]] = {}
    for i in live_indices:
        for module, names in module_to_names_per_spec[i].items():
            union_modules.setdefault(module, set()).update(names)

    # Per-spec direct constructions, deduped by var_idx.
    direct_per_spec: list[list[native.ConstructionIdxRef]] = [[] for _ in specs]
    direct_seen_per_spec: list[set[int]] = [set() for _ in specs]
    for module, all_names in union_modules.items():
        for ref in (
            native.query(ctx)
            .constructions()
            .where_module(module)
            .where_name(sorted(all_names))
            .collect()
        ):
            for i in live_indices:
                if ref.class_name not in module_to_names_per_spec[i].get(module, ()):
                    continue
                seen = direct_seen_per_spec[i]
                if ref.var_idx in seen:
                    continue
                seen.add(ref.var_idx)
                direct_per_spec[i].append(ref)

    # Per-spec factory decls (only for seed_as_entrypoint specs).
    factory_per_spec: list[list[tuple[native.FactoryIdxRef, str]]] = [[] for _ in specs]
    factory_seen_per_spec: list[set[tuple[int, str]]] = [set() for _ in specs]
    seed_indices = [i for i in live_indices if specs[i].seed_as_entrypoint]
    if seed_indices:
        seed_union_modules: dict[str, set[str]] = {}
        for i in seed_indices:
            for module, names in module_to_names_per_spec[i].items():
                seed_union_modules.setdefault(module, set()).update(names)
        for module, all_names in seed_union_modules.items():
            for fref in (
                native.query(ctx)
                .factories()
                .of_module(module)
                .where_name(sorted(all_names))
                .collect()
            ):
                for kind in fref.kinds:
                    for i in seed_indices:
                        if kind not in module_to_names_per_spec[i].get(module, ()):
                            continue
                        key = (fref.decl_idx, kind)
                        if key in factory_seen_per_spec[i]:
                            continue
                        factory_seen_per_spec[i].add(key)
                        factory_per_spec[i].append((fref, kind))

    vars_by_file = _build_vars_by_file(ctx)

    for i in live_indices:
        gathered[i] = DispatchAppGather(
            spec=specs[i],
            direct=direct_per_spec[i],
            factory_decls=factory_per_spec[i],
            handlers=_fetch_handlers(ctx, specs[i].registration_decorators),
            vars_by_file=vars_by_file,
        )
    return gathered


@dataclass(kw_only=True)
class BatchDispatchAppPlugin(Plugin):
    """Run multiple :class:`DispatchAppPlugin` instances with a fused
    gather pass + per-plugin policy.

    Architecture: split each wrapped plugin into a ``spec`` (the
    pure-data description of what to gather, see
    :class:`DispatchAppSpec`) and a ``policy(ctx, gathered)`` method
    (the per-plugin emission, see
    :meth:`DispatchAppPlugin.policy`). This driver runs a fused walk
    across every spec, then calls each plugin's :meth:`policy` with
    its slice of the gather. Subclass overrides of :meth:`policy` are
    honored uniformly — that's the point of the split.

    Fusion strategy (in the gather phase):

    * Shared subclass-walk cache — each ``app_class`` fqn's
      transitive lookup runs at most once regardless of how many
      specs request it.
    * Per-distinct-module construction & factory queries. Two specs
      both targeting ``flask.Flask`` (e.g. Flask + a project-level
      extension) share one query; rows route back to specs by
      matching ``class_name`` against each spec's map.
    * Single project-wide variable scan reused across plugins.

    Per-spec handler queries stay un-fused: the ref API doesn't carry
    which attribute matched, so a fused result can't route to the
    right spec without further work. Handler queries are
    text-prefiltered before any AST parse, so this stays cheap.
    """

    plugins: list[DispatchAppPlugin]
    name: str = "BatchDispatchApp"
    version: int = 1

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        active = [p for p in self.plugins if p._is_active(ctx)]
        if not active:
            return
        gathered = _gather_batched(ctx, [p.spec for p in active])
        for plugin, g in zip(active, gathered, strict=True):
            if g is None:
                continue
            yield from plugin.policy(ctx, g)


@dataclass(kw_only=True)
class LiteralListPlugin(Plugin):
    """Read ``<owner_fqname>.<variable_name>`` (a top-level list/tuple of
    string literals) and treat each entry as a fqname to keep alive.

    Each entry resolves against the assembled graph as either a module
    fqname (whole module surface revived, mirroring
    ``importlib.import_module``) or a single decl fqname.

    ``marker_prefix`` is the short label used in the ``<{marker}>:``
    synthetic fqname the plugin emits.
    """

    marker_prefix: str
    owner_fqname: str = ""
    variable_name: str = ""

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        if not self.owner_fqname or not self.variable_name:
            return
        var_fqname = f"{self.owner_fqname}.{self.variable_name}"
        # Read the variable's RHS directly via the targeted query.
        # The visitor doesn't emit ``var -> referent`` edges for
        # non-``__all__`` string-list assignments, so the plugin can't
        # rely on a descendant walk.
        entries = native.query(ctx).literal_lists().for_fqn(var_fqname).entries()
        if not entries:
            return

        prefix = f"<{self.marker_prefix}>:"
        # One batched scan of the module / decl index maps for every
        # entry, instead of N independent scans. Each entry resolves
        # as either a module fqname (revive the whole surface,
        # mirroring ``importlib.import_module``) or a single decl
        # fqname — try both; some entries may match both (e.g.
        # ``pkg.foo`` where ``foo`` is also a re-exported decl in
        # ``pkg/__init__.py``).
        surfaces = ctx.module_surfaces_indices(entries)
        for entry in entries:
            target_idxs: list[int] = list(surfaces.get(entry, ()))
            target_idxs.extend(native.query(ctx).declarations().with_fqname(entry).indices())
            if not target_idxs:
                continue
            (_kind, marker_path, _fq, _flags) = ctx.node_attrs([target_idxs[0]])[0]
            yield native.AddNodeByIdx(
                fqname=f"{prefix}{entry}",
                path=marker_path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=target_idxs,
            )
