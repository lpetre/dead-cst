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
            module_idx = ctx.module_for_indices(path)
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
            .row_indices()
        ):
            if in_scope(dec_row.path):
                seeds_by_path.setdefault(dec_row.path, []).append(dec_row.decorated_idx)
        for cons_row in (
            native.query(ctx)
            .constructions()
            .where_module(self.decorator_module)
            .where_name(names)
            .row_indices()
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
    """

    marker_prefix: str
    app_classes: tuple[str, ...] = ()
    registration_decorators: frozenset[str] = frozenset()
    seed_as_entrypoint: bool = True

    def _prefix(self, kind: str) -> str:
        return f"<{self.marker_prefix}-{kind}>:"

    def _module_to_names(self, ctx: native.ProjectContext) -> dict[str, set[str]]:
        """Expand each ``app_class`` into ``{module: {simple_name, ...}}``,
        walking subclasses transitively. Lets module-keyed queries
        (``where_module(...).where_name(...)``,
        ``of_module(...).where_name(...)``) discover project / third-party
        subclasses whose import module differs from the base class.

        Subclasses are resolved by inverting the search: instead of
        asking ty's ``find_references`` to walk down from each
        framework class (which forces ty to load + parse the framework
        out of the venv — ~100-400ms per framework on cold cache), we
        do one parallel pass over project files reading each
        ``ClassDef``'s base list and matching against the configured
        app-class fqnames. The framework module is never loaded.
        """
        out: dict[str, set[str]] = {}
        for fqn in self.app_classes:
            module, _, name = fqn.rpartition(".")
            if not module or not name:
                continue
            out.setdefault(module, set()).add(name)
        if not self.app_classes:
            return out
        # One batched attr fetch per framework's subclass set.
        for fqn in self.app_classes:
            sub_idxs = ctx.subclasses_of_fqn_indices(fqn)
            if not sub_idxs:
                continue
            for _kind, _path, sub_fqname, _flags in ctx.node_attrs(sub_idxs):
                sub_module, _, sub_name = sub_fqname.rpartition(".")
                if sub_module and sub_name:
                    out.setdefault(sub_module, set()).add(sub_name)
        return out

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        if not (self.app_classes and self.registration_decorators):
            return
        # Cheap import-presence guard. ``_module_to_names`` triggers
        # ``subclasses().of_fqn(...)`` for every ``app_class``, which
        # forces ty to load the framework from the venv (parse + build
        # SemanticIndex + walk the class hierarchy) — ~100-400ms per
        # framework. If nothing imports the framework's root package,
        # the project can't possibly contain an instance of an
        # ``app_class``, so skip the work.
        app_modules = {fqn.rpartition(".")[0] for fqn in self.app_classes if "." in fqn}
        if not any(native.query(ctx).imports().of(m).exists() for m in app_modules if m):
            return
        decorator_attrs = list(self.registration_decorators)

        module_to_names = self._module_to_names(ctx)
        if not module_to_names:
            return

        # 1. Direct constructions, deduped by var_idx (every construction
        # site has a globally unique node index, so the same physical
        # site that matches under two ``app_classes`` entries — e.g.
        # one is a base of the other — collapses on idx).
        direct_seen: set[int] = set()
        direct: list[native.ConstructionIdxRef] = []
        for module, names in module_to_names.items():
            for ref in (
                native.query(ctx)
                .constructions()
                .where_module(module)
                .where_name(sorted(names))
                .row_indices()
            ):
                if ref.var_idx in direct_seen:
                    continue
                direct_seen.add(ref.var_idx)
                direct.append(ref)

        # 2. Factory functions returning one of the listed classes,
        # deduped by (decl_idx, kind). decl_idx is globally unique;
        # kind is part of the key because one decl can match multiple
        # kinds. Only relevant when we'd actually do something with the
        # markers in step 6.
        factory_seen: set[tuple[int, str]] = set()
        factory_decls: list[tuple[native.FactoryIdxRef, str]] = []
        if self.seed_as_entrypoint:
            for module, names in module_to_names.items():
                for fref in (
                    native.query(ctx)
                    .factories()
                    .of_module(module)
                    .where_name(sorted(names))
                    .row_indices()
                ):
                    for kind in fref.kinds:
                        key = (fref.decl_idx, kind)
                        if key in factory_seen:
                            continue
                        factory_seen.add(key)
                        factory_decls.append((fref, kind))

        handlers = list(
            native.query(ctx).decorators().where_owner_attr(decorator_attrs).row_indices()
        )

        # direct_by_owner: (path, simple_name) -> [var_idx, ...]
        direct_by_owner: dict[tuple[str, str], list[int]] = {}
        if direct:
            direct_attrs = ctx.node_attrs([r.var_idx for r in direct])
            for ref, (_kind, _path, fqname, _flags) in zip(direct, direct_attrs, strict=True):
                simple = fqname.rsplit(".", 1)[-1]
                direct_by_owner.setdefault((ref.path, simple), []).append(ref.var_idx)

        # vars_by_file: (path, simple_name) -> var_idx. First idx wins.
        vars_by_file: dict[tuple[str, str], int] = {}
        var_idxs = native.query(ctx).decls().with_kind("variable").indices()
        if var_idxs:
            var_attrs = ctx.node_attrs(var_idxs)
            for idx, (_kind, path, fqname, _flags) in zip(var_idxs, var_attrs, strict=True):
                simple = fqname.rsplit(".", 1)[-1]
                vars_by_file.setdefault((path, simple), idx)

        app_prefix = self._prefix("app")
        factory_prefix = self._prefix("factory")

        # 3. Entrypoint-promote every direct construction (when enabled).
        if self.seed_as_entrypoint and direct:
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
            if self.seed_as_entrypoint:
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
        if self.seed_as_entrypoint and factory_decls:
            factory_reachers: set[int] = set()
            for fref, _kind in factory_decls:
                factory_reachers.update(ctx.direct_predecessors_idx(fref.decl_idx))

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
        entries = ctx.find_literal_list_entries(var_fqname)
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
            target_idxs.extend(ctx.find_declarations_indices(entry))
            if not target_idxs:
                continue
            (_kind, marker_path, _fq, _flags) = ctx.node_attrs([target_idxs[0]])[0]
            yield native.AddNodeByIdx(
                fqname=f"{prefix}{entry}",
                path=marker_path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=target_idxs,
            )
