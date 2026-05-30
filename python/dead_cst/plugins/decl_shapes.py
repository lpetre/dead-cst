"""Reusable plugin shapes for dynamic-import discovery patterns.

Two abstract bases that target idioms framework-flavoured codebases
keep stumbling into:

  :class:`DecoratedDeclPlugin`
    "Find decorated decls in files matching a search path." The same
    shape Click uses to detect its framework instances.

  :class:`LiteralListPlugin`
    "Read a top-level ``X = [\"...\", \"...\"]`` literal and treat each
    fqname inside it as alive."
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
            fqname = ctx.node_attrs([module_idx])[0].fqname
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
            marker_path = ctx.node_attrs([target_idxs[0]])[0].path
            yield native.AddNodeByIdx(
                fqname=f"{prefix}{entry}",
                path=marker_path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=target_idxs,
            )
