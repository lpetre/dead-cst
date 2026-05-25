"""Plugin: keep Click command and sub-group handlers alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..plugins._base import native
from ..plugins.decl_shapes import DecoratedDeclPlugin

_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"command", "group", "result_callback"})
_SUBGROUP_DECORATOR: frozenset[str] = frozenset({"group"})
_GROUP_DECORATOR_NAMES: frozenset[str] = frozenset({"group", "Group"})
_GROUP_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"Group"})


@dataclass
class ClickPlugin(DecoratedDeclPlugin):
    """Wire Click command and sub-group handlers through their owning group.

    Click groups are not seeded as entrypoints; reachability flows
    through ``[project.scripts]`` / ``__main__`` / ``add_command``.
    """

    marker_prefix: str = "click"
    decorator_module: str = "click"
    decorator_names: frozenset[str] = _GROUP_DECORATOR_NAMES
    constructor_names: frozenset[str] = _GROUP_CONSTRUCTOR_NAMES

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        # Cheap import-presence guard, see ``DecoratedDeclPlugin.run``.
        if not native.query(ctx).imports().of(self.decorator_module).exists():
            return

        # groups_by_owner: (path, simple_name) -> [group_idx, ...]
        groups_by_owner: dict[tuple[str, str], list[int]] = {}

        def add_group(node_idx: int, path: str, fqname: str) -> None:
            simple = fqname.rsplit(".", 1)[-1]
            groups_by_owner.setdefault((path, simple), []).append(node_idx)

        dec_rows = (
            native.query(ctx)
            .decorators()
            .where_module(self.decorator_module)
            .where_name(list(self.decorator_names))
            .row_indices()
        )
        if dec_rows:
            dec_attrs = ctx.node_attrs([r.decorated_idx for r in dec_rows])
            for row, (_k, _p, fqname, _f) in zip(dec_rows, dec_attrs, strict=True):
                add_group(row.decorated_idx, row.path, fqname)

        cons_rows = (
            native.query(ctx)
            .constructions()
            .where_module(self.decorator_module)
            .where_name(list(self.constructor_names))
            .row_indices()
        )
        if cons_rows:
            cons_attrs = ctx.node_attrs([r.var_idx for r in cons_rows])
            for row, (_k, _p, fqname, _f) in zip(cons_rows, cons_attrs, strict=True):
                add_group(row.var_idx, row.path, fqname)

        handler_rows = list(
            native.query(ctx)
            .decorators()
            .where_owner_attr(list(_REGISTRATION_DECORATORS))
            .row_indices()
        )
        # Batched fqname fetch for every handler — used both in the
        # fixpoint dispatch and for the subgroup_links key.
        handler_attrs = (
            ctx.node_attrs([h.decorated_idx for h in handler_rows]) if handler_rows else []
        )
        handler_fqnames = [fq for (_k, _p, fq, _f) in handler_attrs]

        # Set of (decorated_idx, owner_name) pairs from the subgroup
        # decorator — used inside the fixpoint to upgrade a handler to
        # a group when its owner is already known.
        subgroup_links: set[tuple[int, str]] = {
            (h.decorated_idx, h.decorator_owner or "")
            for h in native.query(ctx)
            .decorators()
            .where_owner_attr(list(_SUBGROUP_DECORATOR))
            .row_indices()
        }

        # Dedup by (owner_idx, handler_idx) — idx is globally unique.
        emitted: set[tuple[int, int]] = set()
        changed = True
        while changed:
            changed = False
            for h, handler_fqname in zip(handler_rows, handler_fqnames, strict=True):
                owner_name = h.decorator_owner or ""
                for owner_idx in groups_by_owner.get((h.path, owner_name), []):
                    key = (owner_idx, h.decorated_idx)
                    if key in emitted:
                        continue
                    emitted.add(key)
                    yield native.AddEdgeByIdx(owner_idx, h.decorated_idx)
                    if (h.decorated_idx, owner_name) in subgroup_links:
                        add_group(h.decorated_idx, h.path, handler_fqname)
                        changed = True
