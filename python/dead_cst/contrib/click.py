"""Plugin: keep Click command and sub-group handlers alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..plugins.decl_shapes import DecoratedDeclPlugin

if TYPE_CHECKING:
    from dead_cst import _native as native

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

    name: str = "click"
    version: int = 1777760307
    decorator_module: str = "click"
    decorator_names: frozenset[str] = _GROUP_DECORATOR_NAMES
    constructor_names: frozenset[str] = _GROUP_CONSTRUCTOR_NAMES

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        groups_by_owner: dict[tuple[str, str], list[native.SymbolNode]] = {}

        def add_group(node: "native.SymbolNode") -> None:
            simple = node.fqname.rsplit(".", 1)[-1]
            groups_by_owner.setdefault((node.path, simple), []).append(node)

        for dec_ref in (
            native.query(ctx)
            .decorators()
            .where_module(self.decorator_module)
            .where_name(list(self.decorator_names))
        ):
            add_group(dec_ref.decorated)
        for cons_ref in (
            native.query(ctx)
            .constructions()
            .where_module(self.decorator_module)
            .where_name(list(self.constructor_names))
        ):
            add_group(cons_ref.var)

        handlers: list[tuple[str, native.SymbolNode]] = [
            (h.decorator_owner or "", h.decorated)
            for h in native.query(ctx).decorators().where_owner_attr(list(_REGISTRATION_DECORATORS))
        ]
        # Precompute (path, fqname, owner_name) triples for handlers
        # decorated with the subgroup decorator — used inside the
        # fixpoint to upgrade a handler to a group when its owner is
        # already known. Querying inside the loop would be O(N²).
        subgroup_links: set[tuple[str, str, str]] = {
            (h.decorated.path, h.decorated.fqname, h.decorator_owner or "")
            for h in native.query(ctx).decorators().where_owner_attr(list(_SUBGROUP_DECORATOR))
        }

        emitted: set[tuple[str, str, str, str]] = set()
        changed = True
        while changed:
            changed = False
            for owner_name, handler_func in handlers:
                for owner in groups_by_owner.get((handler_func.path, owner_name), []):
                    key = (owner.path, owner.fqname, handler_func.path, handler_func.fqname)
                    if key in emitted:
                        continue
                    emitted.add(key)
                    yield native.AddEdge(owner, handler_func)
                    if (handler_func.path, handler_func.fqname, owner_name) in subgroup_links:
                        add_group(handler_func)
                        changed = True
