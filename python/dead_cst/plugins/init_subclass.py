"""Plugin: keep subclasses of ``__init_subclass__``-defining classes alive.

A class that defines ``__init_subclass__`` runs custom code every time
it is subclassed -- typically a registry pattern. Subclasses look
unused to a static analyzer because the registration is invisible; the
plugin makes every transitive subclass alive whenever its parent is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._base import Plugin, native

_INIT_SUBCLASS = "__init_subclass__"
INIT_SUBCLASS_PREFIX = "<__init_subclass__>:"


@dataclass
class InitSubclassPlugin(Plugin):
    """Wire subclasses of ``__init_subclass__``-defining classes through a marker."""

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        parent_idxs = native.query(ctx).classes().defining_method(_INIT_SUBCLASS).indices()
        # One batched attr fetch per matched parent — pulls fqname / path
        # in a single FFI hop instead of N per-attribute borrows.
        attrs = ctx.node_attrs(parent_idxs)
        for parent_idx, attr in zip(parent_idxs, attrs, strict=True):
            subclass_idxs = native.query(ctx).subclasses().of_idx(parent_idx).indices()
            yield native.AddNodeByIdx(
                fqname=f"{INIT_SUBCLASS_PREFIX}{attr.fqname}",
                path=attr.path,
                edges_from_idx=[parent_idx],
                edges_to_idx=subclass_idxs,
            )
