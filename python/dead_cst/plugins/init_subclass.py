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
        for parent in native.query(ctx).classes().defining_method(_INIT_SUBCLASS).collect():
            yield native.AddNode(
                fqname=f"{INIT_SUBCLASS_PREFIX}{parent.fqname}",
                path=parent.path,
                edges_from=[parent],
                edges_to=native.query(ctx).subclasses().of_node(parent).collect(),
            )
