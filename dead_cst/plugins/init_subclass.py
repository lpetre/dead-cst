"""Plugin: keep subclasses of ``__init_subclass__``-defining classes alive.

A class that defines ``__init_subclass__`` runs custom code every time
it is subclassed -- typically a registry pattern. Subclasses look
unused to a static analyzer because the registration is invisible; the
plugin makes every transitive subclass alive whenever its parent is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from dead_cst import _native as native

_INIT_SUBCLASS = "__init_subclass__"
INIT_SUBCLASS_PREFIX = "<__init_subclass__>:"


@dataclass
class InitSubclassPlugin:
    """Wire subclasses of ``__init_subclass__``-defining classes through a marker."""

    name: str = "init_subclass"
    version: int = 1777760307

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        for parent in ctx.find_classes_defining_method(_INIT_SUBCLASS):
            yield native.AddNode(
                fqname=f"{INIT_SUBCLASS_PREFIX}{parent.fqname}",
                path=parent.path,
                edges_from=[parent],
                edges_to=ctx.find_subclasses_of(parent),
            )
