"""Plugin: keep module-level dunder variables and ``__future__`` imports alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import dead_cst_ty_native as native

DUNDER_PREFIX = "<dunder>:"


@dataclass
class ModuleDundersPlugin:
    """Keep module-level dunder variables and ``__future__`` imports alive.

    Variables named ``__xxx__`` at module scope (``__all__``, ``__version__``,
    ``__author__``, ``__license__``, ...) are read by external tooling and
    ``from __future__ import ...`` statements are compile-time directives;
    removing either would be observable even when no source references them.
    """

    name: str = "module_dunders"
    version: int = 1777760307

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        import dead_cst_ty_native as native

        for target in [
            *native.query(ctx).module_dunders(),
            *native.query(ctx).imports().of("__future__").collect(),
        ]:
            yield native.AddEntrypoint(target, marker="<dunder>")
