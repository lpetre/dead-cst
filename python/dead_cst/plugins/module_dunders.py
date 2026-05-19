"""Plugin: keep module-level dunder variables and ``__future__`` imports alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._base import Plugin, native

DUNDER_PREFIX = "<dunder>:"


@dataclass
class ModuleDundersPlugin(Plugin):
    """Keep module-level dunder variables and ``__future__`` imports alive.

    Variables named ``__xxx__`` at module scope (``__all__``, ``__version__``,
    ``__author__``, ``__license__``, ...) are read by external tooling and
    ``from __future__ import ...`` statements are compile-time directives;
    removing either would be observable even when no source references them.
    """

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        for target in [
            *ctx.find_module_dunders(),
            *native.query(ctx).imports().of("__future__").collect(),
        ]:
            yield native.AddEntrypoint(target, marker="<dunder>")
