"""Plugin: keep module-level dunder variables and ``__future__`` imports alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._base import Plugin, native


@dataclass
class ModuleDundersPlugin(Plugin):
    """Keep module-level dunder names and ``__future__`` imports alive.

    Variables named ``__xxx__`` at module scope (``__all__``, ``__version__``,
    ``__author__``, ``__license__``, ...) are read by external tooling, and
    PEP 562 module-level dunder *functions* (``__getattr__``, ``__dir__``)
    are called by the import / attribute-access machinery. ``from __future__
    import ...`` statements are compile-time directives. All three are
    observable even when no source reference points at them, so the plugin
    pins them as entrypoints.
    """

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        for target_idx in [
            *ctx.find_module_dunders_indices(),
            *native.query(ctx).imports().of("__future__").indices(),
        ]:
            yield native.AddEntrypointByIdx(target_idx, marker="<dunder>")
