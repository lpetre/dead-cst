"""Plugin: treat ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..graph import NodeFlags
from ._base import Plugin, native

MAIN_BLOCK_PREFIX = "<__main__>:"


@dataclass
class MainBlockPlugin(Plugin):
    """Treat ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module containing a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint with edges to (a) the containing
    module and (b) every top-level decl whose binding site falls inside
    the block's source range.
    """

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        for module, block_decls in ctx.find_main_blocks():
            yield native.AddNode(
                fqname=f"{MAIN_BLOCK_PREFIX}{module.fqname}",
                path=module.path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to=[module, *block_decls],
            )
