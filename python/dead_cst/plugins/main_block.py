"""Plugin: mark ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..graph import NodeFlags
from ._base import Plugin, native

MAIN_BLOCK_PREFIX = "<__main__>:"


@dataclass
class MainBlockPlugin(Plugin):
    """Mark ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module containing a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint with edges to (a) the containing
    module and (b) every top-level decl whose binding site falls inside
    the block's source range.
    """

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        pairs = ctx.find_main_blocks_indices()
        if not pairs:
            return
        # Batch-fetch (fqname, path) for every matched module in one hop.
        module_idxs = [m for (m, _decls) in pairs]
        module_attrs = ctx.node_attrs(module_idxs)
        for (module_idx, decl_idxs), (_kind, module_path, module_fqname, _flags) in zip(
            pairs, module_attrs, strict=True
        ):
            yield native.AddNodeByIdx(
                fqname=f"{MAIN_BLOCK_PREFIX}{module_fqname}",
                path=module_path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=[module_idx, *decl_idxs],
            )
