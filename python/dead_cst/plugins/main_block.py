"""Plugin: mark ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..graph import NodeFlags
from ._base import PerFilePlugin, native

MAIN_BLOCK_PREFIX = "<__main__>:"


@dataclass
class MainBlockPlugin(PerFilePlugin):
    """Mark ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module containing a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint with edges to (a) the containing
    module and (b) every top-level decl whose binding site falls inside
    the block's source range. The synthetic anchor is named
    ``"<__main__>:<module.fqname>"``.

    Runs per file during the populate phase; output salsa-cached so
    re-builds skip files that didn't change.
    """

    def run_per_file(self, file: native.FileScope) -> Iterable[object]:
        main = file.main_block()
        if main is None:
            return
        module_idx, decl_idxs = main
        yield native.PerFileNode(
            f"{MAIN_BLOCK_PREFIX}{file.module_fqname}",
            path=file.path,
            flags=int(NodeFlags.ENTRYPOINT),
            edges_to=[module_idx, *decl_idxs],
        )
