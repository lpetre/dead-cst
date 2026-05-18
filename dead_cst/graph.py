"""Public data types of the symbol graph.

:class:`SymbolNode` is what every node in the project graph is.
:class:`Import` captures a cross-file reference (the dotted module name
plus optional decl). :class:`NodeFlags` and :class:`EdgeFlags` mark
structural attributes (``SHADOWED`` decls, ``DEAD_BRANCH`` edges,
explicit ``ENTRYPOINT``\\s).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from libcst.metadata import CodeRange

logger = logging.getLogger(__name__)


class NodeFlags(enum.IntFlag):
    """Analyzer-internal marker on :class:`SymbolNode`."""

    NONE = 0
    SHADOWED = enum.auto()
    ENTRYPOINT = enum.auto()
    OVERLOAD = enum.auto()
    TESTCASE = enum.auto()
    NOQA = enum.auto()
    NOTEBOOK = enum.auto()


class EdgeFlags(enum.IntFlag):
    """Analyzer-internal marker on graph edges."""

    NONE = 0
    DEAD_BRANCH = enum.auto()
    DYNAMIC_IMPORT = enum.auto()


@dataclass(frozen=True, slots=True)
class Import:
    """Cross-file import reference attached to ``kind="import"`` nodes.

    ``module`` is the absolute dotted target (relative dots resolved);
    ``decl`` is the from-style imported name (``None`` for plain
    ``import`` and for star fan-out); ``star`` flags the implicit star.
    """

    module: str
    decl: str | None = None
    star: bool = False
    _hash: int = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.module, self.decl, self.star)))

    def __hash__(self) -> int:
        return self._hash


@dataclass(frozen=True, slots=True)
class SymbolNode:
    fqname: str
    type: Literal["module", "class", "function", "variable", "type_alias", "import", "synthetic"]
    path: Path
    position: CodeRange
    imports: Import | None = None
    flags: NodeFlags = NodeFlags.NONE
    _hash: int = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.fqname, self.type, self.path, self.position, self.imports, self.flags)),
        )

    def __hash__(self) -> int:
        return self._hash


__all__ = [
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolNode",
]
