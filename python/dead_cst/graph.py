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
    """Analyzer-internal marker on :class:`SymbolNode`.

    Each ``KEEPALIVE`` bit (:data:`ENTRYPOINT`, :data:`TESTCASE`,
    :data:`NOQA`, :data:`NOTEBOOK`) independently says "the reachability
    BFS seeds from nodes carrying this bit." :data:`KEEPALIVE_DEFAULT`
    ORs all of them; pass a subset to
    :meth:`Analysis.reachable(seed_flags=...)` /
    :meth:`Analysis.dead(seed_flags=...)` to restrict the seeds.

    The bits are independent metadata — ``TESTCASE`` alone keeps a node
    alive; it does not have to be ORed with ``ENTRYPOINT``. That lets
    callers ask focused questions: "what's alive ignoring tests?" is
    ``reachable(seed_flags=KEEPALIVE_DEFAULT & ~NodeFlags.TESTCASE)``.
    """

    NONE = 0
    SHADOWED = enum.auto()
    ENTRYPOINT = enum.auto()
    OVERLOAD = enum.auto()
    TESTCASE = enum.auto()
    NOQA = enum.auto()
    NOTEBOOK = enum.auto()


#: Default seed mask for reachability queries. ORs together every
#: :class:`NodeFlags` bit that semantically "keeps a node alive":
#:
#: * :data:`NodeFlags.ENTRYPOINT` — explicit entrypoints (plugin-emitted
#:   seeds, ``-e`` CLI flag, ``[project.scripts]`` targets, ...).
#: * :data:`NodeFlags.TESTCASE` — pytest / unittest discoveries.
#: * :data:`NodeFlags.NOQA` — imports pinned by a ``# noqa: F401``
#:   directive.
#: * :data:`NodeFlags.NOTEBOOK` — notebook cells (run top-to-bottom,
#:   never imported, always alive).
#:
#: :meth:`Analysis.reachable` / :meth:`Analysis.dead` default to this
#: mask. Pass a subset to scope the question — e.g.
#: ``reachable(seed_flags=NodeFlags.ENTRYPOINT)`` asks "what would be
#: alive if the test suite, ``noqa`` pins, and notebooks didn't exist."
KEEPALIVE_DEFAULT: NodeFlags = (
    NodeFlags.ENTRYPOINT | NodeFlags.TESTCASE | NodeFlags.NOQA | NodeFlags.NOTEBOOK
)


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
    "KEEPALIVE_DEFAULT",
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolNode",
]
