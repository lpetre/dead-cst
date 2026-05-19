"""Public data types of the symbol graph.

:class:`SymbolNode` is what every node in the project graph is.
:class:`Import` captures a cross-file reference (the dotted module
name plus optional decl). :class:`NodeFlags` and :class:`EdgeFlags`
mark structural attributes (``SHADOWED`` decls, ``DEAD_BRANCH`` edges,
explicit ``ENTRYPOINT``\\s).

All four are re-exported straight from the rust extension
(:mod:`dead_cst._native`) — there is no parallel Python copy anymore.
"""

from __future__ import annotations

from dead_cst._native import (
    EdgeFlags,
    Import,
    NodeFlags,
    SymbolNode,
)

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
KEEPALIVE_DEFAULT: int = (
    NodeFlags.ENTRYPOINT | NodeFlags.TESTCASE | NodeFlags.NOQA | NodeFlags.NOTEBOOK
)


__all__ = [
    "KEEPALIVE_DEFAULT",
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolNode",
]
