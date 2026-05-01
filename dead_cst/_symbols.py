from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx
from libcst.metadata import CodeRange

logger = logging.getLogger(__name__)


class NodeFlags(enum.IntFlag):
    """Analyzer-internal marker on :class:`SymbolNode`.

    ``SHADOWED`` decls are emitted into the graph (with their parent
    module edge) but are excluded from the lookup trie, so cross-module
    imports never resolve to them.

    ``ENTRYPOINT`` flags a node as a reachability seed: ``_apply_payload``
    sets ``graph.nodes[node]["entrypoint"] = True`` when it sees the
    flag, so :func:`find_reachable` starts its BFS there. Plugins emit
    flagged synthetic nodes via their per-file payloads to declare
    entrypoints without a separate API surface.
    """

    NONE = 0
    SHADOWED = enum.auto()
    ENTRYPOINT = enum.auto()


class EdgeFlags(enum.IntFlag):
    """Analyzer-internal marker on graph edges.

    ``DEAD_BRANCH`` flags an edge whose reference originated inside a
    statically-dead suite (per :func:`dead_cst._branches.unreachable_suites`).
    Default :func:`dead_cst.find_reachable` does **not** filter by this
    flag -- today's behavior, where dead-code references propagate
    liveness through the enclosing decl, is preserved. The opt-in
    :func:`dead_cst.find_kept_alive_by_dead_branches` returns the
    "blast radius" of removing every dead suite by skipping these edges.
    """

    NONE = 0
    DEAD_BRANCH = enum.auto()


@dataclass(frozen=True, slots=True)
class Import:
    path: Path | str
    module: str
    decl: str | None = None
    star: bool = False


@dataclass(frozen=True, slots=True)
class SymbolNode:
    fqname: str
    type: Literal["module", "class", "function", "variable", "import", "synthetic"]
    path: Path
    position: CodeRange
    imports: Import | None = None
    flags: NodeFlags = NodeFlags.NONE


@dataclass(slots=True)
class SymbolTrie:
    """
    Trie that tracks module hierarchy.
    Each node represents a potential module and can have declarations.
    """

    # Children represent submodules/subpackages
    children: dict[str, SymbolTrie] = field(default_factory=dict)

    # Module information at this node (if this represents an actual module)
    module: SymbolNode | None = None

    # Declarations in this module, keyed by simple name. A name maps to the
    # set of decls live at module exit: usually one decl, but multiple when
    # every branch of a conditional binds the same name (``if X: def f...
    # else: def f...``). All branches are legitimate runtime values and a
    # ``from mod import name`` should reach each of them. Decls displaced
    # before module exit are tagged with ``NodeFlags.SHADOWED`` and tracked
    # outside the trie -- the trie only stores entries that should be
    # reachable from cross-module imports.
    declarations: dict[str, list[SymbolNode]] = field(default_factory=dict)

    def add_declaration(self, decl: SymbolNode) -> None:
        """Add a declaration to the trie.

        Decls flagged ``SHADOWED`` should never reach this method; the
        visitor flags them and routes them around the trie. De-duplicates
        so the visitor's double-push for ``Assign`` (one push for the LHS
        name, one for the RHS value) does not register the same decl twice.
        """
        parts = decl.fqname.split(".")
        match decl.type:
            case "module":
                node = self._touch(parts)
                assert node.module is None, f"Module already exists at this node {decl.path}"
                node.module = decl
            case _:
                parent, child = parts[:-1], parts[-1]
                node = self._touch(parent)
                assert node.module is not None, (
                    f"Module should exist when adding {decl.type} {decl.fqname}"
                )
                bucket = node.declarations.setdefault(child, [])
                if decl not in bucket:
                    bucket.append(decl)

    def remove_declaration(self, decl: SymbolNode) -> None:
        """Remove a single declaration entry by identity / equality.

        Used by the visitor when flow analysis determines a previously
        added decl is shadowed: the SHADOWED-flagged copy stays out of
        the trie, and any unflagged version that reached the trie via
        ``add_declaration`` is dropped here.
        """
        parts = decl.fqname.split(".")
        node = self._get(parts[:-1])
        if node is None:
            return
        bucket = node.declarations.get(parts[-1])
        if bucket is None:
            return
        try:
            bucket.remove(decl)
        except ValueError:
            return
        if not bucket:
            del node.declarations[parts[-1]]

    def merge(self, other: SymbolTrie) -> SymbolTrie:
        """Merge another SymbolTrie into this one.

        When both sides hold a module at the same FQN (a real packaging
        collision -- two exported roots both shipping a package with the
        same top-level name), the already-merged module wins and the
        incoming one is dropped with a warning. Callers control the
        precedence order by the order of their ``merge()`` calls; today
        :func:`build_symbol_graph` merges the consumer's own trie first
        and each dep's exported trie afterwards, so own-module always
        beats dep-module on legitimate conflict.
        """
        for part, child in other.children.items():
            if part not in self.children:
                self.children[part] = SymbolTrie()
            self.children[part].merge(child)
        if other.module is None:
            return self
        if self.module is not None:
            logger.warning(
                "SymbolTrie collision at %s: keeping %s, dropping %s",
                self.module.fqname,
                self.module.path,
                other.module.path,
            )
            return self
        self.module = other.module
        self.declarations = {k: list(v) for k, v in other.declarations.items()}
        return self

    def _touch(self, parts: list[str]) -> SymbolTrie:
        node = self
        for p in parts:
            node = node.children.setdefault(p, SymbolTrie())
        return node

    def _get(self, parts: list[str]) -> SymbolTrie | None:
        node = self
        for p in parts:
            node = node.children.get(p)
            if node is None:
                return None
        return node

    def add_module_hierarchy_edges(self, symbol_graph: nx.DiGraph) -> None:
        """
        Walk the trie and add edges from each submodule to its parent module.

        For a structure like:
            v6/
            ├── __init__.py
            └── nntree/
                ├── __init__.py
                └── extern/
                    ├── __init__.py
                    └── clip.py

        This will add edges:
            v6.nntree -> v6
            v6.nntree.extern -> v6.nntree
            v6.nntree.extern.clip -> v6.nntree.extern
        """

        def _walk_and_add_edges(node: SymbolTrie, parent_module: SymbolNode | None):
            """Recursive helper to walk the trie and add edges."""
            # If this node represents a module and has a parent, add an edge
            if node.module and parent_module:
                symbol_graph.add_edge(node.module, parent_module)

            # Recurse into children, passing this node's module as the parent
            for child_node in node.children.values():
                _walk_and_add_edges(child_node, node.module)

        # Start the walk from the root
        _walk_and_add_edges(self, None)
