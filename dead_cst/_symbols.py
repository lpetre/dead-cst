from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx


@dataclass(frozen=True, slots=True)
class Import:
    path: Path | str
    module: str
    decl: str | None = None


@dataclass(frozen=True, slots=True)
class SymbolNode:
    fqname: str
    type: Literal["module", "class", "function", "variable", "import"]
    path: Path
    imports: Import | None = None


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

    # Declarations in this module (classes, functions, variables, imports)
    # Keyed by the simple name (not fully qualified)
    declarations: dict[str, SymbolNode] = field(default_factory=dict)

    def add_declaration(self, decl: SymbolNode) -> None:
        """Add a declaration to the trie."""
        parts = decl.fqname.split(".")
        match decl.type:
            case "module":
                node = self._touch(parts)
                assert node.module is None, f"Module already exists at this node {decl.path}"
                node.module = decl
            case _:
                # Store declarations by their simple name
                parent, child = parts[:-1], parts[-1]
                node = self._touch(parent)
                assert node.module is not None, (
                    f"Module should exist when adding {decl.type} {decl.fqname}"
                )
                node.declarations[child] = decl

    def merge(self, other: SymbolTrie) -> SymbolTrie:
        """Merge another SymbolTrie into this one."""
        module_count = sum(1 for n in (self.module, other.module) if n is not None)
        assert module_count <= 1, "Cannot merge two SymbolTries with conflicting modules"
        for part, child in other.children.items():
            if part not in self.children:
                self.children[part] = SymbolTrie()
            self.children[part].merge(child)
        if other.module:
            assert self.module is None, "Cannot merge module into a node that already has a module"
            self.module = other.module
            self.declarations = other.declarations.copy()
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
