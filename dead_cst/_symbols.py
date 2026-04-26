from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx
from libcst.metadata import CodeRange

logger = logging.getLogger(__name__)


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
    # ``from mod import name`` should reach each of them.
    declarations: dict[str, list[SymbolNode]] = field(default_factory=dict)

    # Decls strictly displaced by a later same-name decl on every path to
    # module exit. They still belong to this module -- the symbol graph
    # needs them so the ``decl -> module`` parent edge is emitted for
    # shadowed decls too -- but they are not exported.
    shadowed: list[SymbolNode] = field(default_factory=list)

    def add_declaration(self, decl: SymbolNode) -> None:
        """Add a declaration to the trie.

        For non-module decls this just collects: the trie does not yet
        know which decls are live at module exit. ``finalize_declarations``
        partitions same-name decls into ``declarations`` (live exports)
        and ``shadowed`` (strictly displaced). De-duplicates so the
        visitor's double-push for ``Assign`` (one push for the LHS name,
        one for the RHS value) does not register the same decl twice.
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

    def finalize_declarations(
        self,
        name: str,
        live: list[SymbolNode],
        shadowed: list[SymbolNode],
    ) -> None:
        """Partition collected decls for ``name`` into live vs shadowed.

        ``live`` is the set of decls that bind ``name`` on at least one
        path to module exit. ``shadowed`` is the rest. The caller is
        expected to derive both from a flow-sensitive walk of the
        module body (see :func:`dead_cst._flow.live_at_exit`).
        """
        if live:
            self.declarations[name] = list(live)
        else:
            self.declarations.pop(name, None)
        self.shadowed.extend(shadowed)

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
        self.shadowed = list(other.shadowed)
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
