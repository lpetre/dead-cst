"""Public data types of the symbol graph.

:class:`SymbolNode` is what every node in the graph
:func:`dead_cst.analyze.build_symbol_graph` returns is. :class:`Import`
captures a cross-file reference at visitor time, before the per-package
edge stitcher resolves it. :class:`NodeFlags` and :class:`EdgeFlags`
mark structural attributes (``SHADOWED`` decls, ``DEAD_BRANCH`` edges,
explicit ``ENTRYPOINT``\\s).

:class:`VisitorPayload` is the analyzer's per-file output -- the
serializable shape stored in :class:`~dead_cst.cache.GraphCache` and
the type :meth:`dead_cst.plugins.EdgePlugin.observe` returns.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import rustworkx as rx
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

    ``OVERLOAD`` flags a ``typing.overload`` stub (or any same-name
    displaced sibling). Excluded from the lookup trie like
    ``SHADOWED``, but its lifetime is anchored to the matching impl
    via explicit ``impl -> overload`` edges so the codemod removes
    the overload alongside a dead impl rather than in isolation.

    ``TESTCASE`` tags an entrypoint as test-only (pytest / unittest
    discovery seeds, fixture seeds, etc.). It is metadata on top of
    ``ENTRYPOINT``: default :func:`find_reachable` treats those seeds
    the same as any other entrypoint, but the opt-in
    ``Analysis.kept_alive_by_flags_only(NodeFlags.TESTCASE)`` query
    returns the "blast radius" of dropping the test suite.

    ``NOQA`` tags an entrypoint as preserved by an explicit user
    directive -- today, an import whose source line carries a
    ruff/pyflakes ``# noqa[: ...F401...]`` (or whose file carries a
    ``# ruff: noqa`` / ``# flake8: noqa``). Like ``TESTCASE`` it is
    metadata on top of ``ENTRYPOINT``, queried via
    ``Analysis.kept_alive_by_flags_only(NodeFlags.NOQA)`` for the
    "blast radius" of removing every F401-pinned import.

    ``NOTEBOOK`` tags every node sourced from a Jupyter ``.ipynb`` file.
    Notebooks are not importable modules; they're stamped
    ``NOTEBOOK | ENTRYPOINT`` on every node so the whole file's content
    is a reachability seed. The codemod uses this flag to skip
    notebooks (cell-aware writeback is out of scope today).
    """

    NONE = 0
    SHADOWED = enum.auto()
    ENTRYPOINT = enum.auto()
    OVERLOAD = enum.auto()
    TESTCASE = enum.auto()
    NOQA = enum.auto()
    NOTEBOOK = enum.auto()


class EdgeFlags(enum.IntFlag):
    """Analyzer-internal marker on graph edges.

    ``DEAD_BRANCH`` flags an edge whose reference originated inside a
    statically-dead suite (per :func:`dead_cst.branches.unreachable_suites`).
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
    """Raw, pre-resolution record of one cross-file reference.

    Every field is what the source code literally said: ``module`` is
    the dotted name written in the ``from <module> import ...`` (or
    ``import <module>``) clause -- no submodule-vs-name disambiguation
    has happened yet. The edge stitcher
    (:func:`dead_cst._edges.resolve_edges`) is the single place that
    classifies the target (first-party module, decl in a module,
    submodule, stdlib, external dist, ...) and may rewrite ``module`` /
    ``decl`` to the canonical ``deepest-module + remainder`` split.

    ``speculative`` is set on the synthetic star imports
    :class:`~dead_cst._visitor.SymbolVisitor` produces for
    ``__import__(name, fromlist=[...])`` / ``importlib.import_module``
    fromlist entries that may or may not be submodules. The stitcher
    silently drops a speculative entry when neither the trie nor the
    resolver can place it; non-speculative misses are surfaced as
    ``[unresolved] <top-level>`` synthetic nodes instead.
    """

    module: str
    decl: str | None = None
    star: bool = False
    speculative: bool = False
    # Memoized: ``resolve_edges._emit`` probes ``(src, dst, flags)`` once
    # per emission and ``SymbolNode.__hash__`` recurses into this, so the
    # same instance gets re-hashed many times per analysis.
    _hash: int = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_hash", hash((self.module, self.decl, self.star, self.speculative))
        )

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
    # See ``Import._hash`` for the rationale.
    _hash: int = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.fqname, self.type, self.path, self.position, self.imports, self.flags)),
        )

    def __hash__(self) -> int:
        return self._hash


@dataclass(frozen=True, slots=True)
class VisitorPayload:
    """Serializable per-file output of the analyzer's symbol visitor.

    Four fields cover everything the analyzer needs to reconstruct one
    file's contribution to the symbol graph:

    * ``nodes`` -- every real ``SymbolNode`` for this file (module +
      top-level decls). Decls displaced by flow analysis are flagged
      :data:`NodeFlags.SHADOWED`; the apply step uses that flag to keep
      them out of the lookup trie while still emitting the parent-module
      edge for the graph.
    * ``edges`` -- ``(src, dst, access_pos)`` triples for resolved
      decl-to-decl references. ``access_pos`` is the source location
      of the reference; the apply step compares it against
      ``dead_suites`` to decide whether the resulting graph edge gets
      :data:`EdgeFlags.DEAD_BRANCH`.
    * ``imports`` -- ``(src, Import, access_pos)`` triples for
      cross-file references. Each :class:`Import` carries only the
      raw dotted names from source; classification + canonicalization
      happen in :func:`dead_cst._edges.resolve_edges`, which the
      apply step feeds along with the derived flag.
    * ``dead_suites`` -- positions of every statically-dead suite in
      the file (including ones with no outgoing references). Used both
      for flag derivation and for surfacing "this file has unreachable
      code at line X" reports without per-edge attribution.
    """

    nodes: tuple[SymbolNode, ...]
    edges: tuple[tuple[SymbolNode, SymbolNode, CodeRange], ...]
    imports: tuple[tuple[SymbolNode, Import, CodeRange], ...]
    dead_suites: tuple[CodeRange, ...]


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

        Module-FQN collisions across files (``foo.py`` alongside
        ``foo/__init__.py``) are resolved upstream by
        :func:`dead_cst._refresh.shadowed_paths`: the loser's payload is
        applied to the graph but skipped at the trie, so this method
        never sees two ``module`` decls at the same node.
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

    def add_module_hierarchy_edges(self, symbol_graph: SymbolGraph) -> None:
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


class _NodesView:
    """View over :class:`SymbolGraph`'s nodes.

    Iterable (yields :class:`SymbolNode`), indexable
    (``view[node]`` returns the node's mutable attribute dict, creating
    it on first access), callable (``view(data=True)`` yields
    ``(node, attrs)`` pairs). Mirrors ``networkx``'s ``NodeView`` for
    the patterns the analyzer relies on.
    """

    __slots__ = ("_graph",)

    def __init__(self, graph: SymbolGraph) -> None:
        self._graph = graph

    def __iter__(self) -> Iterator[SymbolNode]:
        g = self._graph._g
        for idx in g.node_indices():
            yield g[idx]

    def __len__(self) -> int:
        return self._graph._g.num_nodes()

    def __contains__(self, node: object) -> bool:
        return node in self._graph._idx

    def __getitem__(self, node: SymbolNode) -> dict[str, Any]:
        idx = self._graph._idx[node]
        attrs = self._graph._node_attrs.get(idx)
        if attrs is None:
            attrs = {}
            self._graph._node_attrs[idx] = attrs
        return attrs

    def __call__(self, *, data: bool = False) -> Iterator[Any]:
        g = self._graph._g
        node_attrs = self._graph._node_attrs
        if data:
            for idx in g.node_indices():
                yield g[idx], node_attrs.get(idx, _EMPTY_ATTRS)
        else:
            for idx in g.node_indices():
                yield g[idx]


class _EdgesView:
    """View over :class:`SymbolGraph`'s edges.

    Callable: ``view()`` / ``view(data=True)`` / ``view(keys=True)`` /
    ``view(data=True, keys=True)`` yield 2-, 3-, 3-, or 4-tuples
    respectively, one entry per edge instance (parallel edges are not
    deduped, matching ``networkx.MultiDiGraph``).
    """

    __slots__ = ("_graph",)

    def __init__(self, graph: SymbolGraph) -> None:
        self._graph = graph

    def __call__(self, *, data: bool = False, keys: bool = False) -> Iterator[Any]:
        g = self._graph._g
        if data and keys:
            for edge_idx, (u_idx, v_idx, payload) in g.edge_index_map().items():
                yield g[u_idx], g[v_idx], edge_idx, payload
        elif data:
            for u_idx, v_idx, payload in g.weighted_edge_list():
                yield g[u_idx], g[v_idx], payload
        elif keys:
            for edge_idx, (u_idx, v_idx, _payload) in g.edge_index_map().items():
                yield g[u_idx], g[v_idx], edge_idx
        else:
            for u_idx, v_idx in g.edge_list():
                yield g[u_idx], g[v_idx]


# Shared sentinel for ``nodes(data=True)`` on un-attributed nodes. The
# default dict has to be read-only at the call site -- callers either
# query ``.get("entrypoint")`` (safe) or use ``nodes[node]["..."] = ...``
# which routes through :meth:`_NodesView.__getitem__` and materializes
# a fresh dict.
_EMPTY_ATTRS: dict[str, Any] = {}


class SymbolGraph:
    """The analyzer's directed multigraph of symbol references.

    Backed by :class:`rustworkx.PyDiGraph` with ``multigraph=True``;
    nodes are :class:`SymbolNode` instances, edges carry an attribute
    dict (typically ``{"flags": EdgeFlags}``). The wrapper owns the
    ``SymbolNode -> int`` index map so callers address nodes by their
    domain identity while rustworkx operates on integer indices
    internally. Per-node attributes (``entrypoint``, ``testcase``)
    live in a side table rather than on the rustworkx payload.

    Duplicate ``(src, dst, attrs)`` inserts are silently dropped.
    Metadata-distinct parallel edges (e.g. one ``DEAD_BRANCH`` plus
    one ``NONE``) are preserved -- strict-mode reachability filters
    on attrs, so collapsing those would lose fidelity.
    """

    __slots__ = ("_g", "_idx", "_node_attrs", "_edge_keys", "graph")

    def __init__(self) -> None:
        self._g: rx.PyDiGraph = rx.PyDiGraph(multigraph=True)
        self._idx: dict[SymbolNode, int] = {}
        self._node_attrs: dict[int, dict[str, Any]] = {}
        self._edge_keys: set[tuple[int, int, tuple]] = set()
        # ``graph`` mirrors networkx's graph-level attribute dict
        # (``g.graph["dead_suites"]``).
        self.graph: dict[str, Any] = {}

    @property
    def nodes(self) -> _NodesView:
        return _NodesView(self)

    @property
    def edges(self) -> _EdgesView:
        return _EdgesView(self)

    def add_node(self, node: SymbolNode) -> int:
        """Idempotent: returns the existing index if ``node`` is already in the graph."""
        idx = self._idx.get(node)
        if idx is None:
            idx = self._g.add_node(node)
            self._idx[node] = idx
        return idx

    @staticmethod
    def _freeze_attrs(attrs: dict[str, Any]) -> tuple:
        """Hashable key for an edge's attrs dict (order-independent).

        Fast-paths the empty and single-key cases -- our edges carry
        at most ``{"flags": EdgeFlags.X}``, so ``sorted`` is wasted
        work in the dominant path.
        """
        size = len(attrs)
        if size == 0:
            return ()
        if size == 1:
            return next(iter(attrs.items()))
        return tuple(sorted(attrs.items()))

    def _insert_edge(self, src_idx: int, dst_idx: int, payload: dict[str, Any]) -> bool:
        """Insert one edge under the uniqueness invariant. Returns True
        if a new edge was added, False if it was a duplicate."""
        key = (src_idx, dst_idx, self._freeze_attrs(payload))
        if key in self._edge_keys:
            return False
        self._edge_keys.add(key)
        self._g.add_edge(src_idx, dst_idx, payload)
        return True

    def add_edge(self, src: SymbolNode, dst: SymbolNode, **attrs: Any) -> None:
        s = self.add_node(src)
        d = self.add_node(dst)
        # ``attrs`` is a fresh dict produced by Python's kwarg
        # collection; pass it through as the rustworkx edge payload.
        self._insert_edge(s, d, attrs)

    def add_edges_from(
        self, edges: Iterable[tuple[SymbolNode, SymbolNode, dict[str, Any]]]
    ) -> None:
        """Bulk-add ``(src, dst, attrs)`` triples. ``attrs`` is the edge payload dict."""
        for src, dst, payload in edges:
            s = self.add_node(src)
            d = self.add_node(dst)
            self._insert_edge(s, d, payload)

    def has_edge(self, src: SymbolNode, dst: SymbolNode) -> bool:
        s = self._idx.get(src)
        d = self._idx.get(dst)
        if s is None or d is None:
            return False
        return self._g.has_edge(s, d)

    def remove_edge(self, src: SymbolNode, dst: SymbolNode) -> None:
        """Remove a single edge ``src -> dst``. Matches ``MultiDiGraph.remove_edge``.

        rustworkx's ``remove_edge(s, d)`` drops one of the parallel
        edges (impl-defined choice when several share the endpoints).
        We mirror that choice into ``_edge_keys`` by diffing the
        ``(s, d)`` payload set across the removal -- the missing entry
        is the one rustworkx just removed. Edge removal is a cold path
        (no in-tree plugin emits ``RemoveEdge`` today), so the small
        scan over parallel edges at the pair is fine.
        """
        s = self._idx[src]
        d = self._idx[dst]
        pre = {self._freeze_attrs(p) for _u, v_idx, p in self._g.out_edges(s) if v_idx == d}
        self._g.remove_edge(s, d)
        post = {self._freeze_attrs(p) for _u, v_idx, p in self._g.out_edges(s) if v_idx == d}
        for removed in pre - post:
            self._edge_keys.discard((s, d, removed))

    def successors(self, node: SymbolNode) -> Iterator[SymbolNode]:
        idx = self._idx.get(node)
        if idx is None:
            return iter(())
        return iter(self._g.successors(idx))

    def predecessors(self, node: SymbolNode) -> Iterator[SymbolNode]:
        idx = self._idx.get(node)
        if idx is None:
            return iter(())
        return iter(self._g.predecessors(idx))

    def out_edges(self, node: SymbolNode, *, data: bool = False) -> Iterator[Any]:
        idx = self._idx.get(node)
        if idx is None:
            return
        g = self._g
        if data:
            for u_idx, v_idx, payload in g.out_edges(idx):
                yield g[u_idx], g[v_idx], payload
        else:
            for u_idx, v_idx, _payload in g.out_edges(idx):
                yield g[u_idx], g[v_idx]

    def in_degree(self, node: SymbolNode) -> int:
        idx = self._idx.get(node)
        if idx is None:
            return 0
        return self._g.in_degree(idx)

    def number_of_nodes(self) -> int:
        return self._g.num_nodes()

    def subgraph(self, nodes: Iterable[SymbolNode]) -> SymbolGraph:
        """Induced subgraph over ``nodes``: includes every edge whose
        endpoints both survive the filter. Returns a fresh
        :class:`SymbolGraph`; not a view.

        Source-graph edges are already deduped under the uniqueness
        invariant, so the rebuild populates ``_edge_keys`` directly
        without re-probing it on every edge.
        """
        old_indices: list[int] = []
        for n in nodes:
            idx = self._idx.get(n)
            if idx is not None:
                old_indices.append(idx)
        sub = SymbolGraph()
        sub.graph = dict(self.graph)
        sub._g = self._g.subgraph(old_indices)
        # rustworkx's ``subgraph`` renumbers; rebuild the side tables
        # by walking the new graph.
        for new_idx in sub._g.node_indices():
            node = sub._g[new_idx]
            sub._idx[node] = new_idx
            old_idx = self._idx[node]
            attrs = self._node_attrs.get(old_idx)
            if attrs:
                sub._node_attrs[new_idx] = dict(attrs)
        for u, v, payload in sub._g.weighted_edge_list():
            sub._edge_keys.add((u, v, SymbolGraph._freeze_attrs(payload)))
        return sub

    def update(
        self,
        *,
        edges: Iterable[tuple[SymbolNode, SymbolNode, dict[str, Any]]] | None = None,
        nodes: Iterable[tuple[SymbolNode, dict[str, Any]]] | None = None,
    ) -> None:
        """Merge ``nodes`` and ``edges`` from another graph into this one.

        ``nodes`` yields ``(node, attrs_dict)`` (the shape returned by
        :meth:`nodes` under ``data=True``); ``edges`` yields
        ``(u, v, attrs)`` (the shape returned by :meth:`edges` under
        ``data=True``).
        """
        if nodes is not None:
            for n, attrs in nodes:
                idx = self.add_node(n)
                if attrs:
                    bucket = self._node_attrs.setdefault(idx, {})
                    bucket.update(attrs)
        if edges is not None:
            for u, v, payload in edges:
                s = self.add_node(u)
                d = self.add_node(v)
                self._insert_edge(s, d, payload)


# ``SymbolTrie`` is intentionally absent from ``__all__`` -- it's an
# internal data structure shared between the visitor, edge stitcher, and
# plugin context, but not part of the public surface.
__all__ = [
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolGraph",
    "SymbolNode",
    "VisitorPayload",
]
