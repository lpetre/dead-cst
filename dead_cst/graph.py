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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from libcst.metadata import CodeRange

logger = logging.getLogger(__name__)


class NodeFlags(enum.IntFlag):
    """Analyzer-internal marker on :class:`SymbolNode`.

    ``SHADOWED`` decls are emitted into the graph (with their parent
    module edge) but are excluded from the lookup trie, so cross-module
    imports never resolve to them.

    ``ENTRYPOINT`` flags a node as a reachability seed: :func:`find_reachable`
    starts its BFS from every node carrying this bit. Plugins emit
    flagged synthetic nodes via their per-file payloads (or via
    :class:`AddNode` ops constructing :func:`synthetic_node` with
    ``flags=NodeFlags.ENTRYPOINT``) to declare entrypoints.

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

    ``EXPORTED`` tags every node sourced from a file that the package
    exports to its consumers (per :attr:`Package.exported`). Set via
    the visitor's ``default_flags`` mechanism, so it lands on every
    node a file produces. The single per-package lookup trie carries
    both exported and non-exported decls; consumer-side merges use
    :meth:`SymbolTrie.merge_exported` to filter to entries with this
    bit set, while the package's own self-lookup uses :meth:`merge`
    and sees everything.

    ``STAR_REEXPORT`` tags an import decl that ``build_contribution``
    materialized from a ``from X import *`` statement: each name in
    the star target's trie becomes a synthetic ``"import"`` node in
    the importing module so cross-module lookups like
    ``from <importer> import <name>`` resolve through to the real
    source. These nodes have no literal counterpart in the source
    file, so the codemod skips them (the ``RemoveImportsVisitor``
    pass would otherwise hunt for a non-existent line).
    """

    NONE = 0
    SHADOWED = enum.auto()
    ENTRYPOINT = enum.auto()
    OVERLOAD = enum.auto()
    TESTCASE = enum.auto()
    NOQA = enum.auto()
    NOTEBOOK = enum.auto()
    EXPORTED = enum.auto()
    STAR_REEXPORT = enum.auto()


class EdgeFlags(enum.IntFlag):
    """Analyzer-internal marker on graph edges.

    ``DEAD_BRANCH`` flags an edge whose reference originated inside a
    statically-dead suite (per :func:`dead_cst.branches.unreachable_suites`).
    Default :func:`dead_cst.find_reachable` does **not** filter by this
    flag -- today's behavior, where dead-code references propagate
    liveness through the enclosing decl, is preserved. The opt-in
    :func:`dead_cst.find_kept_alive_by_dead_branches` returns the
    "blast radius" of removing every dead suite by skipping these edges.

    ``DYNAMIC_IMPORT`` flags an edge emitted from a runtime-import
    call (``__import__('X')`` / ``importlib.import_module('X')``).
    Visitor behavior is minimal — one edge per *explicit symbol* the
    call mentions, no fan-out to upstream decls — so reachability
    captures only what the literal call expresses. A contrib plugin
    can extend the graph by reading the flag and fanning each
    flagged module-edge out to its exports.
    """

    NONE = 0
    DEAD_BRANCH = enum.auto()
    DYNAMIC_IMPORT = enum.auto()


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
        :func:`dead_cst._package.eclipsed_paths`: the loser's payload is
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
        precedence order by the order of their ``merge()`` calls;
        :meth:`Analysis._build_symbol_lookup` merges the consumer's own
        trie first and each dep's trie via :meth:`merge_exported`
        afterwards, so own-module always beats dep-module on
        legitimate conflict.
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

    def merge_exported(self, other: SymbolTrie) -> SymbolTrie:
        """Merge entries flagged :data:`NodeFlags.EXPORTED` from ``other``.

        Used to fold a dep's per-package trie into a consumer's lookup:
        the dep stores every decl in one trie, and consumers see only
        the subset its source files marked exported (via the visitor's
        ``default_flags`` for files under :attr:`Package.exported`).

        Walks children recursively even when an intermediate module
        isn't exported -- ``pkg/__init__.py`` may not be exported while
        ``pkg/api/__init__.py`` is, and the consumer needs the path
        ``pkg`` -> ``api`` to exist so the descendant resolves.
        Non-exported decls and non-exported intermediate modules are
        dropped; exported descendants still get through.
        """
        for part, child in other.children.items():
            if part not in self.children:
                self.children[part] = SymbolTrie()
            self.children[part].merge_exported(child)
        if other.module is None or not (other.module.flags & NodeFlags.EXPORTED):
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
        for k, v in other.declarations.items():
            exported = [d for d in v if d.flags & NodeFlags.EXPORTED]
            if exported:
                self.declarations[k] = exported
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

    def module_hierarchy_edges(self) -> Iterator[tuple[SymbolNode, SymbolNode]]:
        """Yield ``(child_module, parent_module)`` for every submodule edge.

        For a structure like:
            v6/
            ├── __init__.py
            └── nntree/
                ├── __init__.py
                └── extern/
                    ├── __init__.py
                    └── clip.py

        Yields:
            v6.nntree -> v6
            v6.nntree.extern -> v6.nntree
            v6.nntree.extern.clip -> v6.nntree.extern
        """

        def _walk(
            node: SymbolTrie, parent_module: SymbolNode | None
        ) -> Iterator[tuple[SymbolNode, SymbolNode]]:
            if node.module and parent_module:
                yield (node.module, parent_module)
            for child_node in node.children.values():
                yield from _walk(child_node, node.module)

        yield from _walk(self, None)


def _claim_edge(
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]],
    src: SymbolNode,
    dst: SymbolNode,
    flags: EdgeFlags,
) -> bool:
    """Record ``(src, dst, flags)`` in ``emitted``; return ``True`` iff new.

    Single dedup point shared by the three edge sources in the compose
    pass: contribution edges (``_compose_contribution``), import
    resolution (``resolve_edges._emit``), and plugin ``AddEdge`` ops
    (``apply_ops``). Lives here rather than in ``_edges`` because
    ``plugins._core`` would otherwise pull in a circular import.
    """
    key = (src, dst, flags)
    if key in emitted:
        return False
    emitted.add(key)
    return True


# ``SymbolTrie`` is intentionally absent from ``__all__`` -- it's an
# internal data structure shared between the visitor, edge stitcher, and
# plugin context, but not part of the public surface.
__all__ = [
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolNode",
    "VisitorPayload",
]
