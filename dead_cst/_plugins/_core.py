"""Shared types and helpers for edge plugins.

Defines the protocols every plugin satisfies, the :class:`GraphOp` value
objects plugins emit, and small utilities (:func:`apply_ops`,
:func:`synthetic_node`) used by both the analyzer and the plugins
themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Union, runtime_checkable

import networkx as nx
from libcst.metadata import CodePosition, CodeRange, FullRepoManager

from .._symbols import SymbolNode, SymbolTrie

SYNTHETIC_POSITION = CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0))


class FileTextCache:
    """Lazy byte-level text cache over a fixed set of files.

    Plugins use this to skip files that obviously can't match before
    paying the cost of parsing the CST or walking it. The set of files
    over which :meth:`grep` iterates by default is fixed at construction
    -- normally every ``.py`` file the analyzer just processed -- but
    individual reads accept arbitrary paths and cache them too.

    Files are read on first access and the bytes retained, so the
    common pattern (each plugin greps once for its keyword) only ever
    touches the disk once per file across the whole plugin pass.
    """

    def __init__(self, paths: Iterable[Path] = ()) -> None:
        self._paths: tuple[Path, ...] = tuple(paths)
        self._content: dict[Path, bytes] = {}

    @property
    def paths(self) -> tuple[Path, ...]:
        """The default iteration set passed at construction."""
        return self._paths

    def read(self, path: Path) -> bytes:
        """Return ``path``'s raw bytes; missing or unreadable files yield ``b""``."""
        cached = self._content.get(path)
        if cached is None:
            try:
                cached = path.read_bytes()
            except OSError:
                cached = b""
            self._content[path] = cached
        return cached

    def contains(self, path: Path, needle: str | bytes) -> bool:
        """Return True if ``path``'s contents contain ``needle`` as a literal substring."""
        if isinstance(needle, str):
            needle = needle.encode("utf-8")
        return needle in self.read(path)

    def grep(
        self,
        pattern: str | bytes | re.Pattern[bytes],
        *,
        paths: Iterable[Path] | None = None,
    ) -> Iterator[Path]:
        """Yield paths whose contents match ``pattern``.

        ``str`` / ``bytes`` patterns are matched as literal substrings
        (very cheap -- no regex compile, no Unicode decode). A compiled
        bytes :class:`re.Pattern` triggers a regex search instead. When
        ``paths`` is omitted the default set passed at construction is
        scanned.
        """
        candidates = self._paths if paths is None else paths
        if isinstance(pattern, re.Pattern):
            for p in candidates:
                if pattern.search(self.read(p)):
                    yield p
            return
        if isinstance(pattern, str):
            pattern = pattern.encode("utf-8")
        for p in candidates:
            if pattern in self.read(p):
                yield p


@dataclass
class PluginContext:
    """Read-only view of the analyzer state passed to every plugin."""

    graph: nx.DiGraph
    symbol_lookup: SymbolTrie
    paths: dict[Path, list[Path]]
    project_root: Path
    file_cache: FileTextCache = field(default_factory=FileTextCache)

    def find_module(self, fqname: str) -> SymbolNode | None:
        node = self.symbol_lookup._get(fqname.split("."))
        return node.module if node else None

    def find_declarations(self, fqname: str) -> list[SymbolNode]:
        """Look up top-level declarations by dotted name.

        ``pkg.mod.func`` is split into module ``pkg.mod`` and decl ``func``.
        Returns every :class:`SymbolNode` bound to ``func`` at module
        exit -- normally one, but multiple when each branch of a
        conditional defines the same name. Empty list if nothing matches.
        """
        parts = fqname.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module_parts, decl_name = parts[:split], parts[split]
            node = self.symbol_lookup._get(module_parts)
            if node and node.module and decl_name in node.declarations:
                return list(node.declarations[decl_name])
        return []

    def grep(
        self,
        pattern: str | bytes | re.Pattern[bytes],
        *,
        paths: Iterable[Path] | None = None,
    ) -> Iterator[Path]:
        """Yield paths whose contents match ``pattern`` (see :meth:`FileTextCache.grep`)."""
        return self.file_cache.grep(pattern, paths=paths)


@dataclass(frozen=True)
class AddNode:
    """Add a node to the graph. When ``entrypoint=True``, mark the node so
    :func:`find_reachable` seeds its BFS from it."""

    node: SymbolNode
    entrypoint: bool = False


@dataclass(frozen=True)
class AddEdge:
    src: SymbolNode
    dst: SymbolNode


@dataclass(frozen=True)
class RemoveEdge:
    src: SymbolNode
    dst: SymbolNode


GraphOp = Union[AddNode, AddEdge, RemoveEdge]


@runtime_checkable
class EdgePlugin(Protocol):
    name: str

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]: ...


@runtime_checkable
class CSTAwareEdgePlugin(Protocol):
    name: str
    # marker attribute used by ``isinstance`` to distinguish this protocol from
    # the plain ``EdgePlugin`` -- runtime_checkable Protocols only look at
    # attribute presence, not method signatures, so an extra attribute is the
    # simplest way to disambiguate.
    cst_aware: bool

    def contribute(
        self, ctx: PluginContext, managers: dict[Path, FullRepoManager]
    ) -> Iterable[GraphOp]: ...


def apply_ops(graph: nx.DiGraph, ops: Iterable[GraphOp]) -> None:
    for op in ops:
        match op:
            case AddNode(node, entrypoint):
                graph.add_node(node)
                if entrypoint:
                    graph.nodes[node]["entrypoint"] = True
            case AddEdge(src, dst):
                graph.add_edge(src, dst)
            case RemoveEdge(src, dst):
                if graph.has_edge(src, dst):
                    graph.remove_edge(src, dst)


def synthetic_node(fqname: str, path: Path) -> SymbolNode:
    """Create a placeholder node plugins can attach edges to."""
    return SymbolNode(fqname=fqname, type="synthetic", path=path, position=SYNTHETIC_POSITION)
