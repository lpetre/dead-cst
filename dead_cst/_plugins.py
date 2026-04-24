"""Pluggable edge contributors.

An ``EdgePlugin`` adds (and, optionally, removes) nodes and edges in the symbol
graph after regular analysis is complete. Plugins exist to encode knowledge
the static analyzer cannot infer from CST alone -- dynamic dispatch, framework
conventions, entry-point metadata, etc.

Two flavors exist:

* :class:`EdgePlugin` -- receives a :class:`PluginContext` only. Cheap.
* :class:`CSTAwareEdgePlugin` -- additionally receives the per-base
  :class:`FullRepoManager` map, letting the plugin re-walk source with
  libcst metadata providers.

``build_symbol_graph`` iterates the plugin list and dispatches to whichever
protocol each plugin satisfies (``isinstance``).

Plugins emit :class:`GraphOp` values rather than mutating the graph directly;
this keeps them easy to test and lets the analyzer report back on what was
added.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Union, runtime_checkable

import libcst as cst
import networkx as nx
from libcst.metadata import CodePosition, CodeRange, FullRepoManager

from ._symbols import SymbolNode, SymbolTrie

logger = logging.getLogger(__name__)

SYNTHETIC_POSITION = CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0))


@dataclass
class PluginContext:
    """Read-only view of the analyzer state passed to every plugin."""

    graph: nx.DiGraph
    symbol_lookup: SymbolTrie
    paths: dict[Path, list[Path]]
    project_root: Path

    def find_module(self, fqname: str) -> SymbolNode | None:
        node = self.symbol_lookup._get(fqname.split("."))
        return node.module if node else None

    def find_declaration(self, fqname: str) -> SymbolNode | None:
        """Look up a top-level declaration by dotted name.

        ``pkg.mod.func`` is split into module ``pkg.mod`` and decl ``func``.
        Returns the declaration's :class:`SymbolNode` or ``None``.
        """
        parts = fqname.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module_parts, decl_name = parts[:split], parts[split]
            node = self.symbol_lookup._get(module_parts)
            if node and node.module and decl_name in node.declarations:
                return node.declarations[decl_name]
        return None


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


@dataclass
class MainBlockPlugin:
    """Treat ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module that contains a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint node with an edge to the containing
    module. The module's existing internal edges (collected by the regular
    visitor) then keep every symbol referenced in the block reachable.

    This is a :class:`CSTAwareEdgePlugin` so it can reuse the
    :class:`FullRepoManager` instances the analyzer already built rather than
    re-parsing files.
    """

    name: str = "main_block"
    cst_aware: bool = True

    def contribute(
        self, ctx: PluginContext, managers: dict[Path, FullRepoManager]
    ) -> Iterable[GraphOp]:
        modules_by_path: dict[Path, SymbolNode] = {}
        for node in ctx.graph.nodes:
            if node.type == "module":
                modules_by_path[node.path] = node

        for base, mgr in managers.items():
            for path, module_node in modules_by_path.items():
                if not path.is_relative_to(base):
                    continue
                try:
                    wrapper = mgr.get_metadata_wrapper_for_path(path)
                except Exception:
                    continue
                if not _has_main_block(wrapper.module):
                    continue
                synth = synthetic_node(
                    fqname=f"<__main__>:{module_node.fqname}",
                    path=path,
                )
                yield AddNode(synth, entrypoint=True)
                yield AddEdge(synth, module_node)


def _has_main_block(module: cst.Module) -> bool:
    for stmt in module.body:
        if not isinstance(stmt, cst.If):
            continue
        if _is_name_eq_main(stmt.test):
            return True
    return False


def _is_name_eq_main(expr: cst.BaseExpression) -> bool:
    if not isinstance(expr, cst.Comparison):
        return False
    if len(expr.comparisons) != 1:
        return False
    op = expr.comparisons[0]
    if not isinstance(op.operator, cst.Equal):
        return False
    left, right = expr.left, op.comparator
    return (_is_dunder_name(left, "__name__") and _is_string(right, "__main__")) or (
        _is_string(left, "__main__") and _is_dunder_name(right, "__name__")
    )


def _is_dunder_name(expr: cst.BaseExpression, name: str) -> bool:
    return isinstance(expr, cst.Name) and expr.value == name


def _is_string(expr: cst.BaseExpression, value: str) -> bool:
    if isinstance(expr, cst.SimpleString):
        return expr.evaluated_value == value
    if isinstance(expr, cst.ConcatenatedString):
        try:
            return expr.evaluated_value == value
        except Exception:
            return False
    return False


@dataclass
class ProjectScriptsPlugin:
    """Treat every ``[project.scripts]`` entry in ``pyproject.toml`` as an
    entrypoint.

    For each ``name = "pkg.mod:func"`` mapping, look up ``pkg.mod.func`` in the
    symbol trie and wire a synthetic entrypoint node to it.
    """

    name: str = "project_scripts"
    pyproject_path: Path | None = None

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        pyproject = self.pyproject_path or ctx.project_root / "pyproject.toml"
        if not pyproject.is_file():
            return

        try:
            import tomllib
        except ImportError:  # pragma: no cover
            return

        with pyproject.open("rb") as f:
            data = tomllib.load(f)

        scripts = data.get("project", {}).get("scripts", {})
        for script_name, target in scripts.items():
            module_part, _, decl_part = target.partition(":")
            fqname = f"{module_part}.{decl_part}" if decl_part else module_part
            target_node = ctx.find_declaration(fqname) or ctx.find_module(module_part)
            if target_node is None:
                logger.warning(
                    "ProjectScriptsPlugin: %s -> %r not found in symbol graph",
                    script_name,
                    target,
                )
                continue
            synth = synthetic_node(
                fqname=f"<project.scripts>:{script_name}",
                path=pyproject,
            )
            yield AddNode(synth, entrypoint=True)
            yield AddEdge(synth, target_node)


@dataclass
class ExplicitEntrypointPlugin:
    """Mark user-specified symbols as entrypoints.

    ``specs`` accepts the same three forms the CLI's ``-e`` flag used to:

    * ``str`` -- matches either an exact fully-qualified name
      (``pkg.mod.func``) or a file path relative to ``project_root``.
    * :class:`pathlib.Path` -- matches an exact absolute path.
    * :class:`re.Pattern` -- matched against the file path relative to
      ``project_root``.

    For every matching :class:`SymbolNode`, a synthetic entrypoint node is
    added with an edge pointing at the match.
    """

    specs: list[str | Path | re.Pattern[str]] = field(default_factory=list)
    name: str = "explicit"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        root = ctx.project_root
        for node in ctx.graph.nodes:
            if not self._matches(node, root):
                continue
            synth = synthetic_node(
                fqname=f"<entrypoint>:{node.fqname}",
                path=node.path,
            )
            yield AddNode(synth, entrypoint=True)
            yield AddEdge(synth, node)

    def _matches(self, sym: SymbolNode, root: Path) -> bool:
        try:
            rel = str(sym.path.relative_to(root))
        except ValueError:
            rel = str(sym.path)
        for spec in self.specs:
            if isinstance(spec, re.Pattern):
                if spec.match(rel):
                    return True
            elif isinstance(spec, Path):
                if spec == sym.path:
                    return True
            elif isinstance(spec, str):
                if spec == rel or spec == sym.fqname:
                    return True
        return False


@dataclass
class DunderAllPlugin:
    """Keep module-level ``__all__`` variables alive.

    ``__all__`` declares a module's public export list; removing it would be
    observable even when no source references it. For each top-level
    ``__all__`` variable, emit a synthetic entrypoint node with an edge to
    the variable so :func:`find_reachable` preserves it.
    """

    name: str = "dunder_all"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for node in ctx.graph.nodes:
            if node.type != "variable":
                continue
            if not node.fqname.endswith(".__all__") and node.fqname != "__all__":
                continue
            synth = synthetic_node(
                fqname=f"<__all__>:{node.fqname}",
                path=node.path,
            )
            yield AddNode(synth, entrypoint=True)
            yield AddEdge(synth, node)


BUILTIN_PLUGINS: dict[str, type] = {
    MainBlockPlugin.name: MainBlockPlugin,
    ProjectScriptsPlugin.name: ProjectScriptsPlugin,
    ExplicitEntrypointPlugin.name: ExplicitEntrypointPlugin,
    DunderAllPlugin.name: DunderAllPlugin,
}


def load_plugin(name: str):
    """Load a plugin by name. Checks builtins first, then entry points."""
    if name in BUILTIN_PLUGINS:
        return BUILTIN_PLUGINS[name]()

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.plugins"):
        if ep.name == name:
            cls = ep.load()
            return cls()
    raise KeyError(f"Unknown edge plugin: {name!r}")
