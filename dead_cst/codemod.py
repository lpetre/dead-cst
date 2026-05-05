from __future__ import annotations

from pathlib import Path

import libcst as cst
import networkx as nx
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import RemoveImportsVisitor
from libcst.metadata import CodeRange, FullRepoManager, PositionProvider, QualifiedNameSource

from ._fqn import FixedFullyQualifiedNameProvider
from .graph import SymbolNode


class RemoveDeadSymbols(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (FixedFullyQualifiedNameProvider, PositionProvider)

    def __init__(self, dead_decls: set[tuple[str, CodeRange]]):
        # ``(fqname, position)`` pairs. The position disambiguates same-name
        # decls so a shadowed dead binding does not drag its live sibling
        # out with it (and vice versa).
        self.dead_decls = dead_decls
        self._dead_positions = {pos for _, pos in dead_decls}

    def _should_remove(self, node: cst.CSTNode) -> bool:
        fqnames = self.get_metadata(FixedFullyQualifiedNameProvider, node, default=[])
        pos = self.get_metadata(PositionProvider, node, default=None)
        return any(
            (qn.name, pos) in self.dead_decls
            for qn in fqnames
            if qn.source == QualifiedNameSource.LOCAL
        )

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        if self._should_remove(original_node):
            return cst.RemoveFromParent()
        return updated_node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef):
        if self._should_remove(original_node):
            return cst.RemoveFromParent()
        return updated_node

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign):
        new_targets = []
        for orig_target, new_target in zip(original_node.targets, updated_node.targets):
            if not self._should_remove(orig_target.target):
                new_targets.append(new_target)

        if not new_targets:
            return cst.RemoveFromParent()

        return updated_node.with_changes(targets=new_targets)

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign):
        if self._should_remove(original_node.target):
            return cst.RemoveFromParent()
        return updated_node

    def leave_TypeAlias(self, original_node: cst.TypeAlias, updated_node: cst.TypeAlias):
        # ``FixedFullyQualifiedNameProvider`` does not name-bind ``cst.TypeAlias``,
        # so ``_should_remove``'s FQN-based lookup misses. Match on position alone:
        # a top-level decl's ``Name`` position is unique within the file, and this
        # method only fires on TypeAlias nodes, so cross-shape collisions are not
        # possible.
        pos = self.get_metadata(PositionProvider, original_node.name, default=None)
        if pos in self._dead_positions:
            return cst.RemoveFromParent()
        return updated_node


def _import_remove_args(node: SymbolNode) -> tuple[str, str | None, str | None]:
    """Convert an ``"import"``-typed node to ``RemoveImportsVisitor`` args.

    Returns ``(module, obj, asname)`` matching the signature of
    :meth:`RemoveImportsVisitor.remove_unused_import`. ``asname`` is set
    only when the local bound name differs from the natural binding
    (``obj`` for ``from X import obj``, the leftmost segment of
    ``module`` for bare ``import X``).
    """
    assert node.imports is not None, f"Import node missing imports metadata: {node.fqname}"
    module = str(node.imports.module)
    obj = node.imports.decl
    bound = node.fqname.rsplit(".", 1)[-1]
    natural = obj if obj is not None else module.split(".", 1)[0]
    asname = bound if bound != natural else None
    return module, obj, asname


def remove_code(G: nx.Graph, base: Path) -> None:
    """Delete every symbol in ``G`` from the source files under ``base``.

    Modules are removed by unlinking the file. Functions, classes, and
    top-level variables are dropped by a LibCST transformer matching on
    ``(fqname, position)`` so a shadowed dead binding does not drag its
    live sibling out with it. Surviving statements, comments, and
    formatting around the deletions are preserved.

    Imports are handled in a second pass via libcst's
    :class:`RemoveImportsVisitor`, which walks scopes itself and skips
    anything still referenced (defensive -- the graph already classifies
    these as dead, but the scope check is cheap insurance).

    ``G`` is typically the unreachable subgraph of the graph from
    :func:`build_symbol_graph`; only the nodes are inspected, edges are
    ignored. Symbols whose path is not under ``base`` are skipped, so call
    once per base when analysing several packages together. The
    transformation is destructive -- back the files up first, or run on a
    clean working tree.
    """
    by_file: dict[Path, list[SymbolNode]] = {}
    for node in G.nodes:
        if not node.path.is_relative_to(base):
            continue
        if not node.path.exists():
            continue
        match node.type:
            case "function" | "class" | "variable" | "type_alias" | "import":
                by_file.setdefault(node.path, []).append(node)
            case "module":
                node.path.unlink()

    mgr = FullRepoManager(str(base), [str(p) for p in by_file], {FixedFullyQualifiedNameProvider})
    for path, nodes in sorted(by_file.items(), key=lambda x: x):
        if not path.exists():
            continue

        # Pass 1: drop dead defs / classes / variables. Imports they
        # used to reference become eligible for removal in pass 2.
        wrapper = mgr.get_metadata_wrapper_for_path(str(path))
        dead_decls = {(n.fqname, n.position) for n in nodes if n.type != "import"}
        result = wrapper.visit(RemoveDeadSymbols(dead_decls))

        # Pass 2: hand the dead-import set to libcst's stock import
        # remover. It walks scopes itself, so it'll skip anything still
        # referenced after pass 1 (defensive -- if the graph said
        # something is dead, no live user remains).
        dead_imports = [n for n in nodes if n.type == "import"]
        if dead_imports:
            ctx = CodemodContext()
            for imp in dead_imports:
                module, obj, asname = _import_remove_args(imp)
                RemoveImportsVisitor.remove_unused_import(ctx, module, obj, asname)
            result = RemoveImportsVisitor(ctx).transform_module(result)

        with path.open("w") as f:
            f.write(result.code)


__all__ = ["remove_code"]
