from pathlib import Path

import libcst as cst
import networkx as nx
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import RemoveImportsVisitor
from libcst.metadata import FullRepoManager, QualifiedNameSource

from ._fqn import FixedFullyQualifiedNameProvider
from ._symbols import SymbolNode


class RemoveDeadSymbols(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (FixedFullyQualifiedNameProvider,)

    def __init__(self, dead_fqnames: set[str]):
        self.dead_fqnames = dead_fqnames

    def _should_remove(self, node: cst.CSTNode) -> bool:
        fqnames = self.get_metadata(FixedFullyQualifiedNameProvider, node, default=[])
        return any(
            qn.name in self.dead_fqnames for qn in fqnames if qn.source == QualifiedNameSource.LOCAL
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
            target_node = orig_target.target
            fqnames = self.get_metadata(FixedFullyQualifiedNameProvider, target_node, default=[])
            if not any(
                qn.name in self.dead_fqnames
                for qn in fqnames
                if qn.source == QualifiedNameSource.LOCAL
            ):
                new_targets.append(new_target)

        if not new_targets:
            return cst.RemoveFromParent()

        return updated_node.with_changes(targets=new_targets)

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign):
        fqnames = self.get_metadata(
            FixedFullyQualifiedNameProvider, original_node.target, default=[]
        )
        if any(
            qn.name in self.dead_fqnames for qn in fqnames if qn.source == QualifiedNameSource.LOCAL
        ):
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
    by_file: dict[Path, list[SymbolNode]] = {}
    for node in G.nodes:
        if not node.path.is_relative_to(base):
            continue
        if not node.path.exists():
            continue
        match node.type:
            case "function" | "class" | "variable" | "import":
                by_file.setdefault(node.path, []).append(node)
            case "module":
                node.path.unlink()

    mgr = FullRepoManager(base, by_file.keys(), {FixedFullyQualifiedNameProvider})
    for path, nodes in sorted(by_file.items(), key=lambda x: x):
        if not path.exists():
            continue

        # Pass 1: drop dead defs / classes / variables. Imports they
        # used to reference become eligible for removal in pass 2.
        wrapper = mgr.get_metadata_wrapper_for_path(path)
        dead_fqnames = {n.fqname for n in nodes if n.type != "import"}
        result = wrapper.visit(RemoveDeadSymbols(dead_fqnames))

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
