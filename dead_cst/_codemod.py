from pathlib import Path

import libcst as cst
import networkx as nx
from libcst.metadata import FullRepoManager, QualifiedNameSource

from ._fqn import FixedFullyQualifiedNameProvider


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


def remove_code(G: nx.Graph, base: Path) -> None:
    by_file = {}
    for node in G.nodes:
        if not node.path.is_relative_to(base):
            continue
        if not node.path.exists():
            continue
        match node.type:
            case "function" | "class" | "variable":
                by_file.setdefault(node.path, []).append(node)
            case "module":
                node.path.unlink()

    mgr = FullRepoManager(base, by_file.keys(), {FixedFullyQualifiedNameProvider})
    for path, nodes in sorted(by_file.items(), key=lambda x: x):
        if not path.exists():
            continue
        wrapper = mgr.get_metadata_wrapper_for_path(path)
        dead_fqnames = {n.fqname for n in nodes}
        mod_removed = wrapper.visit(RemoveDeadSymbols(dead_fqnames))
        with path.open("w") as f:
            f.write(mod_removed.code)
