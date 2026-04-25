from collections.abc import Iterable
from pathlib import Path

import libcst as cst
import networkx as nx
from libcst.metadata import FullRepoManager, QualifiedNameSource

from ._fqn import FixedFullyQualifiedNameProvider


def _with_simple_comma(alias: cst.ImportAlias) -> cst.ImportAlias:
    """Normalise an alias's trailing comma for single-line use."""
    if alias.comma is cst.MaybeSentinel.DEFAULT:
        return alias
    return alias.with_changes(
        comma=cst.Comma(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(" "),
        )
    )


def _alias_bound_name(alias: cst.ImportAlias) -> str:
    """Return the local name introduced by ``alias``.

    For ``import foo`` / ``from x import foo`` this is ``"foo"``. For
    ``import foo.bar`` the bound name is the leftmost segment
    (``"foo"``). When an ``as`` clause is present the alias name wins.
    """
    target: cst.BaseExpression = alias.asname.name if alias.asname is not None else alias.name
    while isinstance(target, cst.Attribute):
        target = target.value
    assert isinstance(target, cst.Name), f"unexpected ImportAlias target: {type(target).__name__}"
    return target.value


class RemoveDeadSymbols(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (FixedFullyQualifiedNameProvider,)

    def __init__(
        self,
        dead_fqnames: set[str],
        dead_import_names: Iterable[str] = (),
    ):
        self.dead_fqnames = dead_fqnames
        # Imports are matched by their local bound name rather than FQN
        # because libcst's FQN provider does not surface metadata for
        # ``ImportAlias`` targets.
        self.dead_import_names = set(dead_import_names)

    def _should_remove(self, node: cst.CSTNode) -> bool:
        fqnames = self.get_metadata(FixedFullyQualifiedNameProvider, node, default=[])
        return any(
            qn.name in self.dead_fqnames for qn in fqnames if qn.source == QualifiedNameSource.LOCAL
        )

    def _filter_aliases(self, aliases: Iterable[cst.ImportAlias]) -> list[cst.ImportAlias] | None:
        kept = [a for a in aliases if _alias_bound_name(a) not in self.dead_import_names]
        if not kept:
            return None
        # The original last alias may have been kept or dropped; either
        # way the new last alias must not carry a trailing comma.
        if kept[-1].comma is not cst.MaybeSentinel.DEFAULT:
            kept[-1] = kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return kept

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

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import):
        kept = self._filter_aliases(updated_node.names)
        if kept is None:
            return cst.RemoveFromParent()
        if len(kept) == len(updated_node.names):
            return updated_node
        return updated_node.with_changes(names=kept)

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom):
        # ``from foo import *`` is opaque; leave it alone.
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        kept = self._filter_aliases(updated_node.names)
        if kept is None:
            return cst.RemoveFromParent()
        if len(kept) == len(updated_node.names):
            return updated_node
        # Stripping aliases from a parenthesised multi-line import
        # leaves the surviving aliases carrying ``ParenthesizedWhitespace``
        # commas that are invalid once the parens are gone. Renormalise
        # to a single-line import -- always valid for the surviving names.
        if updated_node.lpar is not None:
            kept = [_with_simple_comma(a) for a in kept]
            updated_node = updated_node.with_changes(lpar=None, rpar=None)
        return updated_node.with_changes(names=kept)


def remove_code(G: nx.Graph, base: Path) -> None:
    by_file: dict[Path, list] = {}
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
        wrapper = mgr.get_metadata_wrapper_for_path(path)
        dead_fqnames = {n.fqname for n in nodes if n.type != "import"}
        # Imports are addressed by their local bound name -- the trailing
        # segment of the FQN, e.g. ``mod.x`` -> ``"x"``.
        dead_import_names = {n.fqname.rsplit(".", 1)[-1] for n in nodes if n.type == "import"}
        mod_removed = wrapper.visit(RemoveDeadSymbols(dead_fqnames, dead_import_names))
        with path.open("w") as f:
            f.write(mod_removed.code)
