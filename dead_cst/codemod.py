from __future__ import annotations

import difflib
from pathlib import Path

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import RemoveImportsVisitor
from libcst.metadata import CodeRange, FullRepoManager, PositionProvider, QualifiedNameSource

from ._fqn import FixedFullyQualifiedNameProvider
from ._graphstore import SymbolGraph
from .graph import NodeFlags, SymbolNode


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


def _select_files(G: SymbolGraph, base: Path) -> tuple[dict[Path, list[SymbolNode]], list[Path]]:
    """Group ``G``'s nodes under ``base`` into files-to-rewrite vs. files-to-delete.

    Used by both :func:`remove_code` and :func:`generate_patch`. Symbols
    outside ``base`` (other packages, vendored deps) are dropped, as are
    nodes whose source file no longer exists. Synthetic nodes are
    ignored implicitly -- they don't appear in the type ``match``.

    ``NodeFlags.NOTEBOOK`` nodes are dropped: cell-aware writeback into
    the notebook JSON envelope is not implemented today.
    """
    by_file: dict[Path, list[SymbolNode]] = {}
    deleted_modules: list[Path] = []
    for node in G.nodes:
        if not node.path.is_relative_to(base):
            continue
        if not node.path.exists():
            continue
        if node.flags & NodeFlags.NOTEBOOK:
            continue
        match node.type:
            case "function" | "class" | "variable" | "type_alias" | "import":
                by_file.setdefault(node.path, []).append(node)
            case "module":
                deleted_modules.append(node.path)
    # A file marked for outright deletion needs no per-decl rewrite on top.
    for path in deleted_modules:
        by_file.pop(path, None)
    return by_file, deleted_modules


def _rewrite_one(wrapper, nodes: list[SymbolNode]) -> tuple[str, str]:
    """Run the two-pass dead-symbol + dead-import transform on one file.

    Returns ``(original, new)`` source pairs read straight off
    ``wrapper.module.code`` -- no extra disk read, and no TOCTOU window
    against a concurrent edit between the parser load and the diff base.
    Pass 1 (:class:`RemoveDeadSymbols`) drops dead defs/classes/variables;
    pass 2 (:class:`RemoveImportsVisitor`) prunes imports they used to
    reference. Pass 2 is skipped when there are no dead imports so the
    Module passes through unchanged.
    """
    original = wrapper.module.code
    dead_decls = {(n.fqname, n.position) for n in nodes if n.type != "import"}
    result = wrapper.visit(RemoveDeadSymbols(dead_decls))

    dead_imports = [n for n in nodes if n.type == "import"]
    if dead_imports:
        ctx = CodemodContext()
        for imp in dead_imports:
            module, obj, asname = _import_remove_args(imp)
            RemoveImportsVisitor.remove_unused_import(ctx, module, obj, asname)
        result = RemoveImportsVisitor(ctx).transform_module(result)

    return original, result.code


def remove_code(G: SymbolGraph, package_path: Path) -> None:
    """Delete every symbol in ``G`` from the source files under ``package_path``.

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
    ignored. Symbols whose path is not under ``package_path`` are
    skipped, so call once per package when analysing several packages
    together. The transformation is destructive -- back the files up
    first, or run on a clean working tree.
    """
    by_file, deleted_modules = _select_files(G, package_path)

    for path in deleted_modules:
        path.unlink()

    if not by_file:
        return

    mgr = FullRepoManager(
        str(package_path), [str(p) for p in by_file], {FixedFullyQualifiedNameProvider}
    )
    for path, nodes in sorted(by_file.items()):
        wrapper = mgr.get_metadata_wrapper_for_path(str(path))
        original, new = _rewrite_one(wrapper, nodes)
        if new != original:
            path.write_text(new)


def generate_patch(G: SymbolGraph, root: Path) -> str:
    """Return a ``git apply``-compatible unified diff that removes ``G``'s nodes.

    Same selection logic as :func:`remove_code` -- group dead nodes by
    file, run :class:`RemoveDeadSymbols` then :class:`RemoveImportsVisitor`
    -- but instead of writing the rewritten source back to disk, compare
    it to the original and emit a unified diff. Module nodes become
    file-deletion hunks (``+++ /dev/null`` plus the ``deleted file
    mode 100644`` extended header).

    Patch paths are emitted as ``a/<rel>`` / ``b/<rel>`` where ``rel``
    is each file's path relative to ``root``; ``git apply`` should be
    run from that same directory. Nodes whose source path is not under
    ``root`` are skipped.

    Selection is driven entirely by ``G.nodes`` -- only those nodes are
    candidates for removal -- so callers can slice the unreachable
    graph however they like (e.g. ``G.subgraph(scc)`` for one SCC at a
    time) to review a big codebase as a series of focused patches. The
    underlying file rewrite still uses ``FullRepoManager`` against the
    real source, so a partial slice removes only the decls in the slice
    and leaves their siblings (and any imports the slice does not
    cover) intact.

    The returned string is the concatenation of every per-file diff,
    sorted by path. An empty string means there was nothing to remove.
    """
    by_file, deleted_modules = _select_files(G, root)
    chunks: list[str] = []

    for path in sorted(deleted_modules):
        rel = path.relative_to(root).as_posix()
        original = path.read_text().splitlines(keepends=True)
        body = "".join(
            difflib.unified_diff(
                original,
                [],
                fromfile=f"a/{rel}",
                tofile="/dev/null",
            )
        )
        chunks.append(f"diff --git a/{rel} b/{rel}\ndeleted file mode 100644\n{body}")

    if by_file:
        mgr = FullRepoManager(
            str(root), [str(p) for p in by_file], {FixedFullyQualifiedNameProvider}
        )
        for path, nodes in sorted(by_file.items()):
            wrapper = mgr.get_metadata_wrapper_for_path(str(path))
            original, new = _rewrite_one(wrapper, nodes)
            if new == original:
                continue
            rel = path.relative_to(root).as_posix()
            body = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
            chunks.append(f"diff --git a/{rel} b/{rel}\n{body}")

    return "".join(chunks)


__all__ = ["generate_patch", "remove_code"]
