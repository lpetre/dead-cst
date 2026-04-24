from __future__ import annotations

import logging
from functools import cache
from importlib.util import resolve_name
from pathlib import Path
from typing import Generator, Literal

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    FullyQualifiedNameProvider,
    ParentNodeProvider,
    PositionProvider,
    ScopeProvider,
)

from ._resolve import resolve_import
from ._symbols import Import, SymbolNode, SymbolTrie

logger = logging.getLogger(__name__)


def _dotted_name_parts(
    prefix: str, node: cst.BaseExpression
) -> Generator[tuple[str, cst.CSTNode], None, None]:
    if isinstance(node, cst.Name):
        full = f"{prefix}{node.value}" if prefix else node.value
        yield full, node
    elif isinstance(node, cst.Attribute):
        # Recurse on the value first
        for nm, n in _dotted_name_parts(prefix, node.value):
            yield nm, n
        # then append this attribute to the last prefix
        full = f"{nm}.{node.attr.value}"
        yield full, node.attr


def _pair_targets(
    target: cst.BaseExpression, rhs: cst.BaseExpression | None
) -> Generator[tuple[cst.Name, cst.BaseExpression | None], None, None]:
    """Yield (name_node, value_node) pairs for an assignment target pattern.

    Handles ``Name`` leaves, tuple / list patterns (including nested ones),
    and starred elements. Non-name leaves (``Attribute``, ``Subscript``)
    are skipped. When the RHS is a tuple / list of matching arity we pair
    element-wise; otherwise the entire RHS is broadcast to every name.
    """
    if isinstance(target, cst.Name):
        yield target, rhs
        return

    if isinstance(target, (cst.Tuple, cst.List)):
        if isinstance(rhs, (cst.Tuple, cst.List)) and len(rhs.elements) == len(target.elements):
            for te, ve in zip(target.elements, rhs.elements):
                yield from _pair_targets(te.value, ve.value)
        else:
            for te in target.elements:
                yield from _pair_targets(te.value, rhs)


class SymbolVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        FullyQualifiedNameProvider,
        ScopeProvider,
        ParentNodeProvider,
        PositionProvider,
    )

    def _pos(self, node: cst.CSTNode):
        return self.get_metadata(PositionProvider, node, default=None)

    def __init__(self, path: Path, search_paths: list[Path]):
        self.path = path
        self.search_paths = search_paths
        self.node_to_symbols: dict[cst.CSTNode, list[SymbolNode]] = {}
        self.node_to_frames: dict[cst.CSTNode, list[list[SymbolNode]]] = {}
        self.decl_stack: list[list[SymbolNode]] = []
        self.nearest_decls: dict[cst.CSTNode, list[SymbolNode]] = {}
        self.import_lookup: dict[cst.CSTNode, Import] = {}
        self.import_edges: set[tuple[SymbolNode, Import]] = set()
        self.internal_edges: set[tuple[SymbolNode, SymbolNode]] = set()
        self.trie: SymbolTrie = SymbolTrie()

    @property
    def module_node(self) -> SymbolNode:
        if not self.decl_stack:
            raise ValueError("Module node has not been set yet.")
        return self.decl_stack[0][0]

    @cache
    def resolve_import(self, name: str) -> str | Path:
        return resolve_import(name, self.search_paths)

    def _push_decl(self, node: cst.CSTNode, decl: SymbolNode):
        self._push_decls(node, [decl])

    def _push_decls(self, node: cst.CSTNode, decls: list[SymbolNode]) -> None:
        """Push a frame of decls for ``node``.

        A frame is a group of decls that are simultaneously "active" for any
        accesses occurring within the subtree rooted at ``node``. A single
        ``_push_decl`` pushes a one-element frame; chained assignments like
        ``b = c = f`` push both ``b`` and ``c`` as a single frame so the RHS
        is attributed to both.
        """
        frame = list(decls)
        self.node_to_symbols.setdefault(node, []).extend(frame)
        self.node_to_frames.setdefault(node, []).append(frame)
        self.decl_stack.append(frame)
        for d in frame:
            self.trie.add_declaration(d)

    def _add_decl(
        self,
        node: cst.CSTNode,
        type_: Literal["module", "class", "function", "variable"],
    ):
        # Only collect top-level declarations, skip nested ones
        if len(self.decl_stack) > 1:
            return

        fqns = self.get_metadata(FullyQualifiedNameProvider, node, default=[])
        pos = self._pos(node)
        for fqn in fqns:
            self._push_decl(node, SymbolNode(fqn.name, type_, self.path, pos))

    def _add_variable(self, node: cst.Assign | cst.AnnAssign):
        # Only collect top-level declarations, skip nested ones
        if len(self.decl_stack) > 1:
            return

        if isinstance(node, cst.Assign):
            targets = [t.target for t in node.targets]
        else:
            targets = [node.target]

        # For `x: T` (AnnAssign without a value) treat the annotation as the
        # rhs so references inside it are attributed to the new symbol.
        rhs = node.value
        if rhs is None and isinstance(node, cst.AnnAssign):
            rhs = node.annotation

        # Flatten each top-level target against the rhs into (name, value) pairs.
        # For chained assignment ``b = c = f`` every target shares the same rhs.
        pairs: list[tuple[cst.Name, cst.BaseExpression | None]] = []
        for target in targets:
            full_name = get_full_name_for_node(target)
            if full_name and "." in full_name:
                continue
            pairs.extend(_pair_targets(target, rhs))

        # Build the symbol for each name and record which value(s) point at it.
        name_to_syms: dict[cst.Name, list[SymbolNode]] = {}
        value_to_syms: dict[cst.CSTNode, list[SymbolNode]] = {}
        for name, value in pairs:
            fqns = self.get_metadata(FullyQualifiedNameProvider, name, default=[])
            pos = self._pos(name)
            for fqn in fqns:
                sym = SymbolNode(fqn.name, "variable", self.path, pos)
                name_to_syms.setdefault(name, []).append(sym)
                if value is not None:
                    value_to_syms.setdefault(value, []).append(sym)

        # Push frames in reverse CST-visit order so on_leave pops them in LIFO.
        # Values are visited after targets, so their frames go first (popped last).
        for value, syms in reversed(value_to_syms.items()):
            self._push_decls(value, syms)
        for name, syms in reversed(name_to_syms.items()):
            for sym in syms:
                self._push_decls(name, [sym])

    def _add_import(self, from_prefix: str, node: cst.Import | cst.ImportFrom) -> None:
        current_decl = self.decl_stack[-1][-1] if self.decl_stack else None

        module_path, module_name = None, None
        if from_prefix:
            module_path = self.resolve_import(from_prefix)
            module_name = from_prefix if module_path else None

        for alias in reversed(node.names):
            alias_name = get_full_name_for_node(alias.name)
            full_name = f"{from_prefix}.{alias_name}" if from_prefix else alias_name

            if resolved := self.resolve_import(full_name):
                module_path = resolved
                module_name = full_name if module_path else None

            if not module_path:
                code = cst.Module([]).code_for_node(alias)
                logger.warning("Failed to resolve cst.Import: '%s' in %s", code, self.path)
                continue

            # if there is an asname, that is the decl being added
            # and we resolve the entire import
            if alias.asname:
                decl_name = alias.asname.name

            # if there is not an asname, the first name is the decl
            else:
                decl_name = alias.name

            # add the import lookup so we can resolve it in on_leave
            self.import_lookup[decl_name] = import_info = Import(
                path=module_path,
                module=module_name,
                decl=(
                    full_name[len(module_name) + 1 :]
                    if module_name and module_name != full_name
                    else None
                ),
            )

            # get the first name of the decl, eg 'google' for 'google.cloud'
            while isinstance(decl_name, cst.Attribute):
                decl_name = decl_name.value

            # add a decl if the import is in the module context
            if current_decl and current_decl.type == "module":
                sym = SymbolNode(
                    f"{self.module_node.fqname}.{decl_name.value}",
                    "import",
                    self.path,
                    self._pos(alias),
                    import_info,
                )
                self._push_decl(alias, sym)

            # add the import edge to the last decl on the stack
            self.import_edges.add((self.decl_stack[-1][-1], import_info))

    def visit_Module(self, node: cst.Module) -> None:
        assert not self.decl_stack, "Module node should be the first visited node"
        fqns = self.get_metadata(FullyQualifiedNameProvider, node, default=[])
        sym = SymbolNode(next(iter(fqns)).name, "module", self.path, self._pos(node))
        self._push_decl(node, sym)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._add_decl(node, "function")

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._add_decl(node, "class")

    def visit_Assign(self, node: cst.Assign) -> None:
        self._add_variable(node)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self._add_variable(node)

    def visit_Import(self, node: cst.Import) -> None:
        self._add_import("", node)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # FIXME support import star
        if isinstance(node.names, cst.ImportStar):
            return

        module = ""
        if node.module:
            module = get_full_name_for_node(node.module) or ""

        if node.relative:
            if self.path.name == "__init__.py":
                current_package = self.module_node.fqname
            else:
                current_package = self.module_node.fqname.rpartition(".")[0]

            prefix = "." * len(node.relative)
            module = resolve_name(f"{prefix}{module}", current_package)

        self._add_import(module, node)

    def on_leave(self, original_node: cst.CSTNode) -> None:
        self.nearest_decls[original_node] = list(self.decl_stack[-1]) if self.decl_stack else []
        for frame in reversed(self.node_to_frames.get(original_node, [])):
            last = self.decl_stack.pop()
            assert last == frame, f"Expected {last} to match {frame} on leave of {original_node}"

        # only run once for the Module node
        if not isinstance(original_node, cst.Module):
            return

        parent_map = self.metadata[ParentNodeProvider]
        references = set()
        for scope in self.metadata[ScopeProvider].values():
            for access in scope.accesses:
                for referent in access.referents:
                    if isinstance(referent, cst.metadata.scope_provider.BuiltinAssignment):
                        continue
                    references.add((access, referent))

        for access, referent in references:
            owner_symbols = self.nearest_decls.get(access.node, [])
            target_node = referent.node
            if isinstance(referent, cst.metadata.scope_provider.ImportAssignment):
                target_node = referent.as_name
                original_import = self.import_lookup.get(referent.as_name)
                if not original_import:
                    code = cst.Module([]).code_for_node(referent.as_name)
                    logger.warning("Failed to resolve import access: '%s' in %s", code, self.path)

                else:
                    accessed_attrs = [] if not original_import.decl else [original_import.decl]

                    if isinstance(access.node, (cst.Name, cst.Attribute)):
                        # figure out what is being accessed
                        prev_access, curr_access = None, access.node
                        while prev_access := parent_map.get(curr_access):
                            if not isinstance(prev_access, cst.Attribute):
                                break
                            accessed_attrs.append(prev_access.attr.value)
                            curr_access = prev_access

                    # Create the new Import with the specific symbol being accessed
                    resolved_import = Import(
                        path=original_import.path,
                        module=original_import.module,
                        decl=".".join(accessed_attrs) if accessed_attrs else None,
                    )

                    for owner_symbol in owner_symbols:
                        self.import_edges.add((owner_symbol, resolved_import))

            target_symbols = self.node_to_symbols.get(target_node)
            if not target_symbols:
                fallback = self.nearest_decls.get(target_node, [])
                target_symbols = fallback[:1]

            if not target_symbols:
                logger.debug(
                    "Missing target symbol for referent %s %s %s",
                    referent,
                    referent.node,
                    target_node,
                )

            for target_symbol in target_symbols:
                for owner_symbol in owner_symbols:
                    if target_symbol != owner_symbol and target_symbol and owner_symbol:
                        self.internal_edges.add((owner_symbol, target_symbol))
