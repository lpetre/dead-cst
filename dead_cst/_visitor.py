from __future__ import annotations

import logging
from functools import cache
from importlib.util import resolve_name
from pathlib import Path
from typing import Generator, Literal

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import FullyQualifiedNameProvider, ParentNodeProvider, ScopeProvider

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


class SymbolVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        FullyQualifiedNameProvider,
        ScopeProvider,
        ParentNodeProvider,
    )

    def __init__(self, path: Path, search_paths: list[Path]):
        self.path = path
        self.search_paths = search_paths
        self.node_to_symbols: dict[cst.CSTNode, list[SymbolNode]] = {}
        self.decl_stack: list[SymbolNode] = []
        self.nearest_decl: dict[cst.CSTNode, SymbolNode] = {}
        self.import_lookup: dict[cst.CSTNode, Import] = {}
        self.import_edges: set[tuple[SymbolNode, Import]] = set()
        self.internal_edges: set[tuple[SymbolNode, SymbolNode]] = set()
        self.dunder_all_refs: list[tuple[SymbolNode, list[str]]] = []
        self.trie: SymbolTrie = SymbolTrie()

    @property
    def module_node(self) -> SymbolNode:
        if not self.decl_stack:
            raise ValueError("Module node has not been set yet.")
        return self.decl_stack[0]

    @cache
    def resolve_import(self, name: str) -> str | Path:
        return resolve_import(name, self.search_paths)

    def _push_decl(self, node: cst.CSTNode, decl: SymbolNode):
        self.node_to_symbols.setdefault(node, []).append(decl)
        self.decl_stack.append(decl)
        self.trie.add_declaration(decl)

    def _add_decl(
        self,
        node: cst.CSTNode,
        type_: Literal["module", "class", "function", "variable"],
    ):
        # Only collect top-level declarations, skip nested ones
        if len(self.decl_stack) > 1:
            return

        fqns = self.get_metadata(FullyQualifiedNameProvider, node, default=[])
        for fqn in fqns:
            self._push_decl(node, SymbolNode(fqn.name, type_, self.path))

    @staticmethod
    def _extract_string_sequence(value: cst.BaseExpression) -> list[str] | None:
        """Extract string elements from a list/tuple literal, e.g. ['f', 'g']."""
        if not isinstance(value, (cst.List, cst.Tuple)):
            return None
        names = []
        for element in value.elements:
            if not isinstance(element, cst.Element):
                return None
            inner = element.value
            if not isinstance(inner, cst.SimpleString):
                return None
            try:
                evaluated = inner.evaluated_value
            except Exception:
                return None
            if not isinstance(evaluated, str):
                return None
            names.append(evaluated)
        return names

    def _add_variable(self, node: cst.Assign | cst.AnnAssign):
        # Only collect top-level declarations, skip nested ones
        if len(self.decl_stack) > 1:
            return

        if isinstance(node, cst.Assign):
            targets = [t.target for t in node.targets]
        else:
            targets = [node.target]

        value_to_syms = {}
        for target in reversed(targets):
            full_name = get_full_name_for_node(target)
            if full_name and "." in full_name:
                continue

            # see if we're unpacking, eg `a, b = ...`
            names = [target]
            if isinstance(target, cst.Tuple):
                names = [t.value for t in target.elements]

            values = [node.value]

            # this can happen with unpacking, e.g. `a, b = (1, 2)`
            if len(names) > len(values) and isinstance(node.value, cst.Tuple):
                values = [v.value for v in node.value.elements]

            if len(names) > len(values) and len(values) == 1:
                # This can happen with unpacking, e.g. `a, b = f()`
                # In this case, we assume the first value applies to all names
                values = [values[0]] * len(names)

            value_name_pairs = list(zip(values, names))
            for value, name in reversed(value_name_pairs):
                fqns = self.get_metadata(FullyQualifiedNameProvider, name, default=[])
                for fqn in fqns:
                    sym = SymbolNode(fqn.name, "variable", self.path)
                    self._push_decl(name, sym)
                    value_to_syms.setdefault(value, []).append(sym)

                    if (
                        isinstance(name, cst.Name)
                        and name.value == "__all__"
                        and (referenced := self._extract_string_sequence(value)) is not None
                    ):
                        self.dunder_all_refs.append((sym, referenced))

        values = [node.value]
        if isinstance(node.value, cst.Tuple):
            values += [v.value for v in node.value.elements]

        for value in reversed(values):
            if syms := value_to_syms.get(value):
                for sym in syms:
                    self._push_decl(value, sym)

    def _add_import(self, from_prefix: str, node: cst.Import | cst.ImportFrom) -> None:
        current_decl = self.decl_stack[-1] if self.decl_stack else None

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
                    f"{self.module_node.fqname}.{decl_name.value}", "import", self.path, import_info
                )
                self._push_decl(alias, sym)

            # add the import edge to the last decl on the stack
            self.import_edges.add((self.decl_stack[-1], import_info))

    def visit_Module(self, node: cst.Module) -> None:
        assert not self.decl_stack, "Module node should be the first visited node"
        fqns = self.get_metadata(FullyQualifiedNameProvider, node, default=[])
        sym = SymbolNode(next(iter(fqns)).name, "module", self.path)
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
        self.nearest_decl[original_node] = self.decl_stack[-1]
        for decl in reversed(self.node_to_symbols.get(original_node, [])):
            last = self.decl_stack.pop()
            assert last == decl, f"Expected {last} to match {decl} on leave of {original_node}"

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
            owner_symbol = self.nearest_decl.get(access.node)
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

                    self.import_edges.add((owner_symbol, resolved_import))

            target_symbols = self.node_to_symbols.get(target_node)
            if not target_symbols:
                target_symbol = self.nearest_decl.get(target_node)
                if target_symbol:
                    target_symbols = {target_symbol}

            if not target_symbols:
                logger.debug(
                    "Missing target symbol for referent %s %s %s",
                    referent,
                    referent.node,
                    target_node,
                )

            for target_symbol in target_symbols:
                if target_symbol != owner_symbol and target_symbol and owner_symbol:
                    self.internal_edges.add((owner_symbol, target_symbol))

        # Resolve __all__ string references to declarations in the current module
        if self.dunder_all_refs:
            module_sym = self.node_to_symbols[original_node][0]
            module_node = self.trie._get(module_sym.fqname.split("."))
            if module_node is not None:
                for owner, names in self.dunder_all_refs:
                    for name in names:
                        target = module_node.declarations.get(name)
                        if target and target != owner:
                            self.internal_edges.add((owner, target))
