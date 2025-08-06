import os
import sys
import sysconfig
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from importlib.util import resolve_name
from pathlib import Path
from typing import Generator, Literal

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    FullyQualifiedNameProvider,
    ParentNodeProvider,
    PositionProvider,
    QualifiedNameProvider,
    ScopeProvider,
)

SITE_PACKAGES = Path(sysconfig.get_path("purelib")).resolve()
STDLIB = Path(sysconfig.get_path("stdlib")).resolve()


@dataclass(frozen=True, slots=True)
class SymbolNode:
    fqname: str
    type: Literal["module", "class", "function", "variable", "import"]
    path: Path


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


def safe_resolve_module(fullname: str) -> ModuleSpec | None:
    parts = fullname.split(".")
    search_paths = list(sys.path)

    # emulate namespace __path__ resolution
    for i, part in enumerate(parts[:-1]):
        candidate_paths = []
        for base in search_paths:
            subdir = os.path.join(base, parts[i])
            if os.path.isdir(subdir):
                candidate_paths.append(subdir)
        search_paths = candidate_paths

    # print(fullname, "in [", search_paths, "]")
    # Final part resolution
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if not find_spec:
            continue
        try:
            spec = find_spec(fullname, search_paths)
            if spec:
                return spec
        except Exception:
            continue

    return None


class SymbolVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        FullyQualifiedNameProvider,
        QualifiedNameProvider,
        ScopeProvider,
        ParentNodeProvider,
        PositionProvider,
    )

    def __init__(self, path: Path, search_paths: list[Path]):
        self.path = path
        self.search_paths = search_paths
        self.decls: set[SymbolNode] = set()
        self.node_to_symbols: dict[cst.CSTNode, list[SymbolNode]] = {}
        self.decl_stack: list[SymbolNode] = []
        self.nearest_decl: dict[cst.CSTNode, SymbolNode] = {}
        self.import_edges: list[tuple[SymbolNode, str]] = []
        self.internal_edges: set[tuple[SymbolNode, SymbolNode]] = set()

    @property
    def module_node(self) -> SymbolNode:
        if not self.decl_stack:
            raise ValueError("Module node has not been set yet.")
        return self.decl_stack[0]

    def _push_decl(self, node: cst.CSTNode, decl: SymbolNode):
        print("->", decl.fqname)
        if self.decl_stack:
            self.internal_edges.add((decl, self.decl_stack[0]))
        self.node_to_symbols.setdefault(node, []).append(decl)
        self.decl_stack.append(decl)
        self.decls.add(decl)

    def resolve_import(self, name: str) -> str | Path:
        spec = safe_resolve_module(name)
        if spec is None:
            return None
        if spec.origin is None:
            return None
        if spec.origin in {"built-in", "frozen"}:
            return f"[stdlib] {name}"
        path = Path(spec.origin).resolve()
        if path.is_relative_to(STDLIB):
            return f"[stdlib] {name}"
        if path.is_relative_to(SITE_PACKAGES):
            return f"[external] {name}"
        for search in self.search_paths:
            if path.is_relative_to(search):
                return path
        raise Exception(
            f"Module {name} resolved to an unexpected path: {path} (not in {self.base})"
        )

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

        values = [node.value]
        if isinstance(node.value, cst.Tuple):
            values += [v.value for v in node.value.elements]

        for value in reversed(values):
            if syms := value_to_syms.get(value):
                for sym in syms:
                    print("Target", sym.fqname, "for value", value)
                    self._push_decl(value, sym)

    def visit_Import(self, node: cst.Import) -> None:
        current_decl = self.decl_stack[-1] if self.decl_stack else None

        for name in reversed(node.names):
            found = None
            for curr_name, curr_node in _dotted_name_parts("", name.name):
                if not self.resolve_import(curr_name):
                    continue
                found = curr_name, curr_node
                break

            if not found:
                print(f"Failed to resolve import: {name.name} in {self.path}")
                continue

            if current_decl and current_decl.type == "module":
                sym = SymbolNode(f"{self.module_node.fqname}.{found[0]}", "import", self.path)
                self._push_decl(found[1], sym)

            # add the import edge to the last decl on the stack
            self.import_edges.append((self.decl_stack[-1], found[0]))

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        current_decl = self.decl_stack[-1] if self.decl_stack else None

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

        # FIXME support import star
        if isinstance(node.names, cst.ImportStar):
            return

        found_module = self.resolve_import(module)
        for alias in reversed(node.names):
            real_name = get_full_name_for_node(alias.name)
            if not real_name:
                continue

            if not found_module:
                found_module = self.resolve_import(f"{module}.{real_name}")

            if not found_module:
                print(f"Failed to resolve import: {found_module} in {self.path}")
                continue

            as_name = real_name
            if alias and alias.asname:
                if eval_alias := alias.evaluated_alias:
                    as_name = eval_alias

            if current_decl and current_decl.type == "module":
                dst = f"{self.module_node.fqname}.{as_name}"
                sym = SymbolNode(dst, "import", self.path)
                self._push_decl(alias, sym)

            src = f"{module}.{real_name}" if module else real_name
            self.import_edges.append((self.decl_stack[-1], src))

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

    def on_leave(self, original_node: cst.CSTNode) -> None:
        self.nearest_decl[original_node] = self.decl_stack[-1]
        for decl in reversed(self.node_to_symbols.get(original_node, [])):
            last = self.decl_stack.pop()
            print("<-", last.fqname)
            assert last == decl, f"Expected {last} to match {decl} on leave of {original_node}"

        # only run once for the Module node
        if not isinstance(original_node, cst.Module):
            return

        for scope in self.metadata[ScopeProvider].values():
            for access in scope.accesses:
                owner_symbol = self.nearest_decl.get(access.node)
                for referent in access.referents:
                    if isinstance(referent, cst.metadata.scope_provider.BuiltinAssignment):
                        continue

                    # print("Owner:", access.node)
                    # print("Referent:", referent, referent.node)

                    target_node = referent.node
                    if isinstance(referent, cst.metadata.scope_provider.ImportAssignment):
                        target_node = referent.as_name

                    target_symbols = self.node_to_symbols.get(target_node)
                    if not target_symbols:
                        target_symbol = self.nearest_decl.get(target_node)
                        if target_symbol:
                            target_symbols = {target_symbol}

                    if not target_symbols:
                        print(
                            "Missing target symbol for referent",
                            referent,
                            referent.node,
                            target_node,
                        )

                    for target_symbol in target_symbols:
                        if target_symbol != owner_symbol and target_symbol and owner_symbol:
                            self.internal_edges.add((owner_symbol, target_symbol))
