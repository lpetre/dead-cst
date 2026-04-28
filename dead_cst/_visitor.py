from __future__ import annotations

import logging
from functools import cache
from importlib.util import resolve_name
from pathlib import Path
from typing import Generator, Literal, Mapping, cast

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    ParentNodeProvider,
    PositionProvider,
    ScopeProvider,
)
from libcst.metadata.scope_provider import (
    Assignment,
    ClassScope,
    FunctionScope,
    GlobalScope,
    ImportAssignment,
    Scope,
)

from ._branches import make_unreachable_node, unreachable_suites
from ._flow import live_at_exit, live_referents
from ._fqn import FixedFullyQualifiedNameProvider
from ._plugins._core import UNRESOLVED_PREFIX
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
        for nm, n in _dotted_name_parts(prefix, node.value):
            yield nm, n
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
        FixedFullyQualifiedNameProvider,
        ScopeProvider,
        ParentNodeProvider,
        PositionProvider,
    )

    def _pos(self, node: cst.CSTNode):
        return self.get_metadata(PositionProvider, node, default=None)

    @staticmethod
    def _scope_body(scope, module_node: cst.Module) -> list | None:
        """Statement list for a scope, or ``None`` if flow analysis is unsupported."""
        if isinstance(scope, GlobalScope):
            return list(module_node.body)
        if isinstance(scope, (FunctionScope, ClassScope)):
            return list(scope.node.body.body)
        return None

    def __init__(self, path: Path, search_paths: list[Path]):
        self.path = path
        self.search_paths = search_paths
        self.node_to_frames: dict[cst.CSTNode, list[list[SymbolNode]]] = {}
        self.decl_stack: list[list[SymbolNode]] = []
        self.nearest_decls: dict[cst.CSTNode, list[SymbolNode]] = {}
        self.import_lookup: dict[cst.CSTNode, Import] = {}
        self.import_edges: set[tuple[SymbolNode, Import]] = set()
        self.internal_edges: set[tuple[SymbolNode, SymbolNode]] = set()
        self.dunder_all_refs: list[tuple[SymbolNode, list[str]]] = []
        self.trie: SymbolTrie = SymbolTrie()
        # CST node used as the flow-analysis "binding site" for each
        # top-level decl. Functions/classes use the def itself, variables
        # use the LHS Name, imports use the ImportAlias. All are
        # descendants of the containing statement, which is what
        # ``live_at_exit`` matches against.
        self.symbol_referent_nodes: dict[SymbolNode, cst.CSTNode] = {}
        # Synthetic graph nodes for statically-unreachable branches.
        # ``dead_suite_owner`` maps the ``id()`` of each dead suite node
        # to the synthetic ``SymbolNode`` that "owns" any references
        # made inside it; ``unreachable_nodes`` is the flat list the
        # analyzer pulls into the graph.
        self.unreachable_nodes: list[SymbolNode] = []
        self.dead_suite_owner: dict[int, SymbolNode] = {}
        self.unreachable_internal_edges: set[tuple[SymbolNode, SymbolNode]] = set()
        self.unreachable_import_edges: set[tuple[SymbolNode, Import]] = set()

    @property
    def module_node(self) -> SymbolNode:
        if not self.decl_stack:
            raise ValueError("Module node has not been set yet.")
        return self.decl_stack[0][0]

    @cache
    def resolve_import(self, name: str) -> str | Path | None:
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
        self.node_to_frames.setdefault(node, []).append(frame)
        self.decl_stack.append(frame)
        for d in frame:
            self.trie.add_declaration(d)

    def _add_decl(
        self,
        node: cst.CSTNode,
        type_: Literal["module", "class", "function", "variable"],
    ):
        if len(self.decl_stack) > 1:
            return

        fqns = self.get_metadata(FixedFullyQualifiedNameProvider, node, default=[])
        pos = self._pos(node)
        for fqn in fqns:
            sym = SymbolNode(fqn.name, type_, self.path, pos)
            self.symbol_referent_nodes[sym] = node
            self._push_decl(node, sym)

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
        if len(self.decl_stack) > 1:
            return

        if isinstance(node, cst.Assign):
            targets = [t.target for t in node.targets]
        else:
            targets = [node.target]

        # For `x: T` (AnnAssign without a value) treat the annotation expression
        # as the rhs so references inside it are attributed to the new symbol.
        rhs = node.value
        if rhs is None and isinstance(node, cst.AnnAssign):
            rhs = node.annotation.annotation

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
            fqns = self.get_metadata(FixedFullyQualifiedNameProvider, name, default=[])
            pos = self._pos(name)
            for fqn in fqns:
                sym = SymbolNode(fqn.name, "variable", self.path, pos)
                self.symbol_referent_nodes[sym] = name
                name_to_syms.setdefault(name, []).append(sym)
                if value is not None:
                    value_to_syms.setdefault(value, []).append(sym)

                if (
                    isinstance(name, cst.Name)
                    and name.value == "__all__"
                    and value is not None
                    and (referenced := self._extract_string_sequence(value)) is not None
                ):
                    # ModuleDundersPlugin keeps __all__ itself alive; we just
                    # need to thread it through to the listed declarations
                    # once the module's trie is populated.
                    self.dunder_all_refs.append((sym, referenced))

        # Push frames in reverse CST-visit order so on_leave pops them in LIFO.
        # Values are visited after targets, so their frames go first (popped last).
        for value, syms in reversed(value_to_syms.items()):
            self._push_decls(value, syms)
        for name, syms in reversed(name_to_syms.items()):
            for sym in syms:
                self._push_decls(name, [sym])

    def _add_import(self, from_prefix: str, node: cst.Import | cst.ImportFrom) -> None:
        current_decl = self.decl_stack[-1][-1] if self.decl_stack else None

        module_path: str | Path | None = None
        module_name: str | None = None
        if from_prefix:
            if path := self.resolve_import(from_prefix):
                module_path = path
                module_name = from_prefix

        # ``visit_ImportFrom`` routes ``from X import *`` to ``_add_star_import``,
        # so by the time we get here ``names`` is always the alias sequence form.
        assert not isinstance(node.names, cst.ImportStar)
        for alias in reversed(node.names):
            alias_name = get_full_name_for_node(alias.name)
            # alias.name is always Name | Attribute, both of which produce a
            # dotted-name string; the helper only returns None for unsupported
            # node types we never see here.
            assert alias_name is not None
            full_name = f"{from_prefix}.{alias_name}" if from_prefix else alias_name

            if resolved := self.resolve_import(full_name):
                module_path = resolved
                module_name = full_name

            if not module_path:
                code = cst.Module([]).code_for_node(alias)
                logger.warning("Failed to resolve cst.Import: '%s' in %s", code, self.path)
                # Surface as a synthetic ``[unresolved] <top-level>`` node
                # anyway so plugins can still answer "which files tried to
                # import X?". The top-level package name is used (mirroring
                # how ``[external dist] fastapi`` collapses every fastapi
                # submodule import into one node) so a plugin's
                # ``importers("fastapi")`` finds them all. Reachability is
                # unaffected (the synthetic has no outbound edges).
                top_level = full_name.split(".", 1)[0]
                module_path = f"{UNRESOLVED_PREFIX}{top_level}"
                module_name = full_name
            assert module_name is not None

            if alias.asname:
                decl_name = alias.asname.name
            else:
                decl_name = alias.name

            self.import_lookup[decl_name] = import_info = Import(
                path=module_path,
                module=module_name,
                decl=(
                    full_name[len(module_name) + 1 :]
                    if module_name and module_name != full_name
                    else None
                ),
            )

            # ``import google.cloud`` binds ``google`` in the local scope; the
            # decl is stored under that bare name, not the dotted path.
            while isinstance(decl_name, cst.Attribute):
                decl_name = decl_name.value

            if current_decl and current_decl.type == "module":
                sym = SymbolNode(
                    f"{self.module_node.fqname}.{decl_name.value}",
                    "import",
                    self.path,
                    self._pos(alias),
                    import_info,
                )
                self.symbol_referent_nodes[sym] = alias
                self._push_decl(alias, sym)

            self.import_edges.add((self.decl_stack[-1][-1], import_info))

    def visit_Module(self, node: cst.Module) -> None:
        assert not self.decl_stack, "Module node should be the first visited node"
        fqns = self.get_metadata(FixedFullyQualifiedNameProvider, node, default=[])
        sym = SymbolNode(next(iter(fqns)).name, "module", self.path, self._pos(node))
        # Cache so ``_finalize_module_declarations`` can locate the trie
        # node after ``on_leave`` has popped the module frame.
        self._module_fqname = sym.fqname
        self._push_decl(node, sym)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._add_decl(node, "function")

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._add_decl(node, "class")

    def visit_Assign(self, node: cst.Assign) -> None:
        self._add_variable(node)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self._add_variable(node)

    def visit_If(self, node: cst.If) -> None:
        self._record_dead_suites(node)

    def visit_While(self, node: cst.While) -> None:
        self._record_dead_suites(node)

    def _record_dead_suites(self, stmt: cst.BaseStatement) -> None:
        """Create a synthetic ``unreachable`` graph node for each dead suite.

        Records the suite's ``id()`` so :meth:`_unreachable_owner` can
        later attribute references made inside the suite to it. Nested
        dead suites are handled implicitly: we visit outer ``If`` /
        ``While`` first, then descend, so an inner dead suite registers
        afterwards and the innermost match wins at lookup time.
        """
        for suite in unreachable_suites(stmt):
            pos = self._pos(suite)
            if pos is None:
                continue
            sym = make_unreachable_node(self.module_node.fqname, self.path, pos)
            self.unreachable_nodes.append(sym)
            self.dead_suite_owner[id(suite)] = sym

    def _unreachable_owner(
        self, node: cst.CSTNode, parent_map: Mapping[cst.CSTNode, object]
    ) -> SymbolNode | None:
        """Return the synthetic node owning ``node`` if it lives inside one.

        Walks up via ``ParentNodeProvider`` looking for a recorded dead
        suite. Returns the innermost match, or ``None`` if ``node`` is
        not inside any dead suite.
        """
        current: object = parent_map.get(node)
        while isinstance(current, cst.CSTNode):
            owner = self.dead_suite_owner.get(id(current))
            if owner is not None:
                return owner
            current = parent_map.get(current)
        return None

    def visit_Import(self, node: cst.Import) -> None:
        self._add_import("", node)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
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

        if isinstance(node.names, cst.ImportStar):
            self._add_star_import(module)
            return

        self._add_import(module, node)

    def _finalize_module_declarations(self, module_node: cst.Module) -> None:
        """Partition same-name top-level decls into live / shadowed at exit.

        For names with more than one decl, ask :func:`live_at_exit` which
        binding sites survive on at least one path to module exit. Live
        decls stay in ``trie.declarations[name]`` (multi-valued for
        conditional bindings); the rest move to ``trie.shadowed`` so the
        graph keeps their parent-module edge but cross-module imports do
        not reach them.
        """
        trie_node = self.trie._get(self._module_fqname.split("."))
        if trie_node is None:
            return

        name_decls = {n: list(d) for n, d in trie_node.declarations.items()}
        for name, decls in name_decls.items():
            if len(decls) <= 1:
                continue

            referent_nodes: list[cst.CSTNode] = []
            for d in decls:
                ref = self.symbol_referent_nodes.get(d)
                if ref is not None:
                    referent_nodes.append(ref)

            live_ids = {id(n) for n in live_at_exit(list(module_node.body), referent_nodes)}

            live_decls: list[SymbolNode] = []
            shadowed_decls: list[SymbolNode] = []
            for d in decls:
                ref = self.symbol_referent_nodes.get(d)
                if ref is not None and id(ref) in live_ids:
                    live_decls.append(d)
                else:
                    shadowed_decls.append(d)

            trie_node.finalize_declarations(name, live_decls, shadowed_decls)

    def _add_star_import(self, module: str) -> None:
        module_path = self.resolve_import(module) if module else None
        if not module_path:
            logger.warning(
                "Failed to resolve star import: 'from %s import *' in %s", module, self.path
            )
            return
        star = Import(path=module_path, module=module, star=True)
        self.import_edges.add((self.decl_stack[-1][-1], star))

    def on_leave(self, original_node: cst.CSTNode) -> None:
        self.nearest_decls[original_node] = list(self.decl_stack[-1]) if self.decl_stack else []
        for frame in reversed(self.node_to_frames.get(original_node, [])):
            last = self.decl_stack.pop()
            assert last == frame, f"Expected {last} to match {frame} on leave of {original_node}"

        # only run once for the Module node
        if not isinstance(original_node, cst.Module):
            return

        self._finalize_module_declarations(original_node)

        parent_map = self.metadata[ParentNodeProvider]
        references = set()
        # ScopeProvider's metadata is typed loosely upstream; the values are
        # always ``Scope`` instances.
        scopes = cast("set[Scope]", set(self.metadata[ScopeProvider].values()))
        for scope in scopes:
            for access in scope.accesses:
                # ``Assignment`` and ``BuiltinAssignment`` are the only
                # ``BaseAssignment`` subclasses; selecting ``Assignment``
                # excludes builtins and gives us a typed ``.node``.
                referents = [r for r in access.referents if isinstance(r, Assignment)]
                if len(referents) > 1:
                    body = self._scope_body(referents[0].scope, original_node)
                    if body is not None:
                        live_ids = {
                            id(n)
                            for n in live_referents(body, access.node, [r.node for r in referents])
                        }
                        referents = [r for r in referents if id(r.node) in live_ids]
                for referent in referents:
                    references.add((access, referent))

        for access, referent in references:
            owner_symbols = self.nearest_decls.get(access.node, [])
            unreachable_owner = self._unreachable_owner(access.node, parent_map)
            target_node = referent.node
            if isinstance(referent, ImportAssignment):
                target_node = referent.as_name
                original_import = self.import_lookup.get(referent.as_name)
                if not original_import:
                    code = cst.Module([]).code_for_node(referent.as_name)
                    logger.warning("Failed to resolve import access: '%s' in %s", code, self.path)

                else:
                    accessed_attrs = [] if not original_import.decl else [original_import.decl]

                    if isinstance(access.node, (cst.Name, cst.Attribute)):
                        curr_access = access.node
                        while parent := parent_map.get(curr_access):
                            if not isinstance(parent, cst.Attribute):
                                break
                            accessed_attrs.append(parent.attr.value)
                            curr_access = parent

                    # Create the new Import with the specific symbol being accessed
                    resolved_import = Import(
                        path=original_import.path,
                        module=original_import.module,
                        decl=".".join(accessed_attrs) if accessed_attrs else None,
                    )

                    for owner_symbol in owner_symbols:
                        self.import_edges.add((owner_symbol, resolved_import))
                    if unreachable_owner is not None:
                        self.unreachable_import_edges.add((unreachable_owner, resolved_import))

            target_symbols = [
                s for frame in self.node_to_frames.get(target_node, ()) for s in frame
            ]
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
                if unreachable_owner is not None and target_symbol is not None:
                    self.unreachable_internal_edges.add((unreachable_owner, target_symbol))

        # Resolve __all__ string references to declarations in the current module
        if self.dunder_all_refs:
            module_sym = self.node_to_frames[original_node][0][0]
            module_trie = self.trie._get(module_sym.fqname.split("."))
            if module_trie is not None:
                for owner, names in self.dunder_all_refs:
                    for name in names:
                        for target in module_trie.declarations.get(name, []):
                            if target != owner:
                                self.internal_edges.add((owner, target))
