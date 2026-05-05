from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from functools import cache
from importlib.util import resolve_name
from pathlib import Path
from typing import Generator, Literal, cast

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    CodeRange,
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

from ._branches import DefaultUnreachableRegionDetector, UnreachableRegionDetector
from ._flow import live_at_exit, live_referents
from ._fqn import FixedFullyQualifiedNameProvider
from ._plugins._core import UNRESOLVED_PREFIX
from ._resolvers import ImportResolver, default_resolve_import
from ._symbols import Import, NodeFlags, SymbolNode, SymbolTrie

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


@dataclass(frozen=True, slots=True)
class VisitorPayload:
    """Serializable per-file output of :class:`SymbolVisitor`.

    Four fields cover everything the analyzer needs to reconstruct one
    file's contribution to the symbol graph:

    * ``nodes`` -- every real ``SymbolNode`` for this file (module +
      top-level decls). Decls displaced by flow analysis are flagged
      :data:`NodeFlags.SHADOWED`; the apply step uses that flag to keep
      them out of the lookup trie while still emitting the parent-module
      edge for the graph.
    * ``edges`` -- ``(src, dst, access_pos)`` triples for resolved
      decl-to-decl references. ``access_pos`` is the source location
      of the reference; the apply step compares it against
      ``dead_suites`` to decide whether the resulting graph edge gets
      :data:`EdgeFlags.DEAD_BRANCH`.
    * ``imports`` -- ``(src, Import, access_pos)`` triples for
      unresolved cross-file references. The apply step feeds them into
      ``resolve_edges`` along with the derived flag.
    * ``dead_suites`` -- positions of every statically-dead suite in
      the file (including ones with no outgoing references). Used both
      for flag derivation and for surfacing "this file has unreachable
      code at line X" reports without per-edge attribution.
    """

    nodes: tuple[SymbolNode, ...]
    edges: tuple[tuple[SymbolNode, SymbolNode, CodeRange], ...]
    imports: tuple[tuple[SymbolNode, Import, CodeRange], ...]
    dead_suites: tuple[CodeRange, ...]


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

    # ``Cacheable`` (name, version) fingerprint contributed to the
    # per-run cache key. ``__version__`` already covers tagged releases,
    # but it doesn't move between commits, so a behaviour-affecting
    # visitor change between two installs at the same dev version
    # would be served from stale ``VisitorPayload`` blobs. Bump
    # ``version`` (to the current Unix epoch) on any change to the
    # visitor's per-file output -- new node kinds surfaced as decls,
    # edge-attribution rules, flow-analysis fixes, etc. Concurrent
    # bumps on different branches merge with ``max()`` semantics.
    name: str = "default"
    version: int = 1777971949

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

    def __init__(
        self,
        path: Path,
        search_paths: list[Path],
        import_resolver: ImportResolver = default_resolve_import,
        unreachable_detector: UnreachableRegionDetector | None = None,
        wrapper: cst.MetadataWrapper | None = None,
    ):
        self.path = path
        self.search_paths = search_paths
        self._import_resolver = import_resolver
        self._unreachable_detector = (
            unreachable_detector
            if unreachable_detector is not None
            else DefaultUnreachableRegionDetector()
        )
        # Stash the wrapper that's about to drive ``visit`` so the
        # unreachable-region detector can reuse the same resolved
        # metadata cache (notably ``PositionProvider``) instead of
        # paying O(file) to re-resolve on a sibling wrapper.
        self._wrapper = wrapper
        self.node_to_frames: dict[cst.CSTNode, list[list[SymbolNode]]] = {}
        self.decl_stack: list[list[SymbolNode]] = []
        self.nearest_decls: dict[cst.CSTNode, list[SymbolNode]] = {}
        self.import_lookup: dict[cst.CSTNode, Import] = {}
        # Edges carry the access position so the apply step can decide
        # whether the reference originated inside a dead suite. Set
        # semantics still dedupe identical triples; same-decl refs from
        # different positions remain distinct (which matters when one is
        # live and one is in a dead branch).
        self.import_edges: set[tuple[SymbolNode, Import, CodeRange]] = set()
        self.internal_edges: set[tuple[SymbolNode, SymbolNode, CodeRange]] = set()
        self.dunder_all_refs: list[tuple[SymbolNode, list[str]]] = []
        self.trie: SymbolTrie = SymbolTrie()
        # CST node used as the flow-analysis "binding site" for each
        # top-level decl. Functions/classes use the def itself, variables
        # use the LHS Name, imports use the ImportAlias. All are
        # descendants of the containing statement, which is what
        # ``live_at_exit`` matches against.
        self.symbol_referent_nodes: dict[SymbolNode, cst.CSTNode] = {}
        self.dead_suites: list[CodeRange] = []
        # Decls displaced by flow analysis. Tracked here (rather than on
        # the trie) so the trie holds only entries cross-module imports
        # should resolve to. Stored unflagged; ``to_payload`` produces
        # ``NodeFlags.SHADOWED`` copies and remaps any edge endpoints
        # that point at them, so the graph keeps consistent identity.
        self.shadowed_decls: list[SymbolNode] = []
        # Module-scope walrus targets keyed by name. PEP 572 says a
        # walrus inside a comprehension binds in the comprehension's
        # enclosing scope, but ``ScopeProvider`` doesn't propagate the
        # binding out -- the access ``return last`` in
        # ``def use(): return last`` lands with empty referents when
        # ``last`` was bound by a walrus inside a module-level
        # comprehension. We patch that gap in ``on_leave`` by routing
        # any unresolved Name access whose ``.value`` matches a leaked
        # walrus target back to the corresponding decl.
        self.walrus_leak_targets: dict[str, list[SymbolNode]] = {}

    @property
    def module_node(self) -> SymbolNode:
        if not self.decl_stack:
            raise ValueError("Module node has not been set yet.")
        return self.decl_stack[0][0]

    @cache
    def resolve_import(self, name: str) -> str | Path | None:
        return self._import_resolver(name, self.search_paths)

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

            self.import_edges.add((self.decl_stack[-1][-1], import_info, self._pos(alias)))

    def visit_Module(self, node: cst.Module) -> None:
        assert not self.decl_stack, "Module node should be the first visited node"
        fqns = self.get_metadata(FixedFullyQualifiedNameProvider, node, default=[])
        sym = SymbolNode(next(iter(fqns)).name, "module", self.path, self._pos(node))
        # Cache so ``_finalize_module_declarations`` can locate the trie
        # node after ``on_leave`` has popped the module frame.
        self._module_fqname = sym.fqname
        self._push_decl(node, sym)
        # Reuse the wrapper that's driving this visit when one was
        # provided -- it already has ``PositionProvider`` resolved, so
        # the detector hits the cache. ``unsafe_skip_copy`` is fine for
        # the fallback path because the detector reads metadata only.
        wrapper = self._wrapper or cst.MetadataWrapper(node, unsafe_skip_copy=True)
        self.dead_suites = list(self._unreachable_detector.find_regions(wrapper))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._add_decl(node, "function")

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._add_decl(node, "class")

    def visit_Assign(self, node: cst.Assign) -> None:
        self._add_variable(node)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self._add_variable(node)

    def visit_TypeAlias(self, node: cst.TypeAlias) -> None:
        self._add_type_alias(node)

    def visit_NamedExpr(self, node: cst.NamedExpr) -> None:
        self._add_walrus(node)

    def _add_type_alias(self, node: cst.TypeAlias) -> None:
        """Surface a PEP 695 ``type X = ...`` statement as a top-level decl.

        ``FixedFullyQualifiedNameProvider`` does not name-bind ``cst.TypeAlias``,
        so the FQN is constructed from the enclosing module. ``ScopeProvider``
        reports the binding site as the whole ``TypeAlias`` node, so the frame
        is pushed there -- this matches what cross-references look up later.
        """
        if len(self.decl_stack) > 1:
            return
        fqname = f"{self.module_node.fqname}.{node.name.value}"
        sym = SymbolNode(fqname, "type_alias", self.path, self._pos(node.name))
        self.symbol_referent_nodes[sym] = node
        self._push_decl(node, sym)

    def _walrus_module_scope(self, node: cst.NamedExpr) -> tuple[bool, bool]:
        """``(at_module, in_comprehension)`` for the walrus's binding scope.

        Per PEP 572, a walrus inside a comprehension binds in the
        comprehension's enclosing scope, so we ignore comprehension
        scopes when deciding "at module". The first ``FunctionDef`` /
        ``Lambda`` / ``ClassDef`` ancestor terminates the walk;
        reaching ``Module`` without hitting one of those means the
        binding leaks to the module namespace. The second flag is
        ``True`` iff the walk passed through any comprehension on the
        way -- callers use it to decide whether pushing a value frame
        would steal attribution from the comprehension target's
        fallback (``Comprehension targets share their enclosing decl's
        frame, so a walrus's value frame would create spurious
        last -> result edges in ``[last := n for n in nums]``).
        """
        parent_map = cast("dict[cst.CSTNode, cst.CSTNode]", self.metadata[ParentNodeProvider])
        parent = parent_map.get(node)
        in_comp = False
        while parent is not None:
            if isinstance(parent, (cst.FunctionDef, cst.Lambda, cst.ClassDef)):
                return False, in_comp
            if isinstance(parent, (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)):
                in_comp = True
            parent = parent_map.get(parent)
        return True, in_comp

    def _add_walrus(self, node: cst.NamedExpr) -> None:
        """Surface a module-scope walrus target as a top-level decl.

        Pushes a frame on the target ``Name`` so accesses elsewhere
        that resolve to the binding via ``ScopeProvider`` find the
        correct symbol. For walruses outside any comprehension a
        frame is also pushed on the value subtree so RHS accesses
        attribute to the walrus decl (mirroring ``_add_variable``).
        Comprehension-leaked walruses skip the value frame: their
        for-loop targets fall back to the enclosing decl's frame, and
        layering the walrus's frame on top would route those fallback
        edges through the walrus decl instead.
        """
        at_module, in_comp = self._walrus_module_scope(node)
        if not at_module:
            return
        if not isinstance(node.target, cst.Name):
            return

        # Walrus targets in comprehensions don't get a module-scoped
        # FQN from the provider -- it returns ``mod.<comprehension>.x``
        # there. Build the module-scoped FQN directly to keep the
        # decl's identity consistent with what other top-level decls
        # use.
        fqname = f"{self.module_node.fqname}.{node.target.value}"
        pos = self._pos(node.target)
        sym = SymbolNode(fqname, "variable", self.path, pos)
        self.symbol_referent_nodes[sym] = node.target
        self.walrus_leak_targets.setdefault(node.target.value, []).append(sym)

        # Push frames in reverse CST-visit order: the target ``Name`` is
        # visited before the value, so the value frame goes on first
        # (popped last). Mirrors the ordering in ``_add_variable``.
        if not in_comp:
            self._push_decls(node.value, [sym])
        self._push_decls(node.target, [sym])

    def visit_Import(self, node: cst.Import) -> None:
        self._add_import("", node)

    def visit_Call(self, node: cst.Call) -> None:
        """Treat ``__import__`` / ``importlib.import_module`` as star imports.

        Conservative over-approximation: the actual attribute pulled
        out via ``getattr(__import__('m'), 'x')`` is opaque, so every
        top-level decl in the target module is kept alive. Relative
        names are resolved against the file's enclosing package the
        same way ``from .x import *`` is. Non-literal arguments are
        skipped with a warning.
        """
        callee = self._dynamic_import_call_name(node.func)
        if callee is None:
            return
        if not node.args:
            return
        raw = self._string_literal_value(node.args[0].value)
        if not raw:
            logger.warning(
                "Skipping dynamic import '%s(...)' in %s: name is not a string literal",
                callee,
                self.path,
            )
            return
        module = self._resolve_dynamic_import_name(raw, callee, node)
        if module is None:
            return
        self._add_star_import(module, self._pos(node))
        if callee == "__import__":
            self._handle_dunder_import_fromlist(node, module)

    def _resolve_dynamic_import_name(
        self,
        raw: str,
        callee: Literal["__import__", "importlib.import_module"],
        node: cst.Call,
    ) -> str | None:
        """Resolve a dynamic-import name to an absolute module path.

        ``importlib.import_module`` encodes relativity as leading dots
        in ``raw``; ``__import__`` encodes it in the ``level`` int
        kwarg (the name itself is always absolute). Both forms resolve
        against the file's enclosing package, or against an explicit
        literal ``package=`` for ``importlib.import_module``. Returns
        ``None`` after warning when any required component is
        non-literal or the resolution fails.
        """
        if callee == "__import__":
            if raw.startswith("."):
                logger.warning(
                    "Skipping '__import__(%r)' in %s: leading dots are invalid for __import__",
                    raw,
                    self.path,
                )
                return None
            level_expr = self._call_arg(node, position=4, keyword="level")
            if level_expr is None:
                return raw
            level = self._int_literal_value(level_expr)
            if level is None:
                logger.warning(
                    "Skipping '__import__(%r, ..., level=...)' in %s: level is not an int literal",
                    raw,
                    self.path,
                )
                return None
            if level == 0:
                return raw
            return self._resolve_relative_name("." * level + raw, package=None)
        # importlib.import_module
        if not raw.startswith("."):
            return raw
        package_expr = self._call_arg(node, position=1, keyword="package")
        package: str | None = None
        if package_expr is not None and not self._is_none_literal(package_expr):
            package = self._string_literal_value(package_expr)
            if package is None:
                logger.warning(
                    "Skipping 'importlib.import_module(%r, package=...)' in %s: "
                    "package is not a string literal",
                    raw,
                    self.path,
                )
                return None
        return self._resolve_relative_name(raw, package=package)

    def _resolve_relative_name(self, name: str, package: str | None) -> str | None:
        if package is None:
            package = self._current_package()
        try:
            return resolve_name(name, package)
        except (ImportError, ValueError):
            logger.warning(
                "Skipping dynamic import %r in %s: could not resolve against package %r",
                name,
                self.path,
                package,
            )
            return None

    def _current_package(self) -> str:
        """Package the current file resolves relative imports against.

        ``__init__.py`` *is* its own package; everything else uses the
        parent package.
        """
        if self.path.name == "__init__.py":
            return self.module_node.fqname
        return self.module_node.fqname.rpartition(".")[0]

    @staticmethod
    def _string_literal_value(arg: cst.BaseExpression) -> str | None:
        if not isinstance(arg, cst.SimpleString):
            return None
        try:
            value = arg.evaluated_value
        except Exception:
            return None
        return value if isinstance(value, str) else None

    @staticmethod
    def _int_literal_value(arg: cst.BaseExpression) -> int | None:
        if not isinstance(arg, cst.Integer):
            return None
        try:
            return arg.evaluated_value
        except ValueError:
            return None

    @staticmethod
    def _is_none_literal(arg: cst.BaseExpression) -> bool:
        return isinstance(arg, cst.Name) and arg.value == "None"

    def _handle_dunder_import_fromlist(self, node: cst.Call, module: str) -> None:
        fromlist_expr = self._call_arg(node, position=3, keyword="fromlist")
        if fromlist_expr is None:
            return
        entries = self._extract_string_sequence(fromlist_expr)
        if entries is None:
            logger.warning(
                "'__import__(%r, fromlist=...)' in %s: fromlist is not a literal "
                "list/tuple, submodule entries are not resolved",
                module,
                self.path,
            )
            return
        access_pos = self._pos(node)
        for entry in entries:
            if not entry:
                continue
            submod = f"{module}.{entry}"
            if self.resolve_import(submod):
                self._add_star_import(submod, access_pos)

    @staticmethod
    def _dynamic_import_call_name(
        func: cst.BaseExpression,
    ) -> Literal["__import__", "importlib.import_module"] | None:
        if isinstance(func, cst.Name) and func.value == "__import__":
            return "__import__"
        if (
            isinstance(func, cst.Attribute)
            and isinstance(func.value, cst.Name)
            and func.value.value == "importlib"
            and func.attr.value == "import_module"
        ):
            return "importlib.import_module"
        return None

    @staticmethod
    def _call_arg(node: cst.Call, *, position: int, keyword: str) -> cst.BaseExpression | None:
        """Return a positional-index-``position``-or-keyword-``keyword`` arg, or ``None``."""
        for i, arg in enumerate(node.args):
            if arg.keyword is not None:
                if isinstance(arg.keyword, cst.Name) and arg.keyword.value == keyword:
                    return arg.value
            elif i == position:
                return arg.value
        return None

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = ""
        if node.module:
            module = get_full_name_for_node(node.module) or ""

        if node.relative:
            prefix = "." * len(node.relative)
            module = resolve_name(f"{prefix}{module}", self._current_package())

        if isinstance(node.names, cst.ImportStar):
            self._add_star_import(module, self._pos(node))
            return

        self._add_import(module, node)

    def _finalize_module_declarations(self, module_node: cst.Module) -> None:
        """Partition same-name top-level decls into live / shadowed at exit.

        For names with more than one decl, ask :func:`live_at_exit` which
        binding sites survive on at least one path to module exit. Live
        decls stay in ``trie.declarations[name]`` (multi-valued for
        conditional bindings); the rest move to
        ``self.shadowed_decls`` so the graph keeps their parent-module
        edge but cross-module imports do not reach them.
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
            shadowed_here: list[SymbolNode] = []
            for d in decls:
                ref = self.symbol_referent_nodes.get(d)
                if ref is not None and id(ref) in live_ids:
                    live_decls.append(d)
                else:
                    shadowed_here.append(d)

            if live_decls:
                trie_node.declarations[name] = list(live_decls)
            else:
                del trie_node.declarations[name]
            self.shadowed_decls.extend(shadowed_here)

    def _add_star_import(self, module: str, access_pos: CodeRange) -> None:
        module_path = self.resolve_import(module) if module else None
        if not module_path:
            logger.warning(
                "Failed to resolve star import: 'from %s import *' in %s", module, self.path
            )
            return
        star = Import(path=module_path, module=module, star=True)
        self.import_edges.add((self.decl_stack[-1][-1], star, access_pos))

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

        # Walrus bindings that leaked from a comprehension don't appear
        # in the enclosing scope's assignment list (libcst's
        # ``ScopeProvider`` keeps them in the ``ComprehensionScope``),
        # so any access to the leaked name lands here with empty
        # referents. Route those accesses to the matching top-level
        # walrus decl so the graph mirrors the runtime semantics.
        if self.walrus_leak_targets:
            for scope in scopes:
                for access in scope.accesses:
                    if not isinstance(access.node, cst.Name):
                        continue
                    if any(isinstance(r, Assignment) for r in access.referents):
                        continue
                    targets = self.walrus_leak_targets.get(access.node.value)
                    if not targets:
                        continue
                    owner_symbols = self.nearest_decls.get(access.node, [])
                    access_pos = self._pos(access.node)
                    for target_symbol in targets:
                        for owner_symbol in owner_symbols:
                            if target_symbol != owner_symbol and target_symbol and owner_symbol:
                                self.internal_edges.add((owner_symbol, target_symbol, access_pos))

        for access, referent in references:
            owner_symbols = self.nearest_decls.get(access.node, [])
            access_pos = self._pos(access.node)
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
                        self.import_edges.add((owner_symbol, resolved_import, access_pos))

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
                        self.internal_edges.add((owner_symbol, target_symbol, access_pos))

        # Resolve __all__ string references to declarations in the current module.
        # The owner's own position stands in as the access position -- the
        # ``__all__`` literal is a single source location, not a per-name
        # one, so the per-string subexpression isn't worth tracking.
        if self.dunder_all_refs:
            module_sym = self.node_to_frames[original_node][0][0]
            module_trie = self.trie._get(module_sym.fqname.split("."))
            if module_trie is not None:
                for owner, names in self.dunder_all_refs:
                    for name in names:
                        for target in module_trie.declarations.get(name, []):
                            if target != owner:
                                self.internal_edges.add((owner, target, owner.position))

    def to_payload(self) -> VisitorPayload:
        """Materialize visitor state into a serializable :class:`VisitorPayload`.

        Decls in :attr:`shadowed_decls` are emitted as
        :data:`NodeFlags.SHADOWED` flagged copies and any edge endpoint
        pointing at them is remapped to the same flagged identity, so
        the resulting graph nodes and edges line up. The per-edge
        :class:`CodeRange` (the access position) is preserved as-is;
        the apply step in :mod:`dead_cst._analyze` derives the
        :data:`EdgeFlags.DEAD_BRANCH` flag from it by checking
        containment against :attr:`dead_suites` (populated by the
        configured :class:`~dead_cst._branches.UnreachableRegionDetector`
        in :meth:`visit_Module`).
        """
        flag_map: dict[SymbolNode, SymbolNode] = {
            d: dataclasses.replace(d, flags=d.flags | NodeFlags.SHADOWED)
            for d in self.shadowed_decls
        }

        def remap(sym: SymbolNode) -> SymbolNode:
            return flag_map.get(sym, sym)

        nodes: list[SymbolNode] = []
        stack: list[SymbolTrie] = [self.trie]
        while stack:
            tnode = stack.pop()
            if tnode.module is not None:
                nodes.append(tnode.module)
            for decls in tnode.declarations.values():
                nodes.extend(decls)
            stack.extend(tnode.children.values())
        nodes.extend(remap(d) for d in self.shadowed_decls)

        edges = tuple((remap(src), remap(dst), pos) for src, dst, pos in self.internal_edges)
        imports = tuple((remap(src), dst, pos) for src, dst, pos in self.import_edges)

        return VisitorPayload(
            nodes=tuple(nodes),
            edges=edges,
            imports=imports,
            dead_suites=tuple(self.dead_suites),
        )
