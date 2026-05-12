"""Shared types and helpers for edge plugins.

Defines the :class:`EdgePlugin` protocol every plugin satisfies, the
:class:`PluginContext` plugins receive once per package, the
:class:`GraphOp` value objects plugins emit, and small utilities
(:func:`apply_ops`, :func:`synthetic_node`) used by both the analyzer
and the plugins.

The plugin pass runs once per package after the analyzer has resolved
that package's edges. Plugins see the symbol lookup that was valid at
that package's resolution time (the current package + its dependencies'
exports). The parsed-module cache on :class:`PluginContext` is
request-scope: a file is read and parsed on first
:meth:`PluginContext.parse` call and the result memoized for the rest
of the analysis. Warm cache hits in the per-file visitor pass
deliberately skip parsing entirely, so plugins that need a
``cst.Module`` will pay the parse cost the first time they ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Container,
    Iterable,
    Iterator,
    Protocol,
    runtime_checkable,
)

import libcst as cst
import networkx as nx
from libcst.metadata import CodePosition, CodeRange

from .._cacheable import Cacheable
from ..graph import NodeFlags, SymbolNode, SymbolTrie

if TYPE_CHECKING:
    from ..graph import VisitorPayload
    from ..resolvers._core import Package

SYNTHETIC_POSITION = CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0))

# Synthetic-node fqname prefixes, consumed by ``importers`` and the path
# resolver. Anything in :data:`SYNTHETIC_PATH_PREFIXES` is also a valid
# value for ``Import.path`` -- the analyzer surfaces non-first-party
# imports as ``[external dist] X`` / ``[external file] X`` /
# ``[unresolved] X`` so plugins can answer "which files import X?".
# Stdlib imports (``[stdlib] X``) are *not* surfaced as graph nodes;
# the prefix exists for the resolver only.
STDLIB_PREFIX = "[stdlib] "
EXTERNAL_DIST_PREFIX = "[external dist] "
EXTERNAL_FILE_PREFIX = "[external file] "
EXTERNAL_PREFIXES = (EXTERNAL_DIST_PREFIX, EXTERNAL_FILE_PREFIX)
UNRESOLVED_PREFIX = "[unresolved] "
# ``[unparseable] <module fqname>`` marks a file libcst could not parse.
# The analyser emits one synthetic with this prefix per failing file --
# tagged ``ENTRYPOINT`` so the file stays alive (we cannot prove its
# contents are dead) and edged at the module node so importers and
# ``why-alive`` queries can still locate it. Distinct from
# ``SYNTHETIC_PATH_PREFIXES`` because these nodes are never import
# targets -- they're internal markers, not classifications.
UNPARSEABLE_PREFIX = "[unparseable] "
SYNTHETIC_PATH_PREFIXES = (*EXTERNAL_PREFIXES, UNRESOLVED_PREFIX)


@dataclass(slots=True)
class PluginContext:
    """Per-package view of the analyzer state passed to every plugin.

    The plugin pass runs once for each package in :attr:`Analysis.packages`
    order (deps before dependents wherever the package graph is
    acyclic). Each invocation gets a fresh context whose
    ``symbol_lookup`` matches what was visible to that package's import
    resolution. Plugins should normally scope iteration to
    :attr:`package` -- :meth:`package_modules` is the easy way --
    because :attr:`graph` accumulates nodes across packages.

    :attr:`_modules` is a request-scope memo for :meth:`parse`: nothing
    is pre-populated, so the first plugin that asks for a given file's
    ``cst.Module`` pays the read + parse, and subsequent calls within
    the same analysis return the cached result.
    """

    graph: nx.DiGraph
    package_graph: nx.MultiDiGraph
    symbol_lookup: SymbolTrie
    package: Package
    project_root: Path
    _modules: dict[Path, cst.Module | None] = field(default_factory=dict, repr=False)
    # Lazy ``fqname -> SymbolNode`` index over synthetic nodes (built on
    # first ``importers`` call).  Plugins that add their own synthetic
    # nodes during the same pass won't see them through this index, which
    # is fine in practice -- ``importers`` is for prefiltering against
    # the analyzer's already-resolved dep markers.
    _synthetic_index: dict[str, SymbolNode] | None = field(default=None, init=False, repr=False)
    # Cached materialization of ``package_modules`` / ``package_nodes``.
    # Plugins may add nodes during their pass (entrypoint synthetics,
    # etc.) but those are never re-iterated by these helpers; we
    # snapshot once at first call to keep iteration cheap when ``graph``
    # accumulates nodes across packages.
    _package_modules_cache: list[tuple[Path, SymbolNode]] | None = field(
        default=None, init=False, repr=False
    )
    _package_nodes_cache: list[SymbolNode] | None = field(default=None, init=False, repr=False)

    def find_module(self, fqname: str) -> SymbolNode | None:
        node = self.symbol_lookup._get(fqname.split("."))
        return node.module if node else None

    def find_declarations(self, fqname: str) -> list[SymbolNode]:
        """Look up top-level declarations by dotted name.

        ``pkg.mod.func`` is split into module ``pkg.mod`` and decl ``func``.
        Returns every :class:`SymbolNode` bound to ``func`` at module
        exit -- normally one, but multiple when each branch of a
        conditional defines the same name. Empty list if nothing matches.
        """
        parts = fqname.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module_parts, decl_name = parts[:split], parts[split]
            node = self.symbol_lookup._get(module_parts)
            if node and node.module and decl_name in node.declarations:
                return list(node.declarations[decl_name])
        return []

    def module_surface(self, fqname: str) -> list[SymbolNode]:
        """Return the module + every decl + every transitive submodule decl.

        Models ``importlib.import_module(fqname)``: the module's whole
        top-level surface (decls, imports, side-effecting assignments)
        runs at import time, plus every submodule the package's own
        imports bring in transitively. Walks the symbol trie in
        O(declarations_in_subtree); cheaper than scanning the graph.
        Returns an empty list when ``fqname`` doesn't resolve to a
        first-party module.
        """
        trie_node = self.symbol_lookup._get(fqname.split("."))
        if trie_node is None or trie_node.module is None:
            return []
        out: list[SymbolNode] = []
        stack = [trie_node]
        while stack:
            node = stack.pop()
            if node.module is not None:
                out.append(node.module)
            for bucket in node.declarations.values():
                out.extend(bucket)
            stack.extend(node.children.values())
        return out

    def package_modules(self) -> Iterator[tuple[Path, SymbolNode]]:
        """Yield ``(path, module_node)`` for every module under :attr:`package`."""
        if self._package_modules_cache is None:
            self._package_modules_cache = [
                (node.path, node) for node in self.package_graph.nodes if node.type == "module"
            ]
        return iter(self._package_modules_cache)

    def package_nodes(self) -> Iterator[SymbolNode]:
        """Yield every graph node under :attr:`package`.

        ``graph`` accumulates nodes across packages; plugins that need
        to iterate "everything in this package" should use this instead
        of ``ctx.graph.nodes`` so they don't pay O(N) for nodes
        belonging to sibling packages.
        """
        if self._package_nodes_cache is None:
            self._package_nodes_cache = list(self.package_graph.nodes)
        return iter(self._package_nodes_cache)

    def importers(self, target: str) -> set[Path]:
        """Return paths under :attr:`package` whose imports reach ``target``.

        ``target`` is matched first as a first-party module fqname
        (e.g. ``pkg.mod``); if no first-party module matches, it is
        matched against the synthetic markers the analyzer adds for
        non-first-party imports -- ``[external dist] <target>``,
        ``[external file] <target>``, and ``[unresolved] <target>``
        (for imports the resolver couldn't pin to an installed dist,
        which still tells us "this file tried to import X"). The result
        is the natural prefilter for framework plugins ("only look at
        files that import fastapi") -- strictly more accurate than
        substring matching, and free because the import edges are
        already in the graph.

        Stdlib imports (``[stdlib] <target>``) are *not* surfaced as
        synthetic nodes by the resolver, so this method cannot prefilter
        on them; plugins that care about stdlib imports must walk the
        import nodes themselves.
        """
        target_node = self.find_module(target)
        if target_node is None:
            for prefix in SYNTHETIC_PATH_PREFIXES:
                node = self._synthetic(f"{prefix}{target}")
                if node is not None:
                    target_node = node
                    break
        if target_node is None:
            return set()
        package_path = self.package.path
        return {
            pred.path
            for pred in self.graph.predecessors(target_node)
            # Exclude same-file predecessors -- for a first-party module
            # node, every decl inside that module is a predecessor (via
            # the standard ``decl -> module`` edge), but we want
            # *importers* of the module, not its contents.
            if pred.path != target_node.path and pred.path.is_relative_to(package_path)
        }

    def parse(self, path: Path) -> cst.Module | None:
        """Return the parsed :class:`libcst.Module` for ``path``.

        First access reads + parses the file; the result is memoized on
        the context for the rest of the analysis so repeat calls within
        the same plugin pass are free. Failures (unreadable file, syntax
        error) are cached as ``None`` so a flaky file isn't re-attempted.
        Warm cache hits in the visitor pass skip parsing entirely, so
        plugins that need a ``cst.Module`` for a hit file pay the parse
        cost the first time they ask.
        """
        if path in self._modules:
            return self._modules[path]
        try:
            module = cst.parse_module(path.read_text())
        except (OSError, cst.ParserSyntaxError):
            module = None
        self._modules[path] = module
        return module

    def _synthetic(self, fqname: str) -> SymbolNode | None:
        if self._synthetic_index is None:
            self._synthetic_index = {n.fqname: n for n in self.graph.nodes if n.type == "synthetic"}
        return self._synthetic_index.get(fqname)


class UnresolvedDependencyError(RuntimeError):
    """A plugin needs ``package`` resolved to an installed distribution but
    only the ``[unresolved] <package>`` synthetic exists.

    This means at least one file under the analyzed package does
    ``import <package>`` / ``from <package> import ...``, but no resolver
    found the distribution on ``sys.path`` -- typically the user hasn't
    activated their venv (or hasn't run ``uv sync``). The plugin can't
    function without the resolver having pinned the package to a real
    site-packages location, so we surface the failure rather than
    silently producing wrong results.
    """


def require_resolved_dep(ctx: PluginContext, package: str) -> SymbolNode | None:
    """Return the resolved ``[external ...]`` synthetic for ``package``.

    Three outcomes:

    * Returns the ``[external dist] <package>`` / ``[external file] <package>``
      node if the analyzer's resolver pinned ``package`` to an installed
      distribution.
    * Returns ``None`` if no file in this package imports ``package`` (no
      synthetic was created at all). The plugin has nothing to do.
    * Raises :class:`UnresolvedDependencyError` if only the
      ``[unresolved] <package>`` synthetic exists -- the plugin's
      precondition (a resolved import of ``package``) is unmet, so we
      stop rather than guess.

    Plugins that wrap framework conventions (FastAPI, Flask, Click,
    Typer, ...) should use this in place of ``ctx.importers(package)``
    so that misconfigured environments fail loudly.
    """
    for prefix in EXTERNAL_PREFIXES:
        node = ctx._synthetic(f"{prefix}{package}")
        if node is not None:
            return node
    if ctx._synthetic(f"{UNRESOLVED_PREFIX}{package}") is None:
        return None
    raise UnresolvedDependencyError(
        f"'{package}' is imported in this package but the analyzer only "
        f"found '[unresolved] {package}'. Activate your project's "
        f"virtual environment (or run `uv sync --all-packages`) so "
        f"'{package}' is importable, then retry."
    )


@dataclass(frozen=True, slots=True)
class AddNode:
    """Add a node to the graph. When ``entrypoint=True``, mark the node so
    :func:`find_reachable` seeds its BFS from it. ``testcase=True`` tags
    the node as a test-only entrypoint -- it still seeds the default
    BFS, but ``Analysis.kept_alive_by_flags_only(NodeFlags.TESTCASE)``
    excludes those seeds to surface the "blast radius" of removing tests."""

    node: SymbolNode
    entrypoint: bool = False
    testcase: bool = False


@dataclass(frozen=True, slots=True)
class AddEdge:
    src: SymbolNode
    dst: SymbolNode


@dataclass(frozen=True, slots=True)
class RemoveEdge:
    src: SymbolNode
    dst: SymbolNode


GraphOp = AddNode | AddEdge | RemoveEdge


@dataclass(frozen=True, slots=True)
class ObserveContext:
    """Per-file context handed to :meth:`EdgePlugin.observe`.

    Plugins inspect the just-parsed CST and the visitor's per-file
    :class:`VisitorPayload` and return a *new* :class:`VisitorPayload`
    with their additional nodes and edges (or ``None`` if the file
    contributes nothing). The returned payload is concatenated with
    the visitor's, cached together, and applied to the graph.

    Plugins should keep observe outputs file-local. Cross-file work
    (looking through the assembled symbol graph, resolving factory
    chains, computing transitive subclass closures) belongs in
    :meth:`EdgePlugin.finalize`, which runs once per package after all
    files' payloads have been applied and the per-package import edges
    resolved -- and which never reads CSTs.
    """

    path: Path
    module: cst.Module
    payload: "VisitorPayload"
    package: Package
    project_root: Path


@runtime_checkable
class EdgePlugin(Cacheable, Protocol):
    """A plugin that contributes nodes/edges per file plus an optional
    per-package graph-finalize pass.

    Two phases:

    * :meth:`observe` runs inside the analyzer's per-file loop with
      the file's parsed :class:`libcst.Module` and just-built
      :class:`VisitorPayload`. It returns a new payload (or ``None``)
      whose ``nodes``/``edges`` extend the file's contribution. The
      result is cached alongside the visitor's payload, so warm runs
      skip both visiting and observing.

    * :meth:`finalize` runs once per package after :func:`resolve_edges`
      has stitched the cross-file import edges. It operates purely on
      the assembled graph -- no CST access -- and emits
      :class:`GraphOp` values. Plugins use this for cross-file work
      that needs the full graph (factory walks, transitive subclass
      closure, ``[project.scripts]`` lookups).

    Inherits the ``(name, version)`` contract from :class:`Cacheable`
    so the analyzer's per-file cache invalidates when a plugin's
    ``observe`` output changes (bump the epoch ``version``).
    """

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None": ...

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]: ...


def apply_ops(graph: nx.DiGraph, ops: Iterable[GraphOp]) -> None:
    for op in ops:
        match op:
            case AddNode(node, entrypoint, testcase):
                graph.add_node(node)
                if entrypoint:
                    graph.nodes[node]["entrypoint"] = True
                if testcase:
                    graph.nodes[node]["testcase"] = True
            case AddEdge(src, dst):
                graph.add_edge(src, dst)
            case RemoveEdge(src, dst):
                if graph.has_edge(src, dst):
                    graph.remove_edge(src, dst)


def synthetic_node(
    fqname: str,
    path: Path,
    *,
    flags: NodeFlags = NodeFlags.NONE,
    position: CodeRange = SYNTHETIC_POSITION,
) -> SymbolNode:
    """Create a placeholder node plugins can attach edges to.

    ``flags`` lets callers stamp ``NodeFlags.ENTRYPOINT`` directly on
    the node so that ``_apply_payload`` will seed reachability from it
    when the node is added to the graph. ``position`` defaults to the
    file-wide :data:`SYNTHETIC_POSITION`; pass a real :class:`CodeRange`
    when the synthetic stands in for a specific source location (e.g.
    a string literal in a registry list) so ``why-alive`` and the
    codemod report it correctly.
    """
    return SymbolNode(fqname=fqname, type="synthetic", path=path, position=position, flags=flags)


def module_node(payload: "VisitorPayload") -> SymbolNode | None:
    """Return the file's synthetic module node, or ``None`` for empty payloads."""
    return next((n for n in payload.nodes if n.type == "module"), None)


def dotted_parts(expr: cst.CSTNode | None) -> list[str] | None:
    """Walk an ``Attribute`` chain rooted in a ``Name`` and return its dotted parts.

    Returns the parts in source order (``a.b.c`` -> ``["a", "b", "c"]``)
    or ``None`` if the chain doesn't bottom out in a bare ``Name``.
    """
    parts: list[str] = []
    current = expr
    while isinstance(current, cst.Attribute):
        parts.append(current.attr.value)
        current = current.value
    if not isinstance(current, cst.Name):
        return None
    parts.append(current.value)
    parts.reverse()
    return parts


def dotted_name(expr: cst.CSTNode | None) -> str | None:
    """Like :func:`dotted_parts` but returns ``"a.b.c"`` (or ``None``)."""
    parts = dotted_parts(expr)
    return ".".join(parts) if parts is not None else None


def string_value(expr: cst.BaseExpression) -> str | None:
    """Return the value of a ``SimpleString`` / ``ConcatenatedString``, else ``None``.

    Bytes literals (which share the ``SimpleString`` node) and
    unparseable concatenations resolve to ``None``.
    """
    if not isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
        return None
    try:
        value = expr.evaluated_value
    except (SyntaxError, UnicodeDecodeError):
        return None
    return value if isinstance(value, str) else None


def payload_imports_module(
    payload: "VisitorPayload", module_name: str, *, include_star: bool = True
) -> bool:
    """True iff any import in ``payload`` targets ``module_name`` or a submodule."""
    prefix = module_name + "."
    for _src, imp, _pos in payload.imports:
        if not include_star and imp.star:
            continue
        if imp.module == module_name or imp.module.startswith(prefix):
            return True
    return False


def decls_by_simple_name(nodes: Iterable[SymbolNode]) -> dict[str, list[SymbolNode]]:
    """Index ``nodes`` by the rightmost dotted segment of their fqname.

    Used by every framework plugin (Click, FastAPI, Flask, Typer) to
    map a top-level name spotted in source (e.g. a Click group's
    handler function) back to its :class:`SymbolNode` so an
    ``instance -> handler`` edge can be emitted. Skips ``module``
    and ``synthetic`` types because their fqnames don't denote
    file-local declarations.
    """
    out: dict[str, list[SymbolNode]] = {}
    for n in nodes:
        if n.type in ("class", "function", "variable", "import"):
            out.setdefault(simple_name(n.fqname), []).append(n)
    return out


def mark_entrypoints(
    seed_fqname: str, path: Path, targets: Iterable[SymbolNode]
) -> Iterator[GraphOp]:
    """Emit a synthetic entrypoint node and edges from it to each target.

    Used by plugins that mark a set of decls alive without a natural caller --
    pytest discovery, unittest discovery, and similar. This helper is for
    :meth:`EdgePlugin.finalize` (which emits :class:`GraphOp` values);
    :func:`entrypoint_payload` is the per-file :meth:`EdgePlugin.observe`
    equivalent.
    """
    targets = list(targets)
    if not targets:
        return
    synth = synthetic_node(fqname=seed_fqname, path=path)
    yield AddNode(synth, entrypoint=True)
    for target in targets:
        yield AddEdge(synth, target)


def entrypoint_payload(
    seed_fqname: str,
    path: Path,
    targets: Iterable[SymbolNode],
) -> "VisitorPayload | None":
    """Per-file equivalent of :func:`mark_entrypoints`.

    Returns a :class:`VisitorPayload` with one ``ENTRYPOINT``-flagged
    synthetic node and an edge from it to every target, or ``None``
    when ``targets`` is empty (so callers can ``return entrypoint_payload(...)``
    directly from :meth:`EdgePlugin.observe`).
    """
    targets = list(targets)
    if not targets:
        return None
    synth = synthetic_node(seed_fqname, path, flags=NodeFlags.ENTRYPOINT)
    return make_payload(
        nodes=[synth],
        edges=[(synth, t, SYNTHETIC_POSITION) for t in targets],
    )


def make_payload(
    *,
    nodes: Iterable[SymbolNode] = (),
    edges: Iterable[tuple[SymbolNode, SymbolNode, CodeRange]] = (),
) -> "VisitorPayload":
    """Build a plugin-style :class:`VisitorPayload`.

    Plugins only contribute ``nodes`` and ``edges``; ``imports`` and
    ``dead_suites`` stay empty so they don't disturb the visitor's
    cross-file resolution or the dead-branch report.
    """
    from ..graph import VisitorPayload

    return VisitorPayload(
        nodes=tuple(nodes),
        edges=tuple(edges),
        imports=(),
        dead_suites=(),
    )


def simple_name(fqname: str) -> str:
    """Return the rightmost dotted segment of ``fqname`` (``pkg.mod.f`` -> ``f``)."""
    return fqname.rpartition(".")[2]


def is_name(node: cst.CSTNode | None, value: str) -> bool:
    """Return ``True`` if ``node`` is a bare ``Name`` with the given value."""
    return isinstance(node, cst.Name) and node.value == value


def is_from_module(node: cst.ImportFrom, module_name: str) -> bool:
    """Return ``True`` if ``node`` is ``from <module_name> import ...`` (non-relative).

    ``module_name`` may be a dotted path (``"discord.ext.commands"``);
    the comparison resolves the import's module reference via
    :func:`dotted_name` so both ``from flask`` and
    ``from discord.ext.commands`` round-trip correctly.
    """
    return not node.relative and dotted_name(node.module) == module_name


def _asname_value(alias: cst.ImportAlias) -> str | None:
    """Return ``alias.asname.name.value`` when it's a bare ``Name``, else ``None``.

    ``AsName.name`` is typed ``Name | Tuple | List`` to share with ``with``-statement
    unpacking, but in import contexts only ``Name`` is legal.
    """
    if alias.asname is None:
        return None
    name = alias.asname.name
    return name.value if isinstance(name, cst.Name) else None


def single_target_assignment(
    stmt: cst.BaseSmallStatement,
) -> tuple[str | None, cst.BaseExpression | None]:
    """Extract ``(name, rhs)`` for ``X = ...`` / ``X: T = ...``; else ``(None, None)``."""
    if isinstance(stmt, cst.Assign):
        if len(stmt.targets) != 1:
            return None, None
        target = stmt.targets[0].target
        if isinstance(target, cst.Name):
            return target.value, stmt.value
    elif isinstance(stmt, cst.AnnAssign):
        if isinstance(stmt.target, cst.Name) and stmt.value is not None:
            return stmt.target.value, stmt.value
    return None, None


def decorator_owner(expr: cst.BaseExpression, valid_attrs: Container[str]) -> str | None:
    """For ``@X.<attr>(...)`` / ``@X.<attr>`` return ``"X"`` when ``attr`` is in
    ``valid_attrs`` and ``X`` is a bare ``Name``. Returns ``None`` otherwise."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if not isinstance(expr, cst.Attribute):
        return None
    if expr.attr.value not in valid_attrs:
        return None
    if not isinstance(expr.value, cst.Name):
        return None
    return expr.value.value


def find_handlers(
    module: cst.Module,
    instance_vars: Container[str] | None,
    valid_attrs: Container[str],
) -> dict[str, list[str]]:
    """Return ``{owner_var: [handler_func_name, ...]}`` for top-level functions
    decorated with ``@<owner_var>.<attr>(...)`` where ``attr`` is in ``valid_attrs``.

    ``instance_vars=None`` accepts any owner -- useful when the caller will
    classify owners by reachability / kind in a second pass.
    """
    handlers: dict[str, list[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            owner = decorator_owner(dec.decorator, valid_attrs)
            if owner is None:
                continue
            if instance_vars is not None and owner not in instance_vars:
                continue
            handlers.setdefault(owner, []).append(stmt.name.value)
            break
    return handlers


def matched_attr_call(
    expr: cst.BaseExpression,
    imports: dict[str, str],
    valid_targets: Container[str],
    *,
    unwrap_call: bool = True,
) -> str | None:
    """Return the matched target name for ``expr`` against ``imports`` / ``valid_targets``.

    Recognizes both forms produced by ``collect_module_imports``:

    * bare ``Name`` (e.g. ``Flask``) where ``imports[name]`` is a real
      target in ``valid_targets`` (from ``from <mod> import Flask``);
    * module-prefixed ``Attribute`` (e.g. ``flask.Flask``) where
      ``imports[<bare module>] == "<module>"`` and the rightmost attr
      is in ``valid_targets``.

    ``unwrap_call=True`` (default) handles ``Foo(...)`` by inspecting
    ``.func``; pass ``False`` for callers that have already unwrapped.
    Returns ``None`` if no match.
    """
    if unwrap_call and isinstance(expr, cst.Call):
        expr = expr.func
    if isinstance(expr, cst.Name):
        target = imports.get(expr.value)
        if target is not None and target in valid_targets:
            return target
    elif isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
        if imports.get(expr.value.value) == "<module>":
            attr = expr.attr.value
            if attr in valid_targets:
                return attr
    return None


def find_call_assignments(
    module: cst.Module, imports: dict[str, str], valid_targets: Container[str]
) -> dict[str, str]:
    """Return ``{var_name: target_name}`` for top-level ``X = <call>`` where
    ``<call>`` matches one of ``valid_targets`` via :func:`matched_attr_call`."""
    instances: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target_name, value = single_target_assignment(small)
            if target_name is None or not isinstance(value, cst.Call):
                continue
            kind = matched_attr_call(value.func, imports, valid_targets, unwrap_call=False)
            if kind is not None:
                instances[target_name] = kind
    return instances


def collect_module_imports(
    module: cst.Module, module_name: str, allowed_targets: Container[str]
) -> dict[str, str]:
    """Return ``{local_name: target}`` for names imported from ``module_name``.

    Each ``from <module_name> import X [as Y]`` whose ``X`` is in
    ``allowed_targets`` adds ``{Y or X: X}``. Each ``import <module_name> [as Y]``
    adds ``{Y or module_name: '<module>'}`` so callers can recognize the bare
    module-prefixed form (``<module_name>.X(...)``).
    """
    bindings: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                if not is_from_module(small, module_name):
                    continue
                if isinstance(small.names, cst.ImportStar):
                    continue
                for alias in small.names:
                    target = alias.name.value if isinstance(alias.name, cst.Name) else None
                    if target is None or target not in allowed_targets:
                        continue
                    local = _asname_value(alias) or target
                    bindings[local] = target
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    if not is_name(alias.name, module_name):
                        continue
                    local = _asname_value(alias) or module_name
                    bindings[local] = "<module>"
    return bindings


def walk_to_instance_kind(
    graph: nx.DiGraph,
    start: SymbolNode,
    terminal: SymbolNode,
    module_name: str,
    instance_kinds: Container[str],
    *,
    factory_marker_prefix: str | None = None,
) -> str | None:
    """Walk forward from ``start`` until hitting an ``import`` node bound to
    ``module_name`` and one of ``instance_kinds``; return the matched decl name.

    Used by framework plugins to classify a variable as a ``Flask`` /
    ``Blueprint`` / ``FastAPI`` / ``APIRouter`` / etc. instance via the
    analyzer's existing reference edges. The factory case
    (``X = create_app()``) drops out because the factory's body
    references the framework class and that edge is already in the
    graph. Returns ``None`` when the chain doesn't reach a discriminating
    import -- callers treat that as "not an instance" rather than guessing.

    The ``terminal`` cutoff (the framework's external-dist synthetic) is
    skipped during traversal so the walk doesn't fan out across
    *every* file that imports the framework.

    ``factory_marker_prefix`` opts in to recognizing the
    ``<prefix><kind>:<owner.fqname>`` synthetic markers
    :func:`find_factory_decls` emits. The factory-marker check kicks in
    when the import-node discriminator is unreliable -- typically the
    ``import <module>; <module>.<Cls>()`` (attribute-form) cases that
    collapse to a bare ``[external dist] <module>`` edge after
    :func:`resolve_edges` drops the ``decl`` half.

    Plugins requiring this should also call :func:`require_resolved_dep`
    so the visitor's ``Import.module`` is the canonical ``"flask"`` /
    ``"fastapi"`` / etc. (the unresolved fallback uses the dotted full
    name and would never match here).
    """
    seen: set[SymbolNode] = set()
    stack: list[SymbolNode] = [start]
    while stack:
        node = stack.pop()
        if node in seen or node is terminal:
            continue
        seen.add(node)
        if node.type == "import" and node.imports is not None:
            decl = node.imports.decl
            if decl is not None and node.imports.module == module_name and decl in instance_kinds:
                return decl
        if (
            factory_marker_prefix is not None
            and node.type == "synthetic"
            and node.fqname.startswith(factory_marker_prefix)
        ):
            kind = node.fqname[len(factory_marker_prefix) :].split(":", 1)[0]
            if kind in instance_kinds:
                return kind
        stack.extend(graph.successors(node))
    return None


class _ConstructorFinder(cst.CSTVisitor):
    """Collect construction kinds for ``valid_targets`` inside a decl body."""

    def __init__(self, imports: dict[str, str], valid_targets: Container[str]) -> None:
        super().__init__()
        self._imports = imports
        self._valid_targets = valid_targets
        self.kinds: set[str] = set()

    def visit_Call(self, node: cst.Call) -> bool | None:
        kind = matched_attr_call(node.func, self._imports, self._valid_targets, unwrap_call=False)
        if kind is not None:
            self.kinds.add(kind)
        return None


def find_factory_decls(
    module: cst.Module, imports: dict[str, str], valid_targets: Container[str]
) -> dict[str, set[str]]:
    """Return ``{decl_name: {kind, ...}}`` for top-level decls whose body
    constructs one of ``valid_targets``.

    Scans every top-level ``def`` / ``class`` body for ``<Cls>(...)`` /
    ``<mod>.<Cls>(...)`` call shapes (named or module-prefixed, both
    forms produced by :func:`collect_module_imports`). Skips files whose
    ``imports`` map is empty -- :func:`matched_attr_call` would reject
    every candidate anyway, so the AST walk would be wasted.

    Used by framework plugins (FastAPI, Flask, ...) to anchor a
    factory-marker synthetic on the constructing decl so
    :func:`walk_to_instance_kind` (with ``factory_marker_prefix=``) can
    classify cross-file consumers even when the framework class is
    reached via the attribute-form ``<mod>.<Cls>()`` -- where
    :func:`resolve_edges` drops the ``decl`` half of the external-edge
    classification and the import-node check alone misses the case.
    """
    if not imports:
        return {}
    out: dict[str, set[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
            continue
        finder = _ConstructorFinder(imports, valid_targets)
        stmt.body.visit(finder)
        if finder.kinds:
            out[stmt.name.value] = finder.kinds
    return out
