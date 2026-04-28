"""Shared types and helpers for edge plugins.

Defines the :class:`EdgePlugin` protocol every plugin satisfies, the
:class:`PluginContext` plugins receive once per base, the :class:`GraphOp`
value objects plugins emit, and small utilities (:func:`apply_ops`,
:func:`synthetic_node`) used by both the analyzer and the plugins.

The plugin pass runs once per base after the analyzer has resolved that
base's edges. Plugins see the symbol lookup that was valid at that base's
resolution time (the current base + its dependencies' exports), and the
parsed module cache is primed with the modules the analyzer just walked,
so plugins never re-read or re-parse source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Container, Iterable, Iterator, Protocol, runtime_checkable

import libcst as cst
import networkx as nx
from libcst.metadata import CodePosition, CodeRange

from .._symbols import SymbolNode, SymbolTrie

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
SYNTHETIC_PATH_PREFIXES = (*EXTERNAL_PREFIXES, UNRESOLVED_PREFIX)


@dataclass
class PluginContext:
    """Per-base view of the analyzer state passed to every plugin.

    The plugin pass runs once for each base in topological order. Each
    invocation gets a fresh context whose ``symbol_lookup`` matches what
    was visible to that base's import resolution. Plugins should normally
    scope iteration to :attr:`base` -- :meth:`base_modules` is the easy
    way -- because :attr:`graph` accumulates nodes across bases.

    The parsed-module cache is pre-populated with the modules the analyzer
    just walked, so :meth:`parse` returns immediately for any file under
    :attr:`base` without re-reading or re-parsing.
    """

    graph: nx.DiGraph
    symbol_lookup: SymbolTrie
    base: Path
    project_root: Path
    _modules: dict[Path, cst.Module | None] = field(default_factory=dict, repr=False)
    # Lazy ``fqname -> SymbolNode`` index over synthetic nodes (built on
    # first ``importers`` call).  Plugins that add their own synthetic
    # nodes during the same pass won't see them through this index, which
    # is fine in practice -- ``importers`` is for prefiltering against
    # the analyzer's already-resolved dep markers.
    _synthetic_index: dict[str, SymbolNode] | None = field(default=None, init=False, repr=False)
    # Cached materialization of ``base_modules`` / ``base_nodes``. Plugins
    # may add nodes during their pass (entrypoint synthetics, etc.) but
    # those are never re-iterated by these helpers; we snapshot once at
    # first call to keep iteration cheap when ``graph`` accumulates nodes
    # across bases.
    _base_modules_cache: list[tuple[Path, SymbolNode]] | None = field(
        default=None, init=False, repr=False
    )
    _base_nodes_cache: list[SymbolNode] | None = field(default=None, init=False, repr=False)

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

    def base_modules(self) -> Iterator[tuple[Path, SymbolNode]]:
        """Yield ``(path, module_node)`` for every module under :attr:`base`."""
        if self._base_modules_cache is None:
            self._base_modules_cache = [
                (node.path, node)
                for node in self.graph.nodes
                if node.type == "module" and node.path.is_relative_to(self.base)
            ]
        return iter(self._base_modules_cache)

    def base_nodes(self) -> Iterator[SymbolNode]:
        """Yield every graph node whose path is under :attr:`base`.

        ``graph`` accumulates nodes across bases; plugins that need to
        iterate "everything in this base" should use this instead of
        ``ctx.graph.nodes`` so they don't pay O(N) for nodes belonging
        to sibling bases.
        """
        if self._base_nodes_cache is None:
            self._base_nodes_cache = [
                node for node in self.graph.nodes if node.path.is_relative_to(self.base)
            ]
        return iter(self._base_nodes_cache)

    def importers(self, target: str) -> set[Path]:
        """Return paths under :attr:`base` whose imports reach ``target``.

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
        return {
            pred.path
            for pred in self.graph.predecessors(target_node)
            # Exclude same-file predecessors -- for a first-party module
            # node, every decl inside that module is a predecessor (via
            # the standard ``decl -> module`` edge), but we want
            # *importers* of the module, not its contents.
            if pred.path != target_node.path and pred.path.is_relative_to(self.base)
        }

    def parse(self, path: Path) -> cst.Module | None:
        """Return the parsed :class:`libcst.Module` for ``path``.

        The analyzer primes the cache with the modules it parsed during
        the visitor pass, so for any file under :attr:`base` this returns
        immediately. Files outside the base are read and parsed on first
        access; failures are cached so a flaky file isn't re-attempted.
        """
        if path in self._modules:
            return self._modules[path]
        try:
            module = cst.parse_module(path.read_text())
        except (OSError, cst.ParserSyntaxError):
            module = None
        self._modules[path] = module
        return module

    def prime_module(self, path: Path, module: cst.Module) -> None:
        """Record an already-parsed module so :meth:`parse` skips re-parsing."""
        self._modules[path] = module

    def _synthetic(self, fqname: str) -> SymbolNode | None:
        if self._synthetic_index is None:
            self._synthetic_index = {n.fqname: n for n in self.graph.nodes if n.type == "synthetic"}
        return self._synthetic_index.get(fqname)


class UnresolvedDependencyError(RuntimeError):
    """A plugin needs ``package`` resolved to an installed distribution but
    only the ``[unresolved] <package>`` synthetic exists.

    This means at least one file under the analyzed base does
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
    * Returns ``None`` if no file in this base imports ``package`` (no
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
        f"'{package}' is imported in this base but the analyzer only "
        f"found '[unresolved] {package}'. Activate your project's "
        f"virtual environment (or run `uv sync --all-packages`) so "
        f"'{package}' is importable, then retry."
    )


@dataclass(frozen=True)
class AddNode:
    """Add a node to the graph. When ``entrypoint=True``, mark the node so
    :func:`find_reachable` seeds its BFS from it."""

    node: SymbolNode
    entrypoint: bool = False


@dataclass(frozen=True)
class AddEdge:
    src: SymbolNode
    dst: SymbolNode


@dataclass(frozen=True)
class RemoveEdge:
    src: SymbolNode
    dst: SymbolNode


GraphOp = AddNode | AddEdge | RemoveEdge


@runtime_checkable
class EdgePlugin(Protocol):
    """A plugin that contributes graph ops once per base.

    The analyzer calls :meth:`contribute` for every base in topological
    order, after that base's import edges have been resolved. Plugins
    use the per-base :class:`PluginContext` to look up symbols, parse
    modules, and find importers of a given dep; they emit
    :class:`GraphOp` values rather than mutating the graph directly.
    """

    name: str

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]: ...


def apply_ops(graph: nx.DiGraph, ops: Iterable[GraphOp]) -> None:
    for op in ops:
        match op:
            case AddNode(node, entrypoint):
                graph.add_node(node)
                if entrypoint:
                    graph.nodes[node]["entrypoint"] = True
            case AddEdge(src, dst):
                graph.add_edge(src, dst)
            case RemoveEdge(src, dst):
                if graph.has_edge(src, dst):
                    graph.remove_edge(src, dst)


def synthetic_node(fqname: str, path: Path) -> SymbolNode:
    """Create a placeholder node plugins can attach edges to."""
    return SymbolNode(fqname=fqname, type="synthetic", path=path, position=SYNTHETIC_POSITION)


def mark_entrypoints(
    seed_fqname: str, path: Path, targets: Iterable[SymbolNode]
) -> Iterator[GraphOp]:
    """Emit a synthetic entrypoint node and edges from it to each target.

    Used by plugins that mark a set of decls alive without a natural caller --
    pytest discovery, unittest discovery, and similar.
    """
    targets = list(targets)
    if not targets:
        return
    synth = synthetic_node(fqname=seed_fqname, path=path)
    yield AddNode(synth, entrypoint=True)
    for target in targets:
        yield AddEdge(synth, target)


def simple_name(fqname: str) -> str:
    """Return the rightmost dotted segment of ``fqname`` (``pkg.mod.f`` -> ``f``)."""
    return fqname.rpartition(".")[2]


def is_name(node: cst.CSTNode | None, value: str) -> bool:
    """Return ``True`` if ``node`` is a bare ``Name`` with the given value."""
    return isinstance(node, cst.Name) and node.value == value


def is_from_module(node: cst.ImportFrom, module_name: str) -> bool:
    """Return ``True`` if ``node`` is ``from <module_name> import ...`` (non-relative)."""
    return not node.relative and is_name(node.module, module_name)


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
        stack.extend(graph.successors(node))
    return None
