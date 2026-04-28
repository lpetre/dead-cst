"""Plugin: keep subclasses of ``__init_subclass__``-defining classes alive.

A class that defines ``__init_subclass__`` runs custom code every time it
is subclassed -- typically a registry pattern (``cls.registry.append(cls)``,
``DISPATCH[cls.kind] = cls``, ...). Static analysis can't see that
registration, so subclasses look unused even though merely defining them
has a side effect that the framework relies on at runtime.

Strategy: for each ``__init_subclass__`` parent ``P`` create a synthetic
*marker* node ``S_P`` with edges ``P -> S_P`` and ``S_P -> C`` for every
(transitive) subclass ``C``. The marker makes "is this class a subclass
of some registry root?" an O(1) graph predecessor lookup, so when we
encounter a new class, we only need to peek at its direct bases -- if a
base is already wired under any marker, the new class inherits the same
markers without any further ancestor walk. The marker also surfaces in
``why-alive`` output as a labeled breadcrumb
(``<__init_subclass__>:pkg.base.Plugin``), making registry-driven
reachability legible.

Parents are not auto-marked as entrypoints; reachability is expected to
flow into ``P`` through whatever already keeps it alive (an explicit
``-e``, an import in a live module, etc.). Once it does, every subclass
registered through ``__init_subclass__`` becomes reachable too.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst

from .._symbols import SymbolNode
from ._core import AddEdge, AddNode, GraphOp, PluginContext, simple_name, synthetic_node

_INIT_SUBCLASS = "__init_subclass__"
INIT_SUBCLASS_PREFIX = "<__init_subclass__>:"


@dataclass
class InitSubclassPlugin:
    """Wire subclasses of ``__init_subclass__``-defining classes through a marker.

    For each top-level class in the project the plugin:

    1. Parses the class body and records whether it defines
       ``__init_subclass__`` (any ``def __init_subclass__`` in the class
       body, with or without ``@classmethod``).
    2. Resolves every base expression to a top-level class node, walking
       through local imports so cross-module inheritance is captured.
    3. Creates one synthetic marker node ``<__init_subclass__>:<P.fqname>``
       per registry parent ``P``, links ``P -> marker``, and links
       ``marker -> C`` for every direct or transitive first-party
       subclass.

    Membership is computed by inheriting markers from a class's direct
    bases: ``C(B)`` joins every marker that ``B`` is already a member of.
    No traversal up the full MRO is needed -- the marker carried on each
    class is the transitive closure. The pass runs to a small fixed
    point so that intra-base sibling chains (``Mid(Root)`` defined
    before ``Leaf(Mid)``) converge regardless of source order.

    Parents are not auto-marked as entrypoints. The expected wiring is
    that something else keeps the parent alive (an entrypoint, an import,
    a ``[project.scripts]`` target). When that happens, every subclass
    -- including ones whose only static use is being defined -- stays
    reachable, which matches the runtime semantics of
    ``__init_subclass__``.

    Limitations: only ``__init_subclass__`` defined in first-party code
    is detected. A subclass of an external base like ``pydantic.BaseModel``
    won't be wired through this plugin (the parent's class body isn't
    visible). Bases written as calls (``class X(make_base()): ...``) are
    skipped; ``Subscript`` bases such as ``Generic[T]`` are unwrapped to
    their value.
    """

    name: str = "init_subclass"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Walk every class node in the graph -- by topological order the
        # graph already contains classes from this base and every dep
        # base, so cross-base inheritance resolves correctly. ``parse``
        # is cached, so re-parsing dep modules in later passes is cheap.
        roots: dict[SymbolNode, SymbolNode] = {}  # parent class -> marker
        bases_of: dict[SymbolNode, list[SymbolNode]] = {}

        for class_node in [n for n in ctx.graph.nodes if n.type == "class"]:
            module = ctx.parse(class_node.path)
            if module is None:
                continue
            class_def = _find_class_def(module, class_node)
            if class_def is None:
                continue
            if _has_init_subclass(class_def):
                roots[class_node] = synthetic_node(
                    fqname=f"{INIT_SUBCLASS_PREFIX}{class_node.fqname}",
                    path=class_node.path,
                )
            module_fqname = class_node.fqname.rsplit(".", 1)[0]
            resolved = _resolve_bases(class_def, ctx, module_fqname)
            if resolved:
                bases_of[class_node] = resolved

        if not roots:
            return

        # ``markers_for[X]`` = the set of markers that already claim ``X``
        # as a (direct or transitive) subclass. Seeded from this pass'
        # registry roots and from any membership previous passes already
        # wired into the graph.
        markers_for: dict[SymbolNode, set[SymbolNode]] = defaultdict(set)
        for parent, marker in roots.items():
            # The parent is its own member so direct subclasses inherit
            # the marker by base-lookup alone; the parent->parent self
            # edge is filtered out at emit time.
            markers_for[parent].add(marker)
        for marker in roots.values():
            if marker not in ctx.graph:
                continue
            for succ in ctx.graph.successors(marker):
                if succ.type == "class":
                    markers_for[succ].add(marker)

        # Fixed point: a class joins every marker its bases are in.
        # Converges in <= max-inheritance-depth iterations.
        changed = True
        while changed:
            changed = False
            for sub, bases in bases_of.items():
                gained: set[SymbolNode] = set()
                for base in bases:
                    gained |= markers_for.get(base, set())
                gained -= markers_for[sub]
                if gained:
                    markers_for[sub] |= gained
                    changed = True

        # Emit the marker node + parent->marker edge once per root.
        # ``AddNode``/``AddEdge`` are idempotent across passes, so re-
        # emitting markers we discovered in an earlier dep base is safe.
        for parent, marker in roots.items():
            yield AddNode(marker)
            yield AddEdge(parent, marker)

        # Emit marker->subclass edges, deduplicating against what's
        # already in the graph (from previous passes) and scoping to
        # subclasses under the current base.
        existing_targets = {
            marker: (set(ctx.graph.successors(marker)) if marker in ctx.graph else set())
            for marker in roots.values()
        }
        parent_of = {marker: parent for parent, marker in roots.items()}
        for sub, markers in markers_for.items():
            if not sub.path.is_relative_to(ctx.base):
                continue
            for marker in markers:
                if parent_of.get(marker) is sub:
                    continue
                if sub in existing_targets.get(marker, set()):
                    continue
                yield AddEdge(marker, sub)


def _find_class_def(module: cst.Module, class_node: SymbolNode) -> cst.ClassDef | None:
    """Return the top-level ``cst.ClassDef`` matching ``class_node`` by name.

    The visitor only emits top-level class nodes, so a body-level scan
    is sufficient. When two top-level class defs share a name (e.g. each
    branch of an ``if/else`` defines the same name), the first match
    wins; both share the same MRO for ``__init_subclass__`` purposes.
    """
    name = simple_name(class_node.fqname)
    for stmt in module.body:
        if isinstance(stmt, cst.ClassDef) and stmt.name.value == name:
            return stmt
    return None


def _has_init_subclass(class_def: cst.ClassDef) -> bool:
    body = class_def.body
    if not isinstance(body, cst.IndentedBlock):
        return False
    for stmt in body.body:
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == _INIT_SUBCLASS:
            return True
    return False


def _resolve_bases(
    class_def: cst.ClassDef,
    ctx: PluginContext,
    module_fqname: str,
) -> list[SymbolNode]:
    resolved: list[SymbolNode] = []
    for arg in class_def.bases:
        # Keyword args are ``metaclass=...`` / PEP 487 ``__init_subclass__``
        # kwargs, not bases.
        if arg.keyword is not None:
            continue
        target = _resolve_class_expr(arg.value, ctx, module_fqname)
        if target is not None:
            resolved.append(target)
    return resolved


def _resolve_class_expr(
    expr: cst.BaseExpression,
    ctx: PluginContext,
    module_fqname: str,
) -> SymbolNode | None:
    """Resolve a base expression to the top-level class node it denotes.

    Handles bare names (``A``), dotted attributes (``pkg.mod.A``,
    ``base.A`` after ``from pkg import base``), and unwraps ``Subscript``
    bases (``Generic[T]`` -> ``Generic``). Returns ``None`` for calls
    and any other shape we don't model.
    """
    if isinstance(expr, cst.Subscript):
        return _resolve_class_expr(expr.value, ctx, module_fqname)
    if isinstance(expr, cst.Name):
        return _follow_to_class(ctx.find_declarations(f"{module_fqname}.{expr.value}"), ctx)
    if isinstance(expr, cst.Attribute):
        target_fqname = _attr_to_fqname(expr, ctx, module_fqname)
        if target_fqname is None:
            return None
        return _follow_to_class(ctx.find_declarations(target_fqname), ctx)
    return None


def _attr_to_fqname(
    expr: cst.Attribute,
    ctx: PluginContext,
    module_fqname: str,
) -> str | None:
    """Translate a dotted attribute base into an absolute fqname.

    Walks local import bindings so ``base.Plugin`` (after
    ``from pkg import base``) resolves to ``pkg.base.Plugin``, and
    ``pkg.base.Plugin`` (after ``import pkg.base``) resolves the same
    way. Falls back to treating ``parts`` as already-absolute when no
    local binding is found.
    """
    parts = _dotted_parts(expr)
    if parts is None:
        return None
    head, rest = parts[0], parts[1:]
    head_decls = ctx.find_declarations(f"{module_fqname}.{head}")
    for decl in head_decls:
        if decl.type != "import" or decl.imports is None:
            continue
        if not isinstance(decl.imports.path, Path):
            continue
        base_fqname = decl.imports.module
        if decl.imports.decl:
            base_fqname = f"{base_fqname}.{decl.imports.decl}"
        if rest:
            return f"{base_fqname}.{'.'.join(rest)}"
        return base_fqname
    return ".".join(parts)


def _dotted_parts(expr: cst.BaseExpression) -> list[str] | None:
    """``a.b.c`` -> ``["a", "b", "c"]``; non-dotted shapes return ``None``."""
    parts: list[str] = []
    current: cst.BaseExpression = expr
    while isinstance(current, cst.Attribute):
        parts.append(current.attr.value)
        current = current.value
    if not isinstance(current, cst.Name):
        return None
    parts.append(current.value)
    parts.reverse()
    return parts


def _follow_to_class(decls: list[SymbolNode], ctx: PluginContext) -> SymbolNode | None:
    """Pick the class behind ``decls``, following imports if needed.

    ``find_declarations`` returns the decls live at module exit -- usually
    one. A ``class`` decl is the answer directly. An ``import`` decl
    points at the source module's class via the resolved import edges
    already in the graph; we follow successors to the first class node.
    """
    for decl in decls:
        if decl.type == "class":
            return decl
    for decl in decls:
        if decl.type != "import":
            continue
        for succ in ctx.graph.successors(decl):
            if succ.type == "class":
                return succ
    return None
