"""Plugin: keep subclasses of ``__init_subclass__``-defining classes alive.

A class that defines ``__init_subclass__`` runs custom code every time it
is subclassed -- typically a registry pattern (``cls.registry.append(cls)``,
``DISPATCH[cls.kind] = cls``, ...). Static analysis can't see that
registration, so subclasses look unused even though merely defining them
has a side effect that the framework relies on at runtime.

Two phases:

* :meth:`observe` (per-file) inspects every top-level class def. For
  each class ``C`` that defines ``__init_subclass__``, it emits a
  ``<__init_subclass__>:C.fqname`` marker plus a ``C -> marker`` edge
  (so the marker stays alive whenever ``C`` is). For each base
  expression on ``C`` whose target can be resolved against this file's
  imports + locals, it also emits a ``<subclass-of:base_fqname>``
  marker plus an edge from that marker to ``C``.

* :meth:`finalize` (per-base) walks the assembled graph: for every
  ``<__init_subclass__>:`` marker ``M`` on parent ``P``, it BFSes the
  ``<subclass-of:>`` graph keyed by ``P.fqname`` to compute the
  transitive subclass closure and emits ``M -> sub`` edges for each
  hit. No CSTs are read in this phase.

The marker also surfaces in ``why-alive`` output as a labeled
breadcrumb (``<__init_subclass__>:pkg.base.Plugin``), making
registry-driven reachability legible.

Parents are not auto-marked as entrypoints; reachability is expected
to flow into ``P`` through whatever already keeps it alive (an explicit
``-e``, an import in a live module, etc.).
"""

from __future__ import annotations

from libcst.metadata import CodeRange
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst

from .._symbols import Import, SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    GraphOp,
    ObserveContext,
    PluginContext,
    make_payload,
    simple_name,
    synthetic_node,
)

if TYPE_CHECKING:
    from .._visitor import VisitorPayload

_INIT_SUBCLASS = "__init_subclass__"
INIT_SUBCLASS_PREFIX = "<__init_subclass__>:"
SUBCLASS_OF_PREFIX = "<subclass-of>:"


@dataclass
class InitSubclassPlugin:
    """Wire subclasses of ``__init_subclass__``-defining classes through a marker.

    The plugin's observe step walks every top-level class def in the
    file. For each class ``C``:

    * If the body defines ``def __init_subclass__``, emit
      ``<__init_subclass__>:C.fqname`` marker + edge ``C -> marker``
      so the marker is alive iff ``C`` is.
    * For each base expression resolvable to fqname ``F`` (using local
      imports + same-file decls), emit a ``<subclass-of>:F`` marker +
      edge ``<subclass-of>:F -> C``. The marker is keyed by the resolved
      base fqname so finalize can BFS the subclass relation cheaply.

    Finalize finds every ``<__init_subclass__>:P.fqname`` marker, then
    BFSes ``<subclass-of>:P.fqname -> sub`` chains across the graph,
    emitting ``marker -> sub`` for every transitive subclass.

    Limitations: only ``__init_subclass__`` defined in first-party code
    is detected (the parent's class body must be visible). Bases written
    as calls (``class X(make_base()): ...``) are skipped; ``Subscript``
    bases such as ``Generic[T]`` are unwrapped to their value. Base
    resolution is local to each file's imports -- re-exports through
    intermediate modules may need an explicit fqname or land in a
    sibling marker that finalize doesn't connect.
    """

    name: str = "init_subclass"
    version: str = "1"

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        nodes_by_simple = _local_class_decls(ctx.payload.nodes)
        local_imports = _local_import_targets(ctx.payload.nodes)
        module_node = next((n for n in ctx.payload.nodes if n.type == "module"), None)
        if module_node is None:
            return None
        module_fqname = module_node.fqname

        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        for stmt in ctx.module.body:
            if not isinstance(stmt, cst.ClassDef):
                continue
            class_decls = nodes_by_simple.get(stmt.name.value, [])
            if not class_decls:
                continue

            has_init = _has_init_subclass(stmt)
            base_fqnames = _resolve_bases(stmt, local_imports, nodes_by_simple, module_fqname)

            for class_decl in class_decls:
                if has_init:
                    marker = synthetic_node(
                        f"{INIT_SUBCLASS_PREFIX}{class_decl.fqname}", class_decl.path
                    )
                    nodes.append(marker)
                    # ``parent -> marker`` so the marker is alive iff
                    # the parent class is.
                    edges.append((class_decl, marker, SYNTHETIC_POSITION))
                for base_fqname in base_fqnames:
                    bucket = synthetic_node(f"{SUBCLASS_OF_PREFIX}{base_fqname}", class_decl.path)
                    nodes.append(bucket)
                    edges.append((bucket, class_decl, SYNTHETIC_POSITION))

        if not nodes and not edges:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Index synth markers by fqname so we can look up
        # ``<subclass-of:F>`` buckets in O(1) per parent fqname.
        subclass_buckets: dict[str, SymbolNode] = {}
        init_markers: list[tuple[SymbolNode, SymbolNode]] = []  # (parent_class, marker)
        for node in ctx.graph.nodes:
            if node.type != "synthetic":
                continue
            if node.fqname.startswith(SUBCLASS_OF_PREFIX):
                # Multiple files may have emitted the same bucket; the
                # graph dedupes by node identity. Keep the first.
                subclass_buckets.setdefault(node.fqname[len(SUBCLASS_OF_PREFIX) :], node)
            elif node.fqname.startswith(INIT_SUBCLASS_PREFIX):
                preds = [p for p in ctx.graph.predecessors(node) if p.type == "class"]
                for parent in preds:
                    init_markers.append((parent, node))

        if not init_markers:
            return

        existing_targets = {
            marker: set(ctx.graph.successors(marker)) for _parent, marker in init_markers
        }

        # BFS subclass closure rooted at each parent's fqname; emit
        # ``marker -> sub`` for every transitive class scoped to the
        # current base. Direct subclasses are graph.successors of
        # ``<subclass-of:parent.fqname>``; their own fqnames key further
        # buckets, recursively.
        for parent, marker in init_markers:
            seen: set[SymbolNode] = set()
            stack = [parent]
            while stack:
                current = stack.pop()
                bucket = subclass_buckets.get(current.fqname)
                if bucket is None:
                    continue
                for sub in ctx.graph.successors(bucket):
                    if sub.type != "class" or sub in seen or sub is parent:
                        continue
                    seen.add(sub)
                    stack.append(sub)
                    if not sub.path.is_relative_to(ctx.base):
                        continue
                    if sub in existing_targets[marker]:
                        continue
                    yield AddEdge(marker, sub)


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
    local_imports: dict[str, Import],
    nodes_by_simple: dict[str, list[SymbolNode]],
    module_fqname: str,
) -> list[str]:
    """Return the list of resolved base fqnames for ``class_def``.

    Resolution is local to this file: bare ``Name`` bases first try the
    file's class/function/variable decls, then fall back to its import
    aliases (``base.Plugin`` after ``from pkg import base``); dotted
    ``Attribute`` bases walk the head through the import map. Bases
    that don't resolve locally are dropped. Re-exports through chained
    modules can stay unresolved here -- callers can add an explicit
    ``-e`` for those cases.
    """
    out: list[str] = []
    for arg in class_def.bases:
        if arg.keyword is not None:
            continue
        target = _resolve_class_expr(arg.value, local_imports, nodes_by_simple, module_fqname)
        if target is not None:
            out.append(target)
    return out


def _resolve_class_expr(
    expr: cst.BaseExpression,
    local_imports: dict[str, Import],
    nodes_by_simple: dict[str, list[SymbolNode]],
    module_fqname: str,
) -> str | None:
    if isinstance(expr, cst.Subscript):
        return _resolve_class_expr(expr.value, local_imports, nodes_by_simple, module_fqname)
    if isinstance(expr, cst.Name):
        name = expr.value
        # Same-file class first.
        for decl in nodes_by_simple.get(name, []):
            if decl.type == "class":
                return decl.fqname
        # Local import alias.
        imp = local_imports.get(name)
        if imp is not None:
            return _import_target_fqname(imp)
        return None
    if isinstance(expr, cst.Attribute):
        parts = _dotted_parts(expr)
        if parts is None:
            return None
        head, rest = parts[0], parts[1:]
        imp = local_imports.get(head)
        if imp is not None:
            base = _import_target_fqname(imp)
            return f"{base}.{'.'.join(rest)}" if rest else base
        return ".".join(parts)
    return None


def _import_target_fqname(imp: Import) -> str:
    if imp.decl:
        return f"{imp.module}.{imp.decl}"
    return imp.module


def _dotted_parts(expr: cst.BaseExpression) -> list[str] | None:
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


def _local_class_decls(nodes) -> dict[str, list[SymbolNode]]:
    out: dict[str, list[SymbolNode]] = {}
    for n in nodes:
        if n.type in ("class", "function", "variable"):
            out.setdefault(simple_name(n.fqname), []).append(n)
    return out


def _local_import_targets(nodes) -> dict[str, Import]:
    """Map of ``local_name -> Import`` for top-level imports in the file."""
    out: dict[str, Import] = {}
    for n in nodes:
        if n.type == "import" and n.imports is not None:
            out[simple_name(n.fqname)] = n.imports
    return out
