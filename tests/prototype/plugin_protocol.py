"""Prototype plugin protocol for the rust backend.

Demonstrates a new plugin shape distinct from the libcst-side
:class:`dead_cst.plugins.EdgePlugin`:

* **Once per project**, not per file or per package. The rust crate
  builds the project-wide graph (`ingest_decls` → `emit_*` phases),
  then invokes each plugin's :meth:`ProjectPlugin.run` exactly once,
  passing a `ProjectContext` whose mutation + query methods are
  implemented in rust against ty's `SemanticIndex` / `parsed_module`.

* The **context is the rust pyclass** (`dead_cst._native.ProjectContext`).
  Plugins ``yield`` ``GraphOp`` values (``AddNode`` / ``AddEdge`` /
  ``AddEntrypoint``) to extend the graph, and call ``native.query(ctx).*`` to
  ask structured questions (``find_module_dunders``,
  ``find_classes_defining_method``, ``find_subclasses_of``,
  ``find_comment_patterns``). Queries return ``SymbolNode`` Python
  objects whose ``idx`` carries the graph identity edges need.

* Configuration mirrors the existing `Project` constructor today
  (root + optional `src_roots` / `python_env` / `python_version` /
  `typeshed` / `extra_paths`); the formal multi-package descriptor
  surface is a follow-up — for the prototype the consumer hands the
  ctx whatever the resolver produced as keyword args.

Typical flow::

    ctx = ProjectContext(project_root, src_roots=["src"], ...)
    ctx.add_plugin(ModuleDundersPlugin())
    ctx.add_plugin(InitSubclassPlugin())
    graph = ctx.materialize()  # builds, runs plugins, snapshots
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from dead_cst import _native as native


@runtime_checkable
class ProjectPlugin(Protocol):
    """One project-scoped pass over the rust-built graph.

    Implementations yield ``GraphOp`` values (``AddNode`` / ``AddEdge``
    / ``AddEntrypoint``) to extend the graph and call ``native.query(ctx).*``
    to query it. Both groups round-trip through rust — queries execute
    against ty's semantic index; ``apply_graph_op`` lands each yielded
    op in the same builder the snapshot reads at the end.

    A plugin's :attr:`name` is informational (used in error messages
    and for ordering / dedup at the call site). Plugins don't carry a
    ``version`` because the rust backend has no per-file payload cache
    to invalidate — ty's Salsa db is the cache.
    """

    name: str

    def run(self, ctx: "native.ProjectContext") -> "Iterable[native.GraphOp] | None": ...


def run_plugins(
    ctx: "native.ProjectContext", plugins: Iterable[ProjectPlugin]
) -> "native.NativeGraph":
    """Register ``plugins`` on ``ctx`` and materialize the final graph.

    Thin convenience over :meth:`ProjectContext.add_plugin` +
    :meth:`ProjectContext.materialize` — useful when plugins are
    assembled by the caller (e.g. resolved from entry points) rather
    than appended one at a time.
    """
    for plugin in plugins:
        ctx.add_plugin(plugin)
    return ctx.materialize()


# ---------------------------------------------------------------------------
# Example plugin — demonstrates a query (``find_comment_patterns``) with
# no libcst-side equivalent. The built-in plugins
# (:class:`dead_cst.plugins.ModuleDundersPlugin`,
# :class:`dead_cst.plugins.InitSubclassPlugin`, ...) implement ``run``
# directly so they satisfy this protocol alongside the libcst-side
# ``observe`` / ``finalize`` contract.
# ---------------------------------------------------------------------------


class KeepAliveCommentPlugin:
    """Keep any decl preceded by ``# dead-cst: keep`` alive.

    Demonstrates the comment-pattern query. ``find_comment_patterns``
    returns ``(decl_node, comment_text)`` pairs where ``decl_node`` is
    the next declaration following each matching comment.
    """

    name = "keep_alive_comment"

    def run(self, ctx: "native.ProjectContext") -> "Iterable[native.GraphOp]":
        for decl, _comment in ctx.find_comment_patterns(r"#\s*dead-cst:\s*keep\b"):
            yield native.AddEntrypoint(decl, marker="<keep>")
