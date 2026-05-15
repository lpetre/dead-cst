"""Prototype plugin protocol for the rust backend.

Demonstrates a new plugin shape distinct from the libcst-side
:class:`dead_cst.plugins.EdgePlugin`:

* **Once per project**, not per file or per package. The rust crate
  builds the project-wide graph (`ingest_decls` → `emit_*` phases),
  then invokes each plugin's :meth:`ProjectPlugin.run` exactly once,
  passing a `ProjectContext` whose mutation + query methods are
  implemented in rust against ty's `SemanticIndex` / `parsed_module`.

* The **context is the rust pyclass** (`dead_cst_ty_native.ProjectContext`).
  Plugins call its methods to mint nodes (`add_node`), wire edges
  (`add_edge`), and ask structured questions
  (`find_module_dunders`, `find_classes_defining_method`,
  `find_subclasses_of`, `find_comment_patterns`). Queries return
  `NativeNode` Python objects whose `idx` carries the graph identity
  edges need.

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

from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:
    import dead_cst_ty_native as native


@runtime_checkable
class ProjectPlugin(Protocol):
    """One project-scoped pass over the rust-built graph.

    Implementations call ``ctx.add_node`` / ``ctx.add_edge`` to extend
    the graph and ``ctx.find_*`` to query it. Both groups round-trip
    through rust — queries execute against ty's semantic index;
    mutations land in the same builder the snapshot reads at the end.

    A plugin's :attr:`name` is informational (used in error messages
    and for ordering / dedup at the call site). Plugins don't carry a
    ``version`` because the rust backend has no per-file payload cache
    to invalidate — ty's Salsa db is the cache.
    """

    name: str

    def run(self, ctx: "native.ProjectContext") -> None: ...


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
# Example plugins — same intent as the libcst-side builtins, ported to
# the new shape so the size delta is visible.
# ---------------------------------------------------------------------------


class ModuleDundersPlugin:
    """Keep module-level dunder variables alive.

    Same intent as :class:`dead_cst.plugins.ModuleDundersPlugin` but
    the per-file CST inspection is gone: ``ctx.find_module_dunders``
    already returns the matching variable nodes from the rust side.
    """

    name = "module_dunders"

    def run(self, ctx: "native.ProjectContext") -> None:
        for dunder in ctx.find_module_dunders():
            marker = ctx.add_node(
                fqname=f"<dunder>:{dunder.fqname}",
                path=dunder.path,
                flags=2,  # NodeFlags.ENTRYPOINT
            )
            ctx.add_edge(marker, dunder)


class InitSubclassPlugin:
    """Keep transitive subclasses of ``__init_subclass__``-defining classes alive.

    Replaces the two-phase libcst plugin (observe + finalize) with one
    query each: ``find_classes_defining_method("__init_subclass__")``
    locates the parents, then ``find_subclasses_of`` returns the
    transitive closure straight from ty's `type_hierarchy_subtypes`.
    """

    name = "init_subclass"

    def run(self, ctx: "native.ProjectContext") -> None:
        for parent in ctx.find_classes_defining_method("__init_subclass__"):
            marker = ctx.add_node(
                fqname=f"<__init_subclass__>:{parent.fqname}",
                path=parent.path,
            )
            ctx.add_edge(parent, marker)
            for sub in ctx.find_subclasses_of(parent):
                ctx.add_edge(marker, sub)


class KeepAliveCommentPlugin:
    """Keep any decl preceded by ``# dead-cst: keep`` alive.

    Demonstrates the comment-pattern query. ``find_comment_patterns``
    returns ``(decl_node, comment_text)`` pairs where ``decl_node`` is
    the next declaration following each matching comment.
    """

    name = "keep_alive_comment"

    def run(self, ctx: "native.ProjectContext") -> None:
        for decl, _comment in ctx.find_comment_patterns(r"#\s*dead-cst:\s*keep\b"):
            marker = ctx.add_node(
                fqname=f"<keep>:{decl.fqname}",
                path=decl.path,
                flags=2,  # NodeFlags.ENTRYPOINT
            )
            ctx.add_edge(marker, decl)
