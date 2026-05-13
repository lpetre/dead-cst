"""Plugin: keep Celery task handlers and ``shared_task`` callables alive.

Strategy: mirror :class:`FlaskPlugin` / :class:`FastAPIPlugin` for the
``app = Celery(...)`` / ``@app.task`` shape, and add a third channel for
the appless ``@shared_task`` decorator that registers a task into
Celery's global registry regardless of which app loads it.

1. Direct ``X = Celery(...)`` (named import) and ``X = celery.Celery(...)``
   (module-prefixed) assignments are classified per-file in
   :meth:`observe`. The Celery worker process loads the app via
   ``celery -A package.module:X``, so a Celery instance is always an
   entrypoint -- matching how :class:`FlaskPlugin` and
   :class:`FastAPIPlugin` treat their app instances.
2. ``@<X>.task`` and ``@<X>.task(...)`` decorators on top-level functions
   produce ``X -> handler`` edges so the task callable lives as long as
   the owning app does. Variables that are decorated but not directly
   classified get a :data:`CELERY_PENDING_PREFIX` marker the finalize
   pass resolves once cross-file edges are stitched -- this covers the
   ``X = make_celery()`` factory shape.
3. Factory functions / classes whose body constructs ``Celery(...)``
   get a :data:`CELERY_FACTORY_PREFIX` marker so the cross-package walk
   has a discriminator even when the factory uses
   ``import celery; celery.Celery()`` and the external-edge classifier
   drops the ``decl='Celery'`` half (see ``fastapi.py`` for the
   rationale -- same mechanism).
4. Top-level functions decorated ``@shared_task`` / ``@shared_task(...)``
   (with ``shared_task`` imported from ``celery``) are seeded as
   entrypoints via a per-file :data:`CELERY_SHARED_PREFIX` synthetic.
   ``shared_task`` registers into Celery's global registry and is
   invoked by name by any worker that imports the module, so there is
   no owning ``app`` instance to wire through.

Limitations: only top-level ``X = Celery(...)`` assignments with a single
``Name`` target are detected. Class-attribute apps
(``self.app = Celery(...)``) and dynamic registrations
(``app.register_task(MyTask())``, ``app.tasks["name"] = ...``,
``app.conf.beat_schedule = {"sched": {"task": "module.func"}}``,
``celery.signals.task_prerun.connect(handler)``) are not handled --
users can keep those alive with explicit ``-e`` entrypoints. Nested-
attribute decorators such as ``@app.on_after_configure.connect`` and
``@app.task(base=MyBase)``-style classifications are also not
recognized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst
from libcst.metadata import CodeRange

from ..graph import NodeFlags, SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    AddNode,
    GraphOp,
    ObserveContext,
    PluginContext,
    collect_module_imports,
    decls_by_simple_name,
    find_call_assignments,
    find_factory_decls,
    find_handlers,
    make_payload,
    matched_attr_call,
    require_resolved_dep,
    synthetic_node,
    walk_to_instance_kind,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload

# Attribute names a Celery app uses to register a task callable. Matched
# as the rightmost attribute of ``@<instance>.<name>(...)``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"task"})

# Classes from ``celery`` that produce an app instance. Value records
# whether the instance should be seeded as an entrypoint -- always
# ``True`` for Celery: the worker process loads ``module:app`` by import
# path, so every constructed app is reachable from the framework side.
_INSTANCE_KINDS: dict[str, bool] = {"Celery": True}

# Module-level decorator names from ``celery`` that register an appless
# task into the global registry.
_SHARED_TASK_NAMES: frozenset[str] = frozenset({"shared_task"})

CELERY_APP_PREFIX = "<celery-app>:"
CELERY_PENDING_PREFIX = "<celery-pending>:"
# Cross-package factory discriminator -- see ``fastapi.py`` for the
# rationale. Format: ``<celery-factory>:Celery:<owner.fqname>``.
CELERY_FACTORY_PREFIX = "<celery-factory>:"
# Per-file synthetic that anchors every ``@shared_task`` decorated
# top-level function in the file. ``shared_task`` is appless: it
# registers into Celery's global registry, so there's no owning
# variable to wire through. Format: ``<celery-shared>:<file.name>``.
CELERY_SHARED_PREFIX = "<celery-shared>:"


@dataclass
class CeleryPlugin:
    """Mark Celery apps as entrypoints and wire task handlers through them.

    Two-phase shape mirrors :class:`FlaskPlugin`:

    * :meth:`observe` (per-file) classifies direct ``X = Celery(...)``
      assignments and emits ``X -> handler`` edges for every
      ``@<X>.task(...)`` decorator. Direct hits get a
      :data:`CELERY_APP_PREFIX` synthetic entrypoint pointed at the
      variable. Variables that have handlers but no direct kind get a
      :data:`CELERY_PENDING_PREFIX` marker synthetic linked to the
      variable, deferred to :meth:`finalize`. Top-level decls whose
      body constructs a Celery instance get a
      :data:`CELERY_FACTORY_PREFIX` marker that survives cross-package
      walks. Top-level functions decorated ``@shared_task`` /
      ``@shared_task(...)`` (with ``shared_task`` imported from
      ``celery``) are seeded via a single :data:`CELERY_SHARED_PREFIX`
      synthetic per file.
    * :meth:`finalize` (per-package) walks each pending marker forward
      through the graph and promotes any Celery factory chain to an
      entrypoint via :func:`walk_to_instance_kind` (with
      :data:`CELERY_FACTORY_PREFIX` so factory markers count as
      discriminators).
    """

    name: str = "celery"
    version: int = 1779000000

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        celery_app_imports = collect_module_imports(ctx.module, "celery", _INSTANCE_KINDS)
        celery_shared_imports = collect_module_imports(ctx.module, "celery", _SHARED_TASK_NAMES)
        direct = find_call_assignments(ctx.module, celery_app_imports, _INSTANCE_KINDS)
        decorated = find_handlers(ctx.module, None, _REGISTRATION_DECORATORS)
        factory_kinds = find_factory_decls(ctx.module, celery_app_imports, _INSTANCE_KINDS)
        shared_task_names = _find_shared_task_handlers(ctx.module, celery_shared_imports)
        if not direct and not decorated and not factory_kinds and not shared_task_names:
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        for var_name in direct.keys() | decorated.keys():
            var_decls = decls_by_name.get(var_name, [])
            kind = direct.get(var_name)
            for var_decl in var_decls:
                if kind is None:
                    pending = synthetic_node(f"{CELERY_PENDING_PREFIX}{var_decl.fqname}", ctx.path)
                    nodes.append(pending)
                    edges.append((pending, var_decl, SYNTHETIC_POSITION))
                elif _INSTANCE_KINDS[kind]:
                    seed = synthetic_node(
                        f"{CELERY_APP_PREFIX}{var_decl.fqname}",
                        ctx.path,
                        flags=NodeFlags.ENTRYPOINT,
                    )
                    nodes.append(seed)
                    edges.append((seed, var_decl, SYNTHETIC_POSITION))

                for handler_name in decorated.get(var_name, ()):
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((var_decl, handler_decl, SYNTHETIC_POSITION))

        # Factory markers: see fastapi.py for the rationale.
        for decl_name, kinds in factory_kinds.items():
            for decl in decls_by_name.get(decl_name, []):
                for kind in kinds:
                    marker = synthetic_node(
                        f"{CELERY_FACTORY_PREFIX}{kind}:{decl.fqname}", ctx.path
                    )
                    nodes.append(marker)
                    edges.append((decl, marker, SYNTHETIC_POSITION))

        # ``@shared_task`` decorated functions: one entrypoint synthetic
        # per file, with an edge to each appless task callable.
        if shared_task_names:
            shared_targets: list[SymbolNode] = []
            for name in shared_task_names:
                for decl in decls_by_name.get(name, []):
                    if decl.type == "function":
                        shared_targets.append(decl)
            if shared_targets:
                shared_seed = synthetic_node(
                    f"{CELERY_SHARED_PREFIX}{ctx.path.name}",
                    ctx.path,
                    flags=NodeFlags.ENTRYPOINT,
                )
                nodes.append(shared_seed)
                for target in shared_targets:
                    edges.append((shared_seed, target, SYNTHETIC_POSITION))

        if not nodes and not edges:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        celery_node = require_resolved_dep(ctx, "celery")
        if celery_node is None:
            return

        for synth in list(ctx.package_nodes()):
            if synth.type != "synthetic" or not synth.fqname.startswith(CELERY_PENDING_PREFIX):
                continue
            for var in list(ctx.graph.successors(synth)):
                kind = walk_to_instance_kind(
                    ctx.graph,
                    var,
                    celery_node,
                    "celery",
                    _INSTANCE_KINDS,
                    factory_marker_prefix=CELERY_FACTORY_PREFIX,
                )
                if kind is None or not _INSTANCE_KINDS[kind]:
                    continue
                seed = synthetic_node(f"{CELERY_APP_PREFIX}{var.fqname}", var.path)
                yield AddNode(seed, entrypoint=True)
                yield AddEdge(seed, var)


def _find_shared_task_handlers(module: cst.Module, imports: dict[str, str]) -> list[str]:
    """Return the names of top-level functions decorated ``@shared_task``.

    Recognizes the four binding forms :func:`collect_module_imports`
    produces: ``from celery import shared_task`` (bare ``Name``),
    ``from celery import shared_task as st`` (aliased ``Name``),
    ``import celery`` (``@celery.shared_task``), and the called variants
    ``@shared_task(bind=True)`` / ``@celery.shared_task(...)``.
    :func:`matched_attr_call` handles all four shapes through the
    ``imports`` map, so we just feed it each decorator expression.
    """
    if not imports:
        return []
    out: list[str] = []
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            if matched_attr_call(dec.decorator, imports, _SHARED_TASK_NAMES):
                out.append(stmt.name.value)
                break
    return out
