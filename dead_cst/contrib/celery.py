"""Plugin: keep Celery task handlers and ``shared_task`` callables alive.

Strategy: mirror :class:`FlaskPlugin` / :class:`FastAPIPlugin` for the
``app = Celery(...)`` / ``@app.task`` shape via the factory-aware
:class:`DispatchAppPlugin` base, and add a third channel for the
appless ``@shared_task`` decorator that registers a task into Celery's
global registry regardless of which app loads it.

1. Direct ``X = Celery(...)`` (named import) and ``X = celery.Celery(...)``
   (module-prefixed) assignments are classified per-file. The Celery
   worker process loads the app via ``celery -A package.module:X``, so
   a Celery instance is always an entrypoint -- matching how
   :class:`FlaskPlugin` and :class:`FastAPIPlugin` treat their app
   instances.
2. ``@<X>.task`` and ``@<X>.task(...)`` decorators on top-level functions
   produce ``X -> handler`` edges so the task callable lives as long as
   the owning app does. Variables that are decorated but not directly
   classified get a ``<celery-pending>:`` marker the finalize pass
   resolves once cross-file edges are stitched -- this covers the
   ``X = make_celery()`` factory shape.
3. Factory functions / classes whose body constructs ``Celery(...)``
   get a ``<celery-factory>:`` marker so the cross-package walk has a
   discriminator even when the factory uses
   ``import celery; celery.Celery()`` and the external-edge classifier
   drops the ``decl='Celery'`` half (see ``fastapi.py`` for the
   rationale -- same mechanism).
4. Top-level functions decorated ``@shared_task`` / ``@shared_task(...)``
   (with ``shared_task`` imported from ``celery``) are seeded as
   entrypoints via a per-file ``<celery-shared>:`` synthetic.
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import libcst as cst
from libcst.metadata import CodeRange

from ..graph import NodeFlags, SymbolNode

if TYPE_CHECKING:
    import dead_cst_ty_native as native
from ..plugins._core import (
    SYNTHETIC_POSITION,
    ObserveContext,
    collect_module_imports,
    decls_by_simple_name,
    make_payload,
    matched_attr_call,
    synthetic_node,
)
from ..plugins.decl_shapes import DispatchAppPlugin

if TYPE_CHECKING:
    from ..graph import VisitorPayload

# Classes from ``celery`` that produce an app instance. Value records
# whether the instance should be seeded as an entrypoint -- always
# ``True`` for Celery: the worker process loads ``module:app`` by import
# path, so every constructed app is reachable from the framework side.
_INSTANCE_KINDS: Mapping[str, bool] = {"Celery": True}

# Module-level decorator names from ``celery`` that register an appless
# task into the global registry.
_SHARED_TASK_NAMES: frozenset[str] = frozenset({"shared_task"})

# Per-file synthetic that anchors every ``@shared_task`` decorated
# top-level function in the file. ``shared_task`` is appless: it
# registers into Celery's global registry, so there's no owning
# variable to wire through. Format: ``<celery-shared>:<file.name>``.
CELERY_SHARED_PREFIX = "<celery-shared>:"


@dataclass
class CeleryPlugin(DispatchAppPlugin):
    """Mark Celery apps as entrypoints and wire task handlers through them.

    Configures the factory-aware :class:`DispatchAppPlugin` shape for
    Celery and overrides :meth:`observe` to add the appless
    ``@shared_task`` channel (a per-file ``<celery-shared>:`` synthetic
    that points at every top-level function decorated with
    ``shared_task`` imported from ``celery``).
    """

    name: str = "celery"
    version: int = 1779000000
    app_module: str = "celery"
    registration_decorators: frozenset[str] = frozenset({"task"})
    instance_kinds: Mapping[str, bool] = field(default_factory=lambda: dict(_INSTANCE_KINDS))

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        base = super().observe(ctx)
        shared = self._shared_task_payload(ctx)
        if shared is None:
            return base
        if base is None:
            return shared
        return make_payload(
            nodes=tuple(base.nodes) + tuple(shared.nodes),
            edges=tuple(base.edges) + tuple(shared.edges),
        )

    def run(self, ctx: "native.ProjectContext") -> None:
        DispatchAppPlugin.run(self, ctx)
        # ``@shared_task`` is appless and not covered by DispatchAppPlugin.
        funcs = ctx.find_decorated_decls("celery", list(_SHARED_TASK_NAMES))
        by_path: dict[str, list[native.NativeNode]] = {}
        for func in funcs:
            by_path.setdefault(func.path, []).append(func)
        for path, targets in by_path.items():
            seed = ctx.add_node(
                fqname=f"{CELERY_SHARED_PREFIX}{Path(path).name}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
            )
            for target in targets:
                ctx.add_edge(seed, target)

    def _shared_task_payload(self, ctx: ObserveContext) -> "VisitorPayload | None":
        """Wire ``@shared_task`` decorated top-level functions to a per-file synthetic."""
        celery_shared_imports = collect_module_imports(ctx.module, "celery", _SHARED_TASK_NAMES)
        shared_task_names = _find_shared_task_handlers(ctx.module, celery_shared_imports)
        if not shared_task_names:
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        shared_targets: list[SymbolNode] = []
        for name in shared_task_names:
            for decl in decls_by_name.get(name, []):
                if decl.type == "function":
                    shared_targets.append(decl)
        if not shared_targets:
            return None

        shared_seed = synthetic_node(
            f"{CELERY_SHARED_PREFIX}{ctx.path.name}",
            ctx.path,
            flags=NodeFlags.ENTRYPOINT,
        )
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = [
            (shared_seed, target, SYNTHETIC_POSITION) for target in shared_targets
        ]
        return make_payload(nodes=[shared_seed], edges=edges)


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
