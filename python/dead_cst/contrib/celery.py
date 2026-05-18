"""Plugin: keep Celery task handlers and ``shared_task`` callables alive."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from ..graph import NodeFlags
from ..plugins.decl_shapes import DispatchAppPlugin

if TYPE_CHECKING:
    from dead_cst import _native as native

_INSTANCE_KINDS: Mapping[str, bool] = {"Celery": True}
_SHARED_TASK_NAMES: frozenset[str] = frozenset({"shared_task"})

CELERY_SHARED_PREFIX = "<celery-shared>:"


@dataclass
class CeleryPlugin(DispatchAppPlugin):
    """Mark Celery apps as entrypoints and wire task handlers through them.

    Also seeds every top-level function decorated with
    ``@shared_task`` (imported from ``celery``) as an entrypoint, since
    ``shared_task`` registers into Celery's global registry without an
    owning app instance.
    """

    name: str = "celery"
    version: int = 1779000000
    app_module: str = "celery"
    registration_decorators: frozenset[str] = frozenset({"task"})
    instance_kinds: Mapping[str, bool] = field(default_factory=lambda: dict(_INSTANCE_KINDS))

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        yield from DispatchAppPlugin.run(self, ctx)
        # ``@shared_task`` is appless and not covered by DispatchAppPlugin.
        by_path: dict[str, list[native.NativeNode]] = {}
        for ref in (
            native.query(ctx)
            .decorators()
            .where_module("celery")
            .where_name(list(_SHARED_TASK_NAMES))
        ):
            by_path.setdefault(ref.path, []).append(ref.decorated)
        for path, targets in by_path.items():
            yield native.AddNode(
                fqname=f"{CELERY_SHARED_PREFIX}{Path(path).name}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to=targets,
            )
