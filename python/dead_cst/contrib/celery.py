"""Plugin: keep Celery task handlers and ``shared_task`` callables alive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..graph import NodeFlags
from ..plugins._base import native
from ..plugins.decl_shapes import DispatchAppPlugin

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

    marker_prefix: str = "celery"
    app_classes: tuple[str, ...] = ("celery.Celery",)
    registration_decorators: frozenset[str] = frozenset({"task"})

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        yield from DispatchAppPlugin.run(self, ctx)
        # ``@shared_task`` is appless and not covered by DispatchAppPlugin.
        by_path: dict[str, list[int]] = {}
        for ref in (
            native.query(ctx)
            .decorators()
            .where_module("celery")
            .where_name(list(_SHARED_TASK_NAMES))
            .row_indices()
        ):
            by_path.setdefault(ref.path, []).append(ref.decorated_idx)
        for path, target_idxs in by_path.items():
            yield native.AddNodeByIdx(
                fqname=f"{CELERY_SHARED_PREFIX}{Path(path).name}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=target_idxs,
            )
