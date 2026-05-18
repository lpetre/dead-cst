"""Plugin: mark user-specified symbols (file paths, FQNs, regexes) as
entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import dead_cst_ty_native as native

EXPLICIT_PREFIX = "<entrypoint>:"


@dataclass
class ExplicitEntrypointPlugin:
    """Mark user-specified symbols as entrypoints.

    ``specs`` accepts:

    * ``str`` -- matches an exact fully-qualified name or a file path
      relative to ``project_root``.
    * :class:`pathlib.Path` -- matches an exact absolute path.
    * :class:`re.Pattern` -- matched against the file path relative to
      ``project_root``.
    """

    specs: list[str | Path | re.Pattern[str]] = field(default_factory=list)
    name: str = "explicit"
    version: int = 1777760307

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        import dead_cst_ty_native as native

        root = Path(ctx.project_root)
        for node in ctx.nodes():
            if not self._matches(node, root):
                continue
            yield native.AddEntrypoint(node, marker="<entrypoint>")

    def _matches(self, node: native.NativeNode, root: Path) -> bool:
        path = Path(node.path)
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = node.path
        for spec in self.specs:
            if isinstance(spec, re.Pattern):
                if spec.match(rel):
                    return True
            elif isinstance(spec, Path):
                if spec == path:
                    return True
            elif isinstance(spec, str):
                if spec == rel or spec == node.fqname:
                    return True
        return False
