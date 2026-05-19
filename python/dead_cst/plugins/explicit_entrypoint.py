"""Plugin: mark user-specified symbols (file paths, FQNs, regexes) as
entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ._base import Plugin, native

EXPLICIT_PREFIX = "<entrypoint>:"


@dataclass
class ExplicitEntrypointPlugin(Plugin):
    """Mark user-specified symbols as entrypoints.

    ``specs`` accepts:

    * ``str`` -- matches an exact fully-qualified name or a file path
      relative to ``project_root``.
    * :class:`pathlib.Path` -- matches an exact absolute path.
    * :class:`re.Pattern` -- matched against the file path relative to
      ``project_root``.
    """

    specs: list[str | Path | re.Pattern[str]] = field(default_factory=list)

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        if not self.specs:
            return
        # Push the per-node match loop into rust: bucket specs by type
        # once on the Python side and hand the buckets across the FFI
        # boundary in one call. Avoids the previous
        # ``for node in ctx.nodes(): Path(node.path).relative_to(root)``
        # pattern which paid ~25-50 µs per node.
        regexes: list[str] = []
        str_specs: list[str] = []
        abs_paths: list[str] = []
        for spec in self.specs:
            if isinstance(spec, re.Pattern):
                regexes.append(spec.pattern)
            elif isinstance(spec, Path):
                abs_paths.append(str(spec))
            elif isinstance(spec, str):
                str_specs.append(spec)
        for node in ctx.find_nodes_matching_specs(ctx.project_root, regexes, str_specs, abs_paths):
            yield native.AddEntrypoint(node, marker="<entrypoint>")
