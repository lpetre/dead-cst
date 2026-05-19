"""Plugin: fan out ``EdgeFlags.DYNAMIC_IMPORT`` edges to a module's exports.

The rust backend emits one minimal edge per *explicit symbol* a
``__import__('X')`` / ``importlib.import_module('X')`` call mentions
(tagged ``EdgeFlags.DYNAMIC_IMPORT``). This plugin opts in to the older
libcst semantic of fanning each module-targeted edge out to every export
of that module.

The plugin supports a three-stage rollout:

1. **Catch-all.** Drop the plugin in with no filters.
2. **Catch-all + targeted excludes.** Use :attr:`exclude_sources` /
   :attr:`exclude_targets` to opt specific files / module trees out.
3. **Targeted includes.** Flip to :attr:`include_sources` /
   :attr:`include_targets` to allowlist remaining call sites. Both
   forms can coexist — when both are set, an edge must match an
   ``include_*`` pattern AND must not match any ``exclude_*`` pattern.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from ..graph import EdgeFlags
from ._base import Plugin, native

_DYNAMIC_IMPORT_FLAG: int = int(EdgeFlags.DYNAMIC_IMPORT)


@dataclass
class DynamicImportFallbackPlugin(Plugin):
    """Fan out ``DYNAMIC_IMPORT``-flagged edges to the target module's exports."""

    include_underscore: bool = False
    respect_dunder_all: bool = True
    exclude_sources: tuple[str, ...] = ()
    exclude_targets: tuple[str, ...] = ()
    include_sources: tuple[str, ...] = ()
    include_targets: tuple[str, ...] = ()

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        nodes = ctx.nodes()
        cache: dict[str, list] = {}
        project_root = Path(ctx.project_root)
        for src_idx, dst_idx, flags in ctx.edges():
            if not (flags & _DYNAMIC_IMPORT_FLAG):
                continue
            dst = nodes[dst_idx]
            if dst.kind != "module":
                continue
            src = nodes[src_idx]
            if not self._allowed(Path(src.path), project_root, dst.fqname):
                continue
            for export in self._exports_for(ctx, dst.fqname, cache):
                yield native.AddEdge(src=src, dst=export)

    def _exports_for(
        self,
        ctx: native.ProjectContext,
        module_fqname: str,
        cache: dict[str, list],
    ) -> list:
        cached = cache.get(module_fqname)
        if cached is not None:
            return cached
        exports = None
        if self.respect_dunder_all:
            exports = ctx.find_module_dunder_all_exports(module_fqname)
        if exports is None:
            decls = ctx.find_module_top_level_decls(module_fqname)
            if self.include_underscore:
                exports = decls
            else:
                exports = [d for d in decls if not d.fqname.rsplit(".", 1)[-1].startswith("_")]
        cache[module_fqname] = exports
        return exports

    def _allowed(self, src_path: Path, project_root: Path, target_fqname: str) -> bool:
        try:
            rel = src_path.relative_to(project_root)
        except ValueError:
            rel = src_path
        rel_posix = PurePosixPath(*rel.parts)
        if self.include_sources and not _match_path(rel_posix, self.include_sources):
            return False
        if self.include_targets and not _match_fqname(target_fqname, self.include_targets):
            return False
        if self.exclude_sources and _match_path(rel_posix, self.exclude_sources):
            return False
        if self.exclude_targets and _match_fqname(target_fqname, self.exclude_targets):
            return False
        return True


def _match_path(path: PurePosixPath, patterns: tuple[str, ...]) -> bool:
    return any(path.match(p) for p in patterns)


def _match_fqname(fqname: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(fqname, p) for p in patterns)
