"""Plugin: fan out ``EdgeFlags.DYNAMIC_IMPORT`` edges to a module's exports.

A rust-backend ``__import__('X')`` / ``importlib.import_module('X')``
emits one minimal edge per *explicit symbol* the call mentions
(tagged ``EdgeFlags.DYNAMIC_IMPORT``) — see
``crates/dead-cst-ty-native/CLAUDE.md``. Reachability captures only
what the literal call expresses, so a side-effecting import like
``importlib.import_module('pkg.plugins.email')`` keeps ``pkg.plugins.email``
alive but does not reach the top-level decls inside it.

That's deliberate. Projects that *want* the older libcst semantic
("import-by-name keeps every export alive") opt in by passing this
plugin to :class:`Analysis`. Projects that have a focused custom
plugin (a Click plugin loader, a Celery beat schedule walker, ...)
should write that and leave this one off — the explicit per-feature
plugin produces a tighter graph.

The plugin is designed for the natural three-stage rollout:

1. **Catch-all.** Drop the plugin in with no filters and every
   ``__import__('X')`` / ``importlib.import_module('X')`` keeps every
   export of ``X`` alive — restores libcst's pre-rust fan-out shape
   without baking it into the visitor.
2. **Catch-all + targeted excludes.** As focused plugins land for
   specific dynamic-dispatch idioms (a Click loader, a Celery beat
   walker), the catch-all double-counts those call sites. Use
   :attr:`exclude_sources` / :attr:`exclude_targets` to opt those
   files / module trees *out* of the catch-all so the focused plugin
   produces the tighter graph instead.
3. **Targeted includes.** Once the exclude list gets unwieldy, flip
   to :attr:`include_sources` / :attr:`include_targets` to allowlist
   the remaining call sites the catch-all still owns. Both forms can
   coexist — when both are set, the call site must match an
   ``include_*`` pattern *and* must not match any ``exclude_*``
   pattern.

The plugin is **rust-backend-focused in practice**: the libcst
pipeline inlines fan-out at visit time without flagging the edges,
so on libcst the ``DYNAMIC_IMPORT`` filter matches nothing and
``finalize`` is a no-op.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable

from ..graph import EdgeFlags, SymbolNode
from ._core import AddEdge, GraphOp, ObserveContext, PluginContext, simple_name

if TYPE_CHECKING:
    import dead_cst_ty_native as native

    from ..graph import VisitorPayload

# Mirror of ``EdgeFlags.DYNAMIC_IMPORT`` (``enum.auto()`` = 2) for use
# against the rust backend's raw edge tuples. The two constants are
# pinned together in :class:`EdgeFlags` and in
# ``crates/dead-cst-ty-native/src/lib.rs::EDGE_FLAG_DYNAMIC_IMPORT``.
_DYNAMIC_IMPORT_FLAG: int = int(EdgeFlags.DYNAMIC_IMPORT)


@dataclass
class DynamicImportFallbackPlugin:
    """Fan out ``DYNAMIC_IMPORT``-flagged edges to the target module's exports.

    For each edge ``src -> module`` flagged
    :attr:`EdgeFlags.DYNAMIC_IMPORT`, emit ``src -> export`` for every
    name the target module would surface under ``from module import *``.

    "Export" is computed per-module:

    * If the module declares ``__all__`` and :attr:`respect_dunder_all`
      is true (the default), use the names listed there. The visitor
      already wires ``__all__`` to each listed decl, so this resolves
      via the existing graph edges.
    * Otherwise, use every top-level decl in the module whose simple
      name does not start with ``_`` (matching ``from module import *``
      runtime semantics). Set :attr:`include_underscore` to ``True`` to
      include ``_private`` names as well.

    Edges whose destination is a specific decl (``__import__('p',
    fromlist=['f'])`` already emits ``src -> p.f``) are left alone —
    only module-targeted edges fan out.

    Use :attr:`exclude_sources` / :attr:`exclude_targets` and
    :attr:`include_sources` / :attr:`include_targets` to scope which
    flagged edges participate. See the module docstring for the
    intended catch-all → exclude → include rollout.
    """

    name: str = "dynamic_import_fallback"
    version: int = 1779200000

    include_underscore: bool = False
    """When ``True``, ``_private`` names participate in the fan-out.
    Default ``False`` matches ``from X import *`` runtime semantics.
    Ignored when the target module declares ``__all__`` and
    :attr:`respect_dunder_all` is left at its default."""

    respect_dunder_all: bool = True
    """When ``True`` and the target module declares ``__all__``, use
    those names as the export list. When ``False``, fall back to the
    underscore-filter rule even for modules with ``__all__``."""

    exclude_sources: tuple[str, ...] = ()
    """Path-glob patterns matched (via :meth:`pathlib.PurePosixPath.match`)
    against each call site's source path relative to the project root.
    Any matching edge is skipped — the typical use is opting specific
    files *out* of the catch-all when a focused plugin handles them.
    Example: ``("pkg/loader.py", "pkg/loaders/*.py")``."""

    exclude_targets: tuple[str, ...] = ()
    """fnmatch patterns matched against the target module fqname. Any
    matching edge is skipped. Example: ``("tests.*", "pkg.vendored.*")``
    silences fan-out into test fixtures and vendored bundles."""

    include_sources: tuple[str, ...] = ()
    """When non-empty, only call sites whose source path matches at
    least one of these :meth:`pathlib.PurePosixPath.match` patterns
    participate. Combined with :attr:`exclude_sources` via
    ``include AND NOT exclude``."""

    include_targets: tuple[str, ...] = ()
    """When non-empty, only flagged edges whose target module fqname
    matches at least one of these fnmatch patterns participate.
    Combined with :attr:`exclude_targets` via ``include AND NOT exclude``."""

    # Per-finalize-pass cache of ``(module_fqname, frozenset_of_exports)``.
    # Modules are looked up many times when several callers each do
    # ``importlib.import_module('pkg.plugins.X')``; cache once per pass.
    _export_cache: dict[str, list[SymbolNode]] = field(default_factory=dict, init=False, repr=False)

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # ``_export_cache`` is per-package: clear so a stale entry from
        # the previous package's lookup doesn't leak into this one (the
        # lookup trie changes per package — see ``PluginContext``).
        self._export_cache.clear()
        raw = ctx.graph.raw
        package_path = ctx.contribution.package.path
        project_root = ctx.project_root
        for u, v, flags in raw.weighted_edge_list():
            if not flags & EdgeFlags.DYNAMIC_IMPORT:
                continue
            src = ctx.graph.node(u)
            dst = ctx.graph.node(v)
            if dst.type != "module":
                continue
            # Scope to the current package by source: the same edge
            # would otherwise be re-visited on every package's finalize
            # pass. ``_claim_edge`` would dedupe the result, but
            # skipping early is cheaper.
            if not src.path.is_relative_to(package_path):
                continue
            if not self._allowed(src.path, project_root, dst.fqname):
                continue
            for export in self._exports_for(ctx, dst):
                yield AddEdge(src, export)

    def _exports_for(self, ctx: PluginContext, module: SymbolNode) -> list[SymbolNode]:
        cached = self._export_cache.get(module.fqname)
        if cached is not None:
            return cached
        raw = ctx.graph.raw
        decls = _module_top_level_decls(ctx, module)
        dunder_all = _find_dunder_all(decls) if self.respect_dunder_all else None
        if dunder_all is not None:
            # The visitor wires ``__all__`` to each listed decl; reuse
            # those edges as the canonical export list.
            exports = [raw[i] for i in raw.successor_indices(ctx.graph.index(dunder_all))]
        elif self.include_underscore:
            exports = decls
        else:
            exports = [d for d in decls if not simple_name(d.fqname).startswith("_")]
        self._export_cache[module.fqname] = exports
        return exports

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        """Rust-backend counterpart of :meth:`finalize`.

        Walks ``ctx.edges()`` for ``DYNAMIC_IMPORT``-flagged triples,
        looks up each target module's exports via
        :meth:`ProjectContext.find_module_dunder_all_exports` /
        :meth:`find_module_top_level_decls`, and yields
        :class:`native.AddEdge` per fan-out target. The per-pass cache
        keeps the export query at one lookup per module even when
        several call sites import the same one.
        """
        import dead_cst_ty_native as native

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
            for export in self._native_exports_for(ctx, dst.fqname, cache):
                yield native.AddEdge(src=src, dst=export)

    def _native_exports_for(
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
                exports = [d for d in decls if not simple_name(d.fqname).startswith("_")]
        cache[module_fqname] = exports
        return exports

    def _allowed(self, src_path: Path, project_root: Path, target_fqname: str) -> bool:
        """``include AND NOT exclude`` over the four pattern lists.

        Sources are matched as path globs (``PurePosixPath.match`` with
        forward-slash separators) against the source path relative to
        ``project_root``. Targets are matched as fnmatch patterns
        against the target module fqname. Sources outside the project
        root fall back to using the absolute path's posix form — a
        defensive case that shouldn't trigger for real first-party
        files but keeps the plugin from crashing on edge cases.
        """
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


def _module_top_level_decls(ctx: PluginContext, module: SymbolNode) -> list[SymbolNode]:
    """Return ``module``'s direct top-level decls (no transitive submodules).

    ``PluginContext.module_surface`` walks every transitive submodule
    too — that's the right shape for ``importlib.import_module`` side
    effects but the wrong shape for ``from module import *``, which
    only binds the importing-scope's top-level names.
    """
    trie_node = ctx.symbol_lookup._get(module.fqname.split("."))
    if trie_node is None or trie_node.module is None:
        return []
    out: list[SymbolNode] = []
    for bucket in trie_node.declarations.values():
        out.extend(bucket)
    return out


def _find_dunder_all(decls: list[SymbolNode]) -> SymbolNode | None:
    for d in decls:
        if d.type == "variable" and simple_name(d.fqname) == "__all__":
            return d
    return None


def _match_path(path: PurePosixPath, patterns: tuple[str, ...]) -> bool:
    return any(path.match(p) for p in patterns)


def _match_fqname(fqname: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(fqname, p) for p in patterns)
