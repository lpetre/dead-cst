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

The plugin is **rust-only in practice**: the libcst pipeline inlines
fan-out at visit time without flagging the edges, so on libcst the
``DYNAMIC_IMPORT`` filter matches nothing and ``finalize`` is a
no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    Construct without arguments for the defaults
    (``include_underscore=False``, ``respect_dunder_all=True``) — those
    match libcst's pre-rust fan-out shape.
    """

    name: str = "dynamic_import_fallback"
    version: int = 1779120000

    include_underscore: bool = False
    """When ``True``, ``_private`` names participate in the fan-out.
    Default ``False`` matches ``from X import *`` runtime semantics.
    Ignored when the target module declares ``__all__`` and
    :attr:`respect_dunder_all` is left at its default."""

    respect_dunder_all: bool = True
    """When ``True`` and the target module declares ``__all__``, use
    those names as the export list. When ``False``, fall back to the
    underscore-filter rule even for modules with ``__all__``."""

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
        for src_idx, dst_idx, flags in ctx.edges():
            if not (flags & _DYNAMIC_IMPORT_FLAG):
                continue
            dst = nodes[dst_idx]
            if dst.kind != "module":
                continue
            src = nodes[src_idx]
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
