"""Stitch raw-name imports into ``src -> dst`` edges in the symbol graph.

The visitor pass produces ``(src, Import, flags)`` triples where
``Import`` carries the raw dotted name written in the source -- no
classification has happened yet. :func:`resolve_edges` is the single
place that walks those triples against the per-base
:class:`SymbolTrie`, falls back to a :class:`PathResolver` for cross-
base / external classification, and yields the concrete
``(src_symbol, dst_symbol, flags)`` triples -- following re-exports,
fanning star imports out to every top-level decl in the target
module, and emitting synthetic nodes for stdlib / external / unresolved
targets. The flag is preserved through every emission so the original
"this reference came from a dead suite" attribution survives
resolution.

Centralizing classification here is what lets the per-file visitor
cache survive search-path changes: the visitor's :class:`Import`
records depend only on the source code, and any rebinding of
``sys.path`` / swapping of resolvers re-stitches edges without
re-running the visitor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterable

from .graph import EdgeFlags, Import, SymbolNode, SymbolTrie
from .plugins._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    STDLIB_PREFIX,
    SYNTHETIC_PATH_PREFIXES,
    UNRESOLVED_PREFIX,
    synthetic_node,
)
from .resolvers import ImportResolver, default_resolve_import

logger = logging.getLogger(__name__)


def _canonicalize(imp: Import, symbol_lookup: SymbolTrie) -> Import:
    """Push ``decl`` parts into ``module`` while they resolve as submodules.

    Recovers the pre-refactor shape where ``from p import functions``
    (``functions`` is a submodule of ``p``) collapses to
    ``Import(module="p.functions", decl=None)`` rather than ``module="p"
    decl="functions"``. Walks the trie greedily through ``module +
    decl`` and stops at the deepest node that has a real module
    attached. Star imports are not canonicalized -- the module name is
    already the target. Imports with no ``decl`` are returned as-is.
    """
    if imp.star or imp.decl is None:
        return imp
    parts = imp.module.split(".") + imp.decl.split(".")
    cur = symbol_lookup
    last_module_idx = -1
    for i, part in enumerate(parts):
        child = cur.children.get(part)
        if child is None:
            break
        cur = child
        if cur.module is not None:
            last_module_idx = i
    if last_module_idx < 0:
        return imp
    new_module = ".".join(parts[: last_module_idx + 1])
    remaining = parts[last_module_idx + 1 :]
    new_decl = ".".join(remaining) if remaining else None
    if new_module == imp.module and new_decl == imp.decl:
        return imp
    return Import(
        module=new_module,
        decl=new_decl,
        star=imp.star,
        speculative=imp.speculative,
    )


def _classify_external(
    imp: Import,
    import_resolver: ImportResolver,
    search_paths: list[Path],
) -> str | None:
    """Return the synthetic-node fqname for ``imp.module``, or ``None`` to drop.

    Trie miss already happened when we get here. Consult the resolver:

    * stdlib / unknown stdlib origin -> drop silently (stdlib isn't
      surfaced as graph nodes).
    * ``[external dist] X`` / ``[external file] X`` -> emit synthetic.
    * first-party-shaped Path (resolver thinks it's a project file but
      we don't have it in the trie) -> treat as unresolved.
    * resolver returned ``None`` -> emit ``[unresolved] <top-level>``
      so plugins can still answer "which files tried to import X?".
      Speculative imports skip this fallback (see
      :class:`~dead_cst.graph.Import` ``speculative``).
    """
    classification = import_resolver(imp.module, search_paths)
    if isinstance(classification, str):
        if classification.startswith(STDLIB_PREFIX):
            return None
        if classification.startswith(SYNTHETIC_PATH_PREFIXES):
            return classification
    if imp.speculative:
        return None
    top_level = imp.module.split(".", 1)[0]
    return f"{UNRESOLVED_PREFIX}{top_level}"


def resolve_edges(
    import_edges: Iterable[tuple[SymbolNode, Import, EdgeFlags]],
    symbol_lookup: SymbolTrie,
    base: Path,
    *,
    import_resolver: ImportResolver = default_resolve_import,
    search_paths: list[Path] | None = None,
) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
    """Yield concrete edges from raw-name import triples.

    ``import_resolver`` and ``search_paths`` are consulted only when
    the trie can't place a target -- typical first-party imports stay
    purely trie-driven. The default resolver + empty search paths
    suffice for tests; the analyzer wires its configured resolver
    chain through here.
    """
    if search_paths is None:
        search_paths = []

    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = set()

    def _emit(
        src: SymbolNode, dst: SymbolNode, flags: EdgeFlags
    ) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
        key = (src, dst, flags)
        if key in emitted:
            return
        emitted.add(key)
        yield key

    def _emit_external(
        src: SymbolNode, imp: Import, flags: EdgeFlags
    ) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
        synth_fqname = _classify_external(imp, import_resolver, search_paths)
        if synth_fqname is None:
            if not imp.speculative and imp.star:
                logger.warning("Failed to resolve star import: 'from %s import *'", imp.module)
            elif not imp.speculative and not imp.star:
                logger.warning("Failed to resolve import module: %s", imp.module)
            return
        yield from _emit(src, synthetic_node(synth_fqname, base), flags)

    for src, raw, flags in import_edges:
        dst = _canonicalize(raw, symbol_lookup)

        node = symbol_lookup._get(dst.module.split("."))
        if not node or node.module is None:
            yield from _emit_external(src, dst, flags)
            continue

        yield from _emit(src, node.module, flags)

        # Star import: fan out to every top-level declaration in the target module
        if dst.star:
            for decls in node.declarations.values():
                for decl in decls:
                    yield from _emit(src, decl, flags)
            continue

        # No decl? the edge points at a module
        if not dst.decl:
            continue

        # Resolve the access to the deepest declaration(s) we can find.
        # ``node.declarations[name]`` may have multiple entries when each
        # branch of a conditional binds the same name (``if X: from a
        # import f else: from b import f``); each one is a separate
        # continuation, so the walk is a small DFS.
        worklist: list[tuple[SymbolTrie, list[str]]] = [(node, dst.decl.split("."))]
        while worklist:
            cur, parts = worklist.pop()
            if not parts:
                continue
            part = parts[0]

            decls = cur.declarations.get(part, [])
            if decls:
                for decl in decls:
                    yield from _emit(src, decl, flags)

                    # Concrete decl terminates this continuation;
                    # trailing attrs like ``.build`` are ignored.
                    # ``type_alias`` (PEP 695 ``type X = ...``) is a
                    # concrete declaration, not an import re-export.
                    if decl.type in {"function", "class", "variable", "type_alias"}:
                        continue

                    # Import re-export: follow it without advancing
                    # ``parts`` so the remaining attrs resolve in the
                    # destination module.
                    assert decl.type == "import"
                    assert decl.imports is not None, "import symbol needs Import"

                    # Re-canonicalize the re-export against the lookup
                    # before chasing it -- the original import was
                    # captured raw, so a shape like ``Import(module="p",
                    # decl="functions")`` needs to flatten to
                    # ``module="p.functions"`` here too.
                    chained = _canonicalize(decl.imports, symbol_lookup)
                    dest = symbol_lookup._get(chained.module.split("."))
                    if not dest or dest.module is None:
                        yield from _emit_external(src, chained, flags)
                        continue

                    next_parts = parts[1:]
                    if chained.decl:
                        next_parts = chained.decl.split(".") + next_parts
                    worklist.append((dest, next_parts))
                continue

            # Maybe ``part`` is a submodule under the current package/module.
            # Don't emit the intermediate-module edge here -- canonicalize
            # already pushed every decl prefix that resolves as a submodule
            # into ``Import.module``, so any submodule we still hit during
            # the walk only matters for reaching its descendants.
            if child := cur.children.get(part):
                worklist.append((child, parts[1:]))
                continue

            logger.warning(
                "Failed to resolve import edge: %s + %s via %s in %s",
                dst.module,
                dst.decl,
                part,
                cur.module.fqname if cur.module else "<no module>",
            )


# Keep these re-exports stable for callers that historically reached
# into ``_edges`` for the synthetic-prefix constants.
__all__ = [
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "UNRESOLVED_PREFIX",
    "resolve_edges",
]
