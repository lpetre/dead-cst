"""Stitch resolved imports into ``src -> dst`` edges in the symbol graph.

The visitor pass produces ``(src, Import, flags)`` triples where
``Import.path`` is whatever a :class:`~dead_cst.resolvers.PathResolver`
returned for the imported name and ``flags`` is the
:class:`~dead_cst.graph.EdgeFlags` value the apply step derived from
the access position. :func:`resolve_edges` walks those triples against
the already-built per-base :class:`SymbolTrie` and yields the concrete
``(src_symbol, dst_symbol, flags)`` triples -- following re-exports,
fanning star imports out to every top-level decl in the target module,
and emitting synthetic nodes for stdlib / external / unresolved
targets. The flag is preserved through every emission so the original
"this reference came from a dead suite" attribution survives
resolution.

The ``name -> path`` half of resolution lives in
:mod:`dead_cst.resolvers._imports`; this module is purely about edge
construction in the trie that visitor pass populated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from .graph import EdgeFlags, Import, SymbolNode, SymbolTrie
from .plugins._core import (
    SYNTHETIC_PATH_PREFIXES,
    synthetic_node,
)

logger = logging.getLogger(__name__)


def resolve_edges(
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
    symbol_lookup: SymbolTrie,
    base: Path,
) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = set()

    def _emit(
        src: SymbolNode, dst: SymbolNode, flags: EdgeFlags
    ) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
        key = (src, dst, flags)
        if key in emitted:
            return
        emitted.add(key)
        yield key

    for src, dst, flags in import_edges:
        if not isinstance(dst.path, Path):
            if dst.path.startswith(SYNTHETIC_PATH_PREFIXES):
                yield from _emit(src, synthetic_node(dst.path, base), flags)
            continue

        node = symbol_lookup._get(dst.module.split("."))
        if not node or node.module is None:
            logger.warning("Failed to resolve import module: %s", dst.module)
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
                    if decl.type in {"function", "class", "variable", "type_alias"}:
                        continue

                    # Import re-export: follow it without advancing
                    # ``parts`` so the remaining attrs resolve in the
                    # destination module.
                    assert decl.type == "import"
                    assert decl.imports is not None, "import symbol needs Import"

                    if not isinstance(decl.imports.path, Path):
                        if decl.imports.path.startswith(SYNTHETIC_PATH_PREFIXES):
                            yield from _emit(src, synthetic_node(decl.imports.path, base), flags)
                        continue

                    dest = symbol_lookup._get(decl.imports.module.split("."))
                    if not dest:
                        logger.warning(
                            "Failed to resolve import edge: %s + %s via %s in %s (no %s)",
                            dst.module,
                            dst.decl,
                            part,
                            cur.module.fqname if cur.module else "<no module>",
                            decl.imports.module,
                        )
                        continue

                    next_parts = parts[1:]
                    if decl.imports.decl:
                        next_parts = decl.imports.decl.split(".") + next_parts
                    worklist.append((dest, next_parts))
                continue

            # Maybe ``part`` is a submodule under the current package/module
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
