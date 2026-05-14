"""Stitch raw-name imports into ``src -> dst`` edges in the symbol graph.

The visitor pass produces ``(src, Import, flags)`` triples where
``Import`` carries the raw dotted name written in the source -- no
classification has happened yet. :func:`resolve_edges` is the single
place that walks those triples against the per-package
:class:`SymbolTrie`, falls back to a :class:`PathResolver` for cross-
package / external classification, and yields the concrete
``(src_symbol, dst_symbol, flags)`` triples -- following re-exports,
fanning star imports out to every top-level decl in the target
module, and emitting synthetic nodes for stdlib / external / unresolved
targets. The flag is preserved through every emission so the original
"this reference came from a dead suite" attribution survives
resolution.

Centralizing classification here is what lets the per-file visitor
cache survive search-path changes: the visitor's :class:`Import`
records depend only on the source code, and any rebinding of
``sys.path`` / swapping the resolver re-stitches edges without
re-running the visitor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterable

from .graph import EdgeFlags, Import, SymbolNode, SymbolTrie, _claim_edge
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


def _canonicalize(imp: Import, symbol_lookup: SymbolTrie) -> tuple[Import, SymbolTrie | None]:
    """Canonicalize ``imp`` and return the trie node its module lands on.

    Pushes ``decl`` parts into ``module`` while they resolve as
    submodules in the trie, recovering the pre-refactor shape where
    ``from p import functions`` (``functions`` is a submodule of
    ``p``) collapses to ``Import(module="p.functions", decl=None)``
    rather than ``module="p" decl="functions"``. Returns the trie
    node for the canonical module so the caller doesn't repeat the
    walk; ``None`` when ``imp.module`` doesn't resolve to anything
    in the trie.

    Star imports and imports without a ``decl`` skip the walk -- the
    module name is already the target -- but still benefit from the
    returned trie node.
    """
    if imp.star or imp.decl is None:
        return imp, symbol_lookup._get(imp.module.split("."))

    parts = imp.module.split(".") + imp.decl.split(".")
    cur = symbol_lookup
    last_module_node: SymbolTrie | None = None
    last_module_idx = -1
    for i, part in enumerate(parts):
        child = cur.children.get(part)
        if child is None:
            break
        cur = child
        if cur.module is not None:
            last_module_node = cur
            last_module_idx = i
    if last_module_idx < 0:
        return imp, None
    new_module = ".".join(parts[: last_module_idx + 1])
    remaining = parts[last_module_idx + 1 :]
    new_decl = ".".join(remaining) if remaining else None
    if new_module == imp.module and new_decl == imp.decl:
        return imp, last_module_node
    return (
        Import(
            module=new_module,
            decl=new_decl,
            star=imp.star,
            speculative=imp.speculative,
        ),
        last_module_node,
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
    package_path: Path,
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]],
    *,
    import_resolver: ImportResolver = default_resolve_import,
    search_paths: list[Path] | None = None,
) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
    """Yield concrete edges from raw-name import triples.

    ``import_resolver`` and ``search_paths`` are consulted only when
    the trie can't place a target -- typical first-party imports stay
    purely trie-driven. The default resolver + empty search paths
    suffice for tests; the analyzer wires its configured resolver
    through here.

    ``emitted`` is the compose-pass edge-dedup set (see
    :func:`dead_cst.graph._claim_edge`); triples already in it are
    skipped, yielded triples are recorded. Pass a fresh ``set()`` for
    standalone use.

    Resolution is memoized at three nested layers so the per-package
    compose loop's growth in importer count is additive rather than
    multiplicative. Equal-spelling :class:`Import` instances (the
    visitor produces fresh objects per file) share one
    :func:`_resolve_targets` entry; different :class:`Import` shapes
    that canonicalize to the same trie state share one ``_walk``
    entry; the re-export DFS itself is cycle-protected so a
    pathological pair like ``A.x: from B import x`` /
    ``B.x: from A import x`` terminates after one trip instead of
    spinning.
    """
    if search_paths is None:
        search_paths = []
    synthetic_memo: dict[str, SymbolNode] = {}
    external_memo: dict[tuple[str, bool], SymbolNode | None] = {}
    # ``id(SymbolTrie)`` is safe as a key here: each trie node is held
    # alive for the call by ``symbol_lookup``'s child chain, so id
    # reuse is impossible inside one ``resolve_edges`` invocation.
    walk_memo: dict[tuple[int, tuple[str, ...]], tuple[SymbolNode, ...]] = {}
    target_memo: dict[Import, tuple[SymbolNode, ...]] = {}

    def _synth(fqname: str) -> SymbolNode:
        node = synthetic_memo.get(fqname)
        if node is None:
            node = synthetic_node(fqname, package_path)
            synthetic_memo[fqname] = node
        return node

    def _classify(imp: Import) -> SymbolNode | None:
        key = (imp.module, imp.speculative)
        if key in external_memo:
            return external_memo[key]
        synth_fqname = _classify_external(imp, import_resolver, search_paths)
        node = _synth(synth_fqname) if synth_fqname is not None else None
        external_memo[key] = node
        return node

    def _emit(
        src: SymbolNode, dst: SymbolNode, flags: EdgeFlags
    ) -> Generator[tuple[SymbolNode, SymbolNode, EdgeFlags], None, None]:
        if _claim_edge(emitted, src, dst, flags):
            yield (src, dst, flags)

    def _walk(start_node: SymbolTrie, parts: tuple[str, ...]) -> tuple[SymbolNode, ...]:
        cache_key = (id(start_node), parts)
        cached = walk_memo.get(cache_key)
        if cached is not None:
            return cached
        # Caller (``_resolve_targets``) only invokes ``_walk`` for trie
        # nodes that already passed the ``node.module is None`` gate.
        assert start_node.module is not None
        results: list[SymbolNode] = []
        worklist: list[tuple[SymbolTrie, tuple[str, ...]]] = [(start_node, parts)]
        # ``visited`` breaks re-export cycles by gating every push;
        # the per-walk scope means the still-emitted decls along the
        # cycle remain correct.
        visited: set[tuple[int, tuple[str, ...]]] = {cache_key}

        def _enqueue(target: SymbolTrie, next_parts: tuple[str, ...]) -> None:
            state = (id(target), next_parts)
            if state in visited:
                return
            visited.add(state)
            worklist.append((target, next_parts))

        while worklist:
            cur, parts_now = worklist.pop()
            if not parts_now:
                continue
            part = parts_now[0]

            decls = cur.declarations.get(part, [])
            if decls:
                for decl in decls:
                    results.append(decl)
                    # Only ``import`` decls re-export; concrete decls terminate.
                    if decl.type != "import":
                        continue
                    assert decl.imports is not None, "import symbol needs Import"

                    chained, dest = _canonicalize(decl.imports, symbol_lookup)
                    if dest is None or dest.module is None:
                        ext = _classify(chained)
                        if ext is not None:
                            results.append(ext)
                        continue

                    next_parts: tuple[str, ...] = parts_now[1:]
                    if chained.decl:
                        next_parts = (*chained.decl.split("."), *next_parts)
                    _enqueue(dest, next_parts)
                continue

            # Submodule descent: don't emit the intermediate-module
            # edge -- canonicalize already pushed every decl prefix
            # that resolves as a submodule into ``Import.module``.
            if child := cur.children.get(part):
                _enqueue(child, parts_now[1:])
                continue

            logger.warning(
                "Failed to resolve import edge: %s + %s via %s in %s",
                start_node.module.fqname,
                ".".join(parts),
                part,
                cur.module.fqname if cur.module else "<no module>",
            )
        out = tuple(results)
        walk_memo[cache_key] = out
        return out

    def _resolve_targets(raw: Import) -> tuple[SymbolNode, ...]:
        """Unique dst SymbolNodes a ``raw`` Import contributes, by value.

        ``Import`` is frozen with an eager ``__hash__``, so equal
        spellings across files (the visitor builds fresh objects per
        file) collapse to one cache entry. The result is pre-deduped
        so the per-src ``_emit`` loop stays short.
        """
        cached = target_memo.get(raw)
        if cached is not None:
            return cached
        dst, node = _canonicalize(raw, symbol_lookup)
        targets: dict[SymbolNode, None] = {}
        if node is None or node.module is None:
            ext = _classify(dst)
            if ext is not None:
                targets[ext] = None
        else:
            targets[node.module] = None
            if dst.star:
                # Module-level stars also get one synthetic ``"import"``
                # decl per re-exported name from
                # :func:`dead_cst._package._materialize_star_reexports`,
                # which is what makes ``from <importer> import <name>``
                # resolve through to the original source. We keep the
                # direct decl fan-out alongside the materialization so
                # non-module-level stars (``def a(): __import__(...)``,
                # which the materializer skips because synthetics need
                # a module home) still produce the pessimistic
                # ``<enclosing_decl> -> target.<name>`` keep-alive edges.
                for decls in node.declarations.values():
                    for decl in decls:
                        targets[decl] = None
            elif dst.decl:
                for n in _walk(node, tuple(dst.decl.split("."))):
                    targets[n] = None
        out = tuple(targets)
        target_memo[raw] = out
        return out

    for src, raw, flags in import_edges:
        for dst_node in _resolve_targets(raw):
            yield from _emit(src, dst_node, flags)


# Keep these re-exports stable for callers that historically reached
# into ``_edges`` for the synthetic-prefix constants.
__all__ = [
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "UNRESOLVED_PREFIX",
    "resolve_edges",
]
