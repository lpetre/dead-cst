import contextlib
import logging
import os
import sys
import sysconfig
from functools import cache
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Generator

from ._plugins._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    STDLIB_PREFIX,
    SYNTHETIC_PATH_PREFIXES,
    UNRESOLVED_PREFIX,
    synthetic_node,
)
from ._symbols import Import, SymbolNode, SymbolTrie

logger = logging.getLogger(__name__)

STDLIB = Path(sysconfig.get_path("stdlib")).resolve()
SITE_PACKAGES_MARKERS = ("site-packages", "dist-packages")


@contextlib.contextmanager
def temp_sys_path(paths: list[Path]):
    old = list(sys.path)
    seen = set(old)
    sys.path = [str(p) for p in paths if str(p) not in seen] + sys.path
    try:
        yield
    finally:
        sys.path = old


@cache
def safe_resolve_module(fullname: str) -> ModuleSpec | None:
    parts = fullname.split(".")
    search_paths = list(sys.path)

    # emulate namespace __path__ resolution
    for i, part in enumerate(parts[:-1]):
        candidate_paths = []
        for base in search_paths:
            subdir = os.path.join(base, parts[i])
            if os.path.isdir(subdir):
                candidate_paths.append(subdir)
        search_paths = candidate_paths

    # Final part resolution
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if not find_spec:
            continue
        try:
            spec = find_spec(fullname, search_paths)
            if spec:
                return spec
        except Exception:
            continue

    return None


@cache
def distribution_lookup() -> dict[Path, str]:
    from importlib import metadata

    lookup = {}
    for dist in metadata.distributions():
        for file in dist.files:
            abs_path = Path(dist.locate_file(file)).resolve()
            lookup[abs_path] = dist.metadata["Name"]
    return lookup


def resolve_import(name: str, search_paths: list[Path]) -> str | Path:
    spec = safe_resolve_module(name)
    if spec is None:
        return None
    if spec.origin is None:
        return None
    if spec.origin in {"built-in", "frozen"}:
        return f"{STDLIB_PREFIX}{name}"
    path = Path(spec.origin).resolve()
    if path.is_relative_to(STDLIB):
        return f"{STDLIB_PREFIX}{name}"

    lookup = distribution_lookup()
    if dist := lookup.get(path):
        return f"{EXTERNAL_DIST_PREFIX}{dist}"

    path_str = str(path)
    if any(m in path_str for m in SITE_PACKAGES_MARKERS):
        return f"{EXTERNAL_FILE_PREFIX}{name}"

    for search in search_paths:
        if path.is_relative_to(search):
            return path
    raise Exception(f"Module {name} resolved to an unexpected path: {path}")


def resolve_edges(
    import_edges: set[tuple[SymbolNode, Import]],
    symbol_lookup: SymbolTrie,
    base: Path,
) -> Generator[tuple[SymbolNode, SymbolNode], None, None]:
    emitted: set[tuple[SymbolNode, SymbolNode]] = set()

    def _emit(
        src: SymbolNode, dst: SymbolNode
    ) -> Generator[tuple[SymbolNode, SymbolNode], None, None]:
        key = (src, dst)
        if key in emitted:
            return
        emitted.add(key)
        yield key

    for src, dst in import_edges:
        if not isinstance(dst.path, Path):
            if dst.path.startswith(SYNTHETIC_PATH_PREFIXES):
                yield from _emit(src, synthetic_node(dst.path, base))
            continue

        node = symbol_lookup._get(dst.module.split("."))
        if not node:
            logger.warning("Failed to resolve import module: %s", dst.module)
            continue

        yield from _emit(src, node.module)

        # Star import: fan out to every top-level declaration in the target module
        if dst.star:
            for decls in node.declarations.values():
                for decl in decls:
                    yield from _emit(src, decl)
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
                    yield from _emit(src, decl)

                    # Concrete decl terminates this continuation;
                    # trailing attrs like ``.build`` are ignored.
                    if decl.type in {"function", "class", "variable"}:
                        continue

                    # Import re-export: follow it without advancing
                    # ``parts`` so the remaining attrs resolve in the
                    # destination module.
                    assert decl.type == "import"
                    assert decl.imports is not None, "import symbol needs Import"

                    if not isinstance(decl.imports.path, Path):
                        if decl.imports.path.startswith(SYNTHETIC_PATH_PREFIXES):
                            yield from _emit(src, synthetic_node(decl.imports.path, base))
                        continue

                    dest = symbol_lookup._get(decl.imports.module.split("."))
                    if not dest:
                        logger.warning(
                            "Failed to resolve import edge: %s + %s via %s in %s (no %s)",
                            dst.module,
                            dst.decl,
                            part,
                            cur.module.fqname,
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
                cur.module.fqname,
            )
