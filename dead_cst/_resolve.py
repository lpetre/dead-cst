import contextlib
import logging
import os
import sys
from functools import cache
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Generator

from ._symbols import Import, SymbolNode, SymbolTrie

logger = logging.getLogger(__name__)


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


def resolve_edges(
    import_edges: set[tuple[SymbolNode, Import]], symbol_lookup: SymbolTrie
) -> Generator[tuple[SymbolNode, SymbolNode], None, None]:
    for src, dst in import_edges:
        if not isinstance(dst.path, Path):
            continue

        node = symbol_lookup._get(dst.module.split("."))
        if not node:
            logger.warning("Failed to resolve import module: %s", dst.module)
            continue

        # add the edge to the module
        yield src, node.module

        # No decl? the edge points at a module
        if not dst.decl:
            continue

        # resolve the access to the deepest declaration we can find. this will jump through imports
        parts = dst.decl.split(".")
        while parts:
            part = parts[0]

            # Try declaration at current node (module or package)
            if decl := node.declarations.get(part):
                # We resolved a symbol; emit edge
                yield src, decl

                # If it's a concrete decl, we're done (ignore trailing attrs like .build)
                if decl.type in {"function", "class", "variable"}:
                    break

                # It's an import re-export; follow it but DO NOT advance `i`
                assert decl.type == "import"
                assert decl.imports is not None, "import symbol needs Import"

                if not isinstance(decl.imports.path, Path):
                    break

                dest = symbol_lookup._get(decl.imports.module.split("."))
                if not dest:
                    logger.warning(
                        "Failed to resolve import edge: %s.%s via %s in %s (no %s)",
                        dst.module,
                        dst.decl,
                        part,
                        node.module.fqname,
                        decl.imports.module,
                    )
                    break

                node = dest
                parts = parts[1:]
                if decl.imports.decl:
                    parts = decl.imports.decl.split(".") + parts
                continue

            # Maybe `part` is a submodule under the current package/module
            if child := node.children.get(part):
                node = child
                parts = parts[1:]
                continue

            # Give up: neither a decl nor a submodule
            logger.warning(
                "Failed to resolve import edge: %s.%s via %s in %s",
                dst.module,
                dst.decl,
                part,
                node.module.fqname,
            )
            break
