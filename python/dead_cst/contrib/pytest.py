"""Plugin: keep pytest-discovered tests, conftest decls, and ``@pytest.fixture`` functions alive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..graph import NodeFlags
from ..plugins._base import Plugin, native

PYTEST_CONFTEST_PREFIX = "<pytest:conftest>:"
PYTEST_TESTS_PREFIX = "<pytest:tests>:"
PYTEST_FIXTURES_PREFIX = "<pytest:fixtures>:"


@dataclass
class PytestPlugin(Plugin):
    """Mark pytest-discovered symbols as entrypoints.

    * ``conftest.py``: every top-level function / class / variable.
    * ``test_*.py`` / ``*_test.py``: every top-level ``test_*`` function
      and ``Test*`` class.
    * Top-level functions decorated with ``@pytest.fixture``.
    """

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        # Rust-side fold of the ``kind in {function, class, variable}``
        # filter — one FFI hop, no per-node Python attribute access.
        decls_by_path: dict[str, list[native.SymbolNode]] = {}
        for n in native.query(ctx).decls().with_kinds(["function", "class", "variable"]).collect():
            decls_by_path.setdefault(n.path, []).append(n)

        for path, decls in decls_by_path.items():
            module = ctx.module_for(path)
            if module is None:
                continue
            filename = Path(path).name
            if filename == "conftest.py":
                yield from _mark_seed(f"{PYTEST_CONFTEST_PREFIX}{module.fqname}", path, decls)
            elif _is_test_filename(filename):
                test_decls = [d for d in decls if _is_test_decl(d)]
                if test_decls:
                    yield from _mark_seed(f"{PYTEST_TESTS_PREFIX}{module.fqname}", path, test_decls)

        fixtures_by_path: dict[str, list[native.SymbolNode]] = {}
        for ref in native.query(ctx).decorators().where_module("pytest").where_name("fixture"):
            fixtures_by_path.setdefault(ref.path, []).append(ref.decorated)
        for path, fixtures in fixtures_by_path.items():
            module = ctx.module_for(path)
            if module is None:
                continue
            yield from _mark_seed(f"{PYTEST_FIXTURES_PREFIX}{module.fqname}", path, fixtures)


def _mark_seed(
    fqname: str,
    path: str,
    targets: list[native.SymbolNode],
) -> Iterable[native.GraphOp]:
    if not targets:
        return
    yield native.AddNode(
        fqname=fqname,
        path=path,
        flags=int(NodeFlags.TESTCASE),
        edges_to=targets,
    )


def _is_test_decl(node: native.SymbolNode) -> bool:
    simple = node.fqname.rsplit(".", 1)[-1]
    if node.kind == "function" and simple.startswith("test_"):
        return True
    if node.kind == "class" and simple.startswith("Test"):
        return True
    return False


def _is_test_filename(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")
