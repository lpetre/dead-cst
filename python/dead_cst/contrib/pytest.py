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
        # filter; one batched node_attrs for the (kind, path, fqname)
        # per-row reads the bucketing needs.
        idxs = native.query(ctx).decls().with_kinds(["function", "class", "variable"]).indices()
        if not idxs:
            return
        attrs = ctx.node_attrs(idxs)

        # Per-path bucket: (path, [(idx, kind, fqname), ...])
        decls_by_path: dict[str, list[tuple[int, str, str]]] = {}
        for idx, (kind, path, fqname, _flags) in zip(idxs, attrs, strict=True):
            decls_by_path.setdefault(path, []).append((idx, kind, fqname))

        # Batched module-fqname fetch.
        paths = list(decls_by_path.keys())
        module_idxs = ctx.modules_for_paths(paths)
        present = [(p, m) for p, m in zip(paths, module_idxs) if m is not None]
        module_fqname_by_path: dict[str, str] = {}
        if present:
            module_attrs = ctx.node_attrs([m for _p, m in present])
            for (path, _m), (_k, _p, module_fqname, _f) in zip(present, module_attrs, strict=True):
                module_fqname_by_path[path] = module_fqname

        for path, rows in decls_by_path.items():
            module_fqname = module_fqname_by_path.get(path)
            if module_fqname is None:
                continue
            filename = Path(path).name
            if filename == "conftest.py":
                yield from _mark_seed(
                    f"{PYTEST_CONFTEST_PREFIX}{module_fqname}", path, [i for i, _k, _f in rows]
                )
            elif _is_test_filename(filename):
                test_idxs = [i for (i, kind, fqname) in rows if _is_test_decl(kind, fqname)]
                if test_idxs:
                    yield from _mark_seed(f"{PYTEST_TESTS_PREFIX}{module_fqname}", path, test_idxs)

        # ``@pytest.fixture`` decorators.
        fixtures_by_path: dict[str, list[int]] = {}
        for ref in (
            native.query(ctx)
            .decorators()
            .where_module("pytest")
            .where_name("fixture")
            .row_indices()
        ):
            fixtures_by_path.setdefault(ref.path, []).append(ref.decorated_idx)

        if fixtures_by_path:
            f_paths = list(fixtures_by_path.keys())
            f_module_idxs = ctx.modules_for_paths(f_paths)
            f_present = [(p, m) for p, m in zip(f_paths, f_module_idxs) if m is not None]
            if f_present:
                f_module_attrs = ctx.node_attrs([m for _p, m in f_present])
                for (path, _m), (_k, _p, module_fqname, _f) in zip(
                    f_present, f_module_attrs, strict=True
                ):
                    yield from _mark_seed(
                        f"{PYTEST_FIXTURES_PREFIX}{module_fqname}",
                        path,
                        fixtures_by_path[path],
                    )


def _mark_seed(
    fqname: str,
    path: str,
    target_idxs: list[int],
) -> Iterable[native.GraphOp]:
    if not target_idxs:
        return
    yield native.AddNodeByIdx(
        fqname=fqname,
        path=path,
        flags=int(NodeFlags.TESTCASE),
        edges_to_idx=target_idxs,
    )


def _is_test_decl(kind: str, fqname: str) -> bool:
    simple = fqname.rsplit(".", 1)[-1]
    if kind == "function" and simple.startswith("test_"):
        return True
    if kind == "class" and simple.startswith("Test"):
        return True
    return False


def _is_test_filename(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")
