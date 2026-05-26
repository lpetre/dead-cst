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
        # Path-only bucketing. Most projects have far more non-test
        # decls than test decls, so ``node_paths`` (one str per row)
        # is ~3× cheaper than ``node_attrs`` (a 4-tuple per row) on
        # this initial scan. We pay the per-row ``kind`` / ``fqname``
        # cost later, but only for decls that survive the filename
        # filter.
        idxs = native.query(ctx).decls().with_kinds(["function", "class", "variable"]).indices()
        if not idxs:
            return
        paths = ctx.node_paths(idxs)

        conftest_idxs_by_path: dict[str, list[int]] = {}
        test_idxs_by_path: dict[str, list[int]] = {}
        for idx, path in zip(idxs, paths, strict=True):
            filename = Path(path).name
            if filename == "conftest.py":
                conftest_idxs_by_path.setdefault(path, []).append(idx)
            elif _is_test_filename(filename):
                test_idxs_by_path.setdefault(path, []).append(idx)

        # Filter test-file decls by ``_is_test_decl`` — needs the
        # ``kind`` and ``fqname`` we deliberately didn't fetch above.
        # We pay the 4-tuple node_attrs cost only for this subset.
        test_filtered_by_path: dict[str, list[int]] = {}
        for path, test_idxs in test_idxs_by_path.items():
            attrs = ctx.node_attrs(test_idxs)
            keep = [
                idx
                for idx, (kind, _p, fqname, _f) in zip(test_idxs, attrs, strict=True)
                if _is_test_decl(kind, fqname)
            ]
            if keep:
                test_filtered_by_path[path] = keep

        # Module-fqname fetch — only for paths we'll actually seed
        # (conftest + filtered tests). Skips the cost for irrelevant
        # files entirely.
        seed_paths = list({*conftest_idxs_by_path, *test_filtered_by_path})
        module_fqname_by_path = _module_fqnames(ctx, seed_paths)
        for path, conftest_idxs in conftest_idxs_by_path.items():
            module_fqname = module_fqname_by_path.get(path)
            if module_fqname is None:
                continue
            yield from _mark_seed(f"{PYTEST_CONFTEST_PREFIX}{module_fqname}", path, conftest_idxs)
        for path, test_idxs in test_filtered_by_path.items():
            module_fqname = module_fqname_by_path.get(path)
            if module_fqname is None:
                continue
            yield from _mark_seed(f"{PYTEST_TESTS_PREFIX}{module_fqname}", path, test_idxs)

        # ``@pytest.fixture`` decorators. Bucketed per-file the same way
        # — only fetch module fqnames for paths that have at least one
        # fixture.
        fixtures_by_path: dict[str, list[int]] = {}
        for ref in (
            native.query(ctx).decorators().where_module("pytest").where_name("fixture").collect()
        ):
            fixtures_by_path.setdefault(ref.path, []).append(ref.decorated_idx)

        if fixtures_by_path:
            fixture_module_fqnames = _module_fqnames(ctx, list(fixtures_by_path.keys()))
            for path, fixture_idxs in fixtures_by_path.items():
                module_fqname = fixture_module_fqnames.get(path)
                if module_fqname is None:
                    continue
                yield from _mark_seed(
                    f"{PYTEST_FIXTURES_PREFIX}{module_fqname}", path, fixture_idxs
                )


def _module_fqnames(ctx: native.ProjectContext, paths: list[str]) -> dict[str, str]:
    """``{path: module_fqname}`` for every path that resolves to a
    project module. Two batched FFI hops total (``modules_for_paths`` +
    ``node_attrs``), irrespective of ``len(paths)``."""
    if not paths:
        return {}
    module_idxs = ctx.modules_for_paths(paths)
    present = [(p, m) for p, m in zip(paths, module_idxs) if m is not None]
    if not present:
        return {}
    module_attrs = ctx.node_attrs([m for _p, m in present])
    return {
        path: module_fqname
        for (path, _m), (_k, _p, module_fqname, _f) in zip(present, module_attrs, strict=True)
    }


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
