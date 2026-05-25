"""Plugin: keep stdlib ``unittest`` test classes and lifecycle hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..graph import NodeFlags
from ..plugins._base import Plugin, native

UNITTEST_PREFIX = "<unittest>:"

_MODULE_HOOKS: frozenset[str] = frozenset({"setUpModule", "tearDownModule", "load_tests"})

_UNITTEST_BASE_FQNAMES: frozenset[str] = frozenset(
    {"unittest.TestCase", "unittest.IsolatedAsyncioTestCase"}
)


@dataclass
class UnittestPlugin(Plugin):
    """Mark stdlib ``unittest`` discoveries as entrypoints.

    Walks ty's type hierarchy to find every transitive subclass of
    ``unittest.TestCase`` / ``IsolatedAsyncioTestCase``, and treats
    ``setUpModule`` / ``tearDownModule`` / ``load_tests`` in any file
    that imports ``unittest`` as additional hooks.
    """

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        # Cheap O(1) presence probe — short-circuits before paying
        # for the path-set lookup below.
        if not native.query(ctx).imports().of("unittest").exists():
            return
        import_idxs = native.query(ctx).imports().of("unittest").indices()
        importer_paths = {path for _k, path, _fq, _f in ctx.node_attrs(import_idxs)}

        # decls_by_path: path -> [decl_idx, ...]
        decls_by_path: dict[str, list[int]] = {}
        for base_fqname in _UNITTEST_BASE_FQNAMES:
            sub_idxs = native.query(ctx).subclasses().of_fqn(base_fqname).transitive(True).indices()
            if not sub_idxs:
                continue
            for sub_idx, (_k, sub_path, _fq, _f) in zip(
                sub_idxs, ctx.node_attrs(sub_idxs), strict=True
            ):
                decls_by_path.setdefault(sub_path, []).append(sub_idx)

        hook_idxs = (
            native.query(ctx)
            .decls()
            .with_kind("function")
            .with_paths(list(importer_paths))
            .with_simple_names(list(_MODULE_HOOKS))
            .indices()
        )
        if hook_idxs:
            for hook_idx, (_k, hook_path, _fq, _f) in zip(
                hook_idxs, ctx.node_attrs(hook_idxs), strict=True
            ):
                decls_by_path.setdefault(hook_path, []).append(hook_idx)

        flags = int(NodeFlags.TESTCASE)
        # Batched module-fqname fetch — one node_attrs hop per path bucket.
        paths = list(decls_by_path.keys())
        module_idxs = ctx.modules_for_paths(paths)
        present_modules = [(p, idx) for p, idx in zip(paths, module_idxs) if idx is not None]
        if not present_modules:
            return
        module_attrs = ctx.node_attrs([idx for _p, idx in present_modules])
        for (path, _idx), (_k, _p, module_fqname, _f) in zip(
            present_modules, module_attrs, strict=True
        ):
            yield native.AddNodeByIdx(
                fqname=f"{UNITTEST_PREFIX}{module_fqname}",
                path=path,
                flags=flags,
                edges_to_idx=decls_by_path[path],
            )
