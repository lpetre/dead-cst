"""Plugin: keep stdlib ``unittest`` test classes and lifecycle hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..graph import NodeFlags

if TYPE_CHECKING:
    import dead_cst_ty_native as native

UNITTEST_PREFIX = "<unittest>:"

_MODULE_HOOKS: frozenset[str] = frozenset({"setUpModule", "tearDownModule", "load_tests"})

_UNITTEST_BASE_FQNAMES: frozenset[str] = frozenset(
    {"unittest.TestCase", "unittest.IsolatedAsyncioTestCase"}
)


@dataclass
class UnittestPlugin:
    """Mark stdlib ``unittest`` discoveries as entrypoints.

    Walks ty's type hierarchy to find every transitive subclass of
    ``unittest.TestCase`` / ``IsolatedAsyncioTestCase``, and treats
    ``setUpModule`` / ``tearDownModule`` / ``load_tests`` in any file
    that imports ``unittest`` as additional hooks.
    """

    name: str = "unittest"
    version: int = 1778248994

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        import dead_cst_ty_native as native

        importer_paths = {n.path for n in native.query(ctx).imports().of("unittest").collect()}

        decls_by_path: dict[str, list[native.NativeNode]] = {}
        for base_fqname in _UNITTEST_BASE_FQNAMES:
            for sub in (
                native.query(ctx).subclasses().of_fqn(base_fqname).transitive(True).collect()
            ):
                decls_by_path.setdefault(sub.path, []).append(sub)

        for node in ctx.nodes():
            if node.kind != "function":
                continue
            if node.path not in importer_paths:
                continue
            simple = node.fqname.rsplit(".", 1)[-1]
            if simple in _MODULE_HOOKS:
                decls_by_path.setdefault(node.path, []).append(node)

        flags = int(NodeFlags.ENTRYPOINT | NodeFlags.TESTCASE)
        for path, decls in decls_by_path.items():
            module = ctx.module_for(path)
            if module is None:
                continue
            yield native.AddNode(
                fqname=f"{UNITTEST_PREFIX}{module.fqname}",
                path=path,
                flags=flags,
                edges_to=decls,
            )
