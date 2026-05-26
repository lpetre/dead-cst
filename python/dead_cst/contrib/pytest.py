"""Plugin: keep pytest-discovered tests, conftest decls, and
``@pytest.fixture`` functions alive; model test → fixture dependencies
as graph edges so callers can introspect them."""

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
    """Mark pytest-discovered symbols as entrypoints and wire
    test → fixture edges by parameter-name matching.

    * ``conftest.py``: every top-level function / class / variable is
      seeded alive (covers ``conftest`` fixtures + helpers).
    * ``test_*.py`` / ``*_test.py``: every top-level ``test_*``
      function and ``Test*`` class is seeded alive.
    * Every ``@pytest.fixture``-decorated function is seeded alive
      via a synthetic ``<pytest:fixtures>:<module>`` entrypoint —
      same conservative rule we've shipped before. Catches
      ``autouse=True`` fixtures, ``usefixtures(...)`` markers,
      ``parametrize(..., indirect=...)``, dynamic
      ``request.getfixturevalue(...)`` lookups, and anything else
      we can't statically pin to a parameter name.
    * On top of the seed, ``test → fixture`` and ``class → fixture``
      edges are emitted by parameter-name matching, so the
      dependency structure is queryable (``ancestors(fixture)`` /
      ``descendants(test)``) even though the seed alone would
      already keep the fixture alive. The binding name is the
      ``name=`` kwarg on ``@pytest.fixture(name="alias")`` when set,
      otherwise the function's simple name.
    * For ``Test*`` classes, the union of every method's parameter
      names (minus ``self`` / ``cls``) is name-matched. Class methods
      aren't graph nodes of their own — the class is the rendezvous
      point.

    Names from a test/class signature that don't match any project
    fixture (pytest builtins like ``tmp_path`` / ``capsys``,
    third-party plugin fixtures like ``mocker``, free-form
    ``parametrize`` names) are silently ignored, so they never
    produce spurious edges.
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

        # One pass: bucket conftest decls by path, collect test-file
        # candidates into a flat (idx, path) pair. Test candidates
        # still need a ``_is_test_decl`` filter (which wants ``kind``
        # / ``fqname``), but we defer that to one batched
        # ``node_attrs`` call across every path instead of one call
        # per path inside a loop.
        conftest_idxs_by_path: dict[str, list[int]] = {}
        test_candidate_idxs: list[int] = []
        test_candidate_paths: list[str] = []
        for idx, path in zip(idxs, paths, strict=True):
            filename = Path(path).name
            if filename == "conftest.py":
                conftest_idxs_by_path.setdefault(path, []).append(idx)
            elif _is_test_filename(filename):
                test_candidate_idxs.append(idx)
                test_candidate_paths.append(path)

        # Single batched ``node_attrs`` for every test candidate
        # across every test file — one FFI hop regardless of how many
        # test files the project has. Re-bucket by path after the
        # filter; track function- vs class-kind tests separately so
        # the fixture-edge pass below can wire each kind from its
        # own parameter source (function bodies vs class method
        # bodies).
        test_filtered_by_path: dict[str, list[int]] = {}
        test_function_idxs: list[int] = []
        test_class_idxs: list[int] = []
        if test_candidate_idxs:
            attrs = ctx.node_attrs(test_candidate_idxs)
            for idx, path, attr in zip(
                test_candidate_idxs, test_candidate_paths, attrs, strict=True
            ):
                if _is_test_decl(attr.kind, attr.fqname):
                    test_filtered_by_path.setdefault(path, []).append(idx)
                    if attr.kind == "function":
                        test_function_idxs.append(idx)
                    elif attr.kind == "class":
                        test_class_idxs.append(idx)

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

        # Fixture pass. Single decorator query feeds both the
        # unconditional ``<pytest:fixtures>:<module>`` seed (every
        # ``@pytest.fixture`` is alive, the conservative rule that
        # catches autouse / usefixtures / indirect / dynamic lookups)
        # AND the parameter-name edge emission (queryable
        # ``test → fixture`` / ``class → fixture`` structure on top
        # of the seed).
        #
        # ``.with_args(True)`` is on so we can read the ``name=``
        # kwarg off ``@pytest.fixture(name="alias")`` — pytest binds
        # the fixture under the alias instead of the function name.
        fixture_refs = (
            native.query(ctx)
            .decorators()
            .where_module("pytest")
            .where_name("fixture")
            .with_args(True)
            .collect()
        )
        if not fixture_refs:
            return
        # Per-file seed-alive marker, mirroring the conftest/test
        # seed shape — every fixture in the project stays alive
        # regardless of whether a static parameter match exists.
        fixtures_by_path: dict[str, list[int]] = {}
        for ref in fixture_refs:
            fixtures_by_path.setdefault(ref.path, []).append(ref.decorated_idx)
        fixture_module_fqnames = _module_fqnames(ctx, list(fixtures_by_path.keys()))
        for path, fixture_idxs in fixtures_by_path.items():
            module_fqname = fixture_module_fqnames.get(path)
            if module_fqname is None:
                continue
            yield from _mark_seed(f"{PYTEST_FIXTURES_PREFIX}{module_fqname}", path, fixture_idxs)

        # Parameter-name edges. Skip the rust calls entirely when
        # there's no test signature to inspect.
        if not test_function_idxs and not test_class_idxs:
            return
        # Build ``binding_name -> [fixture_idx, ...]``. The binding
        # name is the ``name=`` kwarg literal when present
        # (``@pytest.fixture(name="alias")``), else the function's
        # simple name. Collisions are kept (a fixture in
        # ``conftest.py`` plus a same-named fixture in a test file:
        # both targets receive edges from any test whose param matches
        # the name — pessimistic, but harmless).
        fixture_idxs_all = [r.decorated_idx for r in fixture_refs]
        fixture_attrs = ctx.node_attrs(fixture_idxs_all)
        fixtures_by_name: dict[str, list[int]] = {}
        for ref, attr in zip(fixture_refs, fixture_attrs, strict=True):
            binding = _fixture_binding_name(ref, attr.fqname)
            fixtures_by_name.setdefault(binding, []).append(ref.decorated_idx)
        # Pull every test function's parameter list in one batched
        # rust call; emit edges per param-name match. Parameters that
        # don't resolve to a project fixture (pytest builtins like
        # ``tmp_path`` / ``capsys``, third-party plugin fixtures like
        # ``mocker``, free-form names from ``parametrize``) are
        # silently skipped.
        if test_function_idxs:
            test_params = ctx.function_parameters(test_function_idxs)
            for test_idx, params in zip(test_function_idxs, test_params, strict=True):
                for param in params:
                    for fixture_idx in fixtures_by_name.get(param, ()):
                        yield native.AddEdgeByIdx(test_idx, fixture_idx)
        # ``Test*`` classes: union of every method's parameter names
        # (minus ``self`` / ``cls``) → ``class → fixture`` edges per
        # match.
        if test_class_idxs:
            class_params = ctx.class_method_parameters(test_class_idxs)
            for cls_idx, params in zip(test_class_idxs, class_params, strict=True):
                for param in params:
                    for fixture_idx in fixtures_by_name.get(param, ()):
                        yield native.AddEdgeByIdx(cls_idx, fixture_idx)


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
    return {path: attr.fqname for (path, _m), attr in zip(present, module_attrs, strict=True)}


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


def _fixture_binding_name(ref: native.DecoratorIdxRef, fixture_fqname: str) -> str:
    """Pytest binding name for a ``@pytest.fixture``-decorated function.

    Defaults to the function's simple name (last fqname segment) and
    is overridden by the ``name=`` kwarg literal on the decorator —
    ``@pytest.fixture(name="alias")\\ndef some_fn(): ...`` is resolved
    by pytest as fixture ``alias`` regardless of ``some_fn``.

    Anything other than a string-literal kwarg (e.g. a non-literal
    expression that the rust extractor surfaces as ``ArgOpaque``)
    falls back to the function name — we can't statically determine
    the runtime binding for non-literal values.
    """
    alias = ref.kwargs.get("name")
    if isinstance(alias, native.ArgLiteral) and isinstance(alias.value, str):
        return alias.value
    return fixture_fqname.rsplit(".", 1)[-1]
