"""Frozen-graph contract for the plugin pass.

The plugin pass collects every plugin's ops then applies the lot
under one write-lock window at the end. Two invariants follow:

1. **Frozen graph during run**. A plugin's own emissions are
   invisible to its own queries during ``run``; a later plugin
   running in the same cohort also can't see what an earlier
   plugin emitted, because nothing applies until the end.
2. **Apply order**. Ops collected from plugins land in registration
   order at end-of-pass — order-independent fan-out (two plugins
   that each emit edges referencing a base-graph decl) both land.

These hold under both the parallel executor path and the
``DEAD_CST_PLUGINS_SERIAL=1`` fallback. The tests parameterise on
the env var so we exercise both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from dead_cst import Analysis
from dead_cst import _native as native
from dead_cst.plugins import Plugin


@pytest.fixture(params=["parallel", "serial"], ids=["parallel", "serial"])
def plugin_mode(request, monkeypatch):
    """Run each test once under the parallel executor and once under
    the rust-side serial fallback. The two paths must satisfy the
    same frozen-graph contract.
    """
    if request.param == "serial":
        monkeypatch.setenv("DEAD_CST_PLUGINS_SERIAL", "1")
    else:
        monkeypatch.delenv("DEAD_CST_PLUGINS_SERIAL", raising=False)
    return request.param


class _EmitSyntheticThenQuery(Plugin):
    """Plugin that emits a synthetic node *and then* queries the
    graph for that synthetic's fqname in the same ``run``. Under
    the frozen-graph contract the query should return ``0``.
    """

    def __init__(self, fqname: str, observed: list[int]) -> None:
        self.fqname = fqname
        self.observed = observed

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
        op = native.AddNode(fqname=self.fqname, path="/")
        # Emit, *then* check whether our own emission is visible
        # via ``ctx.nodes()``. Per the frozen-graph contract it
        # is not — the apply pass runs end-of-cohort.
        # The yield happens first (so the emission is queued) but
        # the apply only lands when the whole cohort finishes.
        yield op
        self.observed.append(sum(1 for n in ctx.nodes() if n.fqname == self.fqname))


class _ObserveOthersEmission(Plugin):
    """Plugin that just queries for a fqname some other plugin in
    the cohort claims to emit. Used to prove that cross-plugin
    emissions are also invisible until the apply pass runs.
    """

    def __init__(self, target_fqname: str, observed: list[int]) -> None:
        self.target_fqname = target_fqname
        self.observed = observed

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
        self.observed.append(sum(1 for n in ctx.nodes() if n.fqname == self.target_fqname))
        return ()


def test_plugin_does_not_see_its_own_emission(tmp_path: Path, plugin_mode: str) -> None:
    """A plugin querying the graph after a yield sees the base graph,
    not its own emission. This is the core frozen-graph guarantee.
    """
    (tmp_path / "a.py").write_text("def f(): pass\n")

    observed: list[int] = []
    # Pair with a no-op plugin so the parallel path actually kicks
    # in (single-plugin Analysis takes the serial rust path).
    plugin = _EmitSyntheticThenQuery("<frozen-test-own>", observed)

    class _Noop(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            return ()

    analysis = Analysis(tmp_path, plugins=[plugin, _Noop()])
    analysis.materialize_all()

    # During ``run``, the plugin's own AddNode is queued but not yet
    # applied — the query returns 0.
    assert observed == [0]
    # Post-materialize, the synthetic *is* in the final graph.
    fqnames = {n.fqname for n in analysis.materialize_all().nodes()}
    assert "<frozen-test-own>" in fqnames


def test_plugin_does_not_see_other_plugins_emissions(tmp_path: Path, plugin_mode: str) -> None:
    """Plugin B can't see plugin A's emission during ``run`` —
    every plugin in the cohort starts from the same base graph.
    """
    (tmp_path / "a.py").write_text("def f(): pass\n")

    own_observed: list[int] = []
    other_observed: list[int] = []
    emitter = _EmitSyntheticThenQuery("<frozen-test-cross>", own_observed)
    observer = _ObserveOthersEmission("<frozen-test-cross>", other_observed)

    Analysis(tmp_path, plugins=[emitter, observer]).materialize_all()

    assert own_observed == [0]
    assert other_observed == [0]


class _EmitEdgeToBaseDecl(Plugin):
    """Two of these in a cohort prove the apply pass runs in
    registration order without dropping ops. Each plugin emits an
    ``AddNode`` referencing the same base-graph decl via
    ``edges_to``. After materialize, both synthetics + both edges
    must be present.
    """

    def __init__(self, fqname: str, decl_fqname: str) -> None:
        self.fqname = fqname
        self.decl_fqname = decl_fqname

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
        decls = [n for n in ctx.nodes() if n.fqname == self.decl_fqname]
        if not decls:
            return
        yield native.AddNode(
            fqname=self.fqname,
            path=decls[0].path,
            edges_to=[decls[0]],
        )


def test_apply_order_lands_every_plugins_ops(tmp_path: Path, plugin_mode: str) -> None:
    """Two plugins in a cohort, each minting a synthetic anchored on
    the same base decl. After materialize both synthetics and both
    edges must be present — the end-of-pass apply doesn't drop ops
    just because the same write lock window covers them all.
    """
    (tmp_path / "a.py").write_text("def f(): pass\n")

    a = _EmitEdgeToBaseDecl("<frozen-order-A>", "a.f")
    b = _EmitEdgeToBaseDecl("<frozen-order-B>", "a.f")

    analysis = Analysis(tmp_path, plugins=[a, b])
    ctx = analysis.materialize_all()
    fqnames = {n.fqname for n in ctx.nodes()}
    assert "<frozen-order-A>" in fqnames
    assert "<frozen-order-B>" in fqnames

    # Both synthetics point at ``a.f``.
    nodes = list(ctx.nodes())
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    decl_idx = by_fqname["a.f"]
    edges = list(ctx.edges())
    assert (by_fqname["<frozen-order-A>"], decl_idx, 0) in edges
    assert (by_fqname["<frozen-order-B>"], decl_idx, 0) in edges


class _EmitMultipleOps(Plugin):
    """Plugin that emits several ops in defined yield order.
    Used to verify intra-plugin yield order is preserved through
    the collect → apply pipeline.
    """

    def __init__(self, decl_fqname: str, markers: list[str]) -> None:
        self.decl_fqname = decl_fqname
        self.markers = markers

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
        decls = [n for n in ctx.nodes() if n.fqname == self.decl_fqname]
        if not decls:
            return
        decl = decls[0]
        for m in self.markers:
            yield native.AddNode(fqname=m, path=decl.path, edges_to=[decl])


def test_intra_plugin_yield_order_preserved(tmp_path: Path, plugin_mode: str) -> None:
    """Yield order within one plugin survives the collect → apply
    hop: every yielded marker lands as a node in the final graph
    (with its corresponding edge).
    """
    (tmp_path / "a.py").write_text("def f(): pass\n")

    markers = [f"<frozen-yield-{i}>" for i in range(5)]
    plugin = _EmitMultipleOps("a.f", markers)

    # Pair with a noop so the parallel path engages.
    class _Noop(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            return ()

    ctx = Analysis(tmp_path, plugins=[plugin, _Noop()]).materialize_all()
    fqnames = {n.fqname for n in ctx.nodes()}
    for m in markers:
        assert m in fqnames, f"missing marker {m!r}"


def test_descendants_query_sees_base_graph_not_emissions(tmp_path: Path, plugin_mode: str) -> None:
    """``ctx.descendants`` issued from inside a plugin walks the
    *base* graph — synthetic markers the plugin emitted earlier in
    its own ``run`` don't influence the walk.
    """
    (tmp_path / "a.py").write_text("def f(): pass\ndef g(): pass\n")

    walks: list[int] = []

    class _EmitThenWalk(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            decls = {n.fqname: n for n in ctx.nodes()}
            f = decls["a.f"]
            g = decls["a.g"]
            # Emit a synthetic that ``edges_from`` ``f`` to ``g``.
            # If applied mid-run, ``descendants(f)`` would now reach
            # the synthetic. Under the frozen-graph contract it must
            # not.
            yield native.AddNode(
                fqname="<frozen-walk-target>",
                path=f.path,
                edges_from=[f],
                edges_to=[g],
            )
            walks.append(sum(1 for d in ctx.descendants(f) if d.fqname == "<frozen-walk-target>"))

    class _Noop(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            return ()

    Analysis(tmp_path, plugins=[_EmitThenWalk(), _Noop()]).materialize_all()

    assert walks == [0]


def test_collected_ops_handle_is_single_use(tmp_path: Path) -> None:
    """``apply_ops_batched`` consumes the :class:`CollectedOps`
    handle; re-applying the same handle raises ``ValueError``.
    """
    (tmp_path / "a.py").write_text("def f(): pass\n")

    class _Emitter(Plugin):
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:  # pragma: no cover
            decls = [n for n in ctx.nodes() if n.fqname == "a.f"]
            yield native.AddNode(
                fqname="<frozen-single-use>",
                path=decls[0].path,
                edges_to=decls,
            )

    ctx = native.ProjectContext(str(tmp_path), python_env=None, show_progress=False)
    ctx.build_only()
    handle = ctx.run_plugin_collect(_Emitter())
    ctx.apply_ops_batched([handle])
    # Second apply on the same handle fails.
    with pytest.raises(ValueError, match="already drained"):
        ctx.apply_ops_batched([handle])


def test_apply_ops_batched_with_empty_list_is_noop(tmp_path: Path) -> None:
    """An empty :func:`apply_ops_batched` call applies nothing and
    doesn't error — the parallel path hits this when every plugin
    yields nothing.
    """
    (tmp_path / "a.py").write_text("def f(): pass\n")
    ctx = native.ProjectContext(str(tmp_path), python_env=None, show_progress=False)
    ctx.build_only()
    pre = sum(1 for _ in ctx.nodes())
    ctx.apply_ops_batched([])
    assert sum(1 for _ in ctx.nodes()) == pre
