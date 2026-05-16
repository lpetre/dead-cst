"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst import Analysis
from dead_cst._graphstore import SymbolGraph
from dead_cst._package import PackageContribution
from dead_cst.analyze import _entrypoint_seeds, _find_reachable as find_reachable
from dead_cst.graph import SymbolNode, SymbolTrie
from dead_cst.resolvers import ManualResolver, Package


@pytest.fixture
def reachable_fqnames():
    """Return ``{fqname for n in find_reachable(graph) if not synthetic}``."""

    def _reachable(graph) -> set[str]:
        reached = find_reachable(graph, _entrypoint_seeds(graph))
        return {n.fqname for n in reached if n.type != "synthetic"}

    return _reachable


@pytest.fixture
def make_contribution():
    """Return a builder for a minimal :class:`PackageContribution`."""

    def _make(
        package: Package,
        nodes: frozenset[SymbolNode] = frozenset(),
    ) -> PackageContribution:
        return PackageContribution(
            package=package,
            trie=SymbolTrie(),
            nodes=nodes,
            edges=frozenset(),
            dead_suites={},
            import_edges=frozenset(),
        )

    return _make


@pytest.fixture
def build_plugin_graph(tmp_path, write_files, backend):
    """Build a SymbolGraph with the given plugins applied, dispatching on ``--backend``.

    libcst path: today's :class:`Analysis` pipeline (visitor +
    ``observe`` / ``finalize``). Rust path: routes through
    :class:`dead_cst_ty_native.ProjectContext`'s plugin protocol — each
    plugin's ``run(ctx)`` method runs once after ty builds the
    project-wide graph. Falls back to skip when the rust extension is
    missing.

    Plugins must satisfy *both* protocols to run on both backends; today
    that's :class:`ModuleDundersPlugin` and :class:`InitSubclassPlugin`.
    Tests that pull in plugins without a rust ``run`` method should
    keep using :func:`make_analysis` directly.
    """

    def _build(files: dict[str, str], plugins: list) -> SymbolGraph:
        write_files(files)
        if backend == "rust":
            return _build_rust_plugin_graph(tmp_path, plugins)
        return Analysis(
            tmp_path, resolver=ManualResolver(specs=["."]), plugins=plugins
        ).materialize_all()

    return _build


def _build_rust_plugin_graph(root: Path, plugins: list) -> SymbolGraph:
    pytest.importorskip(
        "dead_cst_ty_native",
        reason="Run `maturin develop --manifest-path crates/dead-cst-ty-native/Cargo.toml` "
        "to build the prototype rust backend.",
    )
    from tests.prototype._bridge import materialize
    import dead_cst_ty_native as native

    ctx = native.ProjectContext(str(root))
    for plugin in plugins:
        ctx.add_plugin(plugin)
    return materialize(ctx.materialize())
