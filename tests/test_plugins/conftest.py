"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

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
    """Build a SymbolGraph with the given plugins applied.

    Routes through :class:`Analysis(backend=...)` which dispatches to
    the libcst or rust pipeline. On the rust path, plugins without a
    ``run(ctx)`` method are skipped (the libcst-side ``observe`` /
    ``finalize`` pair isn't called by the rust backend) so the test
    report distinguishes "plugin doesn't support the rust backend
    yet" from real regressions.
    """

    def _build(files: dict[str, str], plugins: list) -> SymbolGraph:
        write_files(files)
        if backend == "rust":
            missing = [type(p).__name__ for p in plugins if not hasattr(p, "run")]
            if missing:
                pytest.skip(f"rust backend: plugins missing run(ctx): {', '.join(missing)}")
        return Analysis(
            tmp_path,
            resolver=ManualResolver(specs=["."]),
            plugins=plugins,
            backend=backend,
        ).materialize_all()

    return _build
