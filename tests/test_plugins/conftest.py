"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

import pytest

from dead_cst._package import PackageContribution
from dead_cst.analyze import _entrypoint_seeds, _find_reachable as find_reachable
from dead_cst.graph import SymbolNode, SymbolTrie
from dead_cst.resolvers import Package


@pytest.fixture
def reachable_fqnames():
    """Return ``{fqname for n in find_reachable(graph) if not synthetic}``."""

    def _reachable(graph) -> set[str]:
        reached = find_reachable(graph, seeds=_entrypoint_seeds(graph))
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
