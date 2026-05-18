"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

import pytest

from dead_cst import Analysis
from dead_cst._graphstore import SymbolGraph
from dead_cst.analyze import _find_reachable as find_reachable, _keepalive_seeds
from dead_cst.graph import KEEPALIVE_DEFAULT
from dead_cst.resolvers import ManualResolver


@pytest.fixture
def reachable_fqnames():
    """Return ``{fqname for n in find_reachable(graph) if not synthetic}``."""

    def _reachable(graph) -> set[str]:
        reached = find_reachable(graph, _keepalive_seeds(graph, KEEPALIVE_DEFAULT))
        return {n.fqname for n in reached if n.type != "synthetic"}

    return _reachable


@pytest.fixture
def build_plugin_graph(tmp_path, write_files):
    """Build a SymbolGraph with the given plugins applied."""

    def _build(files: dict[str, str], plugins: list) -> SymbolGraph:
        write_files(files)
        return Analysis(
            tmp_path,
            resolver=ManualResolver(specs=["."]),
            plugins=plugins,
        ).materialize_all()

    return _build
