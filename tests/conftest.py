import textwrap

import networkx as nx

import pytest

from dead_cst import build_symbol_graph


@pytest.fixture
def build_decl_graph(tmp_path):
    def _make_graph(files: dict[str, str]) -> nx.DiGraph:
        # Write each file to the temporary directory
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())

        # Build the declaration graph
        return build_symbol_graph({tmp_path: []})

    return _make_graph


@pytest.fixture
def assert_edges():
    def _check(graph: nx.DiGraph, expected_edges: set[str]):
        """Compare visitor.internal_edges to expected 'a -> b' strings."""
        actual_edges = {f"{src.fqname} -> {dst.fqname}" for src, dst in graph.edges}
        assert actual_edges == expected_edges

    return _check
