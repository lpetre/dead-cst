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


@pytest.fixture
def assert_positional_edges():
    """Like ``assert_edges`` but disambiguates nodes by source position.

    Formats each node as ``fqname@line:col`` when a position is available
    (module nodes keep their bare fqname). Use this for tests where
    multiple top-level decls share a fqname -- e.g. redeclarations and
    shadowing -- so the per-textual-decl identity is visible in the
    assertion.
    """

    def _fmt(sym):
        # Module nodes have a position too (covering the whole file) but
        # rendering it would just be noise. Leave modules as bare fqnames.
        if sym.type == "module":
            return sym.fqname
        start = sym.position.start
        return f"{sym.fqname}@{start.line}:{start.column}"

    def _check(graph: nx.DiGraph, expected_edges: set[str]):
        actual_edges = {f"{_fmt(src)} -> {_fmt(dst)}" for src, dst in graph.edges}
        assert actual_edges == expected_edges

    return _check
