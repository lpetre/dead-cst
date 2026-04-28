import textwrap

import networkx as nx
import pytest

from dead_cst import build_symbol_graph
from dead_cst._branches import is_unreachable_node


@pytest.fixture
def write_files(tmp_path):
    """Write a ``{relpath: source}`` mapping under ``tmp_path``.

    Each value is dedented and stripped, with a trailing newline appended,
    matching the inline-source convention used across the test suite.
    """

    def _write(files: dict[str, str]) -> None:
        for name, src in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src).strip() + "\n")

    return _write


@pytest.fixture
def build_decl_graph(tmp_path):
    def _make_graph(files: dict[str, str]) -> nx.DiGraph:
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())
        return build_symbol_graph({tmp_path: []})

    return _make_graph


def _has_unreachable_endpoint(edge) -> bool:
    src, dst = edge
    return is_unreachable_node(src) or is_unreachable_node(dst)


@pytest.fixture
def assert_edges():
    def _check(graph: nx.DiGraph, expected_edges: set[str]):
        """Compare visitor.internal_edges to expected 'a -> b' strings.

        Edges touching synthetic ``unreachable`` nodes are excluded so
        these "real symbol graph" assertions don't churn whenever a new
        ``if False:`` / ``if True:`` test case is added. Tests covering
        unreachable behavior should use ``assert_unreachable_edges``.
        """
        actual_edges = {
            f"{src.fqname} -> {dst.fqname}"
            for src, dst in graph.edges
            if not _has_unreachable_endpoint((src, dst))
        }
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_positional_edges():
    """Like ``assert_edges`` but disambiguates nodes by source position.

    Formats each node as ``fqname@line:col`` when a position is available
    (module nodes keep their bare fqname). Use this for tests where
    multiple top-level decls share a fqname -- e.g. redeclarations and
    shadowing -- so the per-textual-decl identity is visible in the
    assertion. Synthetic ``unreachable`` nodes are filtered out for the
    same reason as :func:`assert_edges`.
    """

    def _fmt(sym):
        # Module nodes have a position too (covering the whole file) but
        # rendering it would just be noise. Leave modules as bare fqnames.
        if sym.type == "module":
            return sym.fqname
        start = sym.position.start
        return f"{sym.fqname}@{start.line}:{start.column}"

    def _check(graph: nx.DiGraph, expected_edges: set[str]):
        actual_edges = {
            f"{_fmt(src)} -> {_fmt(dst)}"
            for src, dst in graph.edges
            if not _has_unreachable_endpoint((src, dst))
        }
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_unreachable_edges():
    """Assert on edges originating from synthetic ``unreachable`` nodes.

    Format: ``"<line:col> -> target.fqname"``. Per-suite location is the
    only useful identifier for the source side; the target is rendered
    by fqname. Edges that don't originate from an unreachable node are
    ignored.
    """

    def _check(graph: nx.DiGraph, expected_edges: set[str]):
        actual = set()
        for src, dst in graph.edges:
            if not is_unreachable_node(src):
                continue
            start = src.position.start
            actual.add(f"<{start.line}:{start.column}> -> {dst.fqname}")
        assert actual == expected_edges

    return _check
