import dataclasses
import json
import logging
import textwrap

import networkx as nx
import pytest

from dead_cst import Analysis, EdgeFlags
from dead_cst.graph import NodeFlags, SymbolNode
from dead_cst.resolvers import ManualResolver


@pytest.fixture
def mark_entrypoint():
    """Replace ``node`` in ``graph`` with a copy carrying ``NodeFlags.ENTRYPOINT``.

    ``_find_reachable`` reads the flag straight off :class:`SymbolNode`,
    so tests that want to seed reachability without going through a
    plugin's observe/finalize pass swap in a flag-tagged copy via this
    fixture. The returned callable yields the new node so callers can
    re-bind their local reference.
    """

    def _mark(graph: nx.MultiDiGraph, node: SymbolNode) -> SymbolNode:
        new = dataclasses.replace(node, flags=node.flags | NodeFlags.ENTRYPOINT)
        if new == node:
            return node
        in_edges = list(graph.in_edges(node, keys=True, data=True))
        out_edges = list(graph.out_edges(node, keys=True, data=True))
        graph.remove_node(node)
        graph.add_node(new)
        for src, _, key, data in in_edges:
            graph.add_edge(src, new, key=key, **data)
        for _, dst, key, data in out_edges:
            graph.add_edge(new, dst, key=key, **data)
        return new

    return _mark


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
def visitor_warnings(caplog):
    """Capture WARNING records from the visitor and yield a message getter."""
    with caplog.at_level(logging.WARNING, logger="dead_cst._visitor"):
        yield lambda: [r.getMessage() for r in caplog.records]


@pytest.fixture
def make_analysis(tmp_path):
    """Build an :class:`Analysis` rooted at ``tmp_path`` with minimal boilerplate.

    The single positional argument is a list of :class:`ManualResolver`
    spec strings (``"."``, ``"pkg_a:pkg_b"``, etc.); defaults to
    ``["."]`` so the most common single-base case is just
    ``make_analysis()``. Any extra keyword arguments flow straight
    through to :class:`Analysis` (``plugins``, ``cache``,
    ``unreachable_detector``, ``workers``, or an explicit
    ``resolver=...`` to bypass :class:`ManualResolver` entirely).
    """

    def _make(specs: list[str] | None = None, **kwargs) -> Analysis:
        if "resolver" not in kwargs:
            kwargs["resolver"] = ManualResolver(specs=list(specs) if specs else ["."])
        return Analysis(tmp_path, **kwargs)

    return _make


@pytest.fixture
def build_decl_graph(tmp_path, make_analysis):
    def _make_graph(files: dict[str, str]) -> nx.MultiDiGraph:
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())
        return make_analysis().materialize_all()

    return _make_graph


@pytest.fixture
def write_notebook(tmp_path):
    """Write an nbformat-4 notebook to ``tmp_path``.

    Each entry in ``cells`` is either a ``str`` (becomes a code cell with
    that source, dedented) or a ``dict`` (written through unmodified, for
    testing markdown / raw / malformed shapes).
    """

    def _write(relpath: str, cells: list) -> None:
        nb_cells = []
        for cell in cells:
            if isinstance(cell, str):
                nb_cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": textwrap.dedent(cell).strip() + "\n",
                    }
                )
            else:
                nb_cells.append(cell)
        nb = {
            "cells": nb_cells,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        p = tmp_path / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(nb))

    return _write


def _is_dead_branch(attrs) -> bool:
    return bool(attrs.get("flags", EdgeFlags.NONE) & EdgeFlags.DEAD_BRANCH)


@pytest.fixture
def assert_edges():
    def _check(graph: nx.MultiDiGraph, expected_edges: set[str]):
        """Compare graph edges to expected 'a -> b' strings.

        Iterates the full edge set, including ``DEAD_BRANCH``-flagged
        edges. The flag is metadata-only -- default ``find_reachable``
        traverses these edges, and the live-graph view should reflect
        them. Tests that want only the dead-code references use
        :func:`assert_dead_branch_edges`. Parallel edges (same
        ``(u, v)`` pair, different attrs) collapse to one assertion
        entry; ``set`` deduping handles that automatically.
        """
        actual_edges = {f"{src.fqname} -> {dst.fqname}" for src, dst in graph.edges(keys=False)}
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

    def _check(graph: nx.MultiDiGraph, expected_edges: set[str]):
        actual_edges = {f"{_fmt(src)} -> {_fmt(dst)}" for src, dst in graph.edges(keys=False)}
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_dead_branch_edges():
    """Assert on edges flagged ``EdgeFlags.DEAD_BRANCH``.

    Format: ``"src.fqname -> dst.fqname"``. The previous synthetic-node
    model carried per-suite line/column on the edge source; that
    fidelity is intentionally dropped (see plan -- per-suite
    attribution is a payload-level concern, not a per-edge one).
    """

    def _check(graph: nx.MultiDiGraph, expected_edges: set[str]):
        actual = {
            f"{src.fqname} -> {dst.fqname}"
            for src, dst, attrs in graph.edges(data=True)
            if _is_dead_branch(attrs)
        }
        assert actual == expected_edges

    return _check
