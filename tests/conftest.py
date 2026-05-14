import dataclasses
import json
import logging
import textwrap

import pytest

from dead_cst import Analysis, EdgeFlags
from dead_cst._graphstore import SymbolGraph
from dead_cst.graph import NodeFlags, SymbolNode
from dead_cst.resolvers import ManualResolver


@pytest.fixture
def mark_entrypoint():
    """Replace ``node`` in ``graph`` with a copy carrying ``NodeFlags.ENTRYPOINT``.

    Returns a callable ``(graph, node) -> SymbolNode`` so tests can
    re-bind their local reference to the new node.
    """

    def _mark(graph: SymbolGraph, node: SymbolNode) -> SymbolNode:
        new = dataclasses.replace(node, flags=node.flags | NodeFlags.ENTRYPOINT)
        if new == node:
            return node
        graph.relabel(node, new)
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
    def _make_graph(files: dict[str, str]) -> SymbolGraph:
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


@pytest.fixture
def assert_edges():
    def _check(graph: SymbolGraph, expected_edges: set[str]):
        """Compare graph edges to expected 'a -> b' strings.

        Iterates the full edge set, including ``DEAD_BRANCH``-flagged
        edges. The flag is metadata-only -- default ``find_reachable``
        traverses these edges, and the live-graph view should reflect
        them. Tests that want only the dead-code references use
        :func:`assert_dead_branch_edges`. Parallel edges (same
        ``(u, v)`` pair, different attrs) collapse to one assertion
        entry; ``set`` deduping handles that automatically.
        """
        actual_edges = {
            f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
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
    assertion.
    """

    def _fmt(sym):
        # Module nodes have a position too (covering the whole file) but
        # rendering it would just be noise. Leave modules as bare fqnames.
        if sym.type == "module":
            return sym.fqname
        start = sym.position.start
        return f"{sym.fqname}@{start.line}:{start.column}"

    def _check(graph: SymbolGraph, expected_edges: set[str]):
        actual_edges = {
            f"{_fmt(graph.node(u))} -> {_fmt(graph.node(v))}" for u, v in graph.raw.edge_list()
        }
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_dead_branch_edges():
    """Assert on edges flagged ``EdgeFlags.DEAD_BRANCH`` as ``"src.fqname -> dst.fqname"``."""

    def _check(graph: SymbolGraph, expected_edges: set[str]):
        actual = {
            f"{graph.node(u).fqname} -> {graph.node(v).fqname}"
            for u, v, payload in graph.raw.weighted_edge_list()
            if payload & EdgeFlags.DEAD_BRANCH
        }
        assert actual == expected_edges

    return _check
