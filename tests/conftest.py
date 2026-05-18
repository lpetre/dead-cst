import json
import logging
import textwrap

import pytest

from dead_cst import Analysis, EdgeFlags
from dead_cst._graphstore import SymbolGraph
from dead_cst.resolvers import ManualResolver


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
def make_analysis(tmp_path):
    """Build an :class:`Analysis` rooted at ``tmp_path`` with minimal boilerplate."""

    def _make(specs: list[str] | None = None, **kwargs) -> Analysis:
        if "resolver" not in kwargs:
            kwargs["resolver"] = ManualResolver(specs=list(specs) if specs else ["."])
        return Analysis(tmp_path, **kwargs)

    return _make


@pytest.fixture
def build_decl_graph(tmp_path):
    """Build a SymbolGraph from inline ``{relpath: source}`` files."""

    def _make_graph(files: dict[str, str]) -> SymbolGraph:
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())
        return Analysis(tmp_path, resolver=ManualResolver(specs=["."])).materialize_all()

    return _make_graph


@pytest.fixture
def write_notebook(tmp_path):
    """Write an nbformat-4 notebook to ``tmp_path``."""

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
        actual_edges = {
            f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
        }
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_positional_edges():
    def _fmt(sym):
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
    def _check(graph: SymbolGraph, expected_edges: set[str]):
        actual = {
            f"{graph.node(u).fqname} -> {graph.node(v).fqname}"
            for u, v, payload in graph.raw.weighted_edge_list()
            if payload & EdgeFlags.DEAD_BRANCH
        }
        assert actual == expected_edges

    return _check


@pytest.fixture
def assert_dynamic_import_edges():
    def _check(graph: SymbolGraph, expected_edges: set[str]):
        actual = {
            f"{graph.node(u).fqname} -> {graph.node(v).fqname}"
            for u, v, payload in graph.raw.weighted_edge_list()
            if payload & EdgeFlags.DYNAMIC_IMPORT
        }
        assert actual == expected_edges

    return _check


# Suppress logging configured in tests
_ = logging
