from __future__ import annotations

import json
import logging
import textwrap
from typing import TYPE_CHECKING

import pytest

from dead_cst import Analysis, EdgeFlags
from dead_cst.resolvers import ManualResolver

if TYPE_CHECKING:
    from dead_cst import _native as native


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
    """Materialise inline ``{relpath: source}`` files and return the live
    :class:`native.ProjectContext` — same value
    :meth:`Analysis.materialize_all` returns to production callers.
    """

    def _make_graph(files: dict[str, str]) -> "native.ProjectContext":
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
    def _check(graph: "native.ProjectContext", expected_edges: set[str]):
        nodes = graph.nodes()
        actual_edges = {f"{nodes[u].fqname} -> {nodes[v].fqname}" for u, v, _ in graph.edges()}
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_positional_edges():
    def _fmt(sym):
        if sym.kind == "module":
            return sym.fqname
        return f"{sym.fqname}@{sym.start_line}:{sym.start_column}"

    def _check(graph: "native.ProjectContext", expected_edges: set[str]):
        nodes = graph.nodes()
        actual_edges = {f"{_fmt(nodes[u])} -> {_fmt(nodes[v])}" for u, v, _ in graph.edges()}
        assert actual_edges == expected_edges

    return _check


@pytest.fixture
def assert_dead_branch_edges():
    def _check(graph: "native.ProjectContext", expected_edges: set[str]):
        nodes = graph.nodes()
        actual = {
            f"{nodes[u].fqname} -> {nodes[v].fqname}"
            for u, v, payload in graph.edges()
            if payload & EdgeFlags.DEAD_BRANCH
        }
        assert actual == expected_edges

    return _check


@pytest.fixture
def assert_dynamic_import_edges():
    def _check(graph: "native.ProjectContext", expected_edges: set[str]):
        nodes = graph.nodes()
        actual = {
            f"{nodes[u].fqname} -> {nodes[v].fqname}"
            for u, v, payload in graph.edges()
            if payload & EdgeFlags.DYNAMIC_IMPORT
        }
        assert actual == expected_edges

    return _check


@pytest.fixture
def successors_of():
    """Return the one-hop successors of ``node`` in ``ctx`` (test-only).

    Production code uses :meth:`ProjectContext.descendants` for the
    transitive closure; the one-hop accessor lives here because only
    tests reach for it.
    """

    def _of(ctx: "native.ProjectContext", node) -> list:
        nodes = ctx.nodes()
        idx = nodes.index(node)
        return [nodes[v] for u, v, _ in ctx.edges() if u == idx]

    return _of


@pytest.fixture
def predecessors_of():
    def _of(ctx: "native.ProjectContext", node) -> list:
        nodes = ctx.nodes()
        idx = nodes.index(node)
        return [nodes[u] for u, v, _ in ctx.edges() if v == idx]

    return _of


@pytest.fixture
def has_edge():
    def _has(ctx: "native.ProjectContext", src, dst) -> bool:
        nodes = ctx.nodes()
        s, d = nodes.index(src), nodes.index(dst)
        return any(u == s and v == d for u, v, _ in ctx.edges())

    return _has


@pytest.fixture
def edge_flags_between():
    def _flags(ctx: "native.ProjectContext", src, dst) -> list[int]:
        nodes = ctx.nodes()
        s, d = nodes.index(src), nodes.index(dst)
        return [f for u, v, f in ctx.edges() if u == s and v == d]

    return _flags


# Suppress logging configured in tests
_ = logging
