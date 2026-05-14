import json
import logging
import textwrap
from pathlib import Path
from typing import cast

import pytest
from libcst.metadata import CodePosition, CodeRange

from dead_cst import Analysis, EdgeFlags
from dead_cst._fqn import FixedFullyQualifiedNameProvider
from dead_cst._graphstore import SymbolGraph
from dead_cst.graph import Import, NodeFlags, SymbolNode
from dead_cst.resolvers import ManualResolver


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="libcst",
        choices=["libcst", "rust"],
        help=(
            "Backend the build_decl_graph fixture uses. 'libcst' (default) is the "
            "production pipeline; 'rust' routes through dead_cst_ty_native to surface "
            "missing-feature gaps. Tests that exercise Analysis-specific configuration "
            "(plugins, cache, unreachable_detector) bypass this fixture and stay on libcst."
        ),
    )


@pytest.fixture(scope="session")
def backend(request) -> str:
    return request.config.getoption("--backend")


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
def build_decl_graph(tmp_path, make_analysis, backend):
    """Build a SymbolGraph from inline ``{relpath: source}`` files.

    Dispatches to the configured ``--backend``. ``libcst`` (default) runs
    today's :class:`Analysis` pipeline; ``rust`` routes through the
    ``dead_cst_ty_native`` prototype to surface its feature gaps.
    """

    def _make_graph(files: dict[str, str]) -> SymbolGraph:
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())
        if backend == "rust":
            return _build_rust_graph(tmp_path)
        return make_analysis().materialize_all()

    return _make_graph


def _build_rust_graph(root: Path) -> SymbolGraph:
    """Build a SymbolGraph by routing every ``.py`` file under ``root`` through Rust.

    Uses libcst's :class:`FixedFullyQualifiedNameProvider` to derive each
    file's canonical FQN (matching the libcst pipeline), then asks the
    Rust prototype for one ``NativeGraph`` per file and accumulates the
    results into a single :class:`SymbolGraph`. Skips the suite if the
    native extension hasn't been built.

    Tests that hit this path are expected to fail wherever the Rust
    backend doesn't yet emit the edges / nodes the libcst visitor would
    — that is the whole point of the abstraction.
    """
    native = pytest.importorskip(
        "dead_cst_ty_native",
        reason="Run `maturin develop --manifest-path crates/dead-cst-ty-native/Cargo.toml` "
        "to build the prototype rust backend.",
    )

    files = sorted(p for p in root.rglob("*.py") if p.is_file())
    fqn_cache = FixedFullyQualifiedNameProvider.gen_cache(root, [str(f) for f in files], timeout=5)

    project = native.Project(str(root))
    graph = SymbolGraph()
    for f in files:
        fqn = fqn_cache[str(f)].name
        native_graph = project.build_file_graph(str(f), fqn)
        _accumulate_native(graph, native_graph)
    return graph


def _accumulate_native(graph: SymbolGraph, native_graph) -> None:
    """Merge a ``NativeGraph`` envelope into ``graph``.

    Mirrors ``tests/prototype/_bridge.accumulate``; inlined here so
    the conftest doesn't have to reach across into the prototype
    package (no ``tests/__init__.py`` today).
    """
    symbol_nodes: list[SymbolNode] = []
    for n in native_graph.nodes:
        sn = SymbolNode(
            fqname=n.fqname,
            type=cast("SymbolNode.type", n.kind),
            path=Path(n.path),
            position=CodeRange(
                CodePosition(n.start_line, n.start_column),
                CodePosition(n.end_line, n.end_column),
            ),
            imports=(
                Import(
                    module=n.imports.module,
                    decl=n.imports.decl,
                    star=n.imports.star,
                    speculative=n.imports.speculative,
                )
                if n.imports is not None
                else None
            ),
            flags=NodeFlags(n.flags),
        )
        graph.add(sn)
        symbol_nodes.append(sn)
    for src, dst, flags in native_graph.edges:
        graph.add_edge(symbol_nodes[src], symbol_nodes[dst], EdgeFlags(flags))


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
