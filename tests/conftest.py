from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dead_cst import Analysis, EdgeFlags

if TYPE_CHECKING:
    from dead_cst import _native as native


@pytest.fixture
def write_files(tmp_path):
    """Write a ``{relpath: source}`` mapping under ``tmp_path``."""

    def _write(files: dict[str, str]) -> None:
        for name, src in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src).strip() + "\n")

    return _write


def _python_version_tag() -> str:
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


@pytest.fixture
def make_workspace_venv(tmp_path):
    """Create a fake venv populated with editable ``.pth`` entries.

    Mirrors what ``uv sync --all-packages`` produces: one ``.pth``
    file per first-party member, each containing the absolute path
    to that member's published source dir. ty reads these on
    site-packages traversal and uses them as module-resolution
    search paths, which is what makes ``from libx import foo``
    resolve to ``packages/libx/src/libx/__init__.py`` (and what
    makes the file at that path mount as module ``libx`` rather
    than ``packages.libx.src.libx``).

    ``members`` maps each member name to its published dir, given
    as a path RELATIVE to ``tmp_path``. The dir is what consumers
    put on their search path -- e.g. for ``packages/libx/src/libx/``
    (where ``libx/__init__.py`` lives), the published dir is
    ``packages/libx/src``.

    Returns the absolute path of the new ``.venv`` directory.
    """

    def _make(members: dict[str, str]) -> Path:
        site_packages = tmp_path / ".venv" / "lib" / _python_version_tag() / "site-packages"
        site_packages.mkdir(parents=True)
        for name, exported in members.items():
            exported_abs = (tmp_path / exported).resolve()
            (site_packages / f"_editable_impl_{name}.pth").write_text(f"{exported_abs}\n")
        return tmp_path / ".venv"

    return _make


def _members_from_specs(specs: list[str]) -> dict[str, str]:
    """Parse legacy ``"name:dep1,dep2"`` specs into a ``{name: <name>}``
    mapping (deps ignored -- the new model encodes them via the venv's
    ``.pth`` files, not the resolver). Each member's published dir is
    just ``tmp_path / name`` -- if a test writes ``pkg_a/A/...``, the
    wheel content is ``A`` and ``pkg_a/`` is what consumers put on
    their search path."""
    members: dict[str, str] = {}
    for spec in specs:
        name = spec.split(":", 1)[0].strip()
        if not name or name == ".":
            continue
        members[name] = name
    return members


@pytest.fixture
def make_analysis(tmp_path, make_workspace_venv):
    """Build an :class:`Analysis` rooted at ``tmp_path``.

    Accepts a legacy ``specs`` positional arg (a list like
    ``["pkg_a", "pkg_b:pkg_a"]``) for multi-package tests -- each
    entry's first segment becomes a ``.pth``-registered member with
    its published dir at ``tmp_path / name``. Deps are ignored (the
    venv-driven model has no dep concept). Single-package callers
    just pass ``Analysis``-kwargs.
    """

    def _make(specs: list[str] | None = None, **kwargs) -> Analysis:
        if specs:
            members = _members_from_specs(specs)
            if members and "venv" not in kwargs:
                kwargs["venv"] = make_workspace_venv(members)
        return Analysis(tmp_path, **kwargs)

    return _make


@pytest.fixture
def build_decl_graph(tmp_path):
    """Materialise inline ``{relpath: source}`` files and return the live
    :class:`native.ProjectContext` -- same value
    :meth:`Analysis.materialize_all` returns to production callers.
    Single-package shape: no venv, ty auto-discovers ``project_root``.
    """

    def _make_graph(files: dict[str, str]) -> "native.ProjectContext":
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())
        return Analysis(tmp_path).materialize_all()

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

    The transitive closure lives in :func:`descendants_of`; this one-hop
    accessor stays here because only tests reach for it.
    """

    def _of(ctx: "native.ProjectContext", node) -> list:
        nodes = ctx.nodes()
        idx = nodes.index(node)
        return [nodes[v] for u, v, _ in ctx.edges() if u == idx]

    return _of


@pytest.fixture
def descendants_of():
    """Return the transitive forward closure of ``node`` in ``ctx`` as
    ``SymbolNode``s (test-only).

    The native surface is index-keyed (``descendants_indices``); this
    fixture maps the seed node to its index, runs the closure, and maps
    the result back to nodes so tests can assert on ``SymbolNode``s.
    """

    def _of(ctx: "native.ProjectContext", node, *, skip_flags: int = 0) -> list:
        nodes = ctx.nodes()
        idx = nodes.index(node)
        return [nodes[i] for i in ctx.descendants_indices(idx, skip_flags=skip_flags)]

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
