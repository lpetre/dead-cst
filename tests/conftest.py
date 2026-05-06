import logging
import textwrap
from pathlib import Path

import networkx as nx
import pytest

from dead_cst import Analysis, EdgeFlags, SourceTree, SourceTreeFlags


def build_trees(
    paths: dict[Path, list[Path]] | Path,
) -> list[SourceTree]:
    """Translate a legacy ``{path: [search_paths]}`` mapping (or a single
    ``Path``) into a :class:`SourceTree` list.

    Each path becomes its own ``EXPORTED`` tree; the package name is
    derived from the path's stem (or ``"root"`` when empty), with an
    index suffix to keep names unique. Search paths are interpreted
    as references to other trees in the dict by path identity.
    """
    if isinstance(paths, Path):
        paths = {paths: []}

    used: dict[str, int] = {}

    def _pkg(path: Path) -> str:
        stem = path.name or "root"
        if stem in used:
            used[stem] += 1
            return f"{stem}_{used[stem]}"
        used[stem] = 0
        return stem

    keys = list(paths)
    pkg_for: dict[Path, str] = {}
    for p in keys:
        pkg_for[p.resolve()] = _pkg(p)

    out: list[SourceTree] = []
    for p, deps in paths.items():
        out.append(
            SourceTree(
                path=p.resolve(),
                package=pkg_for[p.resolve()],
                flags=SourceTreeFlags.EXPORTED,
                search_trees=tuple(d.resolve() for d in deps),
            )
        )
    return out


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
def build_decl_graph(tmp_path):
    def _make_graph(files: dict[str, str]) -> nx.MultiDiGraph:
        for filename, content in files.items():
            full_path = tmp_path / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(textwrap.dedent(content).strip())
        return Analysis(build_trees(tmp_path)).materialize_all()

    return _make_graph


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
