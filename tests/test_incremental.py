"""Incremental :meth:`Analysis.re_materialize` correctness tests.

Each test builds an initial graph, mutates one or more source files on
disk, calls ``re_materialize(dirty_files)``, and asserts the resulting
graph matches an independent fresh full build of the same end state.
The fresh build is the ground truth — incremental is correct iff it
produces the same edges and the same node set.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst import Analysis
from dead_cst.plugins import MainBlockPlugin


def _write(path: Path, src: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src).strip() + "\n")


def _edges(ctx) -> set[tuple[str, str]]:
    """Return ``{(src_fqname, dst_fqname)}`` for every edge in ``ctx``.

    Reduces edges to plain strings so the comparison is robust to
    node-identity differences between a re-materialized ctx and a
    fresh-built one. Edge flags are dropped — structural equivalence
    is what we care about here.
    """
    nodes = list(ctx.nodes())
    return {(nodes[src].fqname, nodes[dst].fqname) for src, dst, _flags in ctx.edges()}


def _node_keys(ctx) -> set[tuple[str, str, int]]:
    """``{(fqname, kind, start_line)}`` for every node in ``ctx``.

    Stable across rebuilds (independent of node-index allocation)."""
    return {(n.fqname, n.kind, n.start_line) for n in ctx.nodes()}


def _dead_fqnames(analysis: Analysis) -> set[str]:
    return {n.fqname for n in analysis.dead()}


def _build_fresh(tmp_path: Path) -> Analysis:
    fresh = Analysis(tmp_path)
    fresh.materialize_all()
    return fresh


def _assert_matches_fresh(analysis: Analysis, tmp_path: Path) -> None:
    """Ground-truth check: the incremental build must produce the same
    edges, node set, and dead set as a fresh full build of the same
    on-disk source state."""
    fresh = _build_fresh(tmp_path)
    assert _edges(analysis.materialize_all()) == _edges(fresh.materialize_all())
    assert _node_keys(analysis.materialize_all()) == _node_keys(fresh.materialize_all())
    assert _dead_fqnames(analysis) == _dead_fqnames(fresh)


def test_re_materialize_no_op(tmp_path):
    """A re_materialize with no source edits and no dirty files should
    produce a graph identical to the original build."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    ctx1 = analysis.materialize_all()
    edges_before = _edges(ctx1)

    ctx2 = analysis.re_materialize([])

    assert ctx1 is ctx2  # re_materialize rebuilds in place.
    assert _edges(ctx2) == edges_before
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_add_decl(tmp_path):
    """Add a new top-level function in a dirty file."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    analysis.re_materialize([tmp_path / "a.py"])

    _assert_matches_fresh(analysis, tmp_path)
    # Sanity: the new decl appears in the rebuilt graph.
    assert any(n.fqname == "a.g" for n in analysis.materialize_all().nodes())


def test_re_materialize_remove_decl(tmp_path):
    """Remove a decl from a dirty file; the old node and its edges
    should disappear from the rebuild."""
    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()
    assert any(n.fqname == "a.g" for n in analysis.materialize_all().nodes())

    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis.re_materialize([tmp_path / "a.py"])

    assert not any(n.fqname == "a.g" for n in analysis.materialize_all().nodes())
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_transitive_invalidation(tmp_path):
    """Rename ``a.f`` -> ``a.g`` with only ``a.py`` in the dirty list.

    ``b.py`` is unchanged on disk but its old ``from a import f`` no
    longer resolves. Salsa should still invalidate
    ``file_to_ref_edges(b)`` because that query previously read
    ``file_to_nodes(a)``, which changed. The fresh build is the
    ground-truth comparator.
    """
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def g(): pass\n")
    analysis.re_materialize([tmp_path / "a.py"])

    _assert_matches_fresh(analysis, tmp_path)
    # Sanity: a.f is gone, a.g exists.
    fqs = {n.fqname for n in analysis.materialize_all().nodes()}
    assert "a.f" not in fqs
    assert "a.g" in fqs


def test_re_materialize_multi_file_dirty(tmp_path):
    """Mutate two files at once and re_materialize with both in the
    dirty list."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    _write(tmp_path / "c.py", "def h(): pass\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef extra(): pass\n")
    _write(tmp_path / "c.py", "def h(): pass\ndef other(): pass\n")
    analysis.re_materialize([tmp_path / "a.py", tmp_path / "c.py"])

    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_with_entrypoint_plugin(tmp_path):
    """Re-materialize against a non-empty plugin set. Verifies that
    plugin ops are produced correctly on the second build (plugins are
    re-driven, not double-registered)."""
    _write(
        tmp_path / "a.py",
        """
        def f(): pass
        def g(): pass

        if __name__ == "__main__":
            f()
        """,
    )

    analysis = Analysis(tmp_path, plugins=[MainBlockPlugin()])
    analysis.materialize_all()
    dead_before = _dead_fqnames(analysis)
    assert "a.f" not in dead_before  # main block keeps it alive
    assert "a.g" in dead_before

    # Mutation: main block now also calls g, so g becomes alive.
    _write(
        tmp_path / "a.py",
        """
        def f(): pass
        def g(): pass

        if __name__ == "__main__":
            f()
            g()
        """,
    )
    analysis.re_materialize([tmp_path / "a.py"])

    dead_after = _dead_fqnames(analysis)
    assert "a.f" not in dead_after
    assert "a.g" not in dead_after

    fresh = Analysis(tmp_path, plugins=[MainBlockPlugin()])
    fresh.materialize_all()
    assert _dead_fqnames(analysis) == _dead_fqnames(fresh)
    assert _edges(analysis.materialize_all()) == _edges(fresh.materialize_all())


def test_re_materialize_requires_prior_materialize(tmp_path):
    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis = Analysis(tmp_path)
    with pytest.raises(RuntimeError, match="prior materialize_all"):
        analysis.re_materialize([tmp_path / "a.py"])
