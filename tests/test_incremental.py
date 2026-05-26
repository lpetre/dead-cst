"""Incremental :meth:`Analysis.re_materialize` correctness tests.

Each test builds an initial graph, mutates one or more source files on
disk, calls ``re_materialize()`` (which autodetects via
``ctx.detect_changes()`` -> ty's rescan handler), and asserts the
resulting graph matches an independent fresh full build of the same
end state. The fresh build is the ground truth — incremental is
correct iff it produces the same edges and the same node set.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dead_cst import Analysis
from dead_cst import _native as native
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


def _node_keys(ctx) -> list[tuple[str, str, int]]:
    """``[(fqname, kind, start_line), ...]`` for every node in ``ctx``.

    Returns a list (not a set) so duplicate-node bugs don't get
    silently collapsed by set equality."""
    keys = [(n.fqname, n.kind, n.start_line) for n in ctx.nodes()]
    keys.sort()
    return keys


def _dead_fqnames(analysis: Analysis) -> set[str]:
    return {n.fqname for n in analysis.dead()}


def _build_fresh(tmp_path: Path, *, plugins=()) -> Analysis:
    fresh = Analysis(tmp_path, plugins=plugins)
    fresh.materialize_all()
    return fresh


def _assert_matches_fresh(analysis: Analysis, tmp_path: Path, *, plugins=()) -> None:
    """Ground-truth check: the incremental build must produce the same
    edges, node set, and dead set as a fresh full build of the same
    on-disk source state. Pass ``plugins=`` to mirror a plugin-aware
    Analysis."""
    fresh = _build_fresh(tmp_path, plugins=plugins)
    assert _edges(analysis.materialize_all()) == _edges(fresh.materialize_all())
    assert _node_keys(analysis.materialize_all()) == _node_keys(fresh.materialize_all())
    assert _dead_fqnames(analysis) == _dead_fqnames(fresh)


def test_re_materialize_no_op(tmp_path):
    """A re_materialize with no source edits should match a fresh
    build of the same state."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    ctx1 = analysis.materialize_all()
    edges_before = _edges(ctx1)

    ctx2 = analysis.re_materialize()

    assert ctx1 is ctx2  # re_materialize rebuilds in place.
    assert _edges(ctx2) == edges_before
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_add_decl(tmp_path):
    """Add a new top-level function to an existing file."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    analysis.re_materialize()

    _assert_matches_fresh(analysis, tmp_path)
    assert any(n.fqname == "a.g" for n in analysis.materialize_all().nodes())


def test_re_materialize_remove_decl(tmp_path):
    """Remove a decl from an existing file; the node and its edges
    should disappear."""
    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()
    assert any(n.fqname == "a.g" for n in analysis.materialize_all().nodes())

    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis.re_materialize()

    assert not any(n.fqname == "a.g" for n in analysis.materialize_all().nodes())
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_rename_decl(tmp_path):
    """Rename ``a.f`` to ``a.g``; ``b.py`` is unchanged on disk but
    its import no longer resolves. ty's rescan does an mtime-checked
    sync_all so ``b.py``'s file_to_ref_edges re-fires via the
    cross-file salsa dep."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def g(): pass\n")
    analysis.re_materialize()

    _assert_matches_fresh(analysis, tmp_path)
    fqs = {n.fqname for n in analysis.materialize_all().nodes()}
    assert "a.f" not in fqs
    assert "a.g" in fqs


def test_re_materialize_new_file(tmp_path):
    """Create a brand-new file between builds; autodetect should
    discover it via ty's project file re-walk."""
    _write(tmp_path / "a.py", "def f(): pass\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()
    initial_files = {Path(n.path).name for n in analysis.materialize_all().nodes()}
    assert "c.py" not in initial_files

    _write(tmp_path / "c.py", "def h(): pass\nh()\n")
    analysis.re_materialize()

    rebuilt_files = {Path(n.path).name for n in analysis.materialize_all().nodes()}
    assert "c.py" in rebuilt_files
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_deleted_file(tmp_path):
    """Delete a file between builds; the file's nodes must be gone in
    the rebuild."""
    _write(tmp_path / "a.py", "def f(): pass\nf()\n")
    _write(tmp_path / "b.py", "def g(): pass\ng()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()
    assert any(Path(n.path).name == "b.py" for n in analysis.materialize_all().nodes())

    (tmp_path / "b.py").unlink()
    analysis.re_materialize()

    assert not any(Path(n.path).name == "b.py" for n in analysis.materialize_all().nodes())
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_multi_file_mutation(tmp_path):
    """Mutate two files at once. Autodetect picks up both."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    _write(tmp_path / "c.py", "def h(): pass\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef extra(): pass\n")
    _write(tmp_path / "c.py", "def h(): pass\ndef other(): pass\n")
    analysis.re_materialize()

    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_explicit_events(tmp_path):
    """Pass explicit events instead of auto-detecting; the result must
    match the autodetect path."""
    _write(tmp_path / "a.py", "def f(): pass\nf()\n")

    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\nf()\n")
    analysis.re_materialize([native.ChangeEvent.changed(str(tmp_path / "a.py"))])

    _assert_matches_fresh(analysis, tmp_path)


def test_change_event_constructors_and_accessors():
    """Smoke-test the :class:`ChangeEvent` Python-facing API."""
    changed = native.ChangeEvent.changed("/abs/foo.py")
    assert changed.kind == "changed"
    assert changed.path == "/abs/foo.py"

    created = native.ChangeEvent.created("/abs/bar.py")
    assert created.kind == "created"
    assert created.path == "/abs/bar.py"

    deleted = native.ChangeEvent.deleted("/abs/baz.py")
    assert deleted.kind == "deleted"
    assert deleted.path == "/abs/baz.py"

    rescan = native.ChangeEvent.rescan()
    assert rescan.kind == "rescan"
    assert rescan.path is None

    # __repr__ surfaces both kind and path.
    assert "changed" in repr(changed)
    assert "foo.py" in repr(changed)
    assert "rescan" in repr(rescan)


def test_re_materialize_with_entrypoint_plugin(tmp_path):
    """Plugins run cleanly on the second build (no double-registration,
    plugin ops applied to the rebuilt graph)."""
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
    analysis.re_materialize()

    dead_after = _dead_fqnames(analysis)
    assert "a.f" not in dead_after
    assert "a.g" not in dead_after

    _assert_matches_fresh(analysis, tmp_path, plugins=[MainBlockPlugin()])


def test_re_materialize_requires_prior_materialize(tmp_path):
    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis = Analysis(tmp_path)
    with pytest.raises(RuntimeError, match="prior materialize_all"):
        analysis.re_materialize()
