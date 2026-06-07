"""Incremental :meth:`Analysis.re_materialize` correctness tests.

Each test builds an initial graph, mutates one or more source files on
disk, calls ``re_materialize(events)`` (with events from
``ctx.detect_changes()`` or an explicit list), and asserts the
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
    ctx = analysis.materialize_all()
    return {a.fqname for a in ctx.node_attrs(list(analysis.dead()))}


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

    ctx2 = analysis.re_materialize(ctx1.detect_changes())

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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

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

    analysis = Analysis(tmp_path, plugins=[native.NativePlugin.main_block()])
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
    analysis.re_materialize(analysis.materialize_all().detect_changes())

    dead_after = _dead_fqnames(analysis)
    assert "a.f" not in dead_after
    assert "a.g" not in dead_after

    _assert_matches_fresh(analysis, tmp_path, plugins=[native.NativePlugin.main_block()])


def test_re_materialize_requires_prior_materialize(tmp_path):
    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis = Analysis(tmp_path)
    with pytest.raises(RuntimeError, match="prior materialize_all"):
        analysis.re_materialize([])


def test_apply_changes_reports_zero_change(tmp_path):
    """``apply_changes`` returns whether any salsa revision advanced.

    A ``Changed`` event for an untouched file is a no-op (ty only
    bumps the file's revision when mtime / size differ) and reports
    ``False`` — the signal ``re_materialize`` uses to skip the rebuild
    entirely on the watcher/LSP hot path. A content edit reports
    ``True``. ``Rescan`` is conservatively always ``True`` today (the
    project re-walk sets inputs unconditionally), so rescan-driven
    callers never skip.
    """
    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()

    noop = [native.ChangeEvent.changed(str(tmp_path / "a.py"))]
    assert ctx.apply_changes(noop) is False

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    assert ctx.apply_changes(noop) is True
    # Bring the graph current again (apply_changes alone doesn't
    # rebuild) so the analysis isn't left stale for later asserts.
    analysis.re_materialize([native.ChangeEvent.rescan()])
    _assert_matches_fresh(analysis, tmp_path)


def test_re_materialize_zero_change_skips_rebuild(tmp_path):
    """A no-op re_materialize early-returns without re-running the
    build, and the graph it hands back matches a fresh build of the
    unchanged tree."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis = Analysis(tmp_path)
    ctx1 = analysis.materialize_all()
    edges_before = _edges(ctx1)

    ctx2 = analysis.re_materialize([native.ChangeEvent.changed(str(tmp_path / "a.py"))])
    assert ctx2 is ctx1
    assert _edges(ctx2) == edges_before
    _assert_matches_fresh(analysis, tmp_path)


# ---------------------------------------------------------------------------
# Incremental resolve-cache scenarios.
#
# ``re_materialize`` with explicit content-only ``Changed`` events takes
# the incremental resolve path: everything global is refolded from the
# per-file parts, but cross-file resolutions whose read set avoids the
# effectively-changed file set are reused from the previous build. Each
# scenario asserts the rebuilt graph matches a fresh ground-truth build
# (edges, node keys + flags, dead set) — the refold makes the output
# bit-identical by construction, so the comparison is exact.
# ---------------------------------------------------------------------------


def _node_flags(ctx) -> list[tuple[str, str, int, int]]:
    keys = [(n.fqname, n.kind, n.start_line, int(n.flags)) for n in ctx.nodes()]
    keys.sort()
    return keys


def _changed(tmp_path, *names):
    return [native.ChangeEvent.changed(str(tmp_path / n)) for n in names]


#: (initial tree, edits, files-to-signal) per incremental scenario.
#: Edits are content-only — every scenario stays on the resolve-cache
#: reuse path.
_INCREMENTAL_SCENARIOS = {
    "add_decl": (
        {"a.py": "def f(): pass\n", "b.py": "from a import f\nf()\n"},
        {"a.py": "def f(): pass\ndef g(): pass\n"},
        ["a.py"],
    ),
    "remove_decl": (
        {"a.py": "def f(): pass\ndef g(): pass\n", "b.py": "from a import f\nf()\n"},
        {"a.py": "def f(): pass\n"},
        ["a.py"],
    ),
    "rename_decl": (
        {"a.py": "def f(): pass\n", "b.py": "from a import f\nf()\n"},
        {"a.py": "def g(): pass\n"},
        ["a.py"],
    ),
    "unresolved_becomes_resolved": (
        {"a.py": "def other(): pass\n", "b.py": "from a import f\nf()\n"},
        {"a.py": "def other(): pass\ndef f(): pass\n"},
        ["a.py"],
    ),
    "decl_moves_lines": (
        {"a.py": "def f(): pass\n", "b.py": "from a import f\nf()\n"},
        {"a.py": "# pad\n# pad\n\ndef f(): pass\n"},
        ["a.py"],
    ),
    "star_chain_origin_change": (
        # c -> (star) b -> (star) a: editing the chain ORIGIN must
        # re-resolve the unchanged consumer c's binding through the
        # unchanged intermediary b. The reuse gate sees b as
        # effectively changed via its salsa-recomputed resolution-
        # surface fingerprint (b's payload derives from a's exports),
        # not via any hand-built dependency walk.
        {
            "a.py": "def f(): pass\n",
            "b.py": "from a import *\n",
            "c.py": "from b import f\nf()\n",
        },
        {"a.py": "def f(): pass\ndef extra(): pass\n"},
        ["a.py"],
    ),
    "star_origin_loses_name": (
        {
            "a.py": "def f(): pass\n",
            "b.py": "from a import *\n",
            "c.py": "from b import f\nf()\n",
        },
        {"a.py": "def other(): pass\n"},
        ["a.py"],
    ),
    "import_added_to_changed_file": (
        {"a.py": "def f(): pass\n", "b.py": "x = 1\n"},
        {"b.py": "from a import f\nf()\n"},
        ["b.py"],
    ),
    "shadowed_decl": (
        {"a.py": "def f(): pass\ndef f(): pass\n", "b.py": "from a import f\nf()\n"},
        {"a.py": "def f(): pass\ndef f(): pass\ndef f(): pass\n"},
        ["a.py"],
    ),
    "multi_file_batch": (
        {
            "a.py": "def f(): pass\n",
            "b.py": "from a import f\nf()\n",
            "c.py": "import b\n",
        },
        {"a.py": "def f(): pass\ndef g(): pass\n", "c.py": "import b\nimport a\n"},
        ["a.py", "c.py"],
    ),
    "package_init_change": (
        {
            "pkg/__init__.py": "x = 1\n",
            "pkg/sub.py": "y = 2\n",
            "main.py": "from pkg import sub\n",
        },
        {"pkg/__init__.py": "x = 1\nz = 3\n"},
        ["pkg/__init__.py"],
    ),
    "class_base_change": (
        {
            "base.py": "class Base: pass\nclass Other: pass\n",
            "kid.py": "from base import Base\nclass Kid(Base): pass\n",
        },
        {"base.py": "class Base: pass\nclass Other: pass\nclass Third: pass\n"},
        ["base.py"],
    ),
    "class_base_alias_retarget": (
        # `Alias = Base` rebinding to another class is invisible to the
        # export *surface* (same name, same target range) — the event
        # scope is what dirties the class-base read set here.
        {
            "base.py": "class A: pass\nclass B: pass\nAlias = A\n",
            "kid.py": "from base import Alias\nclass Kid(Alias): pass\n",
        },
        {"base.py": "class A: pass\nclass B: pass\nAlias = B\n"},
        ["base.py"],
    ),
    "dynamic_import_change": (
        {
            "a.py": "def f(): pass\n",
            "b.py": "import importlib\nm = importlib.import_module('a')\n",
        },
        {"a.py": "def f(): pass\ndef g(): pass\n"},
        ["a.py"],
    ),
    "pyi_twin_changed": (
        # impl.py gains an export the stub also declares: the stub's
        # stub-only ENTRYPOINT flag for that name must drop (the node
        # fill re-derives it from the twin's fresh exports every pass).
        {
            "impl.py": "def f(): pass\n",
            "impl.pyi": "def f() -> None: ...\ndef g() -> None: ...\n",
            "use.py": "from impl import f\nf()\n",
        },
        {"impl.py": "def f(): pass\ndef g(): pass\n"},
        ["impl.py"],
    ),
    "noqa_pin_toggles": (
        {"a.py": "def f(): pass\n", "b.py": "from a import f  # noqa: F401\n"},
        {"b.py": "from a import f\n"},
        ["b.py"],
    ),
}


@pytest.mark.parametrize("scenario", sorted(_INCREMENTAL_SCENARIOS))
def test_incremental_resolve_matches_fresh(tmp_path, scenario):
    initial, edits, signal = _INCREMENTAL_SCENARIOS[scenario]
    for name, src in initial.items():
        _write(tmp_path / name, src)
    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    for name, src in edits.items():
        _write(tmp_path / name, src)
    ctx = analysis.re_materialize(_changed(tmp_path, *signal))

    _assert_matches_fresh(analysis, tmp_path)
    fresh = _build_fresh(tmp_path)
    assert _node_flags(ctx) == _node_flags(fresh.materialize_all())


def test_incremental_resolve_reuses_clean_entries(tmp_path):
    """The observability contract: a content edit scoped by explicit
    ``Changed`` events re-resolves only entries whose read set touches
    the effectively-changed files, and reuses the rest."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    for i in range(5):
        _write(tmp_path / f"leaf{i}.py", "import json\nx = json.dumps\n")
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()
    resolved_full, reused_full = ctx._last_resolve_counts()
    assert resolved_full > 0
    assert reused_full == 0

    # Edit a leaf: its rows re-gather, but every memo entry's read set
    # avoids the changed file (stdlib imports read nothing), so nothing
    # re-resolves.
    _write(tmp_path / "leaf3.py", "import json\nx = json.dumps\ny = 1\n")
    analysis.re_materialize(_changed(tmp_path, "leaf3.py"))
    resolved, reused = ctx._last_resolve_counts()
    assert resolved < resolved_full
    assert reused > 0
    _assert_matches_fresh(analysis, tmp_path)

    # Edit the imported module: b's entries (read set touches a.py)
    # re-resolve, the leaves' don't.
    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    analysis.re_materialize(_changed(tmp_path, "a.py"))
    resolved, reused = ctx._last_resolve_counts()
    assert 0 < resolved < resolved_full
    assert reused > 0
    _assert_matches_fresh(analysis, tmp_path)


def test_incremental_resolve_rescan_takes_full_path(tmp_path):
    """A ``Rescan`` (or any non-``Changed`` event) cannot bound the
    blast radius — module resolution may flip — so the next build
    resolves everything (``reused == 0``) and is correct."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    analysis.re_materialize([native.ChangeEvent.rescan()])
    _resolved, reused = ctx._last_resolve_counts()
    assert reused == 0
    _assert_matches_fresh(analysis, tmp_path)


def test_incremental_resolve_created_file_takes_full_path(tmp_path):
    """File-set changes invalidate the cache: a new file can shadow a
    module name project-wide, which read sets don't model."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()

    _write(tmp_path / "new.py", "from a import f\nf()\n")
    analysis.re_materialize([native.ChangeEvent.created(str(tmp_path / "new.py"))])
    _resolved, reused = ctx._last_resolve_counts()
    assert reused == 0
    _assert_matches_fresh(analysis, tmp_path)
    assert any(n.fqname == "new" for n in ctx.nodes())


def test_incremental_resolve_unknown_path_takes_full_path(tmp_path):
    """A ``Changed`` naming a non-project path (config files) drops to
    the full path rather than guessing."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    events = _changed(tmp_path, "a.py") + [native.ChangeEvent.changed(str(tmp_path / "x.toml"))]
    analysis.re_materialize(events)
    _resolved, reused = ctx._last_resolve_counts()
    assert reused == 0
    _assert_matches_fresh(analysis, tmp_path)


def test_incremental_resolve_scope_accumulates_across_batches(tmp_path):
    """Two ``apply_changes`` batches before one rebuild merge their
    scopes — the rebuild sees both files as changed."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    _write(tmp_path / "c.py", "import b\n")
    analysis = Analysis(tmp_path)
    ctx = analysis.materialize_all()

    _write(tmp_path / "a.py", "def f(): pass\ndef g(): pass\n")
    assert ctx.apply_changes(_changed(tmp_path, "a.py")) is True
    _write(tmp_path / "c.py", "import b\nimport a\n")
    ctx2 = analysis.re_materialize(_changed(tmp_path, "c.py"))
    assert ctx2 is ctx
    _resolved, reused = ctx._last_resolve_counts()
    assert reused > 0
    _assert_matches_fresh(analysis, tmp_path)


def test_incremental_resolve_with_plugins(tmp_path):
    """The plugin pass refolds from scratch every build, so plugin
    flags/edges track the incremental rebuild exactly."""
    plugins = (native.NativePlugin.main_block(),)
    _write(tmp_path / "a.py", "def f(): pass\n\nif __name__ == '__main__':\n    f()\n")
    _write(tmp_path / "b.py", "from a import f\n")
    analysis = Analysis(tmp_path, plugins=plugins)
    analysis.materialize_all()

    # Remove the main block: the old entrypoint flag must not survive.
    _write(tmp_path / "a.py", "def f(): pass\n")
    ctx = analysis.re_materialize(_changed(tmp_path, "a.py"))
    _assert_matches_fresh(analysis, tmp_path, plugins=plugins)
    fresh = _build_fresh(tmp_path, plugins=plugins)
    assert _node_flags(ctx) == _node_flags(fresh.materialize_all())

    # And add one back in the other file.
    _write(tmp_path / "b.py", "from a import f\n\nif __name__ == '__main__':\n    f()\n")
    ctx = analysis.re_materialize(_changed(tmp_path, "b.py"))
    _assert_matches_fresh(analysis, tmp_path, plugins=plugins)
    fresh = _build_fresh(tmp_path, plugins=plugins)
    assert _node_flags(ctx) == _node_flags(fresh.materialize_all())


def test_incremental_resolve_repeated_edits(tmp_path):
    """Three consecutive incremental rebuilds stay exact."""
    _write(tmp_path / "a.py", "def f(): pass\n")
    _write(tmp_path / "b.py", "from a import f\nf()\n")
    analysis = Analysis(tmp_path)
    analysis.materialize_all()

    for body in (
        "def f(): pass\ndef g(): pass\n",
        "def g(): pass\n",
        "def f(): pass\n",
    ):
        _write(tmp_path / "a.py", body)
        analysis.re_materialize(_changed(tmp_path, "a.py"))
        _assert_matches_fresh(analysis, tmp_path)
