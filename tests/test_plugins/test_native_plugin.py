"""Tests for the rust-side ``NativePlugin`` harness wrapper.

A native plugin is a pyo3 wrapper around a rust :trait:`NativePluginImpl`
that the harness drives through the same ``CollectedOps`` /
``apply_ops_batched`` flow used for every plugin, with the per-op
extraction replaced by a direct ``Vec<PreparedOp>`` push from rust.

These tests pin harness integration: registration acceptance by the
``Analysis`` harness, running several native plugins in one list, the
``name`` attribute used by progress logs, per-file salsa caching, and the
builtin name -> native lookup the CLI resolves through.
"""

from __future__ import annotations

import pytest

from dead_cst import _native as native


def test_native_main_block_emits_entrypoint_marker(build_plugin_graph):
    """The synthetic ``<__main__>:<module>`` marker still lands in the
    graph — verify the node is interned and carries ``ENTRYPOINT``."""
    ctx = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            if __name__ == "__main__":
                main()
            """,
        },
        [native.NativePlugin.main_block()],
    )
    markers = [n for n in ctx.nodes() if n.fqname == "<__main__>:pkg.script"]
    assert len(markers) == 1
    assert markers[0].flags & native.NodeFlags.ENTRYPOINT


def test_native_plugin_name_attribute():
    """``NativePlugin.name`` mirrors the conventional Python plugin
    name, so harness progress logs and ``progress_callback`` events
    are indistinguishable between the two impls."""
    plugin = native.NativePlugin.main_block()
    assert plugin.name == "MainBlockPlugin"


def test_multiple_native_plugins_in_one_list(build_plugin_graph, reachable_fqnames):
    """Several native plugins in the same registration list all run; the
    harness routes each through the shared collect/apply flow and folds
    every plugin's ops into one graph."""
    files = {
        "pkg/__init__.py": "__all__ = ['x']\nx = 1\n",
        "pkg/script.py": """
        def main(): pass
        if __name__ == "__main__":
            main()
        """,
    }
    ctx = build_plugin_graph(
        files,
        [native.NativePlugin.main_block(), native.NativePlugin.module_dunders()],
    )
    reached = reachable_fqnames(ctx)
    # Main-block contribution.
    assert "pkg.script.main" in reached
    # Module-dunders contribution.
    assert "pkg.__all__" in reached
    assert "pkg.x" in reached


def test_native_plugin_no_main_block_no_ops(build_plugin_graph):
    """A project without any ``if __name__ == "__main__":`` block
    produces no synthetic markers — same shape as the Python
    equivalent's empty path."""
    ctx = build_plugin_graph(
        {"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"},
        [native.NativePlugin.main_block()],
    )
    main_markers = [n for n in ctx.nodes() if n.fqname.startswith("<__main__>:")]
    assert main_markers == []


def test_native_plugin_cannot_be_directly_instantiated():
    """``NativePlugin()`` (no-arg) should raise — the wrapper requires
    a factory like :meth:`NativePlugin.main_block` to bind to a
    concrete impl. Pinned so the constructor doesn't silently
    succeed with no inner impl."""
    with pytest.raises(TypeError):
        native.NativePlugin()


# ---------------------------------------------------------------------------
# Per-file salsa caching: the per-file MainBlockPlugin is invoked through a
# salsa-tracked query keyed on (file, kind). Unchanged files reuse the
# cached ops across re_materialize without re-running the impl. These tests
# use the rust-side run-counter (``_main_block_run_count`` / reset) to assert
# the cache actually fires.
# ---------------------------------------------------------------------------


def _write(path, src: str) -> None:
    import textwrap

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src).strip() + "\n")


def test_per_file_main_block_caches_unchanged_files(tmp_path):
    """Editing one file should re-run the per-file MainBlock plugin for
    *only* that file on ``re_materialize`` — every other file's marker is
    served from the salsa cache."""
    from dead_cst import Analysis

    _write(tmp_path / "pkg/__init__.py", "")
    _write(
        tmp_path / "pkg/a.py",
        """
        def main(): pass
        if __name__ == "__main__":
            main()
        """,
    )
    _write(
        tmp_path / "pkg/b.py",
        """
        def serve(): pass
        if __name__ == "__main__":
            serve()
        """,
    )
    _write(tmp_path / "pkg/c.py", "def helper(): pass\n")

    analysis = Analysis(tmp_path, plugins=[native.NativePlugin.main_block()])
    native._reset_main_block_run_count()
    analysis.materialize_all()
    first_pass = native._main_block_run_count()
    # Cold build: the impl runs once per project file (a, b, c, __init__).
    assert first_pass >= 3

    # Edit only c.py (no main block either before or after). On
    # re_materialize, a.py and b.py must hit the salsa cache; only c.py
    # re-runs the impl.
    native._reset_main_block_run_count()
    _write(tmp_path / "pkg/c.py", "def helper(): pass\ndef extra(): pass\n")
    analysis.re_materialize(analysis.materialize_all().detect_changes())
    second_pass = native._main_block_run_count()
    assert second_pass == 1, (
        f"expected exactly 1 per-file re-run (the edited c.py), got {second_pass} "
        "— salsa cache for unchanged a.py/b.py/__init__.py should have served their ops"
    )


def test_per_file_main_block_cache_invalidates_on_edit(tmp_path):
    """Editing the main-block file itself re-runs its per-file plugin and
    reflects the new block contents."""
    from dead_cst import Analysis

    _write(tmp_path / "pkg/__init__.py", "")
    _write(
        tmp_path / "pkg/a.py",
        """
        def main(): pass
        if __name__ == "__main__":
            main()
        """,
    )

    analysis = Analysis(tmp_path, plugins=[native.NativePlugin.main_block()])
    ctx = analysis.materialize_all()
    assert any(n.fqname == "<__main__>:pkg.a" for n in ctx.nodes())

    # Remove the main block entirely; the marker should disappear after
    # re_materialize (cache miss on the edited file).
    native._reset_main_block_run_count()
    _write(tmp_path / "pkg/a.py", "def main(): pass\n")
    ctx2 = analysis.re_materialize(analysis.materialize_all().detect_changes())
    assert native._main_block_run_count() >= 1  # a.py re-ran
    assert not any(n.fqname == "<__main__>:pkg.a" for n in ctx2.nodes())


# ---------------------------------------------------------------------------
# Harness plumbing: plugin-list type validation and the builtin name ->
# native lookup the CLI resolves through. Per-plugin behaviour lives in
# test_main_block.py / test_module_dunders.py / test_init_subclass.py /
# test_unittest.py.
# ---------------------------------------------------------------------------


def test_non_native_plugin_in_list_raises_typeerror(tmp_path):
    """``Analysis`` type-validates its plugins list up front — a non-
    ``NativePlugin`` entry raises ``TypeError`` before any graph build,
    so a ``Pluign()`` typo doesn't slip past."""
    from dead_cst.analyze import Analysis

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("def foo(): pass\n")
    analysis = Analysis(tmp_path, plugins=[object()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Expected a dead_cst._native.NativePlugin"):
        analysis.materialize_all()
    # The graph itself was never built — Analysis._ctx stays None.
    assert analysis._ctx is None  # noqa: SLF001 — testing the invariant


def test_builtin_native_plugin_registry():
    """``_builtin_native_plugin(name)`` resolves ported built-ins to their
    native impl, and returns ``None`` for names not (yet) ported."""
    assert native._builtin_native_plugin("main_block").name == "MainBlockPlugin"
    assert native._builtin_native_plugin("module_dunders").name == "ModuleDundersPlugin"
    assert native._builtin_native_plugin("init_subclass").name == "InitSubclassPlugin"
    assert native._builtin_native_plugin("does_not_exist") is None
