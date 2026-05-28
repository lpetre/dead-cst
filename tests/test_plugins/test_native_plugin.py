"""Tests for the prototype rust-side ``NativePlugin``.

A native plugin is a pyo3 wrapper around a rust :trait:`NativePluginImpl`
that the harness drives through the same ``ThreadPoolExecutor`` /
``CollectedOps`` / ``apply_ops_batched`` flow as a Python plugin —
the only difference is that the per-op extraction step is replaced
by a direct ``Vec<PreparedOp>`` push from rust.

These tests pin: parity with the Python equivalent, registration
acceptance by the ``Analysis`` harness, mix-and-match with Python
plugins in the same registration list.
"""

from __future__ import annotations

from dead_cst import _native as native
from dead_cst.plugins import MainBlockPlugin


def test_native_main_block_matches_python_plugin(build_plugin_graph, reachable_fqnames):
    """``NativePlugin.main_block()`` produces the same reachable set as
    ``MainBlockPlugin()`` — same synthetic entrypoint per main block,
    same edges, same alive set."""
    files = {
        "pkg/__init__.py": "",
        "pkg/script.py": """
        def main(): pass
        def unused(): pass
        if __name__ == "__main__":
            main()
        """,
        "pkg/other.py": "def g(): pass",
    }
    py_ctx = build_plugin_graph(files, [MainBlockPlugin()])
    rs_ctx = build_plugin_graph(files, [native.NativePlugin.main_block()])
    assert reachable_fqnames(py_ctx) == reachable_fqnames(rs_ctx)


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


def test_native_plugin_mixed_with_python_plugins(build_plugin_graph, reachable_fqnames):
    """A native plugin and a Python plugin in the same registration
    list both run; the harness routes each to the appropriate path
    transparently."""
    from dead_cst.plugins import ModuleDundersPlugin

    files = {
        "pkg/__init__.py": "__all__ = ['x']\nx = 1\n",
        "pkg/script.py": """
        def main(): pass
        if __name__ == "__main__":
            main()
        """,
    }
    # Mix: rust native main-block + python module-dunders.
    ctx = build_plugin_graph(files, [native.NativePlugin.main_block(), ModuleDundersPlugin()])
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
    import pytest

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
