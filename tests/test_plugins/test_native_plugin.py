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
    ctx = build_plugin_graph(
        files, [native.NativePlugin.main_block(), ModuleDundersPlugin()]
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
    import pytest

    with pytest.raises(TypeError):
        native.NativePlugin()
