"""External (dylib) native plugin loading.

These tests are gated on the *plugin-host* build. Run them with::

    DEAD_CST_PLUGIN_HOST=1 \\
    PLUGIN_DYLIB="$(uv run dead-cst build-plugin)" \\
    PLUGIN_DYLIB_PER_FILE="$(uv run dead-cst build-plugin \\
        examples/per_file_main_block/src/lib.rs)" \\
    uv run pytest tests/test_plugins/test_external_dylib_plugin.py

``dead-cst build-plugin`` compiles the plugin (the bundled project-wide example
by default; pass a path for the per-file example) against the prebuilt runtime
via ``rustc --extern`` and installs the dynamic ``_native``. The default static
wheel build can't load external plugins (the runtime is statically linked, not
shared), so the load tests skip there; the rejection test runs anywhere.
"""

from __future__ import annotations

import os

import pytest

_PLUGIN_HOST = bool(os.environ.get("DEAD_CST_PLUGIN_HOST"))
_PLUGIN_DYLIB = os.environ.get("PLUGIN_DYLIB", "")
_PLUGIN_DYLIB_PER_FILE = os.environ.get("PLUGIN_DYLIB_PER_FILE", "")


@pytest.mark.skipif(
    not _PLUGIN_HOST,
    reason="external dylib plugins require the plugin-host build (run via `dead-cst build-plugin`)",
)
def test_external_main_block_plugin_keeps_main_alive(build_plugin_graph, reachable_fqnames):
    """The example external plugin (a separately-built dylib linking the
    shared runtime) loads through the airlock and contributes to the graph:
    the ``if __name__`` block keeps ``main`` reachable, ``unused`` stays dead."""
    from dead_cst import _native as native

    plugins = native.load_native_plugins(_PLUGIN_DYLIB)
    assert [p.name for p in plugins] == ["ExternalMainBlockPlugin"]

    files = {
        "pkg/__init__.py": "",
        "pkg/script.py": """
        def main(): pass
        def unused(): pass
        if __name__ == "__main__":
            main()
        """,
    }
    ctx = build_plugin_graph(files, plugins)
    reached = reachable_fqnames(ctx)
    assert "pkg.script.main" in reached
    assert "pkg.script.unused" not in reached


@pytest.mark.skipif(
    not (_PLUGIN_HOST and _PLUGIN_DYLIB_PER_FILE),
    reason="per-file external plugin requires PLUGIN_DYLIB_PER_FILE built via `dead-cst build-plugin`",
)
def test_external_per_file_plugin_keeps_main_alive(build_plugin_graph, reachable_fqnames):
    """The per-file external plugin (`per_file()` -> Some) is dispatched
    through the salsa-cached per-file query, one `run_on_file` per file, and
    produces the same reachability as the project-wide variant: the
    ``if __name__`` block keeps ``main`` alive, ``unused`` stays dead."""
    from dead_cst import _native as native

    plugins = native.load_native_plugins(_PLUGIN_DYLIB_PER_FILE)
    assert [p.name for p in plugins] == ["ExternalPerFileMainBlockPlugin"]

    files = {
        "pkg/__init__.py": "",
        "pkg/script.py": """
        def main(): pass
        def unused(): pass
        if __name__ == "__main__":
            main()
        """,
        # A file with no main block must contribute nothing.
        "pkg/lib.py": "def helper(): pass\n",
    }
    ctx = build_plugin_graph(files, plugins)
    reached = reachable_fqnames(ctx)
    assert "pkg.script.main" in reached
    assert "pkg.script.unused" not in reached
    assert "pkg.lib.helper" not in reached


def test_load_rejects_dylib_without_manifest():
    """The airlock rejects a .so with no plugin manifest cleanly (no crash).
    ``_native`` itself is a real extension module but exports no manifest."""
    from dead_cst import _native as native

    with pytest.raises(RuntimeError, match="not a dead-cst plugin"):
        native.load_native_plugins(native.__file__)
