"""External (dylib) native plugin loading.

These tests are gated on the *plugin-host* build. Run them with::

    DEAD_CST_PLUGIN_HOST=1 \\
    PLUGIN_DYLIB="$(uv run dead-cst build-plugin main_block_plugin)" \\
    uv run pytest tests/test_plugins/test_external_dylib_plugin.py

``dead-cst build-plugin`` does the prefer-dynamic build (shared runtime) and
installs the dynamic ``_native``. The default static wheel build can't load
external plugins (the runtime is statically linked, not shared), so the load
test skips there; the rejection test runs anywhere.
"""

from __future__ import annotations

import os

import pytest

_PLUGIN_HOST = bool(os.environ.get("DEAD_CST_PLUGIN_HOST"))
_PLUGIN_DYLIB = os.environ.get("PLUGIN_DYLIB", "")


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


def test_load_rejects_dylib_without_manifest():
    """The airlock rejects a .so with no plugin manifest cleanly (no crash).
    ``_native`` itself is a real extension module but exports no manifest."""
    from dead_cst import _native as native

    with pytest.raises(RuntimeError, match="not a dead-cst plugin"):
        native.load_native_plugins(native.__file__)
