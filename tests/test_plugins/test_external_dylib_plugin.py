"""External (dylib) native plugin loading.

These tests are gated on the *dynamic-runtime* (shipped macOS/Linux) wheel, in
which the host already runs the shared runtime. Run them against such an install
with::

    DEAD_CST_PLUGIN_HOST=1 \\
    PLUGIN_DYLIB="$(dead-cst build-plugin)" \\
    PLUGIN_DYLIB_PER_FILE="$(dead-cst build-plugin \\
        examples/per_file_main_block/src/lib.rs)" \\
    PLUGIN_DYLIB_PER_FILE_DECORATED="$(dead-cst build-plugin \\
        examples/per_file_decorated/src/lib.rs)" \\
    pytest tests/test_plugins/test_external_dylib_plugin.py

``dead-cst build-plugin`` compiles the plugin (the bundled project-wide example
by default; pass a path for the per-file example) against the in-package runtime
dylib + the ``dead-cst-plugin-host`` rlib closure via ``rustc --extern``. The
dev/static build can't load external plugins (the runtime is statically linked,
not shared) and ``build-plugin`` won't run there (no in-package runtime dylib),
so the load tests skip; the rejection test runs anywhere. The full path is
exercised in CI by the publish workflow.
"""

from __future__ import annotations

import os

import pytest

_PLUGIN_HOST = bool(os.environ.get("DEAD_CST_PLUGIN_HOST"))
_PLUGIN_DYLIB = os.environ.get("PLUGIN_DYLIB", "")
_PLUGIN_DYLIB_PER_FILE = os.environ.get("PLUGIN_DYLIB_PER_FILE", "")
_PLUGIN_DYLIB_PER_FILE_DECORATED = os.environ.get("PLUGIN_DYLIB_PER_FILE_DECORATED", "")


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


# Source layout shared by the two decorated-plugin tests below: a Click
# command, an aliased Click group, an undecorated helper, and a file that
# never imports `click`.
_DECORATED_FILES = {
    "pkg/__init__.py": "",
    "pkg/app.py": """
    import click

    @click.command()
    def cli(): pass

    def helper(): pass
    """,
    # Aliased direct import resolves through the same matcher.
    "pkg/aliased.py": """
    from click import group as grp

    @grp()
    def root(): pass
    """,
    # No `click` import -> the presence guard skips this file entirely.
    "pkg/lib.py": "def untouched(): pass\n",
}


@pytest.mark.skipif(
    not (_PLUGIN_HOST and _PLUGIN_DYLIB_PER_FILE_DECORATED),
    reason="decorated per-file plugin requires PLUGIN_DYLIB_PER_FILE_DECORATED built via `dead-cst build-plugin`",
)
def test_external_per_file_decorated_plugin_keeps_decorated_alive(
    build_plugin_graph, reachable_fqnames
):
    """The decorated example is a *pair* exported from one cdylib: a per-file
    scanner emits a ``click/commands`` fact for each
    `@click.command`/`@click.group` decl (matched via `imports_any_module` +
    `decorated_decls`), and a project-wide keeper reads those facts and stamps
    the declared ``click/command`` node flag — keeping each command reachable.
    An undecorated helper stays dead, and a file that never imports `click` is
    untouched."""
    from dead_cst import _native as native

    plugins = native.load_native_plugins(_PLUGIN_DYLIB_PER_FILE_DECORATED)
    assert [p.name for p in plugins] == [
        "ExternalClickCommandScanner",
        "ExternalClickCommandKeeper",
    ]

    ctx = build_plugin_graph(_DECORATED_FILES, plugins)
    reached = reachable_fqnames(ctx)
    assert "pkg.app.cli" in reached
    assert "pkg.app.helper" not in reached
    assert "pkg.aliased.root" in reached
    assert "pkg.lib.untouched" not in reached


@pytest.mark.skipif(
    not (_PLUGIN_HOST and _PLUGIN_DYLIB_PER_FILE_DECORATED),
    reason="decorated per-file plugin requires PLUGIN_DYLIB_PER_FILE_DECORATED built via `dead-cst build-plugin`",
)
def test_external_per_file_decorated_plugin_topic_fact_round_trip(build_plugin_graph):
    """End-to-end of the topic/fact channel the decorated pair drives: the
    per-file scanner declares + publishes under the ``click/commands`` topic,
    and the project-wide keeper reads the collected facts. Asserts the topic
    is registered, each surviving Click decl produced one fact (value = the
    decl's fqname, pinned to a real node index), and the keeper's
    ``click/command`` flag landed on exactly those nodes."""
    from dead_cst import _native as native

    plugins = native.load_native_plugins(_PLUGIN_DYLIB_PER_FILE_DECORATED)
    ctx = build_plugin_graph(_DECORATED_FILES, plugins)

    # Both plugins declare the one topic (idempotently) -> a single handle.
    registry = ctx.topic_registry()
    assert len(registry) == 1
    name, handle, description = registry[0]
    assert name == "click/commands"
    assert description == "fqname of each top-level Click-decorated command/group decl"
    assert isinstance(handle, int)

    # The keeper's node flag is registered and resolves to a real bit.
    flag_bit = ctx.node_flag("click/command")
    assert flag_bit
    assert "click/command" in {entry[0] for entry in ctx.node_flag_registry()}

    # One fact per surviving Click decl. Each fact's value is the decl's
    # fqname and its decl_idx points at the matching node, which the keeper
    # stamped with the flag.
    facts = ctx.facts_for_topic("click/commands")
    assert {value for (_path, _decl_idx, value) in facts} == {"pkg.app.cli", "pkg.aliased.root"}
    assert {os.path.basename(path) for (path, _decl_idx, _value) in facts} == {
        "app.py",
        "aliased.py",
    }
    for _path, decl_idx, value in facts:
        assert decl_idx is not None
        node = ctx.nodes_at([decl_idx])[0]
        assert node.fqname == value
        assert node.flags & flag_bit, f"{node.fqname} missing the click/command flag"

    # An unknown topic yields no facts (and does not raise).
    assert ctx.facts_for_topic("click/unknown") == []


def test_load_rejects_dylib_without_manifest():
    """The airlock rejects a .so with no plugin manifest cleanly (no crash).
    ``_native`` itself is a real extension module but exports no manifest."""
    from dead_cst import _native as native

    with pytest.raises(RuntimeError, match="not a dead-cst plugin"):
        native.load_native_plugins(native.__file__)
