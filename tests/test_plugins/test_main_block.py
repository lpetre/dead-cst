"""Tests for :class:`MainBlockPlugin`."""

from __future__ import annotations

from dead_cst import MainBlockPlugin, build_symbol_graph


def test_main_block_plugin_marks_module_entrypoint(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            def unused(): pass
            if __name__ == "__main__":
                main()
            """,
            "pkg/other.py": "def g(): pass",
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.script" in reached
    assert "pkg.script.main" in reached
    # `unused` has no reference from inside __main__, so stays dead
    assert "pkg.script.unused" not in reached
    # modules without a main block are not entrypoints
    assert "pkg.other" not in reached


def test_main_block_reversed_comparison(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            if "__main__" == __name__:
                main()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin()],
        project_root=tmp_path,
    )
    assert "pkg.script" in reachable_fqnames(graph)
