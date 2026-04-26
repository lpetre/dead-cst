"""Tests for the plugin surface itself: empty seeds, registry lookup,
composition across multiple plugins."""

from __future__ import annotations

import pytest

from dead_cst import (
    MainBlockPlugin,
    ProjectScriptsPlugin,
    build_symbol_graph,
    find_reachable,
)


def test_no_plugins_means_nothing_reachable(tmp_path, write_files):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = build_symbol_graph({tmp_path: []})
    assert find_reachable(graph) == set()


def test_plugins_compose(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/cli.py": "def main(): pass",
            "pkg/runner.py": """
            import pkg.cli
            if __name__ == "__main__":
                pkg.cli.main()
            """,
            "pyproject.toml": """
            [project]
            name = "x"
            [project.scripts]
            mytool = "pkg.cli:main"
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), ProjectScriptsPlugin()],
        project_root=tmp_path,
    )
    assert {"pkg.runner", "pkg.cli", "pkg.cli.main"} <= reachable_fqnames(graph)


def test_unknown_plugin_raises():
    from dead_cst import load_plugin

    with pytest.raises(KeyError):
        load_plugin("does-not-exist")
