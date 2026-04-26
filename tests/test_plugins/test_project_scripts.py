"""Tests for :class:`ProjectScriptsPlugin`."""

from __future__ import annotations

from dead_cst import ProjectScriptsPlugin, build_symbol_graph


def test_project_scripts_plugin(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/cli.py": "def main(): pass",
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
        plugins=[ProjectScriptsPlugin()],
        project_root=tmp_path,
    )
    assert "pkg.cli.main" in reachable_fqnames(graph)
