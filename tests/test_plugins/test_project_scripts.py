"""Tests for :class:`ProjectScriptsPlugin`."""

from __future__ import annotations

from dead_cst.plugins import ProjectScriptsPlugin


def test_project_scripts_plugin(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/cli.py": "def main(): pass",
            "pyproject.toml": """
            [project]
            name = "x"
            [project.scripts]
            mytool = "pkg.cli:main"
            """,
        },
        [ProjectScriptsPlugin()],
    )
    assert "pkg.cli.main" in reachable_fqnames(graph)
