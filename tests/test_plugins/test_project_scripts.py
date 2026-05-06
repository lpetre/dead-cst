"""Tests for :class:`ProjectScriptsPlugin`."""

from __future__ import annotations

from dead_cst import Analysis
from dead_cst.plugins import ProjectScriptsPlugin
from conftest import build_trees


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
    graph = Analysis(
        build_trees({tmp_path: []}),
        plugins=[ProjectScriptsPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.cli.main" in reachable_fqnames(graph)
