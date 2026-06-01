"""Tests for the native ``project_scripts`` plugin."""

from __future__ import annotations

import pytest

from dead_cst import _native as native


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
        [native.NativePlugin.project_scripts()],
    )
    assert "pkg.cli.main" in reachable_fqnames(graph)


def test_project_scripts_invalid_toml_surfaces_as_valueerror(build_plugin_graph):
    """A malformed ``pyproject.toml`` propagates out of ``materialize`` as a
    ``ValueError``. Regression guard for the swallow bug: the unified fallible
    ``ExternalPlugin.run`` boundary no longer discards a plugin failure."""
    with pytest.raises(ValueError, match="invalid TOML"):
        build_plugin_graph(
            {
                "pkg/__init__.py": "",
                "pkg/cli.py": "def main(): pass",
                "pyproject.toml": "[project.scripts]\nmytool =\n",
            },
            [native.NativePlugin.project_scripts()],
        )


def test_project_scripts_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("project_scripts")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "ProjectScriptsPlugin"
