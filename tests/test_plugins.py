"""Tests for the edge-plugin surface and the builtin plugins."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from dead_cst import (
    DunderAllPlugin,
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
    ProjectScriptsPlugin,
    build_symbol_graph,
    find_reachable,
)


def _write(tmp_path: Path, files: dict[str, str]) -> None:
    for name, src in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def _reachable_fqnames(graph) -> set[str]:
    return {n.fqname for n in find_reachable(graph) if n.type != "synthetic"}


def test_no_plugins_means_nothing_reachable(tmp_path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = build_symbol_graph({tmp_path: []})
    assert find_reachable(graph) == set()


def test_explicit_entrypoint_by_fqname(tmp_path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.a.f"])],
        project_root=tmp_path,
    )
    assert "pkg.a.f" in _reachable_fqnames(graph)


def test_explicit_entrypoint_by_relpath(tmp_path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=["pkg/a.py"])],
        project_root=tmp_path,
    )
    assert {"pkg.a", "pkg.a.f"} <= _reachable_fqnames(graph)


def test_explicit_entrypoint_by_regex(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/entry.py": "from .a import f\nf()",
            "pkg/a.py": "def f(): pass",
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=[re.compile(r".*entry\.py")])],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "pkg.entry" in reached
    assert "pkg.a.f" in reached


def test_main_block_plugin_marks_module_entrypoint(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            def unused(): pass
            if __name__ == "__main__":
                main()
            """,
            "pkg/other.py": "def g(): pass",
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "pkg.script" in reached
    assert "pkg.script.main" in reached
    # `unused` has no reference from inside __main__, so stays dead
    assert "pkg.script.unused" not in reached
    # modules without a main block are not entrypoints
    assert "pkg.other" not in reached


def test_main_block_reversed_comparison(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            if "__main__" == __name__:
                main()
            """,
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin()],
        project_root=tmp_path,
    )
    assert "pkg.script" in _reachable_fqnames(graph)


def test_project_scripts_plugin(tmp_path):
    _write(
        tmp_path,
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
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[ProjectScriptsPlugin()],
        project_root=tmp_path,
    )
    assert "pkg.cli.main" in _reachable_fqnames(graph)


def test_dunder_all_plugin_keeps_all_alive(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": '__all__ = ["a"]\na = 1',
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[DunderAllPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "pkg.__all__" in reached
    # the listed symbol itself is *not* followed -- only __all__ stays alive.
    assert "pkg.a" not in reached


def test_plugins_compose(tmp_path):
    _write(
        tmp_path,
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
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), ProjectScriptsPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert {"pkg.runner", "pkg.cli", "pkg.cli.main"} <= reached


def test_unknown_plugin_raises():
    from dead_cst import load_plugin

    with pytest.raises(KeyError):
        load_plugin("does-not-exist")
