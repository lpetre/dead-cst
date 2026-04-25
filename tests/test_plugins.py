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
    PytestPlugin,
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


def test_pytest_plugin_marks_test_functions(tmp_path):
    _write(
        tmp_path,
        {
            "tests/__init__.py": "",
            "tests/test_things.py": """
            def test_one(): pass
            def test_two(): pass
            def helper(): pass
            """,
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[PytestPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "tests.test_things.test_one" in reached
    assert "tests.test_things.test_two" in reached
    # ``helper`` is not a test function and is not referenced from one
    assert "tests.test_things.helper" not in reached


def test_pytest_plugin_recognizes_underscore_test_suffix(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/things_test.py": "def test_one(): pass",
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[PytestPlugin()],
        project_root=tmp_path,
    )
    assert "pkg.things_test.test_one" in _reachable_fqnames(graph)


def test_pytest_plugin_marks_test_classes(tmp_path):
    _write(
        tmp_path,
        {
            "tests/__init__.py": "",
            "tests/test_cls.py": """
            class TestThing:
                def test_a(self): pass
            class Helper:
                pass
            """,
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[PytestPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "tests.test_cls.TestThing" in reached
    assert "tests.test_cls.Helper" not in reached


def test_pytest_plugin_marks_conftest_decls(tmp_path):
    _write(
        tmp_path,
        {
            "tests/__init__.py": "",
            "tests/conftest.py": """
            import pytest

            @pytest.fixture
            def my_fixture():
                return 1

            def pytest_collection_modifyitems(config, items):
                pass

            collect_ignore = ["legacy.py"]
            """,
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[PytestPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "tests.conftest.my_fixture" in reached
    assert "tests.conftest.pytest_collection_modifyitems" in reached
    assert "tests.conftest.collect_ignore" in reached


def test_pytest_plugin_marks_decorated_fixtures_outside_conftest(tmp_path):
    _write(
        tmp_path,
        {
            "tests/__init__.py": "",
            "tests/fixtures.py": """
            import pytest
            from pytest import fixture

            @pytest.fixture
            def bare_fixture():
                return 1

            @pytest.fixture(scope="module")
            def parametrized_fixture():
                return 2

            @fixture
            def imported_fixture():
                return 3

            def not_a_fixture():
                return 4
            """,
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[PytestPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    assert "tests.fixtures.bare_fixture" in reached
    assert "tests.fixtures.parametrized_fixture" in reached
    assert "tests.fixtures.imported_fixture" in reached
    assert "tests.fixtures.not_a_fixture" not in reached


def test_pytest_plugin_ignores_non_test_modules(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/utils.py": """
            def test_helper(): pass
            class TestData: pass
            """,
        },
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[PytestPlugin()],
        project_root=tmp_path,
    )
    reached = _reachable_fqnames(graph)
    # ``utils.py`` isn't a pytest-discovered file even though its symbols
    # match the test_*/Test* naming
    assert "pkg.utils.test_helper" not in reached
    assert "pkg.utils.TestData" not in reached


def test_pytest_plugin_loads_via_load_plugin():
    from dead_cst import load_plugin

    plugin = load_plugin("pytest")
    assert isinstance(plugin, PytestPlugin)
