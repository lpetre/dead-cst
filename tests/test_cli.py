"""Tests for :mod:`dead_cst.cli`.

The CLI is a thin Typer wrapper around the public API: every command
calls :func:`build_symbol_graph` then renders the result. These tests
exercise the wrapper -- argument parsing, plugin/resolver composition,
output formatting (text and JSON), exit codes, and the confirm/dry-run
prompts on ``remove`` -- against tiny in-memory projects under
``tmp_path``. Tests for the underlying analysis live in
``test_declarations`` / ``test_codemod`` / etc.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path

import pytest
from libcst.metadata import CodePosition, CodeRange
from typer.testing import CliRunner

from dead_cst._plugins import (
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
    ModuleDundersPlugin,
)
from dead_cst._symbols import SymbolNode
from dead_cst.cli import (
    _is_dunder_all,
    _is_external_dep,
    app,
    build_plugins,
    parse_entrypoint,
    parse_paths,
    resolve_paths,
    setup_logging,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path):
    """Write a ``{relpath: source}`` map under ``tmp_path`` and return it.

    Sources are dedented and have a leading newline stripped so triple-
    quoted literals can line up visually inside test cases.
    """

    def _make(files: dict[str, str]) -> Path:
        for rel, src in files.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            text = textwrap.dedent(src)
            if text.startswith("\n"):
                text = text[1:]
            path.write_text(text)
        return tmp_path

    return _make


def _pos() -> CodeRange:
    return CodeRange(start=CodePosition(line=1, column=0), end=CodePosition(line=1, column=1))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParseEntrypoint:
    def test_plain_string_returned_as_is(self):
        assert parse_entrypoint("pkg.mod.func") == "pkg.mod.func"

    def test_path_like_string_returned_as_is(self):
        assert parse_entrypoint("src/main.py") == "src/main.py"

    def test_re_prefix_compiles_pattern(self):
        result = parse_entrypoint("re:.*\\.py$")
        assert isinstance(result, re.Pattern)
        assert result.pattern == ".*\\.py$"

    def test_re_prefix_with_empty_pattern(self):
        result = parse_entrypoint("re:")
        assert isinstance(result, re.Pattern)
        assert result.pattern == ""


class TestParsePaths:
    def test_empty_list_returns_root_only(self, tmp_path):
        assert parse_paths(tmp_path, []) == {tmp_path: []}

    def test_base_only_spec(self, tmp_path):
        assert parse_paths(tmp_path, ["src"]) == {tmp_path / "src": []}

    def test_base_with_deps(self, tmp_path):
        result = parse_paths(tmp_path, ["src:dep1,dep2"])
        assert result == {tmp_path / "src": [tmp_path / "dep1", tmp_path / "dep2"]}

    def test_dep_whitespace_is_stripped(self, tmp_path):
        result = parse_paths(tmp_path, ["src: dep1 , dep2 "])
        assert result == {tmp_path / "src": [tmp_path / "dep1", tmp_path / "dep2"]}

    def test_trailing_comma_drops_empty_dep(self, tmp_path):
        result = parse_paths(tmp_path, ["src:dep1,"])
        assert result == {tmp_path / "src": [tmp_path / "dep1"]}

    def test_multiple_specs_merge_into_dict(self, tmp_path):
        result = parse_paths(tmp_path, ["src:dep1", "lib"])
        assert result == {
            tmp_path / "src": [tmp_path / "dep1"],
            tmp_path / "lib": [],
        }


class TestResolvePaths:
    def test_no_specs_no_resolvers_returns_root(self, tmp_path):
        assert resolve_paths(tmp_path, [], []) == {tmp_path: []}

    def test_explicit_specs_only(self, tmp_path):
        (tmp_path / "src").mkdir()
        result = resolve_paths(tmp_path, ["src"], [])
        assert result == {tmp_path / "src": []}

    def test_unknown_resolver_raises(self, tmp_path):
        with pytest.raises(KeyError):
            resolve_paths(tmp_path, [], ["does-not-exist"])


class TestBuildPlugins:
    def test_default_only_includes_module_dunders(self):
        plugins = build_plugins(entrypoints=[], plugin_names=[])
        assert [type(p) for p in plugins] == [ModuleDundersPlugin]

    def test_named_plugins_run_before_module_dunders(self):
        plugins = build_plugins(entrypoints=[], plugin_names=["main_block"])
        assert [type(p) for p in plugins] == [MainBlockPlugin, ModuleDundersPlugin]

    def test_entrypoints_appended_last(self):
        plugins = build_plugins(entrypoints=["pkg.foo"], plugin_names=["main_block"])
        assert [type(p) for p in plugins] == [
            MainBlockPlugin,
            ModuleDundersPlugin,
            ExplicitEntrypointPlugin,
        ]
        explicit = plugins[-1]
        assert isinstance(explicit, ExplicitEntrypointPlugin)
        assert explicit.specs == ["pkg.foo"]

    def test_regex_entrypoint_is_compiled(self):
        plugins = build_plugins(entrypoints=["re:.*main.*"], plugin_names=[])
        explicit = plugins[-1]
        assert isinstance(explicit, ExplicitEntrypointPlugin)
        assert len(explicit.specs) == 1
        assert isinstance(explicit.specs[0], re.Pattern)

    def test_unknown_plugin_name_raises(self):
        with pytest.raises(KeyError):
            build_plugins(entrypoints=[], plugin_names=["nope"])


class TestPredicates:
    def test_is_dunder_all_true_for_dunder_variable(self):
        node = SymbolNode("pkg.__all__", "variable", Path("/x.py"), _pos())
        assert _is_dunder_all(node) is True

    def test_is_dunder_all_false_for_other_variable(self):
        node = SymbolNode("pkg.foo", "variable", Path("/x.py"), _pos())
        assert _is_dunder_all(node) is False

    def test_is_dunder_all_false_for_non_variable(self):
        node = SymbolNode("pkg.__all__", "function", Path("/x.py"), _pos())
        assert _is_dunder_all(node) is False

    def test_is_external_dep_true_for_external_synthetic(self):
        node = SymbolNode("[external dist] networkx", "synthetic", Path("/x.py"), _pos())
        assert _is_external_dep(node) is True

    def test_is_external_dep_false_for_other_synthetic(self):
        node = SymbolNode("<entrypoint>:foo", "synthetic", Path("/x.py"), _pos())
        assert _is_external_dep(node) is False

    def test_is_external_dep_false_for_real_node(self):
        node = SymbolNode("pkg.foo", "function", Path("/x.py"), _pos())
        assert _is_external_dep(node) is False


class TestSetupLogging:
    """Inspect the kwargs ``setup_logging`` passes to ``basicConfig``.

    Asserting on the root logger's level directly is brittle: pytest's
    ``logging`` plugin attaches its own handler before each test, and
    :func:`logging.basicConfig` is a no-op when the root logger already
    has handlers. Patching ``basicConfig`` sidesteps that interference.
    """

    def test_non_verbose_uses_warning_level(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
        setup_logging(False)
        assert captured["level"] == logging.WARNING

    def test_verbose_uses_debug_level(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
        setup_logging(True)
        assert captured["level"] == logging.DEBUG

    def test_logs_are_routed_to_stderr(self, monkeypatch):
        import sys

        captured: dict = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
        setup_logging(False)
        assert captured["stream"] is sys.stderr


# ---------------------------------------------------------------------------
# Top-level app
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_flag_prints_version_and_exits(self, runner):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.stdout.startswith("dead-cst ")

    def test_help_lists_commands(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("analyze", "why-alive", "dependencies", "unused-exports", "remove"):
            assert cmd in result.stdout


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_clean_project_exits_zero(self, runner, project):
        root = project({"mod.py": "def used():\n    pass\nused()\n"})
        result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used"])
        assert result.exit_code == 0
        assert "function: 1 total" in result.stdout
        # No "dead" qualifier when nothing is dead
        assert "dead" not in result.stdout
        # Banner goes to stderr
        assert "Building symbol graph" in result.stderr

    def test_dead_symbol_listed_and_exits_one(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                def dead():
                    pass

                used()
                """,
            }
        )
        result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used"])
        assert result.exit_code == 1
        assert "function: 2 total, 1 dead" in result.stdout
        assert "Dead symbols (1):" in result.stdout
        assert "mod.dead (function) at mod.py" in result.stdout

    def test_no_entrypoints_marks_everything_dead(self, runner, project):
        root = project({"mod.py": "def f():\n    pass\n"})
        result = runner.invoke(app, ["analyze", str(root)])
        assert result.exit_code == 1
        assert "module: 1 total, 1 dead" in result.stdout
        assert "function: 1 total, 1 dead" in result.stdout

    def test_unreachable_branch_reported(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                if False:
                    dead_branch = 1

                used()
                """,
            }
        )
        result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used"])
        assert result.exit_code == 1
        assert "Unreachable branches (1):" in result.stdout
        # Format: ``mod.py:line:col-line:col``
        assert re.search(r"mod\.py:\d+:\d+-\d+:\d+", result.stdout)

    def test_synthetic_kind_is_excluded_from_summary(self, runner, project):
        root = project({"mod.py": "if False:\n    x = 1\n"})
        result = runner.invoke(app, ["analyze", str(root)])
        # ``synthetic`` nodes show up internally for the dead branch, but
        # the per-kind summary suppresses them.
        for line in result.stdout.splitlines():
            assert not line.strip().startswith("synthetic:")

    def test_json_output_is_parseable_and_has_expected_shape(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                def dead():
                    pass

                used()
                """,
            }
        )
        result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used", "--format", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert set(payload) == {"summary", "dead_symbols", "unreachable_branches"}
        summary = payload["summary"][str(root.resolve())]
        assert summary["function"] == {"total": 2, "dead": 1}
        assert payload["dead_symbols"] == [
            {"fqname": "mod.dead", "type": "function", "path": "mod.py"}
        ]
        assert payload["unreachable_branches"] == []

    def test_json_includes_unreachable_branch_positions(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                if False:
                    x = 1

                used()
                """,
            }
        )
        result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used", "--format", "json"])
        payload = json.loads(result.stdout)
        assert len(payload["unreachable_branches"]) == 1
        branch = payload["unreachable_branches"][0]
        assert branch["path"] == "mod.py"
        assert "line" in branch["start"] and "column" in branch["start"]
        assert "line" in branch["end"] and "column" in branch["end"]

    def test_regex_entrypoint_keeps_modules_alive(self, runner, project):
        root = project({"mod.py": "def f():\n    pass\n"})
        result = runner.invoke(app, ["analyze", str(root), "-e", "re:.*\\.py$"])
        # The regex matches the module file; with the module alive, the
        # function it contains is also alive.
        assert result.exit_code == 0

    def test_path_spec_scopes_summary_to_subdir(self, runner, project):
        root = project({"src/mod.py": "def f():\n    pass\nf()\n"})
        result = runner.invoke(app, ["analyze", str(root), "-p", "src", "-e", "mod.f"])
        assert result.exit_code == 0
        assert f"{root.resolve() / 'src'}:" in result.stdout

    def test_plugin_flag_keeps_main_block_alive(self, runner, project):
        root = project(
            {
                "mod.py": """
                def main():
                    pass

                if __name__ == "__main__":
                    main()
                """,
            }
        )
        result = runner.invoke(app, ["analyze", str(root), "--plugin", "main_block"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# why-alive
# ---------------------------------------------------------------------------


class TestWhyAlive:
    def test_reports_predecessor_chain(self, runner, project):
        root = project({"mod.py": "def alive():\n    pass\nalive()\n"})
        result = runner.invoke(app, ["why-alive", str(root), "mod.alive"])
        assert result.exit_code == 0
        assert "Symbol: mod.alive (function)" in result.stdout
        assert "Path: mod.py" in result.stdout
        assert "Predecessor chain:" in result.stdout
        assert "<- mod.alive (function) at mod.py" in result.stdout
        assert "<- mod (module) at mod.py" in result.stdout

    def test_missing_symbol_exits_one(self, runner, project):
        root = project({"mod.py": "def alive():\n    pass\nalive()\n"})
        result = runner.invoke(app, ["why-alive", str(root), "mod.missing"])
        assert result.exit_code == 1
        assert "Symbol not found: mod.missing" in result.stderr


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------


class TestDependencies:
    def test_no_third_party_deps(self, runner, project):
        root = project({"mod.py": "import os\nimport json\n_used = (os, json)\n"})
        result = runner.invoke(app, ["dependencies", str(root)])
        assert result.exit_code == 0
        assert "(no third-party dependencies found)" in result.stdout

    def test_third_party_dep_listed(self, runner, project):
        root = project({"mod.py": "import networkx\n_x = networkx\n"})
        result = runner.invoke(app, ["dependencies", str(root)])
        assert result.exit_code == 0
        assert "[external dist] networkx" in result.stdout

    def test_json_output_groups_by_base(self, runner, project):
        root = project({"mod.py": "import networkx\n_x = networkx\n"})
        result = runner.invoke(app, ["dependencies", str(root), "--format", "json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload == {str(root.resolve()): ["[external dist] networkx"]}

    def test_json_output_empty_when_no_deps(self, runner, project):
        root = project({"mod.py": "import os\n_x = os\n"})
        result = runner.invoke(app, ["dependencies", str(root), "--format", "json"])
        payload = json.loads(result.stdout)
        assert payload == {str(root.resolve()): []}


# ---------------------------------------------------------------------------
# unused-exports
# ---------------------------------------------------------------------------


class TestUnusedExports:
    def test_reports_nothing_when_no_dunder_all(self, runner, project):
        root = project({"mod.py": "def used():\n    pass\nused()\n"})
        result = runner.invoke(app, ["unused-exports", str(root)])
        assert result.exit_code == 0
        assert "No __all__ entries are kept alive only by __all__." in result.stdout

    def test_reports_entry_alive_only_via_dunder_all(self, runner, project):
        root = project(
            {
                "pkg/__init__.py": 'from .a import foo\n__all__ = ["foo"]\n',
                "pkg/a.py": "def foo():\n    pass\n",
            }
        )
        result = runner.invoke(app, ["unused-exports", str(root)])
        assert result.exit_code == 0
        assert "pkg.__all__" in result.stdout
        assert "pkg.foo" in result.stdout


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_no_dead_code_message_when_clean(self, runner, project):
        root = project({"mod.py": "def f():\n    pass\nf()\n"})
        original = (root / "mod.py").read_text()
        result = runner.invoke(app, ["remove", str(root), "-e", "mod.f"])
        assert result.exit_code == 0
        assert "No dead code found." in result.stdout
        assert (root / "mod.py").read_text() == original

    def test_dry_run_lists_but_keeps_file(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                def dead():
                    pass

                used()
                """,
            }
        )
        original = (root / "mod.py").read_text()
        result = runner.invoke(app, ["remove", str(root), "-e", "mod.used", "--dry-run"])
        assert result.exit_code == 0
        assert "Dead symbols to remove (1):" in result.stdout
        assert "mod.dead (function) at mod.py" in result.stdout
        assert "--dry-run specified, no changes made." in result.stdout
        assert (root / "mod.py").read_text() == original

    def test_decline_confirm_aborts(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                def dead():
                    pass

                used()
                """,
            }
        )
        original = (root / "mod.py").read_text()
        result = runner.invoke(app, ["remove", str(root), "-e", "mod.used"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted." in result.stdout
        assert (root / "mod.py").read_text() == original

    def test_accept_confirm_rewrites_file(self, runner, project):
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                def dead():
                    pass

                used()
                """,
            }
        )
        result = runner.invoke(app, ["remove", str(root), "-e", "mod.used"], input="y\n")
        assert result.exit_code == 0
        assert "Dead code removed." in result.stdout
        rewritten = (root / "mod.py").read_text()
        assert "def dead" not in rewritten
        assert "def used" in rewritten

    def test_dry_run_does_not_prompt(self, runner, project):
        """Pass no stdin: if ``--dry-run`` short-circuits before the prompt
        the command exits cleanly. If it didn't, ``typer.confirm`` would
        hit EOF and abort."""
        root = project(
            {
                "mod.py": """
                def used():
                    pass

                def dead():
                    pass

                used()
                """,
            }
        )
        result = runner.invoke(app, ["remove", str(root), "-e", "mod.used", "--dry-run"])
        assert result.exit_code == 0
