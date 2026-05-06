"""Tests for :mod:`dead_cst.cli`.

The CLI is a thin Typer wrapper around the public API: every command
calls :func:`build_symbol_graph` then renders the result. These tests
exercise the wrapper -- argument parsing, plugin/resolver composition,
output formatting (text and JSON), exit codes, and the confirm/dry-run
prompts on ``remove`` -- against tiny in-memory projects under
``tmp_path``. The underlying analysis is covered by ``test_declarations``
/ ``test_codemod`` / etc.
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

from dead_cst.plugins import (
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
    ModuleDundersPlugin,
)
from dead_cst.plugins._core import EXTERNAL_DIST_PREFIX
from dead_cst.plugins.explicit_entrypoint import EXPLICIT_PREFIX
from dead_cst.graph import SymbolNode
from dead_cst.resolvers import ManualResolver, SourceTreeFlags
from dead_cst.cli import (
    _dead_real,
    _is_dunder_all,
    _is_external_dep,
    _rel_path,
    app,
    build_plugins,
    parse_entrypoint,
    resolve_trees,
    setup_logging,
)


def _normalise(s: str) -> str:
    """Dedent ``s`` and strip up to one leading newline.

    Same convention used by ``test_codemod``: lets a triple-quoted
    literal put its opening quote on its own line without losing genuine
    leading blank lines.
    """
    s = textwrap.dedent(s)
    return s[1:] if s.startswith("\n") else s


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path):
    """Write a ``{relpath: source}`` mapping under ``tmp_path`` and return its path.

    Sources are normalised with ``_normalise`` so triple-quoted literals
    can line up visually inside the test cases.
    """

    def _make(files: dict[str, str]) -> Path:
        for rel, src in files.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_normalise(src))
        return tmp_path

    return _make


def _pos() -> CodeRange:
    return CodeRange(start=CodePosition(line=1, column=0), end=CodePosition(line=1, column=1))


# ---------------------------------------------------------------------------
# Pure helpers -- parse_entrypoint, parse_paths, resolve_paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        pytest.param("pkg.mod.func", "pkg.mod.func", id="fqname-passes-through"),
        pytest.param("src/main.py", "src/main.py", id="path-string-passes-through"),
    ],
)
def test_parse_entrypoint_returns_string_unchanged(spec, expected):
    assert parse_entrypoint(spec) == expected


def test_parse_entrypoint_compiles_re_prefix():
    result = parse_entrypoint("re:.*\\.py$")
    assert isinstance(result, re.Pattern)
    assert result.pattern == ".*\\.py$"


def test_parse_entrypoint_re_prefix_with_empty_pattern():
    # ``re:`` on its own is a degenerate but legal compile target;
    # exercise it so a future mistaken ``[3:]`` slice doesn't pass.
    result = parse_entrypoint("re:")
    assert isinstance(result, re.Pattern)
    assert result.pattern == ""


def test_manual_resolver_empty_specs(tmp_path):
    assert ManualResolver(specs=[]).resolve(tmp_path) == []


@pytest.mark.parametrize(
    "specs, expected",
    [
        pytest.param(["src"], [("src", ())], id="path-only"),
        pytest.param(
            ["src:dep1,dep2"],
            [("src", ("dep1", "dep2"))],
            id="path-with-deps",
        ),
        pytest.param(
            ["src: dep1 , dep2 "],
            [("src", ("dep1", "dep2"))],
            id="dep-whitespace-stripped",
        ),
        pytest.param(
            ["src:dep1,"],
            [("src", ("dep1",))],
            id="trailing-comma-drops-empty-dep",
        ),
        pytest.param(
            ["src:dep1", "lib"],
            [("src", ("dep1",)), ("lib", ())],
            id="multiple-specs-emit-multiple-trees",
        ),
    ],
)
def test_manual_resolver_parses_specs(tmp_path, specs, expected):
    result = ManualResolver(specs=specs).resolve(tmp_path)
    assert [(t.path, t.search_trees) for t in result] == [
        ((tmp_path / p).resolve(), tuple((tmp_path / d).resolve() for d in deps))
        for p, deps in expected
    ]
    for tree in result:
        assert tree.flags & SourceTreeFlags.EXPORTED


def test_resolve_trees_no_specs_no_resolvers_returns_root(tmp_path):
    trees, resolvers = resolve_trees(tmp_path, [], [])
    assert len(trees) == 1
    assert trees[0].path == tmp_path
    assert trees[0].flags & SourceTreeFlags.EXPORTED
    assert resolvers == []


def test_resolve_trees_explicit_specs_only(tmp_path):
    (tmp_path / "src").mkdir()
    trees, resolvers = resolve_trees(tmp_path, ["src"], [])
    assert [t.path for t in trees] == [(tmp_path / "src").resolve()]
    # ``-p`` flows through a ManualResolver so its ``resolve_import``
    # is part of the chain alongside any named resolvers.
    assert [r.name for r in resolvers] == ["manual"]


def test_resolve_trees_unknown_resolver_raises(tmp_path):
    with pytest.raises(KeyError):
        resolve_trees(tmp_path, [], ["does-not-exist"])


# ---------------------------------------------------------------------------
# build_plugins -- composition order
# ---------------------------------------------------------------------------


def test_build_plugins_default_only_includes_module_dunders():
    plugins = build_plugins(entrypoints=[], plugin_names=[])
    assert [type(p) for p in plugins] == [ModuleDundersPlugin]


def test_build_plugins_named_plugins_run_before_module_dunders():
    plugins = build_plugins(entrypoints=[], plugin_names=["main_block"])
    assert [type(p) for p in plugins] == [MainBlockPlugin, ModuleDundersPlugin]


def test_build_plugins_appends_explicit_last():
    # ``-e`` runs after upstream plugins so it can hang entrypoints off
    # nodes those plugins contribute.
    plugins = build_plugins(entrypoints=["pkg.foo"], plugin_names=["main_block"])
    assert [type(p) for p in plugins] == [
        MainBlockPlugin,
        ModuleDundersPlugin,
        ExplicitEntrypointPlugin,
    ]
    explicit = plugins[-1]
    assert isinstance(explicit, ExplicitEntrypointPlugin)
    assert explicit.specs == ["pkg.foo"]


def test_build_plugins_compiles_regex_entrypoint():
    plugins = build_plugins(entrypoints=["re:.*main.*"], plugin_names=[])
    explicit = plugins[-1]
    assert isinstance(explicit, ExplicitEntrypointPlugin)
    assert isinstance(explicit.specs[0], re.Pattern)


def test_build_plugins_unknown_plugin_name_raises():
    with pytest.raises(KeyError):
        build_plugins(entrypoints=[], plugin_names=["nope"])


# ---------------------------------------------------------------------------
# Predicates -- _is_dunder_all, _is_external_dep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fqname, type_, expected",
    [
        pytest.param("pkg.__all__", "variable", True, id="dunder-all-variable"),
        pytest.param("pkg.foo", "variable", False, id="other-variable"),
        pytest.param("pkg.__all__", "function", False, id="dunder-name-non-variable"),
    ],
)
def test_is_dunder_all(fqname, type_, expected):
    node = SymbolNode(fqname, type_, Path("/x.py"), _pos())
    assert _is_dunder_all(node) is expected


@pytest.mark.parametrize(
    "fqname, type_, expected",
    [
        pytest.param(f"{EXTERNAL_DIST_PREFIX}networkx", "synthetic", True, id="external-synthetic"),
        pytest.param(f"{EXPLICIT_PREFIX}foo", "synthetic", False, id="entrypoint-synthetic"),
        pytest.param("pkg.foo", "function", False, id="real-function"),
    ],
)
def test_is_external_dep(fqname, type_, expected):
    node = SymbolNode(fqname, type_, Path("/x.py"), _pos())
    assert _is_external_dep(node) is expected


def test_rel_path_under_root_is_relativized():
    assert _rel_path(Path("/a/b/c.py"), Path("/a")) == Path("b/c.py")


def test_rel_path_outside_root_returned_unchanged():
    p = Path("/elsewhere/c.py")
    assert _rel_path(p, Path("/a")) == p


def test_rel_path_equal_to_root_yields_empty(tmp_path):
    """``Path.relative_to`` of a path against itself is ``Path('.')``."""
    assert _rel_path(tmp_path, tmp_path) == Path(".")


def test_dead_real_filters_synthetic_nodes():
    """Synthetic nodes (entrypoint sentinels, external markers) are
    excluded from the dead-symbol report so we don't surface them
    alongside user-visible declarations."""
    import networkx as nx

    real = SymbolNode("pkg.f", "function", Path("/a.py"), _pos())
    entrypoint_synth = SymbolNode(f"{EXPLICIT_PREFIX}pkg.f", "synthetic", Path("/a.py"), _pos())

    g = nx.MultiDiGraph()
    for n in (real, entrypoint_synth):
        g.add_node(n)

    assert _dead_real(g) == [real]


def test_dead_real_empty_graph_returns_empty_list():
    import networkx as nx

    assert _dead_real(nx.MultiDiGraph()) == []


# ---------------------------------------------------------------------------
# setup_logging
#
# Asserting on root-logger state is brittle: pytest's ``logging`` plugin
# attaches a handler before each test, which makes :func:`logging.basicConfig`
# a no-op. Patch ``basicConfig`` and inspect the kwargs instead.
# ---------------------------------------------------------------------------


def test_setup_logging_non_verbose_uses_warning_level(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
    setup_logging(False)
    assert captured["level"] == logging.WARNING


def test_setup_logging_verbose_uses_debug_level(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
    setup_logging(True)
    assert captured["level"] == logging.DEBUG


def test_setup_logging_streams_to_stderr(monkeypatch):
    import sys

    captured: dict = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
    setup_logging(False)
    assert captured["stream"] is sys.stderr


# ---------------------------------------------------------------------------
# Top-level app -- --version and --help
# ---------------------------------------------------------------------------


def test_version_flag_prints_version_and_exits(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("dead-cst ")


def test_help_lists_all_commands(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("analyze", "why-alive", "dependencies", "unused-exports", "remove"):
        assert cmd in result.stdout


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_clean_project_exits_zero(runner, project):
    root = project({"mod.py": "def used():\n    pass\nused()\n"})
    result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used"])
    assert result.exit_code == 0
    assert "function: 1 total" in result.stdout
    # No "dead" qualifier when nothing is dead.
    assert "dead" not in result.stdout
    # Banner goes to stderr so JSON consumers can pipe stdout cleanly.
    assert "Building symbol graph" in result.stderr


def test_analyze_dead_symbol_listed_and_exits_one(runner, project):
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


def test_analyze_no_entrypoints_marks_everything_dead(runner, project):
    root = project({"mod.py": "def f():\n    pass\n"})
    result = runner.invoke(app, ["analyze", str(root)])
    assert result.exit_code == 1
    assert "module: 1 total, 1 dead" in result.stdout
    assert "function: 1 total, 1 dead" in result.stdout


def test_analyze_unreachable_branch_reported(runner, project):
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
    assert re.search(r"mod\.py:\d+:\d+-\d+:\d+", result.stdout)


def test_analyze_synthetic_kind_excluded_from_summary(runner, project):
    # ``synthetic`` nodes show up internally for the dead branch, but
    # the per-kind summary suppresses them so users don't see e.g.
    # ``synthetic: 1 total``.
    root = project({"mod.py": "if False:\n    x = 1\n"})
    result = runner.invoke(app, ["analyze", str(root)])
    for line in result.stdout.splitlines():
        assert not line.strip().startswith("synthetic:")


def test_analyze_json_output_has_expected_shape(runner, project):
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
    assert payload["summary"][str(root.resolve())]["function"] == {"total": 2, "dead": 1}
    assert payload["dead_symbols"] == [{"fqname": "mod.dead", "type": "function", "path": "mod.py"}]
    assert payload["unreachable_branches"] == []


def test_analyze_json_output_includes_branch_positions(runner, project):
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
    assert {"line", "column"} <= set(branch["start"])
    assert {"line", "column"} <= set(branch["end"])


def test_analyze_regex_entrypoint_keeps_module_alive(runner, project):
    # A regex matching the file makes the module an entrypoint; the
    # function it contains is alive transitively.
    root = project({"mod.py": "def f():\n    pass\n"})
    result = runner.invoke(app, ["analyze", str(root), "-e", "re:.*\\.py$"])
    assert result.exit_code == 0


def test_analyze_path_spec_scopes_summary_to_subdir(runner, project):
    root = project({"src/mod.py": "def f():\n    pass\nf()\n"})
    result = runner.invoke(app, ["analyze", str(root), "-p", "src", "-e", "mod.f"])
    assert result.exit_code == 0
    assert f"{root.resolve() / 'src'}:" in result.stdout


def test_analyze_main_block_plugin_keeps_entry_alive(runner, project):
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


def test_why_alive_reports_predecessor_chain(runner, project):
    root = project({"mod.py": "def alive():\n    pass\nalive()\n"})
    result = runner.invoke(app, ["why-alive", str(root), "mod.alive"])
    assert result.exit_code == 0
    assert "Symbol: mod.alive (function)" in result.stdout
    assert "Path: mod.py" in result.stdout
    assert "Predecessor chain:" in result.stdout
    assert "<- mod.alive (function) at mod.py" in result.stdout
    assert "<- mod (module) at mod.py" in result.stdout


def test_why_alive_missing_symbol_exits_one(runner, project):
    root = project({"mod.py": "def alive():\n    pass\nalive()\n"})
    result = runner.invoke(app, ["why-alive", str(root), "mod.missing"])
    assert result.exit_code == 1
    assert "Symbol not found: mod.missing" in result.stderr


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------


def test_dependencies_no_third_party(runner, project):
    root = project({"mod.py": "import os\nimport json\n_used = (os, json)\n"})
    result = runner.invoke(app, ["dependencies", str(root)])
    assert result.exit_code == 0
    assert "(no third-party dependencies found)" in result.stdout


def test_dependencies_third_party_listed(runner, project):
    root = project({"mod.py": "import networkx\n_x = networkx\n"})
    result = runner.invoke(app, ["dependencies", str(root)])
    assert result.exit_code == 0
    assert "[external dist] networkx" in result.stdout


def test_dependencies_json_output_groups_by_base(runner, project):
    root = project({"mod.py": "import networkx\n_x = networkx\n"})
    result = runner.invoke(app, ["dependencies", str(root), "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {str(root.resolve()): ["[external dist] networkx"]}


def test_dependencies_json_output_empty_when_no_deps(runner, project):
    root = project({"mod.py": "import os\n_x = os\n"})
    result = runner.invoke(app, ["dependencies", str(root), "--format", "json"])
    assert json.loads(result.stdout) == {str(root.resolve()): []}


# ---------------------------------------------------------------------------
# unused-exports
# ---------------------------------------------------------------------------


def test_unused_exports_reports_nothing_when_no_dunder_all(runner, project):
    root = project({"mod.py": "def used():\n    pass\nused()\n"})
    result = runner.invoke(app, ["unused-exports", str(root)])
    assert result.exit_code == 0
    assert "No __all__ entries are kept alive only by __all__." in result.stdout


def test_unused_exports_reports_entry_alive_only_via_dunder_all(runner, project):
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


def test_remove_no_dead_code_when_clean(runner, project):
    root = project({"mod.py": "def f():\n    pass\nf()\n"})
    original = (root / "mod.py").read_text()
    result = runner.invoke(app, ["remove", str(root), "-e", "mod.f"])
    assert result.exit_code == 0
    assert "No dead code found." in result.stdout
    assert (root / "mod.py").read_text() == original


def test_remove_dry_run_lists_but_keeps_file(runner, project):
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


def test_remove_decline_confirm_aborts(runner, project):
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


def test_remove_accept_confirm_rewrites_file(runner, project):
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


def test_remove_dry_run_does_not_prompt(runner, project):
    # Pass no stdin: if ``--dry-run`` short-circuits before the prompt
    # the command exits cleanly. Without that branch ``typer.confirm``
    # would hit EOF and abort.
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
