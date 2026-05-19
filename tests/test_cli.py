"""Tests for :mod:`dead_cst.cli`.

The CLI is a thin Typer wrapper around the public API: each command
builds (or loads) a symbol graph and renders the result. These tests
exercise the wrapper — argument parsing, plugin/resolver composition,
build-vs-load mode, ``--query`` selection, output formatting (text and
JSON), exit codes, the patch emission on ``remove``, and the
round-trip through ``build`` + ``read_graph`` — against tiny in-memory
projects under ``tmp_path``. The underlying analysis is covered by
``test_declarations`` / ``test_codemod`` / etc.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from dead_cst.cli import (
    _dead_real,
    _rel_path,
    app,
    build_plugins,
    build_resolver,
    parse_meta,
    setup_logging,
)
from dead_cst.graph import GraphMetadata, SymbolNode, read_graph
from dead_cst.plugins import (
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
    ModuleDundersPlugin,
)
from dead_cst.resolvers import ManualResolver, UvResolver


def _normalise(s: str) -> str:
    """Dedent ``s`` and strip up to one leading newline.

    Same convention used by ``test_codemod``: lets a triple-quoted
    literal put its opening quote on its own line without losing
    genuine leading blank lines.
    """
    s = textwrap.dedent(s)
    return s[1:] if s.startswith("\n") else s


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path):
    """Write a ``{relpath: source}`` mapping under ``tmp_path``."""

    def _make(files: dict[str, str]) -> Path:
        for rel, src in files.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_normalise(src))
        return tmp_path

    return _make


def _make_node(fqname: str, kind: str, path: str = "/x.py") -> SymbolNode:
    return SymbolNode(
        fqname,
        kind,
        path,
        start_line=1,
        start_column=0,
        end_line=1,
        end_column=1,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_meta_splits_on_first_equals():
    assert parse_meta("k=v") == ("k", "v")


def test_parse_meta_allows_equals_in_value():
    assert parse_meta("k=v=more") == ("k", "v=more")


def test_parse_meta_empty_value_is_legal():
    assert parse_meta("k=") == ("k", "")


def test_parse_meta_missing_equals_raises():
    with pytest.raises(typer.BadParameter):
        parse_meta("kv")


def test_parse_meta_empty_key_raises():
    with pytest.raises(typer.BadParameter):
        parse_meta("=v")


@pytest.mark.parametrize(
    "specs, expected_packages",
    [
        pytest.param(["src"], {"src": ()}, id="base-only"),
        pytest.param(
            ["src:dep1,dep2"],
            {"src": ("dep1", "dep2"), "dep1": (), "dep2": ()},
            id="base-with-deps-auto-promoted",
        ),
    ],
)
def test_manual_resolver_parses_specs(tmp_path, specs, expected_packages):
    result = ManualResolver(specs=specs).resolve(tmp_path)
    by_name = {p.name: p for p in result}
    assert set(by_name) == set(expected_packages)
    for name, deps in expected_packages.items():
        assert by_name[name].deps == deps
        assert by_name[name].path == (tmp_path / name).resolve()


def test_build_resolver_no_specs_returns_default_manual():
    resolver = build_resolver([], None)
    assert isinstance(resolver, ManualResolver)
    assert resolver.specs == ["."]


def test_build_resolver_named_resolver_only():
    resolver = build_resolver([], "uv")
    assert isinstance(resolver, UvResolver)


def test_build_resolver_path_and_name_are_mutually_exclusive():
    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        build_resolver(["src"], "uv")


def test_build_plugins_default_only_includes_module_dunders():
    plugins = build_plugins(entrypoints=[], entrypoint_regexes=[], plugin_names=[])
    assert [type(p) for p in plugins] == [ModuleDundersPlugin]


def test_build_plugins_named_plugins_run_before_module_dunders():
    plugins = build_plugins(entrypoints=[], entrypoint_regexes=[], plugin_names=["main_block"])
    assert [type(p) for p in plugins] == [MainBlockPlugin, ModuleDundersPlugin]


def test_build_plugins_appends_explicit_last():
    plugins = build_plugins(
        entrypoints=["pkg.foo"],
        entrypoint_regexes=[],
        plugin_names=["main_block"],
    )
    assert [type(p) for p in plugins] == [
        MainBlockPlugin,
        ModuleDundersPlugin,
        ExplicitEntrypointPlugin,
    ]
    explicit = plugins[-1]
    assert isinstance(explicit, ExplicitEntrypointPlugin)
    assert explicit.specs == ["pkg.foo"]


def test_build_plugins_compiles_entrypoint_regex():
    import re

    plugins = build_plugins(
        entrypoints=[],
        entrypoint_regexes=[".*main.*"],
        plugin_names=[],
    )
    explicit = plugins[-1]
    assert isinstance(explicit, ExplicitEntrypointPlugin)
    assert isinstance(explicit.specs[0], re.Pattern)


def test_build_plugins_unknown_plugin_name_raises():
    with pytest.raises(KeyError):
        build_plugins(entrypoints=[], entrypoint_regexes=[], plugin_names=["nope"])


def test_rel_path_under_root_is_relativized():
    assert _rel_path(Path("/a/b/c.py"), Path("/a")) == Path("b/c.py")


def test_rel_path_outside_root_returned_unchanged():
    p = Path("/elsewhere/c.py")
    assert _rel_path(p, Path("/a")) == p


def test_dead_real_filters_synthetic_nodes():
    """Synthetic nodes (entrypoint sentinels, external markers) are
    excluded from the dead-symbol report so we don't surface them
    alongside user-visible declarations."""
    real = _make_node("pkg.f", "function", "/a.py")
    synth = _make_node("<entrypoint>:pkg.f", "synthetic", "/a.py")

    assert _dead_real([real, synth]) == [real]


def test_dead_real_empty_input_returns_empty_list():
    assert _dead_real([]) == []


# ---------------------------------------------------------------------------
# setup_logging
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


# ---------------------------------------------------------------------------
# Top-level app — --version, --help, command set
# ---------------------------------------------------------------------------


def test_version_flag_prints_version_and_exits(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("dead-cst ")


def test_help_lists_supported_commands(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("build", "analyze", "remove"):
        assert cmd in result.stdout


def test_help_does_not_list_removed_commands(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("why-alive", "dependencies", "unused-exports"):
        assert cmd not in result.stdout


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_clean_project_exits_zero(runner, project):
    root = project({"mod.py": "def used():\n    pass\nused()\n"})
    result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used"])
    assert result.exit_code == 0
    assert "function: 1 total" in result.stdout
    assert "dead" not in result.stdout
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


def test_analyze_exit_zero_overrides_exit_code(runner, project):
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
    result = runner.invoke(app, ["analyze", str(root), "-e", "mod.used", "--exit-zero"])
    assert result.exit_code == 0
    assert "mod.dead" in result.stdout


def test_analyze_entrypoint_regex_flag(runner, project):
    root = project({"mod.py": "def f():\n    pass\n"})
    result = runner.invoke(app, ["analyze", str(root), "--entrypoint-regex", r".*\.py$"])
    assert result.exit_code == 0


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
    assert set(payload) == {"summary", "dead_symbols"}
    assert payload["summary"][str(root.resolve())]["function"] == {"total": 2, "dead": 1}
    assert payload["dead_symbols"] == [{"fqname": "mod.dead", "type": "function", "path": "mod.py"}]


def test_analyze_test_only_query_flags_function_only_used_by_tests(runner, project):
    """``--query test-only`` surfaces decls whose only path back to a
    keepalive seed runs through a ``TESTCASE`` node."""
    root = project(
        {
            "mod.py": """
            def helper():
                pass
            """,
            "test_mod.py": """
            from mod import helper

            def test_smoke():
                helper()
            """,
        }
    )
    result = runner.invoke(
        app,
        [
            "analyze",
            str(root),
            "--plugin",
            "pytest",
            "--query",
            "test-only",
            "--exit-zero",
        ],
    )
    assert result.exit_code == 0
    assert "mod.helper" in result.stdout


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_no_dead_code_emits_no_patch(runner, project):
    root = project({"mod.py": "def f():\n    pass\nf()\n"})
    original = (root / "mod.py").read_text()
    result = runner.invoke(app, ["remove", str(root), "-e", "mod.f"])
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "No dead code found." in result.stderr
    assert (root / "mod.py").read_text() == original


@pytest.mark.parametrize("output_to_file", [False, True], ids=["stdout", "output-file"])
def test_remove_emits_patch_without_touching_files(runner, project, tmp_path, output_to_file):
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

    args = ["remove", str(root), "-e", "mod.used"]
    out: Path | None = None
    if output_to_file:
        out = tmp_path / "dead.patch"
        args += ["-o", str(out)]

    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert (root / "mod.py").read_text() == original

    if out is not None:
        patch = out.read_text()
        assert result.stdout == ""
        assert f"git apply {out}" in result.stderr
    else:
        patch = result.stdout
        assert "git apply" in result.stderr

    assert "diff --git a/mod.py b/mod.py" in patch
    assert "-def dead():" in patch


def test_remove_test_only_query_drops_test_set(runner, project):
    """``remove --query test-only`` emits the blast-radius patch: both
    the test and the helper that exists only to back it. Helpers that
    are *also* reachable from an entrypoint stay put."""
    root = project(
        {
            "mod.py": """
            def used():
                pass

            def test_only_helper():
                pass
            """,
            "test_mod.py": """
            from mod import test_only_helper

            def test_smoke():
                test_only_helper()
            """,
        }
    )
    result = runner.invoke(
        app,
        [
            "remove",
            str(root),
            "-e",
            "mod.used",
            "--plugin",
            "pytest",
            "--query",
            "test-only",
        ],
    )
    assert result.exit_code == 0
    # The test gets dropped (TESTCASE-flagged) along with the helper
    # that only the test consumed.
    assert "-def test_only_helper" in result.stdout
    assert "diff --git a/test_mod.py" in result.stdout
    # ``used`` is the entrypoint, so neither the function nor any
    # diff against mod.py's body appears.
    assert "-def used" not in result.stdout


# ---------------------------------------------------------------------------
# build + --graph round-trip
# ---------------------------------------------------------------------------


def test_build_writes_file_with_metadata(runner, project, tmp_path):
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
    graph_path = tmp_path / "graph.bin"
    result = runner.invoke(
        app,
        [
            "build",
            str(root),
            "-o",
            str(graph_path),
            "--meta",
            "branch=feature/x",
            "--meta",
            "sha=abc123",
        ],
    )
    assert result.exit_code == 0
    assert graph_path.exists()
    assert "Wrote graph to" in result.stderr

    sg, meta = read_graph(graph_path)
    assert isinstance(meta, GraphMetadata)
    assert meta.node_count == len(sg.nodes())
    assert meta.edge_count == len(sg.edges())
    assert ("branch", "feature/x") in meta.user_meta
    assert ("sha", "abc123") in meta.user_meta


def test_analyze_with_graph_flag_loads_prebuilt(runner, project, tmp_path):
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
    graph_path = tmp_path / "graph.bin"
    runner.invoke(app, ["build", str(root), "-o", str(graph_path), "-e", "mod.used"])
    assert graph_path.exists()

    result = runner.invoke(app, ["analyze", str(root), "--graph", str(graph_path)])
    assert result.exit_code == 1
    assert "Loading symbol graph from" in result.stderr
    assert "mod.dead" in result.stdout


def test_analyze_with_graph_and_build_inputs_is_rejected(runner, project, tmp_path, monkeypatch):
    # Rich wraps long Typer error messages across the terminal width;
    # widen COLUMNS so the substring assertion isn't split by box rules.
    monkeypatch.setenv("COLUMNS", "500")
    root = project({"mod.py": "def f():\n    pass\n"})
    graph_path = tmp_path / "graph.bin"
    runner.invoke(app, ["build", str(root), "-o", str(graph_path)])

    result = runner.invoke(
        app,
        ["analyze", str(root), "--graph", str(graph_path), "-e", "mod.f"],
    )
    assert result.exit_code != 0
    combined = (result.stderr or "") + (result.output or "")
    assert "build inputs are not allowed" in combined


def test_remove_with_graph_flag_emits_patch(runner, project, tmp_path):
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
    graph_path = tmp_path / "graph.bin"
    runner.invoke(app, ["build", str(root), "-o", str(graph_path), "-e", "mod.used"])

    result = runner.invoke(
        app,
        ["remove", str(root), "--graph", str(graph_path)],
    )
    assert result.exit_code == 0
    assert "-def dead():" in result.stdout


def test_read_graph_rejects_non_dead_cst_file(tmp_path):
    bogus = tmp_path / "bogus.bin"
    bogus.write_bytes(b"NOTAGRAPH" * 4)
    with pytest.raises(ValueError, match="not a dead-cst graph file"):
        read_graph(bogus)
