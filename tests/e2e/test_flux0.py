"""End-to-end tests against ``flux0-ai/flux0`` at a pinned SHA.

flux0 is a uv-workspace FastAPI app with several first-party packages
under ``packages/*/src/``. We point dead-cst at the ``flux0_server``
package -- it has a real ``if __name__ == "__main__":`` block calling
``main()``, which gives us a stable level-1 anchor without needing the
workspace's venv (cross-package imports degrade to ``[unresolved]``
synthetic nodes, which is fine for these checks).

Levels:
- **0**: ``analyze`` runs to completion without raising; exit code is
  the documented 0 / 1.
- **1**: ``why-alive`` finds the pinned ``main`` symbol and reports
  the ``MainBlockPlugin`` synthetic in its predecessor chain.
- **2**: project-specific plugins from :mod:`._flux0_plugins` close
  flux0's two ``importlib``-driven blind spots. Each is a tiny
  subclass of an abstract base (:class:`SubpackageDiscoveryPlugin` /
  :class:`FqnRegistryPlugin`) that a future dead-cst could expose
  as a builtin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dead_cst import (
    MainBlockPlugin,
    ModuleDundersPlugin,
    build_symbol_graph,
    find_reachable,
)
from dead_cst.cli import app

from ._flux0_plugins import Flux0CliCommandsPlugin, Flux0InternalModulesPlugin

pytestmark = pytest.mark.e2e


FLUX0_URL = "https://github.com/flux0-ai/flux0.git"
FLUX0_SHA = "8d04176642b091ddb5c5020486f353d4e824460b"


@pytest.fixture(scope="module")
def flux0_repo(clone_repo):
    return clone_repo("flux0", FLUX0_URL, FLUX0_SHA)


@pytest.fixture(scope="module")
def flux0_server_src(flux0_repo) -> str:
    return str(flux0_repo / "packages" / "server" / "src")


@pytest.fixture(scope="module")
def flux0_cli_src(flux0_repo) -> str:
    return str(flux0_repo / "packages" / "cli" / "src")


# ---------------------------------------------------------------------------
# Level 0: runs to completion
# ---------------------------------------------------------------------------


def test_analyze_flux0_server_runs_to_completion(flux0_server_src):
    """Level 0: dead-cst should walk the whole package without crashing."""
    result = CliRunner().invoke(
        app,
        ["analyze", flux0_server_src, "--plugin", "main_block", "--no-cache"],
    )
    # Typer raises SystemExit on non-zero exit; anything else means we crashed.
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    # 0 = no dead code, 1 = dead code found. Anything else is a crash.
    assert result.exit_code in (0, 1), (result.exit_code, result.output)


def test_analyze_flux0_cli_runs_to_completion(flux0_cli_src):
    result = CliRunner().invoke(
        app,
        ["analyze", flux0_cli_src, "--plugin", "main_block", "--no-cache"],
    )
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code in (0, 1), (result.exit_code, result.output)


# ---------------------------------------------------------------------------
# Level 1: known-alive symbols have the expected predecessor
# ---------------------------------------------------------------------------


def test_why_alive_flux0_server_main(flux0_server_src):
    """Level 1: ``main`` is reached via the module's ``if __name__ ...`` block.

    The MainBlockPlugin attaches a synthetic ``<__main__>:<module>`` node
    as a predecessor of every top-level call inside the guard, so we
    expect to see that synthetic in the chain.
    """
    result = CliRunner().invoke(
        app,
        [
            "why-alive",
            flux0_server_src,
            "flux0_server.main.main",
            "--plugin",
            "main_block",
            "--no-cache",
        ],
    )
    assert result.exception is None, result.exception
    assert result.exit_code == 0, result.output
    assert "Symbol: flux0_server.main.main (function)" in result.stdout
    assert "<__main__>:flux0_server.main" in result.stdout


def test_why_alive_flux0_cli_main(flux0_cli_src):
    result = CliRunner().invoke(
        app,
        [
            "why-alive",
            flux0_cli_src,
            "flux0_cli.main.main",
            "--plugin",
            "main_block",
            "--no-cache",
        ],
    )
    assert result.exception is None, result.exception
    assert result.exit_code == 0, result.output
    assert "Symbol: flux0_cli.main.main (function)" in result.stdout
    assert "<__main__>:flux0_cli.main" in result.stdout


# ---------------------------------------------------------------------------
# Level 2: project-specific plugin closes a real blind spot
# ---------------------------------------------------------------------------


def _module_node(graph, fqname):
    return next(
        (n for n in graph.nodes if n.type == "module" and n.fqname == fqname),
        None,
    )


def _build_graph(base: Path, *plugins):
    return build_symbol_graph({base: []}, plugins=list(plugins), project_root=base)


def test_flux0_cli_cmds_dead_without_plugin(flux0_cli_src):
    """Sanity: without the dynamic-loader plugin, the cmds modules are dead."""
    base = Path(flux0_cli_src)
    graph = _build_graph(base, MainBlockPlugin())
    reachable = find_reachable(graph)

    cmds_agents = _module_node(graph, "flux0_cli.cmds.agents")
    assert cmds_agents is not None, "cmds.agents module should be in the graph"
    assert cmds_agents not in reachable, (
        "cmds.agents must be unreachable in the baseline -- "
        "if this fails the test is no longer demonstrating the blind spot"
    )


def test_flux0_dynamic_loader_revives_cli_cmds(flux0_cli_src):
    """The plugin must keep both the modules *and* their contents alive.

    Asserting just the module node would miss the failure mode where a
    plugin attaches the synth to the module (which is a sink in the
    keep-alive graph) and silently leaves every decl inside it dead.
    """
    base = Path(flux0_cli_src)
    graph = _build_graph(base, MainBlockPlugin(), Flux0CliCommandsPlugin())
    reachable = find_reachable(graph)

    for fqname in (
        "flux0_cli.cmds.agents",  # module
        "flux0_cli.cmds.sessions",  # module
        "flux0_cli.cmds.agents.list_agents",  # decl inside the dynamic module
        "flux0_cli.cmds.sessions.create_event",  # decl inside the dynamic module
    ):
        node = next((n for n in graph.nodes if n.fqname == fqname), None)
        assert node is not None, f"{fqname} not in graph"
        assert node in reachable, f"{fqname} should be alive once the plugin runs"


def test_flux0_dynamic_loader_revives_server_internal_modules(flux0_server_src):
    base = Path(flux0_server_src)
    graph = _build_graph(base, MainBlockPlugin(), Flux0InternalModulesPlugin())
    reachable = find_reachable(graph)

    for fqname in (
        "flux0_server.replay_agent",  # __init__ module
        "flux0_server.replay_agent.init_module",  # decl inside __init__
        "flux0_server.replay_agent.replay_agent",  # nested module
        "flux0_server.replay_agent.replay_agent.ReplayAgentRunner",  # decl
    ):
        node = next((n for n in graph.nodes if n.fqname == fqname), None)
        assert node is not None, f"{fqname} not in graph"
        assert node in reachable, f"{fqname} should be alive via INTERNAL_MODULES"


def test_flux0_full_plugin_set_only_finds_real_dead_code(flux0_cli_src, flux0_server_src):
    """With both project plugins + the CLI's defaults active, only real dead code remains.

    Pinning this number guards against future regressions where the
    dynamic-loader logic stops covering something it used to. Two
    constants in flux0_server/main.py and one unused TypeVar in
    flux0_cli/main.py are the genuine findings at SHA 8d04176.
    """
    expected_dead = {
        Path(flux0_server_src): {
            "flux0_server.main.DEFAULT_PORT",
            "flux0_server.main.SERVER_ADDRESS",
        },
        Path(flux0_cli_src): {
            "flux0_cli.main.F",
            "flux0_cli.main.TypeVar",
        },
    }
    plugins = [
        MainBlockPlugin(),
        ModuleDundersPlugin(),
        Flux0CliCommandsPlugin(),
        Flux0InternalModulesPlugin(),
    ]
    for base, expected in expected_dead.items():
        graph = _build_graph(base, *plugins)
        reachable = find_reachable(graph)
        dead = {
            n.fqname
            for n in graph.nodes
            if n not in reachable and n.type != "synthetic" and not n.fqname.startswith("[")
        }
        assert dead == expected, f"unexpected dead set under {base}: {sorted(dead)}"
