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

from dead_cst import Analysis
from dead_cst.analyze import _entrypoint_seeds, _find_reachable as find_reachable
from dead_cst.cache import GraphCache
from dead_cst.cli import app
from dead_cst.plugins import MainBlockPlugin, ModuleDundersPlugin
from dead_cst.resolvers import ManualResolver

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
    return Analysis(
        base, resolver=ManualResolver(specs=["."]), plugins=list(plugins)
    ).materialize_all()


def test_flux0_cli_cmds_dead_without_plugin(flux0_cli_src):
    """Sanity: without the dynamic-loader plugin, the cmds modules are dead."""
    base = Path(flux0_cli_src)
    graph = _build_graph(base, MainBlockPlugin())
    reachable = find_reachable(graph, _entrypoint_seeds(graph))

    cmds_agents = _module_node(graph, "flux0_cli.cmds.agents")
    assert cmds_agents is not None, "cmds.agents module should be in the graph"
    assert cmds_agents not in reachable, (
        "cmds.agents must be unreachable in the baseline -- "
        "if this fails the test is no longer demonstrating the blind spot"
    )


def test_flux0_cli_commands_revives_click_groups(flux0_cli_src):
    """Flux0CliCommandsPlugin marks every click.Group under flux0_cli.cmds alive.

    What the plugin promises is exactly what flux0's
    ``register_commands`` does at runtime: the ``isinstance(cmd,
    click.Group)`` filter says "register every Group attribute as a
    sub-command." The plugin reproduces that statically by parsing
    each cmds submodule's CST and finding ``@click.group()``-decorated
    functions / ``X = click.Group(...)`` assignments.

    It does NOT promise to revive the handler functions decorated with
    flux0's custom ``@get_options(group, ...)`` / ``@create_options(group, ...)``
    wrappers; those are invisible to dead-cst's ClickPlugin and would
    need their own decorator-aware plugin.
    """
    base = Path(flux0_cli_src)
    graph = _build_graph(base, MainBlockPlugin(), Flux0CliCommandsPlugin())
    reachable = find_reachable(graph, _entrypoint_seeds(graph))

    for fqname in (
        "flux0_cli.cmds.agents.agents",  # @click.group() in agents.py
        "flux0_cli.cmds.sessions.sessions",  # @click.group() in sessions.py
        "flux0_cli.cmds.agents",  # module stays alive via the Group decl
        "flux0_cli.cmds.sessions",
    ):
        node = next((n for n in graph.nodes if n.fqname == fqname), None)
        assert node is not None, f"{fqname} not in graph"
        assert node in reachable, f"{fqname} should be alive once the plugin runs"

    # Counter-assertion: the @get_options-decorated handlers stay dead
    # because dead-cst's ClickPlugin does not understand flux0's custom
    # decorator chain. This is documented analyzer behaviour, not a bug
    # in the new plugin -- a future ``Flux0CustomDecoratorsPlugin`` would
    # be the right place to close that gap.
    handler_fqname = "flux0_cli.cmds.agents.list_agents"
    handler = next((n for n in graph.nodes if n.fqname == handler_fqname), None)
    assert handler is not None
    assert handler not in reachable, (
        "list_agents should still appear dead -- @get_options is a flux0-"
        "specific decorator dead-cst can't statically resolve"
    )


def test_flux0_internal_modules_revives_replay_agent(flux0_server_src):
    """Flux0InternalModulesPlugin reads the literal list and revives every entry.

    The plugin parses ``flux0_server.main.INTERNAL_MODULES`` from the
    CST, so adding a new module name to the upstream list flows through
    automatically (the plugin doesn't need a code change, only a re-run).
    """
    base = Path(flux0_server_src)
    graph = _build_graph(base, MainBlockPlugin(), Flux0InternalModulesPlugin())
    reachable = find_reachable(graph, _entrypoint_seeds(graph))

    for fqname in (
        "flux0_server.replay_agent",  # __init__ module
        "flux0_server.replay_agent.init_module",  # called by main.py:152
        "flux0_server.replay_agent.shutdown_module",  # called by main.py:168
        "flux0_server.replay_agent.replay_agent",  # nested module imported by __init__
        "flux0_server.replay_agent.replay_agent.ReplayAgentRunner",  # decl
    ):
        node = next((n for n in graph.nodes if n.fqname == fqname), None)
        assert node is not None, f"{fqname} not in graph"
        assert node in reachable, f"{fqname} should be alive via INTERNAL_MODULES"


def test_flux0_server_dead_set_pins_to_real_findings(flux0_server_src):
    """Server side has a tight, durable dead set: just two unused constants.

    Pinning this catches regressions in either the INTERNAL_MODULES
    parser (would re-introduce the whole replay_agent surface) or the
    dunder handling (would re-introduce ``__version__``).
    """
    base = Path(flux0_server_src)
    graph = _build_graph(
        base,
        MainBlockPlugin(),
        ModuleDundersPlugin(),
        Flux0InternalModulesPlugin(),
    )
    reachable = find_reachable(graph, _entrypoint_seeds(graph))
    dead = {
        n.fqname
        for n in graph.nodes
        if n not in reachable and n.type != "synthetic" and not n.fqname.startswith("[")
    }
    assert dead == {
        "flux0_server.main.DEFAULT_PORT",
        "flux0_server.main.SERVER_ADDRESS",
    }, f"unexpected server dead set: {sorted(dead)}"


def test_flux0_cli_dead_set_includes_real_findings_and_decorator_blind_spot(flux0_cli_src):
    """CLI side has two genuine findings + a known custom-decorator blind spot.

    Genuine: ``flux0_cli.main.F`` (unused TypeVar) and the matching
    ``TypeVar`` import. Blind spot: every ``@get_options(group, ...)``
    handler stays dead because dead-cst can't trace flux0's custom
    decorator chain back to its owning Click group. We assert both
    classes show up so a future ``Flux0CustomDecoratorsPlugin`` will
    visibly shrink this test when added.
    """
    base = Path(flux0_cli_src)
    graph = _build_graph(
        base,
        MainBlockPlugin(),
        ModuleDundersPlugin(),
        Flux0CliCommandsPlugin(),
    )
    reachable = find_reachable(graph, _entrypoint_seeds(graph))
    dead = {
        n.fqname
        for n in graph.nodes
        if n not in reachable and n.type != "synthetic" and not n.fqname.startswith("[")
    }
    # Real findings: present.
    assert {"flux0_cli.main.F", "flux0_cli.main.TypeVar"} <= dead
    # Custom-decorator blind spot: handlers using @get_options /
    # @create_options / @list_options stay dead. Sample a few rather
    # than pinning the whole list, since flux0 may add more handlers.
    expected_blind_spot = {
        "flux0_cli.cmds.agents.list_agents",
        "flux0_cli.cmds.agents.get_agent",
        "flux0_cli.cmds.sessions.create_event",
        "flux0_cli.cmds.sessions.list_sessions",
    }
    assert expected_blind_spot <= dead, (
        f"missing expected blind-spot dead symbols: {expected_blind_spot - dead}"
    )
    # The Click groups themselves must NOT appear dead.
    assert "flux0_cli.cmds.agents.agents" not in dead
    assert "flux0_cli.cmds.sessions.sessions" not in dead


def test_flux0_internal_modules_survives_cache_round_trip(flux0_server_src, tmp_path):
    """LiteralListPlugin must produce the same dead set warm as cold.

    The naive observe-stashes-on-self design silently regresses on
    warm runs (cached payload replays without invoking observe, leaving
    captured fqnames empty). The current design encodes captured
    fqnames as ENTRYPOINT-flagged synthetic decls in the observe
    payload, so the cache replay is fully self-sufficient and finalize
    only walks the graph -- this test guards that property.
    """
    base = Path(flux0_server_src)
    plugins = [
        MainBlockPlugin(),
        ModuleDundersPlugin(),
        Flux0InternalModulesPlugin(),
    ]
    cache_path = tmp_path / "cache.sqlite"

    dead_sets = []
    for _ in range(2):
        with GraphCache(cache_path) as cache:
            graph = Analysis(
                base,
                resolver=ManualResolver(specs=["."]),
                plugins=plugins,
                cache=cache,
            ).materialize_all()
        reachable = find_reachable(graph, _entrypoint_seeds(graph))
        dead_sets.append(
            {
                n.fqname
                for n in graph.nodes
                if n not in reachable and n.type != "synthetic" and not n.fqname.startswith("[")
            }
        )

    cold, warm = dead_sets
    assert cold == warm, f"cache regressed plugin output: cold={cold}, warm={warm}"
    assert cold == {
        "flux0_server.main.DEFAULT_PORT",
        "flux0_server.main.SERVER_ADDRESS",
    }
