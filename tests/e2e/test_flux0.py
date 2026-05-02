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
- **2**: a project-specific :class:`Flux0DynamicLoaderPlugin`, declared
  below as a prototype of "what a dead-cst user would write," revives
  flux0's two ``importlib``-driven blind spots (the ``flux0_cli.cmds``
  Click sub-command auto-loader and the
  ``flux0_server.main.INTERNAL_MODULES`` registry).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest
from typer.testing import CliRunner

from dead_cst import (
    AddEdge,
    AddNode,
    GraphOp,
    MainBlockPlugin,
    PluginContext,
    build_symbol_graph,
    find_reachable,
)
from dead_cst._plugins import synthetic_node
from dead_cst.cli import app

pytestmark = pytest.mark.e2e


FLUX0_URL = "https://github.com/flux0-ai/flux0.git"
FLUX0_SHA = "8d04176642b091ddb5c5020486f353d4e824460b"


# ---------------------------------------------------------------------------
# Project-specific plugin prototype
#
# This is the minimum it takes today to author a project-local plugin
# that closes a static-analysis blind spot. We use it as a forcing
# function for "what scaffolding would shrink this further?" -- see the
# notes in the PR description. Anything that ends up shared across two
# user plugins is a candidate for promotion to ``dead_cst._plugins``.
# ---------------------------------------------------------------------------


@dataclass
class Flux0DynamicLoaderPlugin:
    """Keep flux0's dynamically-loaded modules alive.

    Two blind spots in this repo:

    1. ``flux0_cli/main.py`` discovers Click sub-commands with
       ``pkgutil.iter_modules`` + ``importlib.import_module``, so every
       module under ``flux0_cli.cmds`` is referenced only via runtime
       string interpolation.
    2. ``flux0_server/main.py`` carries an ``INTERNAL_MODULES`` list of
       string FQNs that's consumed by ``importlib`` elsewhere in the
       app; ``flux0_server.replay_agent`` is the only entry today.

    The plugin walks the graph once per base in :meth:`finalize`,
    selects every module whose fqname matches one of the configured
    package prefixes, and hangs an entrypoint synthetic off them so
    :func:`find_reachable` will reach them.
    """

    packages: tuple[str, ...] = (
        "flux0_cli.cmds",
        "flux0_server.replay_agent",
    )
    name: str = "flux0_dynamic"
    version: str = "1"

    def observe(self, ctx) -> None:
        # No per-file work: this plugin is purely a finalize-time
        # entrypoint declaration.
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # ``importlib.import_module(M)`` executes every top-level
        # statement in ``M``, so every top-level decl inside a covered
        # module behaves as if it were called from an entrypoint. The
        # module node itself is a *successor* of its decls (decl -> module
        # in the keep-alive graph), so attaching the synth to the module
        # alone keeps the file from being reported but doesn't revive
        # any of its contents -- we want the decls.
        targets = [node for node in ctx.base_nodes() if self._covers(node.fqname)]
        if not targets:
            return
        # Inlined ``mark_entrypoints`` -- it lives in ``_plugins._core``
        # but isn't re-exported, so a user plugin has to spell the four
        # lines by hand for now.
        synth = synthetic_node(f"<flux0-dynamic>:{ctx.base.name}", ctx.base)
        yield AddNode(synth, entrypoint=True)
        for target in targets:
            yield AddEdge(synth, target)

    def _covers(self, fqname: str) -> bool:
        return any(fqname == pkg or fqname.startswith(pkg + ".") for pkg in self.packages)


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
    graph = _build_graph(base, MainBlockPlugin(), Flux0DynamicLoaderPlugin())
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
    graph = _build_graph(base, MainBlockPlugin(), Flux0DynamicLoaderPlugin())
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
