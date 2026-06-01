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
- **1**: :meth:`Analysis.ancestors` on the pinned ``main`` symbol
  reports the native ``main_block`` synthetic in its predecessor chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dead_cst import Analysis, _native as native
from dead_cst.graph import KEEPALIVE_DEFAULT
from dead_cst.cli import app

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
        ["analyze", flux0_server_src, "--plugin", "main_block"],
    )
    # Typer raises SystemExit on non-zero exit; anything else means we crashed.
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    # 0 = no dead code, 1 = dead code found. Anything else is a crash.
    assert result.exit_code in (0, 1), (result.exit_code, result.output)


def test_analyze_flux0_cli_runs_to_completion(flux0_cli_src):
    result = CliRunner().invoke(
        app,
        ["analyze", flux0_cli_src, "--plugin", "main_block"],
    )
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code in (0, 1), (result.exit_code, result.output)


# ---------------------------------------------------------------------------
# Level 1: known-alive symbols have the expected predecessor
# ---------------------------------------------------------------------------


def _ancestor_fqnames(base: Path, target_fqname: str) -> list[str]:
    """Run the Python equivalent of ``why-alive``.

    Materializes the graph, finds the target node, and returns the
    predecessor chain's fqnames — the same data the dropped CLI
    command rendered, exposed through :meth:`Analysis.ancestors`.
    """
    analysis = Analysis(base, plugins=[native.NativePlugin.main_block()])
    ctx = analysis.materialize_all()
    target_idx = next((i for i, n in enumerate(ctx.nodes()) if n.fqname == target_fqname), None)
    assert target_idx is not None, f"{target_fqname} not in graph"
    return [a.fqname for a in ctx.node_attrs(analysis.ancestors(target_idx))]


def test_why_alive_flux0_server_main(flux0_server_src):
    """Level 1: ``main`` is reached via the module's ``if __name__ ...`` block.

    The native ``main_block`` plugin attaches a synthetic
    ``<__main__>:<module>`` node as a predecessor of every top-level
    call inside the guard.
    """
    ancestors = _ancestor_fqnames(Path(flux0_server_src), "flux0_server.main.main")
    assert any(a.startswith("<__main__>:flux0_server.main") for a in ancestors)


def test_why_alive_flux0_cli_main(flux0_cli_src):
    ancestors = _ancestor_fqnames(Path(flux0_cli_src), "flux0_cli.main.main")
    assert any(a.startswith("<__main__>:flux0_cli.main") for a in ancestors)


# ---------------------------------------------------------------------------
# Baseline: the importlib-loaded cmds modules are dead under static analysis
# ---------------------------------------------------------------------------


def _module_node(graph, fqname):
    return next(
        (n for n in graph.nodes() if n.kind == "module" and n.fqname == fqname),
        None,
    )


def test_flux0_cli_cmds_dead_without_dynamic_seed(flux0_cli_src):
    """flux0 loads its CLI command modules via runtime ``importlib``
    discovery (``register_commands``), which static analysis can't see —
    so with only ``main_block`` the cmds modules are correctly dead."""
    base = Path(flux0_cli_src)
    graph = Analysis(base, plugins=[native.NativePlugin.main_block()]).materialize_all()
    reachable = graph.reachable(seed_flags=KEEPALIVE_DEFAULT)

    cmds_agents = _module_node(graph, "flux0_cli.cmds.agents")
    assert cmds_agents is not None, "cmds.agents module should be in the graph"
    assert cmds_agents not in reachable, (
        "cmds.agents must be unreachable in the baseline -- "
        "it is only reached via flux0's runtime importlib discovery"
    )
