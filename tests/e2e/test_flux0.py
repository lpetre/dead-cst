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
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

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
