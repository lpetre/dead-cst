"""Extensions targeting specific third-party tools.

Contrib modules know about external systems -- frameworks
(FastAPI, Flask, Click, Typer), test runners (pytest, unittest).
Anything that handles a generic Python language convention lives
in :mod:`dead_cst.plugins` instead.
"""

from __future__ import annotations

from .celery import CeleryPlugin
from .click import ClickPlugin
from .cyclopts import cyclopts_plugin
from .discordpy import DiscordPyPlugin
from .fastapi import fastapi_plugin
from .fastmcp import fastmcp_plugin
from .flask import flask_plugin
from .mock_patch import MockPatchPlugin
from .pytest import PytestPlugin
from .server_config import ServerConfigPlugin
from .typer import typer_plugin
from .unittest import UnittestPlugin

__all__ = [
    "CeleryPlugin",
    "ClickPlugin",
    "DiscordPyPlugin",
    "MockPatchPlugin",
    "PytestPlugin",
    "ServerConfigPlugin",
    "UnittestPlugin",
    "cyclopts_plugin",
    "fastapi_plugin",
    "fastmcp_plugin",
    "flask_plugin",
    "typer_plugin",
]
