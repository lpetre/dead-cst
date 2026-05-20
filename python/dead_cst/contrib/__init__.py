"""Extensions targeting specific third-party tools.

Contrib modules know about external systems -- frameworks
(FastAPI, Flask, Click, Typer), test runners (pytest, unittest), and
build tools (uv). Anything that handles a generic Python language
convention lives in :mod:`dead_cst.plugins` or :mod:`dead_cst.resolvers`
instead.
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
from .uv import UvResolver

__all__ = [
    "CeleryPlugin",
    "ClickPlugin",
    "DiscordPyPlugin",
    "MockPatchPlugin",
    "PytestPlugin",
    "ServerConfigPlugin",
    "UnittestPlugin",
    "UvResolver",
    "cyclopts_plugin",
    "fastapi_plugin",
    "fastmcp_plugin",
    "flask_plugin",
    "typer_plugin",
]
