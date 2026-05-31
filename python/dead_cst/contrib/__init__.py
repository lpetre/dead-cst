"""Extensions targeting specific third-party tools.

Contrib modules know about external systems -- frameworks
(discord.py), test runners (pytest, unittest), and the
``mock.patch`` idiom. Anything that handles a generic Python
language convention lives in :mod:`dead_cst.plugins` instead.

Click and the dispatch-app frameworks (Flask, FastAPI, Typer,
Cyclopts, Slack Bolt, FastMCP, Celery) are now native plugins
resolved through ``dead_cst._native`` -- see ``NativePlugin.click()``,
``NativePlugin.flask()`` and friends.
"""

from __future__ import annotations

from .discordpy import DiscordPyPlugin
from .mock_patch import MockPatchPlugin
from .pytest import PytestPlugin
from .unittest import UnittestPlugin

__all__ = [
    "DiscordPyPlugin",
    "MockPatchPlugin",
    "PytestPlugin",
    "UnittestPlugin",
]
