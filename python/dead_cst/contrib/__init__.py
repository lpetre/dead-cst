"""Extensions targeting specific third-party tools.

Contrib modules know about external systems -- test runners
(unittest). Anything that handles a generic Python language
convention lives in :mod:`dead_cst.plugins` instead.

Click, discord.py, pytest, the ``mock.patch`` idiom, and the
dispatch-app frameworks (Flask, FastAPI, Typer, Cyclopts, Slack Bolt,
FastMCP, Celery) are now native plugins resolved through
``dead_cst._native`` -- see ``NativePlugin.click()``,
``NativePlugin.discordpy()``, ``NativePlugin.pytest()`` and friends.
``UnittestPlugin`` is the lone remaining Python contrib plugin (kept
for its native-parity test until the Python ``Plugin`` ABC is removed).
"""

from __future__ import annotations

from .unittest import UnittestPlugin

__all__ = [
    "UnittestPlugin",
]
