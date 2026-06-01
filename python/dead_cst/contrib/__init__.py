"""Third-party-aware extensions.

The framework- and test-runner-aware plugins that used to live here
(Flask, FastAPI, Typer, Cyclopts, Slack Bolt, FastMCP, Celery, Click,
discord.py, pytest, the ``mock.patch`` idiom, unittest) are now native
plugins resolved through :mod:`dead_cst._native` — see
``NativePlugin.flask()`` … ``NativePlugin.unittest()``. The package is
retained as the home for any future third-party-aware contributions.
"""

from __future__ import annotations

__all__: list[str] = []
