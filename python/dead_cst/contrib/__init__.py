"""Extensions targeting specific third-party tools.

Contrib modules know about external systems -- frameworks
(FastAPI, Flask, Click, Typer), test runners (pytest, unittest), and
build tools (uv). Anything that handles a generic Python language
convention lives in :mod:`dead_cst.plugins` or :mod:`dead_cst.resolvers`
instead.

All exports are loaded lazily through ``__getattr__`` so importing
:mod:`dead_cst.resolvers` (which only needs ``UvResolver`` from
contrib) does not pull in every framework plugin's module.
"""

from __future__ import annotations

from importlib import import_module

# Lazy module lookups so importing one piece of contrib doesn't pull
# every plugin module — particularly important because plugins import
# from ``dead_cst.plugins`` which re-exports back through contrib.
_EXPORTS: dict[str, str] = {
    "CeleryPlugin": ".celery",
    "ClickPlugin": ".click",
    "CycloptsPlugin": ".cyclopts",
    "DiscordPyPlugin": ".discordpy",
    "FastAPIPlugin": ".fastapi",
    "FastMCPPlugin": ".fastmcp",
    "FlaskPlugin": ".flask",
    "MockPatchPlugin": ".mock_patch",
    "PytestPlugin": ".pytest",
    "ServerConfigPlugin": ".server_config",
    "TyperPlugin": ".typer",
    "UnittestPlugin": ".unittest",
    "UvResolver": ".uv",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'dead_cst.contrib' has no attribute {name!r}")
    return getattr(import_module(module_path, package=__name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
