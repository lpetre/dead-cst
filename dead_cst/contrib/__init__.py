"""Extensions targeting specific third-party tools.

Contrib modules know about external systems -- frameworks
(FastAPI, Flask, Click, Typer), test runners (pytest, unittest), and
build tools (uv) -- and ship the plugin or resolver that bridges
``dead-cst`` to that system. Anything that handles a generic Python
language convention (``__main__`` blocks, ``[project.scripts]``,
``__init_subclass__`` discovery, ``__all__`` / dunder exports) lives in
:mod:`dead_cst.plugins` or :mod:`dead_cst.resolvers` instead.

The classes here are also re-exported from those packages for
ergonomics, so

.. code-block:: python

    from dead_cst.plugins import FastAPIPlugin
    from dead_cst.resolvers import UvResolver

work alongside

.. code-block:: python

    from dead_cst.contrib import FastAPIPlugin, UvResolver
    from dead_cst.contrib.fastapi import FastAPIPlugin

"""

from __future__ import annotations

from .click import ClickPlugin
from .cyclopts import CycloptsPlugin
from .fastapi import FastAPIPlugin
from .flask import FlaskPlugin
from .mock_patch import MockPatchPlugin
from .pytest import PytestPlugin
from .typer import TyperPlugin
from .unittest import UnittestPlugin

# ``UvResolver`` is re-exported lazily to break an initialization cycle:
# ``uv.py`` imports from ``dead_cst.resolvers``, and
# ``dead_cst.resolvers.__init__`` in turn eagerly re-exports
# ``UvResolver`` from this contrib module. Loading the class here at
# module-init time would deadlock the cycle through partially-initialized
# modules; deferring to ``__getattr__`` lets ``dead_cst.resolvers``
# finish loading ``_core`` (which is what ``uv`` actually depends on)
# before the class itself is pulled.
__all__ = [
    "ClickPlugin",
    "CycloptsPlugin",
    "FastAPIPlugin",
    "FlaskPlugin",
    "MockPatchPlugin",
    "PytestPlugin",
    "TyperPlugin",
    "UnittestPlugin",
    "UvResolver",
]


def __getattr__(name: str):
    if name == "UvResolver":
        from .uv import UvResolver

        return UvResolver
    raise AttributeError(f"module 'dead_cst.contrib' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
