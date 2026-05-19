"""Example plugin: ``@registry.on(event=ImportedClass)`` → inverted edge.

Demonstrates a real-world pattern that's invisible to default
reachability:

* A handler function is decorated with a registration call that takes
  a class as a kwarg: ``@registry.on(event=UserCreated)``.
* Nothing in the codebase *calls* the handler directly — at runtime,
  the registry dispatches by class.
* The handler is "kept alive" by the existence of references to the
  resolved event class anywhere in the project.

The plugin walks decorator refs, reads the resolved ``event`` kwarg
as a :class:`SymbolNode` (the payload-side declref resolution
re-added on top of literal-only matchers), and emits an
``event_class → decorated_function`` edge — the *inverse* of the
natural ``decorated → event_class`` use edge that already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest

native = pytest.importorskip("dead_cst.native")


@pytest.fixture
def make_ctx(tmp_path: Path):
    def make(files: dict[str, str], **kwargs) -> native.ProjectContext:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.ProjectContext(str(tmp_path), **kwargs)

    return make


@dataclass(kw_only=True)
class HandlerByKwargPlugin:
    """For every ``@<owner>.<attr>(<kwarg>=<ImportedClass>)`` decorator,
    emit an ``ImportedClass -> decorated_function`` edge.

    Reads the resolved kwarg as a :class:`native.SymbolNode` from
    ``ref.kwargs``. Non-literal expressions that don't statically
    resolve (anonymous lambdas, runtime objects, etc.) surface as
    ``None`` and are skipped silently.
    """

    name: str = "handler-by-kwarg"
    version: int = 1
    decorator_owner: str = ""
    decorator_attr: str = ""
    kwarg_name: str = ""

    def run(self, ctx: "native.ProjectContext") -> Iterable["native.GraphOp"]:
        from dead_cst import native

        refs = native.query(ctx).decorators().where_owner_attr([self.decorator_attr]).collect()
        for ref in refs:
            if ref.decorator_owner != self.decorator_owner:
                continue
            target = ref.kwargs.get(self.kwarg_name)
            if target is None or not hasattr(target, "fqname"):
                continue
            yield native.AddEdge(target, ref.decorated)


def _edges(graph: "native.NativeGraph") -> set[str]:
    return {f"{graph.nodes[s].fqname} -> {graph.nodes[d].fqname}" for s, d, _ in graph.edges}


def test_inverted_edge_from_kwarg_imported_symbol(make_ctx):
    """The plugin emits ``events.UserCreated -> handlers.on_user_created``
    so reachability flows from the event class to its handler."""

    ctx = make_ctx(
        {
            "events.py": ("class UserCreated: pass\nclass UserDeleted: pass\n"),
            "registry.py": (
                "class Registry:\n"
                "    def on(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "registry = Registry()\n"
            ),
            "handlers.py": (
                "from events import UserCreated, UserDeleted\n"
                "from registry import registry\n"
                "\n"
                "@registry.on(event=UserCreated)\n"
                "def on_user_created(evt): pass\n"
                "\n"
                "@registry.on(event=UserDeleted)\n"
                "def on_user_deleted(evt): pass\n"
            ),
        }
    )
    ctx.add_plugin(
        HandlerByKwargPlugin(
            decorator_owner="registry",
            decorator_attr="on",
            kwarg_name="event",
        )
    )
    graph = ctx.materialize()
    edges = _edges(graph)

    # The inverted edges the plugin emitted.
    assert "events.UserCreated -> handlers.on_user_created" in edges
    assert "events.UserDeleted -> handlers.on_user_deleted" in edges


def test_no_inverted_edge_when_kwarg_is_a_literal(make_ctx):
    """``@registry.on(event="user.created")`` (string literal, not a
    resolvable name) emits no inverted edge — the plugin skips refs
    whose kwarg payload isn't a SymbolNode."""

    ctx = make_ctx(
        {
            "registry.py": (
                "class Registry:\n"
                "    def on(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "registry = Registry()\n"
            ),
            "handlers.py": (
                "from registry import registry\n"
                "\n"
                "@registry.on(event='user.created')\n"
                "def on_user_created(evt): pass\n"
            ),
        }
    )
    ctx.add_plugin(
        HandlerByKwargPlugin(
            decorator_owner="registry",
            decorator_attr="on",
            kwarg_name="event",
        )
    )
    graph = ctx.materialize()
    edges = _edges(graph)

    # No edge whose target is the handler and whose source isn't already
    # in the natural graph (the literal string can't anchor an edge).
    assert not any(
        e.endswith(" -> handlers.on_user_created") and "registry" not in e and "handlers" not in e
        for e in edges
    )
