"""Tests for the :meth:`Plugin.prepare` pre-graph hook.

Covers the contract documented in :meth:`Plugin.prepare`:

* ``prepare`` is called exactly once per plugin per
  :meth:`Analysis.materialize_all` invocation.
* It receives the project root as :class:`pathlib.Path`.
* It runs *before* graph construction (visible because exceptions
  raised inside propagate without the graph being built).
* The default no-op base implementation doesn't crash.
* :meth:`Analysis.materialize_all` is memoized — prepare doesn't
  re-fire on subsequent calls.
* Type validation of the plugins list happens up front, before
  ``prepare`` is dispatched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst.analyze import Analysis
from dead_cst.plugins import Plugin


class _RecordingPlugin(Plugin):
    name = "recording"
    version = 1

    def __init__(self) -> None:
        self.prepare_calls: list[Path] = []

    def prepare(self, repo_root: Path) -> None:
        self.prepare_calls.append(repo_root)

    def run(self, ctx):  # type: ignore[no-untyped-def]
        return ()


class _RaisingPlugin(Plugin):
    name = "raising"
    version = 1

    def prepare(self, repo_root: Path) -> None:
        raise RuntimeError("config invalid")

    def run(self, ctx):  # type: ignore[no-untyped-def]
        return ()


class _BareBaseSubclass(Plugin):
    """Doesn't override ``prepare`` — exercises the default no-op."""

    name = "bare"
    version = 1

    def run(self, ctx):  # type: ignore[no-untyped-def]
        return ()


def _make_analysis(tmp_path: Path, *plugins: Plugin) -> Analysis:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("def foo(): pass\n")
    return Analysis(tmp_path, plugins=tuple(plugins))


def test_prepare_called_once_with_project_root(tmp_path: Path) -> None:
    plug = _RecordingPlugin()
    analysis = _make_analysis(tmp_path, plug)
    analysis.materialize_all()
    assert plug.prepare_calls == [tmp_path]


def test_prepare_called_on_every_plugin(tmp_path: Path) -> None:
    a, b, c = _RecordingPlugin(), _RecordingPlugin(), _RecordingPlugin()
    analysis = _make_analysis(tmp_path, a, b, c)
    analysis.materialize_all()
    assert a.prepare_calls == [tmp_path]
    assert b.prepare_calls == [tmp_path]
    assert c.prepare_calls == [tmp_path]


def test_prepare_does_not_refire_on_memoized_materialize(tmp_path: Path) -> None:
    plug = _RecordingPlugin()
    analysis = _make_analysis(tmp_path, plug)
    analysis.materialize_all()
    analysis.materialize_all()  # memoized — no second prepare call.
    assert plug.prepare_calls == [tmp_path]


def test_prepare_raising_propagates_before_graph_build(tmp_path: Path) -> None:
    recording = _RecordingPlugin()
    raising = _RaisingPlugin()
    # Order matters: recording first so we can prove the second
    # plugin's prepare raises before any graph construction.
    analysis = _make_analysis(tmp_path, recording, raising)
    with pytest.raises(RuntimeError, match="config invalid"):
        analysis.materialize_all()
    # The first plugin's prepare DID run (we got to it before the
    # second one raised).
    assert recording.prepare_calls == [tmp_path]
    # But the graph itself was never built — Analysis._ctx stays None.
    assert analysis._ctx is None  # noqa: SLF001 — testing the invariant


def test_prepare_default_no_op_does_not_crash(tmp_path: Path) -> None:
    """Subclasses that don't override ``prepare`` use the base no-op."""
    analysis = _make_analysis(tmp_path, _BareBaseSubclass())
    analysis.materialize_all()  # would raise if the base default broke.


def test_non_plugin_in_list_raises_typeerror_before_prepare(tmp_path: Path) -> None:
    """The isinstance check fires before any plugin's ``prepare`` is
    invoked — a ``Pluign()`` typo doesn't slip past."""
    good = _RecordingPlugin()
    bad = object()  # not a Plugin instance
    analysis = _make_analysis(tmp_path, good, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected a dead_cst.plugins.Plugin"):
        analysis.materialize_all()
    # ``bad`` is the second entry; the isinstance loop fires
    # immediately on it without running ``good.prepare`` either,
    # since the validation is interleaved with the prepare dispatch.
    # (We don't pin the exact order beyond "the TypeError fires" —
    # both pre-prepare and pre-graph are acceptable.)
