"""Tests for the overlay / what-if API.

Three knobs work together:

* :meth:`Analysis.preview_payloads` regenerates :class:`VisitorPayload`
  for a hand-picked file set, optionally with a substitute
  :class:`UnreachableRegionDetector`, and bypasses the on-disk cache
  entirely.
* :meth:`Analysis.materialize_with` rebuilds only the affected
  packages' contributions with the substitute payloads spliced in
  and re-runs cross-package composition into a fresh graph,
  leaving the baseline graph untouched.
* :meth:`Analysis.preview` chains the two and wraps the result in a
  :class:`GraphView` so callers can ask the same reachability
  questions as :class:`Analysis` against the overlay.

These tests focus on the contract; the underlying truthiness work is
covered by ``test_truthiness_resolver.py`` /
``test_unreachable_branches.py``.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
import pytest

from dead_cst import Analysis
from dead_cst.analyze import GraphView
from dead_cst.branches import DefaultUnreachableRegionDetector, TruthinessResolver
from dead_cst.cache import GraphCache
from dead_cst.plugins import MainBlockPlugin
from dead_cst.resolvers import ManualResolver


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


@dataclass(frozen=True)
class _BakedFlagDetector(DefaultUnreachableRegionDetector):
    """Folds ``check_flag(<literal>)`` to ``True`` if the flag is in ``on``.

    Reads the flag *name* via :meth:`TruthinessResolver.resolve_constant`
    so call sites passing a Name (``check_flag(FEATURE_A)``) fold the
    same way as call sites passing a literal (``check_flag("feature_a")``).
    """

    name: str = "baked-flag"
    version: int = 1
    on: frozenset[str] = field(default_factory=frozenset)

    def resolve(
        self,
        expr: cst.BaseExpression,
        resolver: TruthinessResolver | None = None,
    ) -> bool | None:
        if not isinstance(expr, cst.Call):
            return None
        func = expr.func
        if not (isinstance(func, cst.Name) and func.value == "check_flag"):
            return None
        if len(expr.args) != 1 or resolver is None:
            return None
        const = resolver.resolve_constant(expr.args[0].value)
        if const is None:
            return None
        return const.value in self.on


# ----------------------------------------------------------------------
# preview_payloads: scope, cache bypass, detector substitution.
# ----------------------------------------------------------------------


def test_preview_payloads_returns_payloads_for_requested_files(tmp_path):
    _write(
        tmp_path,
        {
            "app.py": "def f(): return 1\n",
            "other.py": "def g(): return 2\n",
        },
    )
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    payloads = a.preview_payloads([tmp_path / "app.py"])
    assert set(payloads) == {tmp_path / "app.py"}
    payload = payloads[tmp_path / "app.py"]
    assert any(n.fqname == "app.f" for n in payload.nodes)


def test_preview_payloads_empty_files_returns_empty(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    assert a.preview_payloads([]) == {}


def test_preview_payloads_unknown_file_raises(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    other = tmp_path.parent / "outside.py"
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    with pytest.raises(KeyError):
        a.preview_payloads([other])


def test_preview_payloads_does_not_write_cache(tmp_path):
    """Substitute payloads must not poison the on-disk cache.

    Otherwise a one-shot what-if would silently alter every
    subsequent baseline run.
    """
    _write(
        tmp_path,
        {
            "app.py": textwrap.dedent(
                """
                def check_flag(name): return False
                def used():
                    return 1
                if check_flag("feature_a"):
                    used()
                """
            ).strip()
            + "\n",
        },
    )
    cache = GraphCache(tmp_path / ".dead-cst-cache")
    a = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["."]),
        cache=cache,
    )
    a.refresh()
    baseline_payload = cache.get(tmp_path / "app.py", a._fingerprint)
    assert baseline_payload is not None

    detector = _BakedFlagDetector(on=frozenset({"feature_a"}))
    a.preview_payloads([tmp_path / "app.py"], detector=detector)

    after = cache.get(tmp_path / "app.py", a._fingerprint)
    # Cache hit count and contents under the analysis fingerprint are
    # unchanged: preview_payloads neither read nor wrote.
    assert after == baseline_payload


def test_preview_payloads_detector_override_changes_dead_suites(tmp_path):
    """A baked-flag detector marks a branch dead that the default cannot."""
    _write(
        tmp_path,
        {
            "app.py": textwrap.dedent(
                """
                FEATURE_A = "feature_a"
                def check_flag(name): return False
                def hot(): return 1
                def cold(): return 2
                if check_flag(FEATURE_A):
                    hot()
                else:
                    cold()
                """
            ).strip()
            + "\n",
        },
    )
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    detector = _BakedFlagDetector(on=frozenset({"feature_a"}))
    payloads = a.preview_payloads([tmp_path / "app.py"], detector=detector)
    payload = payloads[tmp_path / "app.py"]
    # Exactly one suite (the else: cold()) is now statically dead.
    assert len(payload.dead_suites) == 1


# ----------------------------------------------------------------------
# materialize_with: overlay graph, baseline preserved, scope.
# ----------------------------------------------------------------------


def test_materialize_with_empty_returns_full_graph(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    g_full = a.materialize_all()
    g_empty = a.materialize_with({})
    assert g_full is g_empty


def test_materialize_with_preserves_baseline_graph(tmp_path):
    _write(
        tmp_path,
        {
            "app.py": textwrap.dedent(
                """
                FEATURE_A = "feature_a"
                def check_flag(name): return False
                def hot(): return 1
                def cold(): return 2
                if check_flag(FEATURE_A):
                    hot()
                else:
                    cold()
                """
            ).strip()
            + "\n",
        },
    )
    a = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["."]),
        plugins=[MainBlockPlugin()],
    )
    baseline = a.materialize_all()
    detector = _BakedFlagDetector(on=frozenset({"feature_a"}))
    payloads = a.preview_payloads([tmp_path / "app.py"], detector=detector)
    overlay = a.materialize_with(payloads)
    # Different graph objects; subsequent baseline calls return the
    # original.
    assert overlay is not baseline
    assert a.materialize_all() is baseline


def test_materialize_with_unknown_file_raises(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    a.refresh()
    from dead_cst.graph import VisitorPayload

    rogue = tmp_path.parent / "outside.py"
    with pytest.raises(KeyError):
        a.materialize_with({rogue: VisitorPayload((), (), (), ())})


# ----------------------------------------------------------------------
# preview() end-to-end: GraphView surface, baseline diff.
# ----------------------------------------------------------------------


def test_preview_returns_graph_view(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    view = a.preview([tmp_path / "app.py"])
    assert isinstance(view, GraphView)


def test_preview_kept_alive_by_dead_branches_diff(tmp_path):
    """End-to-end: with the flag baked ON, the else branch's symbol is
    dead-branch kept-alive; with the flag baked OFF, the if branch's is.
    """
    _write(
        tmp_path,
        {
            "app.py": textwrap.dedent(
                """
                FEATURE_A = "feature_a"
                def check_flag(name): return False
                def hot(): return 1
                def cold(): return 2
                def main():
                    if check_flag(FEATURE_A):
                        return hot()
                    else:
                        return cold()
                if __name__ == "__main__":
                    main()
                """
            ).strip()
            + "\n",
        },
    )
    a = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["."]),
        plugins=[MainBlockPlugin()],
    )
    # Baseline: no dead branches because check_flag is opaque.
    assert a.kept_alive_by_dead_branches() == set()

    on = a.preview(
        [tmp_path / "app.py"],
        detector=_BakedFlagDetector(on=frozenset({"feature_a"})),
    )
    on_kept = {n.fqname for n in on.kept_alive_by_dead_branches()}
    assert on_kept == {"app.cold"}

    off = a.preview(
        [tmp_path / "app.py"],
        detector=_BakedFlagDetector(on=frozenset()),
    )
    off_kept = {n.fqname for n in off.kept_alive_by_dead_branches()}
    assert off_kept == {"app.hot"}


# ----------------------------------------------------------------------
# GraphView: same reachability surface as Analysis.
# ----------------------------------------------------------------------


def test_graph_view_reachable_and_dead(tmp_path):
    _write(
        tmp_path,
        {
            "app.py": textwrap.dedent(
                """
                def used(): return 1
                def unused(): return 2
                def main():
                    return used()
                if __name__ == "__main__":
                    main()
                """
            ).strip()
            + "\n",
        },
    )
    a = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["."]),
        plugins=[MainBlockPlugin()],
    )
    view = GraphView(a.materialize_all())
    reachable_names = {n.fqname for n in view.reachable()}
    assert "app.used" in reachable_names
    assert "app.unused" not in reachable_names
    dead_names = {n.fqname for n in view.dead()}
    assert dead_names == {"app.unused"}


def test_graph_view_count_nodes(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    view = GraphView(a.materialize_all())
    counts = view.count_nodes()
    assert counts.get("module") == 1
    assert counts.get("function") == 1


def test_graph_view_holds_reference_to_supplied_graph(tmp_path):
    _write(tmp_path, {"app.py": "def f(): return 1\n"})
    a = Analysis(tmp_path, resolver=ManualResolver(specs=["."]))
    g = a.materialize_all()
    assert GraphView(g).graph is g
