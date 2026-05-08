"""Tests for :class:`ExplicitEntrypointPlugin`."""

from __future__ import annotations

import re

from dead_cst.plugins import ExplicitEntrypointPlugin


def test_explicit_entrypoint_by_fqname(make_analysis, write_files, reachable_fqnames):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["pkg.a.f"])]).materialize_all()
    assert "pkg.a.f" in reachable_fqnames(graph)


def test_explicit_entrypoint_by_relpath(make_analysis, write_files, reachable_fqnames):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["pkg/a.py"])]).materialize_all()
    assert {"pkg.a", "pkg.a.f"} <= reachable_fqnames(graph)


def test_explicit_entrypoint_by_regex(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/entry.py": "from .a import f\nf()",
            "pkg/a.py": "def f(): pass",
        }
    )
    graph = make_analysis(
        plugins=[ExplicitEntrypointPlugin(specs=[re.compile(r".*entry\.py")])]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.entry" in reached
    assert "pkg.a.f" in reached
