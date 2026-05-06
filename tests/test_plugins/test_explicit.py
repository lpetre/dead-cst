"""Tests for :class:`ExplicitEntrypointPlugin`."""

from __future__ import annotations

import re

from dead_cst import Analysis
from dead_cst.plugins import ExplicitEntrypointPlugin


def test_explicit_entrypoint_by_fqname(tmp_path, write_files, reachable_fqnames):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = Analysis(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=["pkg.a.f"])],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.a.f" in reachable_fqnames(graph)


def test_explicit_entrypoint_by_relpath(tmp_path, write_files, reachable_fqnames):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = Analysis(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=["pkg/a.py"])],
        project_root=tmp_path,
    ).materialize_all()
    assert {"pkg.a", "pkg.a.f"} <= reachable_fqnames(graph)


def test_explicit_entrypoint_by_regex(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/entry.py": "from .a import f\nf()",
            "pkg/a.py": "def f(): pass",
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=[re.compile(r".*entry\.py")])],
        project_root=tmp_path,
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.entry" in reached
    assert "pkg.a.f" in reached
