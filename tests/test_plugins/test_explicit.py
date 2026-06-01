"""Tests for the native ``explicit`` entrypoint plugin."""

from __future__ import annotations

import re
from pathlib import Path

from dead_cst import _native as native


def _explicit(specs):
    """Bucket ``str | Path | re.Pattern`` specs the way the CLI does and
    build the native ``explicit`` plugin from the three typed lists."""
    regexes: list[str] = []
    str_specs: list[str] = []
    abs_paths: list[str] = []
    for spec in specs:
        if isinstance(spec, re.Pattern):
            regexes.append(spec.pattern)
        elif isinstance(spec, Path):
            abs_paths.append(str(spec))
        else:
            str_specs.append(spec)
    return native.NativePlugin.explicit(regexes, str_specs, abs_paths)


def test_explicit_entrypoint_by_fqname(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"},
        [_explicit(["pkg.a.f"])],
    )
    assert "pkg.a.f" in reachable_fqnames(graph)


def test_explicit_entrypoint_by_relpath(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"},
        [_explicit(["pkg/a.py"])],
    )
    assert {"pkg.a", "pkg.a.f"} <= reachable_fqnames(graph)


def test_explicit_entrypoint_by_regex(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/entry.py": "from .a import f\nf()",
            "pkg/a.py": "def f(): pass",
        }
    )
    graph = make_analysis(plugins=[_explicit([re.compile(r".*entry\.py")])]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.entry" in reached
    assert "pkg.a.f" in reached
