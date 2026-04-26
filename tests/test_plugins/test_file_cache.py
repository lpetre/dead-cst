"""Tests for :class:`FileTextCache` and the ``ctx.grep`` plugin surface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx

from dead_cst import (
    FileTextCache,
    MainBlockPlugin,
    build_symbol_graph,
)
from dead_cst._plugins import PluginContext
from dead_cst._plugins._core import GraphOp
from dead_cst._symbols import SymbolTrie


def _make_cache(tmp_path, files):
    paths = []
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        paths.append(p)
    return FileTextCache(paths)


def test_file_text_cache_substring_grep(tmp_path):
    cache = _make_cache(
        tmp_path,
        {"a.py": "import fastapi\n", "b.py": "print('hi')\n", "c.py": "from fastapi import X\n"},
    )
    matches = set(cache.grep("fastapi"))
    assert {p.name for p in matches} == {"a.py", "c.py"}


def test_file_text_cache_regex_grep(tmp_path):
    cache = _make_cache(
        tmp_path,
        {"a.py": "x = FastAPI()\n", "b.py": "x = APIRouter()\n", "c.py": "pass\n"},
    )
    pat = re.compile(rb"FastAPI|APIRouter")
    matches = {p.name for p in cache.grep(pat)}
    assert matches == {"a.py", "b.py"}


def test_file_text_cache_caches_reads(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("hello world\n")
    cache = FileTextCache([p])
    assert cache.contains(p, "hello")
    p.write_text("totally different content\n")
    # Second call should still see the cached bytes, not the new content.
    assert cache.contains(p, "hello")
    assert not cache.contains(p, "totally")


def test_file_text_cache_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.py"
    cache = FileTextCache([missing])
    assert cache.read(missing) == b""
    assert list(cache.grep("anything")) == []


def test_file_text_cache_explicit_paths_override_default(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("import fastapi\n")
    b = tmp_path / "b.py"
    b.write_text("import fastapi\n")
    cache = FileTextCache([a])
    assert list(cache.grep("fastapi")) == [a]
    assert set(cache.grep("fastapi", paths=[a, b])) == {a, b}


def test_plugin_context_grep_delegates_to_cache(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("__main__ guard\n")
    b = tmp_path / "b.py"
    b.write_text("nothing here\n")
    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        paths={tmp_path: []},
        project_root=tmp_path,
        file_cache=FileTextCache([a, b]),
    )
    assert list(ctx.grep("__main__")) == [a]


def test_plugin_context_grep_default_cache_is_empty():
    """A bare PluginContext (no cache supplied) yields nothing rather than crashing."""
    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        paths={},
        project_root=Path("/"),
    )
    assert list(ctx.grep("anything")) == []


@dataclass
class _RecordingPlugin:
    """CST-aware plugin that records every path it would have parsed."""

    name: str = "recorder"
    cst_aware: bool = True
    seen: list[Path] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.seen = []

    def contribute(self, ctx: PluginContext, managers) -> Iterable[GraphOp]:
        for path in ctx.grep("__main__"):
            self.seen.append(path)
        return ()


def test_grep_prefilter_skips_unrelated_modules(tmp_path, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/has_main.py": """
            if __name__ == "__main__":
                pass
            """,
            "pkg/no_main.py": "def f(): pass",
        }
    )
    plugin = _RecordingPlugin()
    build_symbol_graph(
        {tmp_path: []},
        plugins=[plugin, MainBlockPlugin()],
        project_root=tmp_path,
    )
    seen = {p.name for p in plugin.seen}
    assert "has_main.py" in seen
    assert "no_main.py" not in seen
