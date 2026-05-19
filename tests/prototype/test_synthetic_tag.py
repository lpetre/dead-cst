"""Tests for the structured `SyntheticTag` plugin-coordination contract.

`SyntheticTag(plugin, kind, payload)` lets plugins stamp typed metadata
onto a synthetic node they emit via `AddNode(..., tag=...)`. Other
plugins (and the emitting plugin's own second-pass code) look the
synthetics up via `ctx.find_synthetic(plugin, kind=None)` instead of
parsing the fqname string.

These tests pin:

* The pyclass round-trips: emit with a tag, observe the same three
  fields back on the resulting `SymbolNode.tag`.
* `find_synthetic(plugin, kind)` and `find_synthetic(plugin)` discriminate
  correctly across plugins and roles.
* Untagged `AddNode` ops produce nodes with `tag is None`.
* The tag participates in the node intern key — two `AddNode` ops that
  share fqname/path but differ in their tag are interned as distinct
  nodes (so a plugin can emit two markers under the same fqname
  template with different structured payloads).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

native = pytest.importorskip("dead_cst._native")


@pytest.fixture
def make_ctx(tmp_path: Path):
    def make(files: dict[str, str], **kwargs) -> native.ProjectContext:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.ProjectContext(str(tmp_path), **kwargs)

    return make


def _tagged_nodes(graph: native.NativeGraph) -> list[native.SymbolNode]:
    return [n for n in graph.nodes if n.tag is not None]


def _find_node(graph: native.NativeGraph, fqname: str) -> native.SymbolNode:
    for n in graph.nodes:
        if n.fqname == fqname:
            return n
    raise AssertionError(f"node {fqname!r} not in graph")


class _OnePlugin:
    """Emits one tagged synthetic per call. Pinned for tag fields."""

    def __init__(self, plugin: str, kind: str, payload: str):
        self._plugin = plugin
        self._kind = kind
        self._payload = payload

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        for n in ctx.nodes():
            if n.fqname == "mod" and n.kind == "module":
                yield native.AddNode(
                    fqname=f"<{self._plugin}-{self._kind}>:marker",
                    path=n.path,
                    edges_to=[n],
                    tag=native.SyntheticTag(
                        plugin=self._plugin,
                        kind=self._kind,
                        payload=self._payload,
                    ),
                )
                break


def test_synthetic_tag_round_trips(make_ctx):
    """A tag set on AddNode is observable on the resulting SymbolNode."""
    ctx = make_ctx({"mod.py": "x = 1\n"})
    ctx.add_plugin(_OnePlugin(plugin="p", kind="k", payload="v"))
    graph = ctx.materialize()

    tagged = _tagged_nodes(graph)
    assert len(tagged) == 1
    node = tagged[0]
    assert node.tag is not None
    assert node.tag.plugin == "p"
    assert node.tag.kind == "k"
    assert node.tag.payload == "v"


def test_find_synthetic_by_plugin_and_kind(make_ctx):
    """Two plugins emit disjoint tagged synthetics; the lookup
    discriminates by (plugin, kind)."""

    class TwoTags:
        """Single plugin emitting two synthetics with different kinds.

        Models how DispatchAppPlugin emits both ``app`` and ``factory``
        markers, then queries for just the ``factory`` ones.
        """

        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
            for n in ctx.nodes():
                if n.fqname == "mod" and n.kind == "module":
                    yield native.AddNode(
                        fqname="<mine-a>:1",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="mine", kind="a", payload="1"),
                    )
                    yield native.AddNode(
                        fqname="<mine-b>:2",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="mine", kind="b", payload="2"),
                    )
                    yield native.AddNode(
                        fqname="<other-a>:3",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="other", kind="a", payload="3"),
                    )
                    break

    seen_by_kind: dict[str, list[str]] = {}

    class Probe:
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
            seen_by_kind["mine-a"] = [n.fqname for n in ctx.find_synthetic(plugin="mine", kind="a")]
            seen_by_kind["mine-b"] = [n.fqname for n in ctx.find_synthetic(plugin="mine", kind="b")]
            seen_by_kind["other-a"] = [
                n.fqname for n in ctx.find_synthetic(plugin="other", kind="a")
            ]
            seen_by_kind["missing"] = [
                n.fqname for n in ctx.find_synthetic(plugin="mine", kind="zzz")
            ]
            return None

    ctx = make_ctx({"mod.py": "x = 1\n"})
    ctx.add_plugin(TwoTags())
    ctx.add_plugin(Probe())
    ctx.materialize()

    assert seen_by_kind["mine-a"] == ["<mine-a>:1"]
    assert seen_by_kind["mine-b"] == ["<mine-b>:2"]
    assert seen_by_kind["other-a"] == ["<other-a>:3"]
    assert seen_by_kind["missing"] == []


def test_find_synthetic_by_plugin_only(make_ctx):
    """``kind=None`` returns every synthetic tagged with the plugin,
    regardless of role."""

    class TwoTags:
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
            for n in ctx.nodes():
                if n.fqname == "mod" and n.kind == "module":
                    yield native.AddNode(
                        fqname="<mine-a>:1",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="mine", kind="a", payload="1"),
                    )
                    yield native.AddNode(
                        fqname="<mine-b>:2",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="mine", kind="b", payload="2"),
                    )
                    yield native.AddNode(
                        fqname="<other-a>:3",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="other", kind="a", payload="3"),
                    )
                    break

    seen: list[str] = []

    class Probe:
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
            seen.extend(n.fqname for n in ctx.find_synthetic(plugin="mine"))
            return None

    ctx = make_ctx({"mod.py": "x = 1\n"})
    ctx.add_plugin(TwoTags())
    ctx.add_plugin(Probe())
    ctx.materialize()

    # Emission order preserved; ``other-a`` excluded by the plugin
    # filter.
    assert seen == ["<mine-a>:1", "<mine-b>:2"]


def test_untagged_synthetic_node_tag_is_none(make_ctx):
    """``AddNode`` without ``tag=`` produces a node with ``tag is None``."""

    class Untagged:
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
            for n in ctx.nodes():
                if n.fqname == "mod" and n.kind == "module":
                    yield native.AddNode(
                        fqname="<untagged>:1",
                        path=n.path,
                        edges_to=[n],
                    )
                    break

    ctx = make_ctx({"mod.py": "x = 1\n"})
    ctx.add_plugin(Untagged())
    graph = ctx.materialize()

    node = _find_node(graph, "<untagged>:1")
    assert node.tag is None
    # The visitor-side module / decl nodes are also untagged.
    module = _find_node(graph, "mod")
    assert module.tag is None


def test_two_addnode_with_same_fqname_different_tag_dedup(make_ctx):
    """Two ``AddNode`` ops with the same fqname/path but different
    tags intern as distinct nodes — tag participates in the intern key.

    Mirrors ``SymbolNode.__eq__`` semantics: nodes that compare equal
    share an intern slot; nodes that differ in any field (including
    tag) do not.
    """

    class TwoOps:
        def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
            for n in ctx.nodes():
                if n.fqname == "mod" and n.kind == "module":
                    # Identical fqname / path; tag differs only in kind.
                    yield native.AddNode(
                        fqname="<dup>:1",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="dup", kind="a", payload="1"),
                    )
                    yield native.AddNode(
                        fqname="<dup>:1",
                        path=n.path,
                        edges_to=[n],
                        tag=native.SyntheticTag(plugin="dup", kind="b", payload="1"),
                    )
                    # And another with no tag — also a third intern
                    # slot (tag=None is distinguishable from tag=Some).
                    yield native.AddNode(
                        fqname="<dup>:1",
                        path=n.path,
                        edges_to=[n],
                    )
                    break

    ctx = make_ctx({"mod.py": "x = 1\n"})
    ctx.add_plugin(TwoOps())
    graph = ctx.materialize()

    dup_nodes = [n for n in graph.nodes if n.fqname == "<dup>:1"]
    assert len(dup_nodes) == 3

    tag_kinds = sorted(n.tag.kind if n.tag is not None else "<none>" for n in dup_nodes)
    assert tag_kinds == ["<none>", "a", "b"]
