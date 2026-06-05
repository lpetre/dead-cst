"""Smoke tests for the topic/fact channel getters on ``ProjectContext``.

The full declare-topic -> emit-fact -> read-fact round trip runs through an
external (dylib) native plugin and is exercised by the CI-gated plugin-host
suite; the dev/static build can't load external plugins. These tests pin the
locally-observable surface: with no topic-declaring plugin registered the
registry is empty, and ``facts_for_topic`` is guarded behind materialization.
"""

from __future__ import annotations

import pytest

from dead_cst import _native as native


def test_topic_registry_empty_without_topic_plugins(build_decl_graph):
    graph = build_decl_graph({"mod.py": "def a(): pass\n"})
    assert graph.topic_registry() == []
    assert graph.facts_for_topic("acme/anything") == []


def test_topic_registry_empty_before_materialize(tmp_path):
    # ``topic_registry`` reads the registry built at construction time, so it
    # decodes even on a never-materialized context (here: empty, no plugins).
    ctx = native.ProjectContext(str(tmp_path))
    assert ctx.topic_registry() == []


def test_facts_for_topic_raises_before_materialize(tmp_path):
    # Facts only exist after a build, so the getter rejects a call made
    # outside an active materialize() (e.g. from a plugin's run()).
    ctx = native.ProjectContext(str(tmp_path))
    with pytest.raises(RuntimeError):
        ctx.facts_for_topic("acme/anything")
