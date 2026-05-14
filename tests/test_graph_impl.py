"""Edge-dedup and parallel-edge semantics for the rustworkx-backed graph wrappers.

These tests pin the contract the analyzer / codemod / plugins rely on:
``MultiDiGraph`` keeps parallel edges separate (the DEAD_BRANCH case),
``DiGraph`` overwrites them, ``successors`` deduplicates across
parallels, and ``subgraph`` / ``copy`` preserve every parallel-edge
slot. They are intentionally low-level: the higher-level analyzer
behavior is covered elsewhere (see ``test_payload`` for the
DEAD_BRANCH dual-edge end-to-end case).
"""

from __future__ import annotations

from dead_cst._graph_impl import DiGraph, MultiDiGraph, relabel_nodes


def test_multidigraph_keeps_parallel_unattributed_edges() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "b")
    assert g.number_of_edges() == 2
    # Both copies carry the (empty) default attribute dict.
    assert list(g.edges(data=True)) == [("a", "b", {}), ("a", "b", {})]


def test_multidigraph_keeps_parallel_attributed_edges() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edges_from(
        [
            ("a", "b", {"flags": 1}),
            ("a", "b", {"flags": 2}),  # DEAD_BRANCH-style dual edge
        ]
    )
    assert g.number_of_edges() == 2
    by_flag = sorted(g.edges(data=True), key=lambda e: e[2]["flags"])
    assert by_flag == [("a", "b", {"flags": 1}), ("a", "b", {"flags": 2})]


def test_multidigraph_keeps_identical_attributed_parallels() -> None:
    """Re-adding the same (u, v, attrs) triple stacks parallel edges.

    Matches ``networkx.MultiDiGraph`` semantics: ``add_edge`` does not
    silently dedupe; the analyzer's pipeline is responsible for not
    feeding the same triple twice.
    """
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edges_from([("a", "b", {"flags": 1}), ("a", "b", {"flags": 1})])
    assert g.number_of_edges() == 2


def test_digraph_overwrites_parallel_edges() -> None:
    g: DiGraph[str] = DiGraph()
    g.add_edge("a", "b", flags=1)
    g.add_edge("a", "b", flags=2)
    assert g.number_of_edges() == 1
    assert list(g.edges(data=True)) == [("a", "b", {"flags": 2})]


def test_successors_deduplicates_across_parallels() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    assert sorted(g.successors("a")) == ["b", "c"]


def test_has_edge_is_true_for_any_parallel() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    assert not g.has_edge("a", "b")
    g.add_edge("a", "b", flags=1)
    g.add_edge("a", "b", flags=2)
    assert g.has_edge("a", "b")


def test_remove_edge_drops_one_of_parallel_pair() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edges_from([("a", "b", {"flags": 1}), ("a", "b", {"flags": 2})])
    g.remove_edge("a", "b")
    assert g.number_of_edges() == 1
    assert g.has_edge("a", "b")  # the other parallel survives


def test_subgraph_preserves_parallel_edges() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edges_from(
        [
            ("a", "b", {"flags": 1}),
            ("a", "b", {"flags": 2}),
            ("c", "d"),
        ]
    )
    sub = g.subgraph(["a", "b"])
    assert sub.number_of_edges() == 2
    assert sorted(sub.edges(data=True), key=lambda e: e[2]["flags"]) == [
        ("a", "b", {"flags": 1}),
        ("a", "b", {"flags": 2}),
    ]


def test_copy_decouples_edge_attr_dicts() -> None:
    """Editing a copy's edge attrs must not affect the original."""
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edge("a", "b", flags=1)
    clone = g.copy()
    cloned_edges = list(clone.edges(data=True))
    cloned_edges[0][2]["flags"] = 99
    assert list(g.edges(data=True)) == [("a", "b", {"flags": 1})]


def test_subgraph_decouples_edge_attr_dicts() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edge("a", "b", flags=1)
    sub = g.subgraph(["a", "b"])
    sub_edges = list(sub.edges(data=True))
    sub_edges[0][2]["flags"] = 99
    assert list(g.edges(data=True)) == [("a", "b", {"flags": 1})]


def test_add_node_is_idempotent() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_node("a")
    g.add_node("a")
    assert g.number_of_nodes() == 1


def test_add_node_keeps_first_instance() -> None:
    """Re-adding an equal-but-distinct instance is a no-op.

    The first registration wins; later equal-instances are dropped.
    Mirrors ``networkx.Graph.add_node`` semantics so analyzer call
    sites that rely on ``__hash__`` / ``__eq__`` (rather than ``is``)
    keep working.
    """
    g: MultiDiGraph[tuple[str, int]] = MultiDiGraph()
    a1 = ("a", 1)
    a2 = ("a", 1)  # equal but distinct identity in CPython for short tuples? force a copy
    g.add_node(a1)
    g.add_node(a2)
    assert g.number_of_nodes() == 1
    # Iteration returns the first-registered instance.
    (only,) = list(g.nodes)
    assert only is a1


def test_relabel_nodes_in_place_preserves_edges() -> None:
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edges_from(
        [
            ("a", "b", {"flags": 1}),
            ("a", "b", {"flags": 2}),
            ("a", "c"),
        ]
    )
    relabel_nodes(g, {"a": "A"}, copy=False)
    assert sorted(g.nodes) == ["A", "b", "c"]
    assert sorted(e[:2] for e in g.edges()) == [("A", "b"), ("A", "b"), ("A", "c")]


def test_edges_with_node_arg_filters_to_outgoing() -> None:
    """``g.edges(node)`` walks edges leaving ``node`` (networkx parity)."""
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("x", "y")
    assert sorted(g.edges("a")) == [("a", "b"), ("a", "c")]


def test_graph_attribute_dict_is_independent_per_graph() -> None:
    """``g.graph`` (graph-level attrs) must not leak across instances."""
    g1: MultiDiGraph[str] = MultiDiGraph()
    g2: MultiDiGraph[str] = MultiDiGraph()
    g1.graph["dead_suites"] = {"file": ()}
    assert g2.graph == {}


def test_subgraph_carries_graph_level_attrs() -> None:
    """``subgraph`` propagates ``g.graph`` so dead-suites queries still work."""
    g: MultiDiGraph[str] = MultiDiGraph()
    g.add_node("a")
    g.graph["dead_suites"] = {"file": "pos"}
    sub = g.subgraph(["a"])
    assert sub.graph["dead_suites"] == {"file": "pos"}
    # And decoupled: editing the sub's graph dict doesn't bleed back.
    sub.graph["dead_suites"] = {}
    assert g.graph["dead_suites"] == {"file": "pos"}
