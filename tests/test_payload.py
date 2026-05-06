"""Tests for the per-file ``VisitorPayload`` shape and its apply step.

The visitor's per-file output is collapsed into :class:`VisitorPayload`
with four fields: ``nodes``, ``edges``, ``imports``, ``dead_suites``.
``_analyze._apply_payload`` reads them, derives
:data:`EdgeFlags.DEAD_BRANCH` per edge by checking each access
position against the file's dead-suite list, and emits the per-file
contribution to the symbol graph.

These tests cover the new layer directly:

* the flag plumbing on :class:`SymbolNode` and the new
  :class:`EdgeFlags` IntFlag,
* the per-file ``VisitorPayload`` shape and pickle round-trip (the
  cache pre-flight),
* edge-flag derivation: refs inside dead suites land with
  ``DEAD_BRANCH``, refs outside don't.

End-to-end behavior of the analyzer is already covered by
``test_declarations`` / ``test_unreachable_branches`` /
``test_imports``; this file focuses on the payload contract itself.
"""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path

from libcst.metadata import CodePosition, CodeRange, FullRepoManager

from dead_cst import Analysis, EdgeFlags, NodeFlags
from dead_cst._fqn import FixedFullyQualifiedNameProvider
from dead_cst.graph import SymbolNode
from dead_cst._visitor import SymbolVisitor
from dead_cst.graph import VisitorPayload
from conftest import manual


def _pos(line: int = 1, column: int = 0) -> CodeRange:
    return CodeRange(start=CodePosition(line, column), end=CodePosition(line, column + 1))


# ---------------------------------------------------------------------------
# Flag enums
# ---------------------------------------------------------------------------


def test_node_flags_default_is_none():
    """Bare ``SymbolNode`` construction leaves ``flags`` cleared.

    The default needs to stay ``NONE`` so existing call sites that
    construct ``SymbolNode`` without flags (plugins, tests) keep
    producing live, unshadowed nodes.
    """
    n = SymbolNode("pkg.f", "function", Path("/a.py"), _pos())
    assert n.flags is NodeFlags.NONE
    assert not (n.flags & NodeFlags.SHADOWED)


def test_node_flags_compose_via_or():
    """An :class:`IntFlag` lets callers OR multiple markers together."""
    combined = NodeFlags.SHADOWED | NodeFlags.SHADOWED
    assert combined & NodeFlags.SHADOWED


def test_edge_flags_dead_branch_distinct_from_none():
    """``DEAD_BRANCH`` is a distinct bit from ``NONE``."""
    assert EdgeFlags.DEAD_BRANCH & EdgeFlags.DEAD_BRANCH
    assert not (EdgeFlags.NONE & EdgeFlags.DEAD_BRANCH)


# ---------------------------------------------------------------------------
# VisitorPayload shape
# ---------------------------------------------------------------------------


def _payload_from_source(tmp_path: Path, src: str) -> tuple[VisitorPayload, Path]:
    """Run ``SymbolVisitor`` over ``src`` and return its payload.

    Bypasses ``build_symbol_graph`` so we can inspect the visitor's own
    output -- i.e. what would be cached -- without the per-base
    apply-and-resolve work on top.
    """
    file = tmp_path / "pkg.py"
    file.write_text(textwrap.dedent(src).strip() + "\n")
    mgr = FullRepoManager(str(tmp_path), [str(file)], {FixedFullyQualifiedNameProvider})
    wrapper = mgr.get_metadata_wrapper_for_path(str(file))
    visitor = SymbolVisitor(file, [tmp_path], wrapper=wrapper)
    wrapper.visit(visitor)
    return visitor.to_payload(), file


def test_payload_module_node_present(tmp_path):
    """Every payload lists exactly one ``type='module'`` node for the file."""
    payload, _ = _payload_from_source(
        tmp_path,
        """
        def f(): pass
        """,
    )
    modules = [n for n in payload.nodes if n.type == "module"]
    assert len(modules) == 1
    assert modules[0].fqname == "pkg"


def test_payload_shadowed_decl_carries_flag(tmp_path):
    """Decls displaced by flow analysis are emitted with ``SHADOWED``.

    The visitor's ``_finalize_module_declarations`` partitions same-name
    decls into live vs. displaced via :func:`live_at_exit`; the
    displaced copies pick up :data:`NodeFlags.SHADOWED` only at
    payload-construction time. Live decls never gain the flag.
    """
    payload, _ = _payload_from_source(
        tmp_path,
        """
        def f():
            return 1

        def f():
            return 2
        """,
    )
    fs = sorted(
        (n for n in payload.nodes if n.fqname == "pkg.f"),
        key=lambda n: n.position.start.line,
    )
    assert len(fs) == 2
    assert fs[0].flags & NodeFlags.SHADOWED
    assert not (fs[1].flags & NodeFlags.SHADOWED)


def test_payload_records_dead_suites(tmp_path):
    """Statically-dead suites land in ``payload.dead_suites``.

    The list is the source of truth for "where in the file are
    statically-unreachable regions" -- the apply step uses it for
    edge-flag derivation, and the CLI surfaces it for the
    "Unreachable branches" report.
    """
    payload, _ = _payload_from_source(
        tmp_path,
        """
        def used(): pass

        if False:
            used()

        if False:
            pass
        """,
    )
    assert len(payload.dead_suites) == 2


def test_payload_edge_carries_access_pos(tmp_path):
    """Each edge entry has an access-position third field.

    The position is what the apply step compares against
    ``dead_suites`` to decide whether the resulting graph edge gets
    ``EdgeFlags.DEAD_BRANCH``. Type-checked here as ``CodeRange``.
    """
    payload, _ = _payload_from_source(
        tmp_path,
        """
        def used(): pass

        used()
        """,
    )
    assert payload.edges
    for src, dst, pos in payload.edges:
        assert isinstance(src, SymbolNode)
        assert isinstance(dst, SymbolNode)
        assert isinstance(pos, CodeRange)


def test_apply_flags_dead_branch_edges(tmp_path, write_files):
    """End-to-end: refs inside dead suites land with ``DEAD_BRANCH``."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
                def helper(): pass

                helper()      # live ref
                if False:
                    helper()  # dead ref
            """,
        }
    )
    graph = Analysis(tmp_path, resolvers=manual()).materialize_all()
    helper = next(n for n in graph.nodes if n.fqname == "pkg.a.helper")
    module = next(n for n in graph.nodes if n.fqname == "pkg.a")

    # MultiDiGraph: parallel edges between the same pair are kept
    # separate. Both live and dead-branch refs to ``helper`` produce
    # edges from the module; the flag distinguishes them.
    edges = list(graph.edges(module, data=True))
    flags_seen = {attrs.get("flags", EdgeFlags.NONE) for _, dst, attrs in edges if dst == helper}
    assert EdgeFlags.NONE in flags_seen
    assert EdgeFlags.DEAD_BRANCH in flags_seen


def test_payload_pickle_round_trip(tmp_path):
    """Payload survives pickle / unpickle.

    This is the sanity check for the upcoming SQLite cache work --
    payloads will be pickled into the ``file_cache`` table. If the
    nested ``CodeRange`` / ``Path`` / ``Import`` types ever stop being
    picklable, this test catches it before the cache PR lands.
    """
    payload, _ = _payload_from_source(
        tmp_path,
        """
        from typing import List

        def used(): pass

        def f():
            return used()

        def f():
            return 2

        if False:
            used()
        """,
    )
    restored = pickle.loads(pickle.dumps(payload))
    assert restored == payload


# ---------------------------------------------------------------------------
# Graph-level integration
# ---------------------------------------------------------------------------


def test_shadowed_decl_in_graph_keeps_consistent_identity(tmp_path, write_files):
    """Shadowed decl + edges referencing it share one graph node.

    The risk of mishandled remapping is two graph nodes for one
    logical decl: an unflagged copy implicitly created by ``add_edge``
    plus the flagged copy added by ``add_node``. Confirm by walking
    the assembled graph and asserting every edge endpoint that
    matches a shadowed decl IS that flagged instance, not a parallel
    unflagged twin.
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
                def f():
                    return 1

                g = f

                def f():
                    return 2
            """,
        }
    )
    g = Analysis(tmp_path, resolvers=manual()).materialize_all()
    shadowed = [n for n in g.nodes if n.flags & NodeFlags.SHADOWED]
    assert len(shadowed) == 1
    s = shadowed[0]
    # Every graph node with this fqname must be either ``s`` itself or
    # the live (unflagged) sibling -- never an unflagged copy of the
    # shadowed binding.
    matches = [n for n in g.nodes if n.fqname == s.fqname]
    assert len(matches) == 2
    flagged_matches = [n for n in matches if n.flags & NodeFlags.SHADOWED]
    assert flagged_matches == [s]


def test_dead_suites_exposed_on_graph(tmp_path, write_files):
    """``graph.graph['dead_suites']`` lists positions per analyzed file."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
                if False:
                    pass
            """,
        }
    )
    g = Analysis(tmp_path, resolvers=manual()).materialize_all()
    suites = g.graph["dead_suites"]
    file = tmp_path / "pkg" / "a.py"
    assert file in suites
    assert len(suites[file]) == 1
