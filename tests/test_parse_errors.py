"""Tests for resilience to non-file paths and unparseable sources.

``rglob('*.py')`` matches by name, not by entry kind -- a directory
literally called ``something.py`` would otherwise sneak through. And
``libcst`` may reject any sufficiently broken syntax (or unsupported
future syntax). Both cases used to crash the analyser; the visitor
now degrades gracefully.
"""

from __future__ import annotations

import logging

# Source libcst cannot parse. Kept simple so the test stays meaningful
# regardless of which exact dialects libcst has caught up with.
BROKEN_SOURCE = "def f(\n"


def test_directory_named_dot_py_is_skipped(tmp_path, make_analysis, assert_edges):
    """A directory whose name ends in ``.py`` must not reach the visitor."""
    (tmp_path / "weird.py").mkdir()
    (tmp_path / "weird.py" / "marker.txt").write_text("not python")
    (tmp_path / "real.py").write_text("def f(): pass\nf()\n")

    graph = make_analysis().materialize_all()
    assert_edges(graph, {"real.f -> real", "real -> real.f"})


def test_unparseable_file_emits_synthetic_marker(tmp_path, make_analysis, assert_edges, caplog):
    """A file libcst can't parse becomes an ``[unparseable]`` placeholder.

    The visitor logs a warning, emits a payload pairing the real module
    node with an ``[unparseable] <module>`` synthetic flagged
    ``ENTRYPOINT``, and continues. Sibling files in the same package
    are unaffected.
    """
    (tmp_path / "broken.py").write_text(BROKEN_SOURCE)
    (tmp_path / "ok.py").write_text("def g(): pass\ng()\n")

    with caplog.at_level(logging.WARNING, logger="dead_cst._refresh"):
        graph = make_analysis().materialize_all()

    assert any("Could not parse" in r.getMessage() for r in caplog.records)
    assert_edges(
        graph,
        {
            "[unparseable] broken -> broken",
            "ok.g -> ok",
            "ok -> ok.g",
        },
    )


def test_unparseable_module_keeps_importer_alive(tmp_path, make_analysis):
    """An importer of an unparseable module still resolves to the module level.

    We can't see the unparseable module's decls, so ``from broken
    import x`` only places the cross-file edge at the module. That is
    enough to keep the importer chain reachable from the unparseable
    file's own ``[unparseable]`` entrypoint synthetic.
    """
    (tmp_path / "broken.py").write_text(BROKEN_SOURCE)
    (tmp_path / "consumer.py").write_text("from broken import greet\n")

    graph = make_analysis().materialize_all()

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    # The import node routes to the module (decl-level resolution is
    # impossible without the module's parsed decls).
    assert "consumer.greet -> broken" in edges
    # The synthetic marker keeps the unparseable module alive.
    assert "[unparseable] broken -> broken" in edges


def test_unparseable_payload_is_cached(tmp_path, make_analysis, caplog):
    """Cached unparseable payloads survive a re-run with the same source.

    Warm runs must not re-parse the failing file (otherwise every
    subsequent run repeats the warning storm). Once the source SHA
    changes -- e.g. after the user fixes the syntax -- the entry
    invalidates and the parse re-runs.
    """
    from dead_cst.cache import CACHE_DIR_NAME, GraphCache

    (tmp_path / "broken.py").write_text(BROKEN_SOURCE)
    db_path = tmp_path / CACHE_DIR_NAME / "cache.db"

    with caplog.at_level(logging.WARNING, logger="dead_cst._refresh"):
        with GraphCache(db_path) as cache:
            make_analysis(cache=cache).materialize_all()

    first_warnings = [r.getMessage() for r in caplog.records if "Could not parse" in r.getMessage()]
    assert len(first_warnings) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dead_cst._refresh"):
        with GraphCache(db_path) as cache:
            make_analysis(cache=cache).materialize_all()

    second_warnings = [
        r.getMessage() for r in caplog.records if "Could not parse" in r.getMessage()
    ]
    # Cache hit -> no re-parse, no warning.
    assert second_warnings == []
