"""Tests for :class:`DunderAllPlugin`."""

from __future__ import annotations

from dead_cst import DunderAllPlugin, build_symbol_graph


def test_dunder_all_plugin_keeps_all_alive(tmp_path, write_files, reachable_fqnames):
    write_files({"pkg/__init__.py": '__all__ = ["a"]\na = 1'})
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[DunderAllPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.__all__" in reached
    # the listed symbol itself is *not* followed -- only __all__ stays alive.
    assert "pkg.a" not in reached
