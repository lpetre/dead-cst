"""Tests for :class:`DynamicImportFallbackPlugin`.

The plugin reads ``EdgeFlags.DYNAMIC_IMPORT`` edges and fans each
flagged ``src -> module`` edge out to the module's exports. Today only
the rust backend emits the flag (the libcst pipeline inlines fan-out
at visit time), so most tests here build a libcst graph, manually
inject ``DYNAMIC_IMPORT`` edges, and exercise the plugin's
``finalize`` pass directly via :func:`apply_ops`.
"""

from __future__ import annotations

from dead_cst.graph import EdgeFlags
from dead_cst.plugins import DynamicImportFallbackPlugin, PluginContext, apply_ops


def _run_finalize(plugin, analysis, graph):
    """Run ``plugin.finalize`` once per package on the materialized graph.

    Mirrors ``Analysis._materialize``'s plugin loop. Tests use this
    after they've injected ``DYNAMIC_IMPORT`` edges into the graph by
    hand. Reaches into ``Analysis._contributions`` /
    ``_build_symbol_lookup`` to reconstruct the same ``PluginContext``
    the analyzer would have handed the plugin during ``_materialize``.
    """
    for pkg in analysis.packages:
        contribution = analysis._contributions[pkg.path]
        ctx = PluginContext(
            graph=graph,
            symbol_lookup=analysis._build_symbol_lookup(pkg.path),
            contribution=contribution,
            project_root=analysis.project_root,
        )
        apply_ops(graph, plugin.finalize(ctx), set())
    return graph


def _inject_dyn_edge(graph, src_fqname: str, dst_fqname: str) -> None:
    """Add ``src -> dst`` to ``graph`` flagged ``EdgeFlags.DYNAMIC_IMPORT``."""
    src = next(n for n in graph.nodes if n.fqname == src_fqname)
    dst = next(n for n in graph.nodes if n.fqname == dst_fqname)
    graph.add_edge(src, dst, flags=EdgeFlags.DYNAMIC_IMPORT)


# ---------------------------------------------------------------------------
# Plugin shape
# ---------------------------------------------------------------------------


def test_plugin_satisfies_protocol():
    plugin = DynamicImportFallbackPlugin()
    assert plugin.name == "dynamic_import_fallback"
    assert isinstance(plugin.version, int)
    # observe must exist and return None for an arbitrary file.
    assert plugin.observe(None) is None  # type: ignore[arg-type]


def test_plugin_loadable_by_name():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("dynamic_import_fallback")
    assert isinstance(plugin, DynamicImportFallbackPlugin)


# ---------------------------------------------------------------------------
# Fan-out behavior with hand-injected DYNAMIC_IMPORT edges
# ---------------------------------------------------------------------------


def test_dynamic_import_edge_fans_out_to_module_exports(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": "def use(): pass\n",
            "pkg/target.py": "def f(): pass\ndef g(): pass\n",
        }
    )
    analysis = make_analysis(plugins=[DynamicImportFallbackPlugin()])
    graph = analysis.materialize_all()
    _inject_dyn_edge(graph, "pkg.loader.use", "pkg.target")

    graph = _run_finalize(DynamicImportFallbackPlugin(), analysis, graph)

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    assert "pkg.loader.use -> pkg.target.f" in edges
    assert "pkg.loader.use -> pkg.target.g" in edges


def test_dynamic_import_skips_underscore_exports_by_default(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": "def use(): pass\n",
            "pkg/target.py": "def public(): pass\ndef _private(): pass\n",
        }
    )
    analysis = make_analysis(plugins=[DynamicImportFallbackPlugin()])
    graph = analysis.materialize_all()
    _inject_dyn_edge(graph, "pkg.loader.use", "pkg.target")

    graph = _run_finalize(DynamicImportFallbackPlugin(), analysis, graph)

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    assert "pkg.loader.use -> pkg.target.public" in edges
    assert "pkg.loader.use -> pkg.target._private" not in edges


def test_include_underscore_fans_to_private_names(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": "def use(): pass\n",
            "pkg/target.py": "def public(): pass\ndef _private(): pass\n",
        }
    )
    analysis = make_analysis()
    graph = analysis.materialize_all()
    _inject_dyn_edge(graph, "pkg.loader.use", "pkg.target")

    graph = _run_finalize(DynamicImportFallbackPlugin(include_underscore=True), analysis, graph)

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    assert "pkg.loader.use -> pkg.target.public" in edges
    assert "pkg.loader.use -> pkg.target._private" in edges


def test_dunder_all_filters_fan_out(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": "def use(): pass\n",
            "pkg/target.py": ("__all__ = ['kept']\ndef kept(): pass\ndef dropped(): pass\n"),
        }
    )
    analysis = make_analysis()
    graph = analysis.materialize_all()
    _inject_dyn_edge(graph, "pkg.loader.use", "pkg.target")

    graph = _run_finalize(DynamicImportFallbackPlugin(), analysis, graph)

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    assert "pkg.loader.use -> pkg.target.kept" in edges
    # `dropped` would have been a non-underscore export, but `__all__`
    # excludes it.
    assert "pkg.loader.use -> pkg.target.dropped" not in edges


def test_respect_dunder_all_false_falls_back_to_underscore_filter(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": "def use(): pass\n",
            "pkg/target.py": (
                "__all__ = ['kept']\ndef kept(): pass\ndef public_but_not_in_all(): pass\n"
            ),
        }
    )
    analysis = make_analysis()
    graph = analysis.materialize_all()
    _inject_dyn_edge(graph, "pkg.loader.use", "pkg.target")

    graph = _run_finalize(DynamicImportFallbackPlugin(respect_dunder_all=False), analysis, graph)

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    assert "pkg.loader.use -> pkg.target.kept" in edges
    # Without __all__ respect, every non-underscore name is a target.
    assert "pkg.loader.use -> pkg.target.public_but_not_in_all" in edges


def test_non_module_dynamic_target_is_left_alone(make_analysis, write_files):
    """``__import__('p', fromlist=['f'])`` already points at the decl;
    nothing to fan out."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": "def use(): pass\n",
            "pkg/target.py": "def f(): pass\ndef g(): pass\n",
        }
    )
    analysis = make_analysis()
    graph = analysis.materialize_all()
    # Inject the edge pointing at the decl ``pkg.target.f`` (not the
    # module), so the plugin should skip it.
    _inject_dyn_edge(graph, "pkg.loader.use", "pkg.target.f")

    graph = _run_finalize(DynamicImportFallbackPlugin(), analysis, graph)

    edges = {f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()}
    assert "pkg.loader.use -> pkg.target.f" in edges  # the original flagged edge
    # No fan-out to sibling ``g`` — the plugin only fans out
    # module-targeted edges.
    assert "pkg.loader.use -> pkg.target.g" not in edges


def test_libcst_pipeline_no_op(make_analysis, write_files):
    """The libcst path inlines fan-out at visit time without setting
    ``DYNAMIC_IMPORT``, so the plugin sees no flagged edges and
    contributes nothing — it's safe to enable on every project."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": (
                "import importlib\ndef main(): importlib.import_module('pkg.target')\n"
            ),
            "pkg/target.py": "def f(): pass\n",
        }
    )
    analysis = make_analysis(plugins=[DynamicImportFallbackPlugin()])
    # Snapshot the edge set before plugin finalize runs, then again
    # after — the plugin should be a complete no-op (no fan-out edges,
    # no exceptions) when no DYNAMIC_IMPORT-flagged edges exist.
    graph = analysis.materialize_all()
    edges = {(u, v) for u, v in graph.raw.edge_list()}
    flagged = [
        (u, v) for u, v, flags in graph.raw.weighted_edge_list() if flags & EdgeFlags.DYNAMIC_IMPORT
    ]
    assert flagged == []
    # Re-run finalize and confirm the edge set is unchanged.
    _run_finalize(DynamicImportFallbackPlugin(), analysis, graph)
    assert {(u, v) for u, v in graph.raw.edge_list()} == edges
