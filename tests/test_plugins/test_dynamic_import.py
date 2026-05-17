"""Tests for :class:`DynamicImportFallbackPlugin`.

The plugin reads ``EdgeFlags.DYNAMIC_IMPORT`` edges and fans each
flagged ``src -> module`` edge out to the module's exports. Today only
the rust backend emits the flag (the libcst pipeline inlines fan-out
at visit time), so most tests here build a libcst graph, manually
inject ``DYNAMIC_IMPORT`` edges, and exercise the plugin's
``finalize`` pass directly via :func:`apply_ops`.
"""

from __future__ import annotations

import pytest

from dead_cst.graph import EdgeFlags
from dead_cst.plugins import (
    DynamicImportFallbackPlugin,
    MainBlockPlugin,
    PluginContext,
    apply_ops,
)


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


# ---------------------------------------------------------------------------
# Rust backend integration — run(ctx) end-to-end via build_plugin_graph.
# The rust path emits DYNAMIC_IMPORT-flagged edges for __import__ /
# importlib.import_module calls, so this is where the plugin's effect
# is observable without hand-injection.
# ---------------------------------------------------------------------------


@pytest.mark.skip_when_backend("libcst")
def test_run_fans_importlib_call_to_module_exports(build_plugin_graph, reachable_fqnames):
    """``importlib.import_module('pkg.target')`` on the rust backend
    emits one DYN-flagged edge to ``pkg.target``. The plugin's
    ``run(ctx)`` should fan that edge out to every non-underscore
    top-level decl of the module, keeping them alive under
    reachability."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": (
                "import importlib\n"
                "def main(): importlib.import_module('pkg.target')\n"
                "if __name__ == '__main__': main()\n"
            ),
            "pkg/target.py": ("def public(): pass\ndef _private(): pass\ndef other(): pass\n"),
        },
        [DynamicImportFallbackPlugin(), MainBlockPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.public" in reached
    assert "pkg.target.other" in reached
    # Underscore name skipped by the default `include_underscore=False`.
    assert "pkg.target._private" not in reached


@pytest.mark.skip_when_backend("libcst")
def test_run_respects_dunder_all(build_plugin_graph, reachable_fqnames):
    """When the target module declares ``__all__``, only the listed
    names participate in the fan-out."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": (
                "import importlib\n"
                "def main(): importlib.import_module('pkg.target')\n"
                "if __name__ == '__main__': main()\n"
            ),
            "pkg/target.py": ("__all__ = ['kept']\ndef kept(): pass\ndef dropped(): pass\n"),
        },
        [DynamicImportFallbackPlugin(), MainBlockPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.kept" in reached
    assert "pkg.target.dropped" not in reached


@pytest.mark.skip_when_backend("libcst")
def test_run_include_underscore_picks_private_names(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": (
                "import importlib\n"
                "def main(): importlib.import_module('pkg.target')\n"
                "if __name__ == '__main__': main()\n"
            ),
            "pkg/target.py": "def public(): pass\ndef _private(): pass\n",
        },
        [
            DynamicImportFallbackPlugin(include_underscore=True),
            MainBlockPlugin(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.public" in reached
    assert "pkg.target._private" in reached


@pytest.mark.skip_when_backend("libcst")
def test_run_no_op_without_dynamic_imports(build_plugin_graph, reachable_fqnames):
    """With no ``__import__`` / ``importlib.import_module`` calls there
    are no DYN-flagged edges; the plugin emits nothing."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": (
                "from pkg.target import public\n"
                "def main(): public()\n"
                "if __name__ == '__main__': main()\n"
            ),
            "pkg/target.py": "def public(): pass\ndef other(): pass\n",
        },
        [DynamicImportFallbackPlugin(), MainBlockPlugin()],
    )
    reached = reachable_fqnames(graph)
    # Direct import of `public` — alive. `other` was never named, so
    # it stays dead.
    assert "pkg.target.public" in reached
    assert "pkg.target.other" not in reached


# ---------------------------------------------------------------------------
# Include / exclude filters
#
# The plugin's intended rollout has three stages: drop-in catch-all,
# catch-all + exclude_sources / exclude_targets to opt files out as
# focused plugins land, and finally include_sources / include_targets
# to flip from a long exclude list to an explicit allowlist.
# ---------------------------------------------------------------------------


@pytest.mark.skip_when_backend("libcst")
def test_exclude_sources_skips_matching_files(build_plugin_graph, reachable_fqnames):
    """A focused plugin handles ``pkg.handled_loader``; the catch-all
    is told to leave that file alone via ``exclude_sources``. The
    other dynamic-import call site still fans out."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/handled_loader.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.handled')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/other_loader.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.other')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/handled.py": "def handled_export(): pass\n",
            "pkg/other.py": "def other_export(): pass\n",
        },
        [
            DynamicImportFallbackPlugin(exclude_sources=("pkg/handled_loader.py",)),
            MainBlockPlugin(),
        ],
    )
    reached = reachable_fqnames(graph)
    # `pkg.other_loader` is not excluded, so its `importlib` call fans
    # out to `pkg.other.other_export`.
    assert "pkg.other.other_export" in reached
    # `pkg.handled_loader` is excluded; the catch-all stays out of it,
    # so `pkg.handled.handled_export` does NOT get a fan-out edge from
    # the loader. (A focused plugin would supply it; we're verifying
    # the catch-all stays silent.)
    assert "pkg.handled.handled_export" not in reached


@pytest.mark.skip_when_backend("libcst")
def test_exclude_sources_supports_glob_patterns(build_plugin_graph, reachable_fqnames):
    """``exclude_sources`` matches ``PurePosixPath.match`` so callers
    can write ``pkg/loaders/*.py`` to silence a whole directory."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/loaders/__init__.py": "",
            "pkg/loaders/a.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.target_a')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/loaders/b.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.target_b')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/target_a.py": "def a_export(): pass\n",
            "pkg/target_b.py": "def b_export(): pass\n",
        },
        [
            DynamicImportFallbackPlugin(exclude_sources=("pkg/loaders/*.py",)),
            MainBlockPlugin(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target_a.a_export" not in reached
    assert "pkg.target_b.b_export" not in reached


@pytest.mark.skip_when_backend("libcst")
def test_exclude_targets_silences_module_tree(build_plugin_graph, reachable_fqnames):
    """``exclude_targets`` filters by the imported module's fqname —
    fan-out is blocked even if the source file isn't excluded."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/loader.py": (
                "import importlib\n"
                "def boot():\n"
                "    importlib.import_module('pkg.live')\n"
                "    importlib.import_module('pkg.vendored.email')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/live.py": "def live_export(): pass\n",
            "pkg/vendored/__init__.py": "",
            "pkg/vendored/email.py": "def vendored_export(): pass\n",
        },
        [
            DynamicImportFallbackPlugin(exclude_targets=("pkg.vendored.*",)),
            MainBlockPlugin(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.live.live_export" in reached
    assert "pkg.vendored.email.vendored_export" not in reached


@pytest.mark.skip_when_backend("libcst")
def test_include_sources_acts_as_allowlist(build_plugin_graph, reachable_fqnames):
    """When ``include_sources`` is non-empty, only matching call sites
    participate. Useful once focused plugins cover most of the codebase
    and the catch-all is reduced to a small allowlist."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/legacy.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.legacy_target')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/modern.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.modern_target')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/legacy_target.py": "def legacy_export(): pass\n",
            "pkg/modern_target.py": "def modern_export(): pass\n",
        },
        [
            DynamicImportFallbackPlugin(include_sources=("pkg/legacy.py",)),
            MainBlockPlugin(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.legacy_target.legacy_export" in reached
    # `pkg/modern.py` is outside the include list; the catch-all
    # doesn't touch it. A focused plugin would handle `modern`.
    assert "pkg.modern_target.modern_export" not in reached


@pytest.mark.skip_when_backend("libcst")
def test_include_and_exclude_combine(build_plugin_graph, reachable_fqnames):
    """``include`` and ``exclude`` compose as ``include AND NOT exclude``
    — useful for "allowlist a directory but punch a hole in it"."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/legacy/__init__.py": "",
            "pkg/legacy/a.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.t_a')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/legacy/b.py": (
                "import importlib\n"
                "def boot(): importlib.import_module('pkg.t_b')\n"
                "if __name__ == '__main__': boot()\n"
            ),
            "pkg/t_a.py": "def a_export(): pass\n",
            "pkg/t_b.py": "def b_export(): pass\n",
        },
        [
            DynamicImportFallbackPlugin(
                include_sources=("pkg/legacy/*.py",),
                exclude_sources=("pkg/legacy/b.py",),
            ),
            MainBlockPlugin(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.t_a.a_export" in reached
    assert "pkg.t_b.b_export" not in reached
