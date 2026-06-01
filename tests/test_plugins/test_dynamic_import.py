"""Tests for the native ``dynamic_import_fallback`` plugin.

The plugin reads ``EdgeFlags.DYNAMIC_IMPORT`` edges and fans each
flagged ``src -> module`` edge out to the module's exports.
"""

from __future__ import annotations

from dead_cst import _native as native


def test_plugin_loadable_by_name():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("dynamic_import_fallback")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "DynamicImportFallbackPlugin"


def test_run_fans_importlib_call_to_module_exports(build_plugin_graph, reachable_fqnames):
    """``importlib.import_module('pkg.target')`` emits one DYN-flagged
    edge to ``pkg.target``. The plugin's ``run(ctx)`` should fan that
    edge out to every non-underscore top-level decl of the module."""
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
        [native.NativePlugin.dynamic_import_fallback(), native.NativePlugin.main_block()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.public" in reached
    assert "pkg.target.other" in reached
    # Underscore name skipped by the default `include_underscore=False`.
    assert "pkg.target._private" not in reached


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
        [native.NativePlugin.dynamic_import_fallback(), native.NativePlugin.main_block()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.kept" in reached
    assert "pkg.target.dropped" not in reached


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
            native.NativePlugin.dynamic_import_fallback(include_underscore=True),
            native.NativePlugin.main_block(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.public" in reached
    assert "pkg.target._private" in reached


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
        [native.NativePlugin.dynamic_import_fallback(), native.NativePlugin.main_block()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target.public" in reached
    assert "pkg.target.other" not in reached


def test_exclude_sources_skips_matching_files(build_plugin_graph, reachable_fqnames):
    """A focused plugin handles ``pkg.handled_loader``; the catch-all
    is told to leave that file alone via ``exclude_sources``."""
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
            native.NativePlugin.dynamic_import_fallback(exclude_sources=("pkg/handled_loader.py",)),
            native.NativePlugin.main_block(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.other.other_export" in reached
    assert "pkg.handled.handled_export" not in reached


def test_exclude_sources_supports_glob_patterns(build_plugin_graph, reachable_fqnames):
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
            native.NativePlugin.dynamic_import_fallback(exclude_sources=("pkg/loaders/*.py",)),
            native.NativePlugin.main_block(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.target_a.a_export" not in reached
    assert "pkg.target_b.b_export" not in reached


def test_exclude_targets_silences_module_tree(build_plugin_graph, reachable_fqnames):
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
            native.NativePlugin.dynamic_import_fallback(exclude_targets=("pkg.vendored.*",)),
            native.NativePlugin.main_block(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.live.live_export" in reached
    assert "pkg.vendored.email.vendored_export" not in reached


def test_include_sources_acts_as_allowlist(build_plugin_graph, reachable_fqnames):
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
            native.NativePlugin.dynamic_import_fallback(include_sources=("pkg/legacy.py",)),
            native.NativePlugin.main_block(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.legacy_target.legacy_export" in reached
    assert "pkg.modern_target.modern_export" not in reached


def test_include_and_exclude_combine(build_plugin_graph, reachable_fqnames):
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
            native.NativePlugin.dynamic_import_fallback(
                include_sources=("pkg/legacy/*.py",),
                exclude_sources=("pkg/legacy/b.py",),
            ),
            native.NativePlugin.main_block(),
        ],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.t_a.a_export" in reached
    assert "pkg.t_b.b_export" not in reached
