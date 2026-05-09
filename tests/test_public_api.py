"""Pin the public surface so accidental ``__all__`` drops are caught.

The test does two things for each public module:

1. Asserts that every name in ``__all__`` is actually importable from
   that module (catches typos and stale entries).
2. Asserts the exact set of names in ``__all__`` against a snapshot
   here. Adding a new public name requires updating this file --
   intentional friction, since these are alpha-stable promises that
   downstream callers may pin.

Module-level ``__all__`` defines the contract. Names imported into a
public module but not listed in ``__all__`` (e.g. ``CodeRange``,
``Path``) are intentionally excluded.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED_PUBLIC_API: dict[str, list[str]] = {
    "dead_cst": [
        "Analysis",
        "Cacheable",
        "EdgeFlags",
        "Import",
        "NodeFlags",
        "PackageView",
        "SymbolNode",
        "__version__",
    ],
    "dead_cst.analyze": [
        "Analysis",
        "PackageView",
    ],
    "dead_cst.branches": [
        "DefaultUnreachableRegionDetector",
        "ResolveExpr",
        "TruthinessResolver",
        "UnreachableRegionDetector",
        "evaluate_truthiness",
        "unreachable_bodies",
        "unreachable_suites",
    ],
    "dead_cst.cache": [
        "GraphCache",
        "SCHEMA_VERSION",
        "clear_cache",
        "compute_fingerprint",
        "default_cache_path",
    ],
    "dead_cst.codemod": ["remove_code"],
    "dead_cst.graph": [
        "EdgeFlags",
        "Import",
        "NodeFlags",
        "SymbolNode",
        "VisitorPayload",
    ],
    "dead_cst.plugins": [
        "AddEdge",
        "AddNode",
        "BUILTIN_PLUGINS",
        "ClickPlugin",
        "CycloptsPlugin",
        "DecoratedDeclPlugin",
        "EXTERNAL_DIST_PREFIX",
        "EXTERNAL_FILE_PREFIX",
        "EXTERNAL_PREFIXES",
        "EdgePlugin",
        "ExplicitEntrypointPlugin",
        "FastAPIPlugin",
        "FlaskPlugin",
        "GraphOp",
        "InitSubclassPlugin",
        "LiteralListPlugin",
        "MainBlockPlugin",
        "MockPatchPlugin",
        "ModuleDundersPlugin",
        "ObserveContext",
        "PluginContext",
        "ProjectScriptsPlugin",
        "PytestPlugin",
        "RemoveEdge",
        "STDLIB_PREFIX",
        "SYNTHETIC_PATH_PREFIXES",
        "SYNTHETIC_POSITION",
        "TyperPlugin",
        "UNPARSEABLE_PREFIX",
        "UNRESOLVED_PREFIX",
        "UnittestPlugin",
        "UnresolvedDependencyError",
        "apply_ops",
        "collect_module_imports",
        "decls_by_simple_name",
        "decorator_owner",
        "dotted_name",
        "dotted_parts",
        "entrypoint_payload",
        "find_call_assignments",
        "find_handlers",
        "is_from_module",
        "is_name",
        "load_plugin",
        "make_payload",
        "mark_entrypoints",
        "matched_attr_call",
        "module_node",
        "payload_imports_module",
        "require_resolved_dep",
        "simple_name",
        "single_target_assignment",
        "string_value",
        "synthetic_node",
        "walk_to_instance_kind",
    ],
    "dead_cst.contrib": [
        "ClickPlugin",
        "CycloptsPlugin",
        "FastAPIPlugin",
        "FlaskPlugin",
        "MockPatchPlugin",
        "PytestPlugin",
        "TyperPlugin",
        "UnittestPlugin",
        "UvResolver",
    ],
    "dead_cst.resolvers": [
        "BUILTIN_RESOLVERS",
        "ImportResolver",
        "ManualResolver",
        "Package",
        "PathResolver",
        "SITE_PACKAGES_MARKERS",
        "STDLIB",
        "UvResolver",
        "clear_path_caches",
        "default_resolve_import",
        "distribution_lookup",
        "editable_distribution_roots",
        "exported_roots",
        "load_resolver",
        "load_toml",
        "safe_resolve_module",
    ],
}


@pytest.mark.parametrize("module_name", sorted(EXPECTED_PUBLIC_API))
def test_public_all_matches_expected(module_name: str) -> None:
    """``module.__all__`` matches the snapshot (sorted)."""
    mod = importlib.import_module(module_name)
    assert sorted(mod.__all__) == sorted(EXPECTED_PUBLIC_API[module_name]), (
        f"{module_name}.__all__ drifted from the snapshot in test_public_api.py. "
        "Update the snapshot if the change is intentional."
    )


@pytest.mark.parametrize("module_name", sorted(EXPECTED_PUBLIC_API))
def test_every_all_entry_is_importable(module_name: str) -> None:
    """Every name in ``__all__`` resolves to an attribute of the module."""
    mod = importlib.import_module(module_name)
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert not missing, f"{module_name}.__all__ lists unresolvable names: {missing}"
