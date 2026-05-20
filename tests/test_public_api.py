"""Pin the public surface so accidental ``__all__`` drops are caught."""

from __future__ import annotations

import importlib

import pytest

EXPECTED_PUBLIC_API: dict[str, list[str]] = {
    "dead_cst": [
        "Analysis",
        "EdgeFlags",
        "Import",
        "NodeFlags",
        "SymbolNode",
        "__version__",
    ],
    "dead_cst.analyze": [
        "Analysis",
    ],
    "dead_cst.codemod": ["generate_patch", "remove_code"],
    "dead_cst.graph": [
        "EdgeFlags",
        "GraphMetadata",
        "Import",
        "KEEPALIVE_DEFAULT",
        "LoadedGraph",
        "NodeFlags",
        "SymbolNode",
        "read_graph",
        "write_graph",
    ],
    "dead_cst.plugins": [
        "DecoratedDeclPlugin",
        "DispatchAppPlugin",
        "DynamicImportFallbackPlugin",
        "EXTERNAL_DIST_PREFIX",
        "EXTERNAL_FILE_PREFIX",
        "EXTERNAL_PREFIXES",
        "ExplicitEntrypointPlugin",
        "InitSubclassPlugin",
        "LiteralListPlugin",
        "MainBlockPlugin",
        "ModuleDundersPlugin",
        "Plugin",
        "ProjectScriptsPlugin",
        "STDLIB_PREFIX",
        "SYNTHETIC_PATH_PREFIXES",
        "UNPARSEABLE_PREFIX",
        "UNRESOLVED_PREFIX",
        "simple_name",
    ],
    "dead_cst.contrib": [
        "CeleryPlugin",
        "ClickPlugin",
        "DiscordPyPlugin",
        "MockPatchPlugin",
        "PytestPlugin",
        "ServerConfigPlugin",
        "UnittestPlugin",
        "UvResolver",
        "cyclopts_plugin",
        "fastapi_plugin",
        "fastmcp_plugin",
        "flask_plugin",
        "typer_plugin",
    ],
    "dead_cst.resolvers": [
        "ManualResolver",
        "Package",
        "PathResolver",
        "load_toml",
    ],
}


@pytest.mark.parametrize("module_name", sorted(EXPECTED_PUBLIC_API))
def test_public_all_matches_expected(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert sorted(mod.__all__) == sorted(EXPECTED_PUBLIC_API[module_name]), (
        f"{module_name}.__all__ drifted from the snapshot in test_public_api.py."
    )


@pytest.mark.parametrize("module_name", sorted(EXPECTED_PUBLIC_API))
def test_every_all_entry_is_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert not missing, f"{module_name}.__all__ lists unresolvable names: {missing}"
