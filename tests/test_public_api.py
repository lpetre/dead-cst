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
        "PackageView",
        "SymbolNode",
        "__version__",
    ],
    "dead_cst.analyze": [
        "Analysis",
        "PackageView",
    ],
    "dead_cst.codemod": ["generate_patch", "remove_code"],
    "dead_cst.graph": [
        "EdgeFlags",
        "Import",
        "NodeFlags",
        "SymbolNode",
    ],
    "dead_cst.plugins": [
        "BUILTIN_PLUGINS",
        "CeleryPlugin",
        "ClickPlugin",
        "CycloptsPlugin",
        "DecoratedDeclPlugin",
        "DiscordPyPlugin",
        "DispatchAppPlugin",
        "DynamicImportFallbackPlugin",
        "EXTERNAL_DIST_PREFIX",
        "EXTERNAL_FILE_PREFIX",
        "EXTERNAL_PREFIXES",
        "ExplicitEntrypointPlugin",
        "FastAPIPlugin",
        "FastMCPPlugin",
        "FlaskPlugin",
        "InitSubclassPlugin",
        "LiteralListPlugin",
        "MainBlockPlugin",
        "MockPatchPlugin",
        "ModuleDundersPlugin",
        "ProjectScriptsPlugin",
        "PytestPlugin",
        "STDLIB_PREFIX",
        "SYNTHETIC_PATH_PREFIXES",
        "ServerConfigPlugin",
        "TyperPlugin",
        "UNPARSEABLE_PREFIX",
        "UNRESOLVED_PREFIX",
        "UnittestPlugin",
        "load_plugin",
        "simple_name",
    ],
    "dead_cst.contrib": [
        "CeleryPlugin",
        "ClickPlugin",
        "CycloptsPlugin",
        "DiscordPyPlugin",
        "FastAPIPlugin",
        "FastMCPPlugin",
        "FlaskPlugin",
        "MockPatchPlugin",
        "PytestPlugin",
        "ServerConfigPlugin",
        "TyperPlugin",
        "UnittestPlugin",
        "UvResolver",
    ],
    "dead_cst.resolvers": [
        "BUILTIN_RESOLVERS",
        "ManualResolver",
        "Package",
        "PathResolver",
        "UvResolver",
        "load_resolver",
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
