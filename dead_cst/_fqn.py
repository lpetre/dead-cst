from __future__ import annotations

from pathlib import PurePath
from typing import Any, Mapping

from libcst.helpers import ModuleNameAndPackage
from libcst.metadata import FullyQualifiedNameProvider

# Synthetic FQN segment that distinguishes a ``.pyi`` stub module from its
# same-named ``.py`` runtime sibling. ``calculate_module_and_package``
# strips the suffix, so ``mod.pyi`` and ``mod.py`` would otherwise both
# claim FQN ``mod`` and collide in the symbol trie. Appending this
# segment puts every ``.pyi`` decl under ``<module>.<PYI_FQN_SEGMENT>.<name>``
# so the runtime and stub namespaces stay disjoint; cross-module imports
# of ``mod.f`` therefore resolve to the runtime decl, not the stub.
PYI_FQN_SEGMENT = "__pyi__"


class FixedFullyQualifiedNameProvider(FullyQualifiedNameProvider):
    # libcst's calculate_module_and_package collapses ``pkg/__main__.py`` to
    # the package name ``pkg``, colliding with ``pkg/__init__.py``. The real
    # module name is ``pkg.__main__`` (or just ``__main__`` at the top level).
    @classmethod
    def gen_cache(
        cls,
        root_path,
        paths: list[str],
        **kwargs: Any,
    ) -> Mapping[str, ModuleNameAndPackage]:
        cache = super().gen_cache(root_path, paths, **kwargs)
        fixed: dict[str, ModuleNameAndPackage] = {}
        for path, mp in cache.items():
            pp = PurePath(path)
            if pp.name == "__main__.py":
                package = mp.name
                name = f"{package}.__main__" if package else "__main__"
                fixed[path] = ModuleNameAndPackage(name=name, package=package)
            elif pp.suffix == ".pyi":
                # Disambiguate from the matching ``.py`` so both can
                # coexist in the same trie. ``mod.py`` keeps FQN ``mod``;
                # ``mod.pyi`` becomes ``mod.__pyi__``. The package field
                # (used to resolve relative imports) is the original
                # module name -- the stub module sits under it as a
                # synthetic submodule, but its imports still resolve as
                # though they ran from the runtime module's package.
                stub_name = f"{mp.name}.{PYI_FQN_SEGMENT}"
                fixed[path] = ModuleNameAndPackage(name=stub_name, package=mp.package)
            else:
                fixed[path] = mp
        return fixed
