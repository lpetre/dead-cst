from __future__ import annotations

from pathlib import PurePath
from typing import Any, Mapping

from libcst.helpers import ModuleNameAndPackage
from libcst.metadata import FullyQualifiedNameProvider

# Synthetic FQN segment appended to a ``.pyi`` stub module's FQN so it
# doesn't collide with its same-named ``.py`` sibling. (libcst's
# ``calculate_module_and_package`` strips the file suffix, so both
# ``mod.py`` and ``mod.pyi`` would otherwise resolve to ``mod``.)
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
                # ``mod.pyi`` becomes ``mod.__pyi__``. ``package`` is
                # left at the runtime name so relative imports inside
                # the stub resolve against the same package as the
                # runtime module would.
                stub_name = f"{mp.name}.{PYI_FQN_SEGMENT}"
                fixed[path] = ModuleNameAndPackage(name=stub_name, package=mp.package)
            else:
                fixed[path] = mp
        return fixed
