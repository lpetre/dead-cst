from pathlib import PurePath
from typing import Any, List, Mapping

from libcst.helpers import ModuleNameAndPackage
from libcst.metadata import FullyQualifiedNameProvider


class FixedFullyQualifiedNameProvider(FullyQualifiedNameProvider):
    # libcst's calculate_module_and_package collapses ``pkg/__main__.py`` to
    # the package name ``pkg``, colliding with ``pkg/__init__.py``. The real
    # module name is ``pkg.__main__`` (or just ``__main__`` at the top level).
    @classmethod
    def gen_cache(
        cls,
        root_path,
        paths: List[str],
        **kwargs: Any,
    ) -> Mapping[str, ModuleNameAndPackage]:
        cache = super().gen_cache(root_path, paths, **kwargs)
        fixed: dict[str, ModuleNameAndPackage] = {}
        for path, mp in cache.items():
            if PurePath(path).name == "__main__.py":
                package = mp.name
                name = f"{package}.__main__" if package else "__main__"
                fixed[path] = ModuleNameAndPackage(name=name, package=package)
            else:
                fixed[path] = mp
        return fixed
