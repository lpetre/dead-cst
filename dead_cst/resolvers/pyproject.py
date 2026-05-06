"""Resolver that reads ``[tool.dead-cst]`` from a project's ``pyproject.toml``.

Two configuration shapes:

* Explicit -- ``[tool.dead-cst].trees`` lists each :class:`SourceTree`
  with its ``path``, ``package``, ``exported`` flag, and
  ``search_trees`` (paths to other configured trees). The resolver
  copies those entries verbatim and returns them.
* Implicit -- when no ``trees`` table is present, the resolver falls
  back to the conventional layout: a single ``EXPORTED`` tree at the
  project's importable directory (``src/`` if present, else the project
  root) plus an optional non-exported ``tests/`` tree if a sibling
  ``tests/`` dir exists. The package's exported directory is
  determined from the build backend's wheel-target metadata so analysis
  matches what the package actually ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._core import SourceTree, SourceTreeFlags, load_toml
from ._imports import default_resolve_import


@dataclass
class PyprojectResolver:
    """Read ``[tool.dead-cst]`` (or fall back to conventional layouts)
    and turn it into a :class:`SourceTree` list.

    Example ``pyproject.toml``::

        [project]
        name = "my-pkg"

        [[tool.dead - cst.trees]]
        path = "src"
        package = "my-pkg"
        exported = true

        [[tool.dead - cst.trees]]
        path = "tests"
        package = "my-pkg"
        search_trees = ["src"]

    Without an explicit ``[tool.dead-cst]`` block, the resolver returns
    the project's importable dir as a single ``EXPORTED`` tree. A
    sibling ``tests/`` directory contributes a non-exported tree under
    the same package, with the exported tree as its search target.
    """

    name: str = "pyproject"
    version: int = 1777985838

    def resolve(self, project_root: Path) -> list[SourceTree]:
        project_root = project_root.resolve()
        data = load_toml(project_root / "pyproject.toml")
        if data is None:
            return []

        tool = (
            data.get("tool", {}).get("dead-cst", {}) if isinstance(data.get("tool"), dict) else {}
        )
        explicit = tool.get("trees") if isinstance(tool, dict) else None
        if isinstance(explicit, list):
            return _trees_from_config(project_root, explicit)

        return _default_trees(project_root, data)

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)


def _trees_from_config(project_root: Path, entries: list) -> list[SourceTree]:
    out: list[SourceTree] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        entry: dict[str, object] = raw
        path = entry.get("path")
        package = entry.get("package")
        if not isinstance(path, str) or not isinstance(package, str) or not package:
            continue
        flags = SourceTreeFlags.NONE
        if entry.get("exported"):
            flags |= SourceTreeFlags.EXPORTED
        raw_search = entry.get("search_trees") or []
        search: list[Path] = []
        if isinstance(raw_search, list):
            for item in raw_search:
                if isinstance(item, str):
                    search.append((project_root / item).resolve())
        out.append(
            SourceTree(
                path=(project_root / path).resolve(),
                package=package,
                flags=flags,
                search_trees=tuple(search),
            )
        )
    return out


def _default_trees(project_root: Path, data: dict) -> list[SourceTree]:
    """Conventional-layout fallback when no ``[tool.dead-cst].trees`` is present.

    Picks the project's exported root from the build backend's wheel
    target (or the universal ``src/`` shortcut), names the package
    after ``[project].name``, and threads a sibling ``tests/`` tree
    (if any) onto the same package.
    """
    package = _project_name(data) or project_root.name or "root"
    exported_root = _exported_root(project_root, data)
    if exported_root is None:
        return []

    trees: list[SourceTree] = [
        SourceTree(
            path=exported_root,
            package=package,
            flags=SourceTreeFlags.EXPORTED,
            search_trees=(),
        )
    ]

    tests = (project_root / "tests").resolve()
    if tests != exported_root and tests.is_dir():
        trees.append(
            SourceTree(
                path=tests,
                package=package,
                flags=SourceTreeFlags.NONE,
                search_trees=(exported_root,),
            )
        )
    return trees


def _project_name(data: dict) -> str | None:
    project = data.get("project") or {}
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    return name if isinstance(name, str) and name else None


def _exported_root(project_root: Path, data: dict) -> Path | None:
    """Pick the directory the build backend would ship as the package.

    Discovery order (first match wins):

    1. ``<project_root>/src/`` exists -> ``<project_root>/src``. The
       universal src-layout convention; no backend introspection.
    2. ``[build-system].build-backend`` is read and dispatched against
       backend-specific keys (``[tool.hatch.build.targets.wheel].packages``,
       ``[tool.setuptools].packages``, ``[tool.poetry].packages``,
       ``[tool.pdm.build].includes``, ``[tool.flit.module].name``). The
       *parent* of the first listed package directory is returned, so
       both ``src/foo`` and a flat ``foo/`` map to the right directory.
    3. ``<project_root>/<normalized [project].name>/__init__.py`` --
       the auto-discovery layout used by hatchling and setuptools when
       no explicit packages list is given.
    4. ``<project_root>`` itself, when nothing else matched but a
       :data:`SourceTree` is still required.
    """
    if (project_root / "src").is_dir():
        return (project_root / "src").resolve()

    backend = (data.get("build-system") or {}).get("build-backend")
    tool = data.get("tool") or {}

    pkg_dirs: list[Path] | None = None
    if backend == "hatchling.build":
        wheel = (tool.get("hatch") or {}).get("build", {}).get("targets", {}).get("wheel", {})
        packages = wheel.get("packages")
        if isinstance(packages, list):
            pkg_dirs = [(project_root / p).resolve() for p in packages if isinstance(p, str)]
    elif backend in ("setuptools.build_meta", "setuptools.build_meta:__legacy__"):
        st = tool.get("setuptools") or {}
        packages = st.get("packages")
        if isinstance(packages, list):
            pkg_dirs = [
                (project_root / p.replace(".", "/")).resolve()
                for p in packages
                if isinstance(p, str)
            ]
    elif backend in ("poetry.core.masonry.api", "poetry_core.masonry.api"):
        packages = (tool.get("poetry") or {}).get("packages")
        if isinstance(packages, list):
            pkg_dirs = []
            for p in packages:
                if isinstance(p, dict):
                    inc = p.get("include")
                    if not isinstance(inc, str):
                        continue
                    frm = p.get("from", "")
                    pkg_dirs.append((project_root / frm / inc).resolve())
                elif isinstance(p, str):
                    pkg_dirs.append((project_root / p).resolve())
    elif backend == "pdm.backend":
        includes = (tool.get("pdm") or {}).get("build", {}).get("includes")
        if isinstance(includes, list):
            pkg_dirs = [
                (project_root / inc).resolve()
                for inc in includes
                if isinstance(inc, str) and "*" not in inc
            ]
    elif backend == "flit_core.buildapi":
        mod = (tool.get("flit") or {}).get("module", {}).get("name")
        if isinstance(mod, str):
            pkg_dirs = [(project_root / mod).resolve()]

    if pkg_dirs:
        # The exported tree is the *parent* of the first listed package
        # directory: ``src/foo`` -> ``src``, ``foo`` -> project root,
        # ``foo/sub`` -> ``foo``. Files under that parent get walked
        # via longest-prefix-match against the tree's path.
        return pkg_dirs[0].parent

    project_name = _project_name(data)
    if project_name:
        normalized = project_name.replace("-", "_").replace(".", "_")
        candidate = project_root / normalized
        if (candidate / "__init__.py").is_file():
            return project_root.resolve()

    return project_root.resolve()
