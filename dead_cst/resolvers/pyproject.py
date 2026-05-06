"""Resolver that reads ``[tool.dead-cst]`` from a project's ``pyproject.toml``.

Two configuration shapes:

* Explicit -- ``[tool.dead-cst].trees`` lists each :class:`SourceTree`
  with its ``path``, ``package``, ``exported`` flag, and
  ``search_trees`` (paths to other configured trees). The resolver
  copies those entries verbatim and returns them.
* Implicit -- when no ``trees`` table is present, the resolver falls
  back to the conventional layout via :func:`exported_tree_root`: a
  single ``EXPORTED`` tree at the project's importable directory
  (``src/`` if present, else what the build backend ships, else the
  project root) plus an optional non-exported ``tests/`` tree if a
  sibling ``tests/`` dir exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._core import SourceTree, SourceTreeFlags, load_toml
from ._exports import exported_tree_root
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
        explicit = data.get("tool", {}).get("dead-cst", {}).get("trees")
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

    Picks the project's exported root from
    :func:`~dead_cst.resolvers._exports.exported_tree_root`, names the
    package after ``[project].name``, and threads a sibling ``tests/``
    tree (if any) onto the same package.
    """
    package = _project_name(data) or project_root.name or "root"
    exported_root = exported_tree_root(project_root)
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
