"""Resolver that reads ``[tool.dead-cst]`` from a project's ``pyproject.toml``.

Two configuration shapes:

* Explicit -- ``[tool.dead-cst].packages`` lists each :class:`Package`
  with its ``path``, ``name``, ``exported`` subdirs, and ``deps``
  (other package names). The resolver copies those entries verbatim.
* Implicit -- when no ``packages`` table is present, the resolver
  falls back to the conventional layout: a single :class:`Package`
  rooted at ``project_root``, with ``exported`` derived from
  :func:`exported_tree_root` (``src/`` if present, otherwise the dir
  the build backend would ship). No deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._core import Package, load_toml
from ._exports import exported_tree_root
from ._imports import default_resolve_import


@dataclass
class PyprojectResolver:
    """Read ``[tool.dead-cst]`` (or fall back to conventional layouts)
    and turn it into a :class:`Package` list.

    Example ``pyproject.toml``::

        [project]
        name = "my-pkg"

        [[tool.dead - cst.packages]]
        path = "."
        name = "my-pkg"
        exported = ["src"]
        deps = []

    Without an explicit ``[tool.dead-cst]`` block, the resolver returns
    a single :class:`Package` rooted at ``project_root`` with
    ``exported`` set to whatever :func:`exported_tree_root` discovers.
    """

    name: str = "pyproject"
    version: int = 1778025600

    def resolve(self, project_root: Path) -> list[Package]:
        project_root = project_root.resolve()
        data = load_toml(project_root / "pyproject.toml")
        if data is None:
            return []
        explicit = data.get("tool", {}).get("dead-cst", {}).get("packages")
        if isinstance(explicit, list):
            return _packages_from_config(project_root, explicit)
        return _default_packages(project_root, data)

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)


def _packages_from_config(project_root: Path, entries: list) -> list[Package]:
    out: list[Package] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        entry: dict[str, object] = raw
        path = entry.get("path")
        name = entry.get("name")
        if not isinstance(path, str) or not isinstance(name, str) or not name:
            continue
        pkg_path = (project_root / path).resolve()
        raw_exported = entry.get("exported") or []
        exported: list[Path] = []
        if isinstance(raw_exported, list):
            for item in raw_exported:
                if isinstance(item, str):
                    exported.append((pkg_path / item).resolve())
        raw_deps = entry.get("deps") or []
        deps: list[str] = []
        if isinstance(raw_deps, list):
            for item in raw_deps:
                if isinstance(item, str) and item:
                    deps.append(item)
        out.append(
            Package(
                path=pkg_path,
                name=name,
                exported=tuple(exported),
                deps=tuple(deps),
            )
        )
    return out


def _default_packages(project_root: Path, data: dict) -> list[Package]:
    """Conventional-layout fallback when no ``[tool.dead-cst].packages`` is present.

    Single package rooted at ``project_root``, named after
    ``[project].name``, with ``exported`` set to the directory the
    build backend ships (per :func:`exported_tree_root`).
    """
    name = _project_name(data) or project_root.name or "root"
    exported_root = exported_tree_root(project_root)
    exported = (exported_root,) if exported_root is not None else ()
    return [
        Package(
            path=project_root,
            name=name,
            exported=exported,
            deps=(),
        )
    ]


def _project_name(data: dict) -> str | None:
    project = data.get("project") or {}
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    return name if isinstance(name, str) and name else None
