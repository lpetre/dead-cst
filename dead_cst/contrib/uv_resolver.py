"""Resolver that reads ``uv.lock`` to discover first-party source trees.

Works for both single-package uv projects (one ``[[package]]`` with
``source = { editable = "." }`` and the project's own dist) and
multi-member workspaces (one ``[[package]]`` per workspace member,
plus the workspace root as ``virtual = "."``). Each first-party
package becomes one :class:`~dead_cst.resolvers.SourceTree`; each
member's first-party dependencies (from the lockfile's
``dependencies`` array) become its ``search_trees`` refs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..resolvers._core import SourceTree, SourceTreeFlags, load_toml
from ..resolvers._exports import exported_tree_root
from ..resolvers._imports import default_resolve_import


@dataclass
class UvResolver:
    """Discover first-party source trees from ``uv.lock``.

    For each ``[[package]]`` entry whose ``source`` is
    ``{ editable = "..." }`` or ``{ virtual = "..." }`` -- uv's two
    markers for first-party (non-PyPI) packages -- emit one
    ``EXPORTED`` :class:`SourceTree`. The exported tree's path comes
    from :func:`~dead_cst.resolvers._exports.exported_tree_root`,
    which reads the package's ``pyproject.toml`` and picks the
    directory the build backend would actually ship (``src/`` /
    ``[tool.hatch.build.targets.wheel].packages`` / etc.).
    ``editable`` packages are installable distributions; ``virtual``
    packages are runnable apps/services that aren't shipped as
    wheels. Both are first-party code that needs to be analyzed.

    The workspace root itself (``virtual = "."`` in a multi-member
    workspace) is skipped -- it's a container that holds
    ``[tool.uv.workspace]``, not a package to walk.

    Direct first-party dependencies come from the lockfile's
    per-package ``dependencies`` array; they translate into
    ``search_trees`` refs against the corresponding members'
    exported tree paths. Transitive deps are reachable through the
    chain of returned trees and don't need to be re-listed per
    member.

    Run ``dead-cst`` with the project's venv active (``uv run
    dead-cst ...``) so third-party imports resolve against the
    workspace's installed distributions via the running Python's
    ``sys.path``.
    """

    lock_path: Path | None = None
    name: str = "uv"
    version: int = 1777985838

    def resolve(self, project_root: Path) -> list[SourceTree]:
        project_root = project_root.resolve()
        data = load_toml(self.lock_path or project_root / "uv.lock")
        if data is None:
            return []

        member_dirs: dict[str, Path] = {}
        member_deps: dict[str, list[str]] = {}
        for pkg in data.get("package", []):
            if not isinstance(pkg, dict):
                continue
            source = pkg.get("source") or {}
            editable = source.get("editable")
            virtual = source.get("virtual")
            if editable is None and virtual is None:
                continue
            # ``virtual = "."`` is the workspace-root marker in
            # multi-member workspaces (a container that holds
            # ``[tool.uv.workspace]``, not a package to walk).
            # ``editable = "."`` for a single-package uv project is
            # fine -- that's the project itself, with its own
            # pyproject.toml and an exported tree.
            if virtual == ".":
                continue
            location = editable if editable is not None else virtual
            if not isinstance(location, str):
                continue
            member_dir = (project_root / location).resolve()
            name = pkg["name"]
            member_dirs[name] = member_dir
            member_deps[name] = [
                d["name"] for d in pkg.get("dependencies", []) if isinstance(d, dict)
            ]

        out: list[SourceTree] = []
        member_paths: dict[str, Path] = {}
        for name, member_dir in member_dirs.items():
            tree_root = _tree_root_for(member_dir)
            if tree_root is None:
                continue
            member_paths[name] = tree_root

        for name, tree_root in member_paths.items():
            search: list[Path] = []
            for dep_name in member_deps[name]:
                dep_path = member_paths.get(dep_name)
                if dep_path is not None:
                    search.append(dep_path)
            out.append(
                SourceTree(
                    path=tree_root,
                    package=name,
                    flags=SourceTreeFlags.EXPORTED,
                    search_trees=tuple(search),
                )
            )
        return out

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)


def _tree_root_for(member_dir: Path) -> Path | None:
    """Pick the exported tree path for one workspace member.

    Defers to :func:`~dead_cst.resolvers._exports.exported_tree_root`
    on the member's ``pyproject.toml`` so build-backend-shipped
    layouts (hatchling ``packages``, setuptools, poetry, pdm, flit)
    are honored. Falls back to the universal "src/ if present, else
    member dir" heuristic when the member has no ``pyproject.toml``
    (uv allows this for ``virtual`` members that aren't installable
    dists).
    """
    derived = exported_tree_root(member_dir)
    if derived is not None:
        return derived
    if (member_dir / "src").is_dir():
        return (member_dir / "src").resolve()
    if member_dir.is_dir():
        return member_dir.resolve()
    return None
