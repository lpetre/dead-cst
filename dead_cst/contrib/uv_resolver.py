"""Resolver that reads ``uv.lock`` to discover first-party packages.

Works for both single-package uv projects (one ``[[package]]`` with
``source = { editable = "." }``) and multi-member workspaces (one
``[[package]]`` per workspace member, plus the workspace root as
``virtual = "."``). Each first-party package becomes one
:class:`~dead_cst.resolvers.Package`; each member's first-party
dependencies (from the lockfile's ``dependencies`` array) become
its ``deps`` (production DAG over package names).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..resolvers._core import Package, load_toml
from ..resolvers._exports import exported_tree_root
from ..resolvers._imports import default_resolve_import


@dataclass
class UvResolver:
    """Discover first-party packages from ``uv.lock``.

    For each ``[[package]]`` entry whose ``source`` is
    ``{ editable = "..." }`` or ``{ virtual = "..." }`` -- uv's two
    markers for first-party (non-PyPI) packages -- emit one
    :class:`Package`. ``editable`` packages are installable
    distributions; ``virtual`` packages are runnable apps/services
    that aren't shipped as wheels. Both are first-party code that
    needs to be analyzed.

    The exported portion of each package comes from
    :func:`~dead_cst.resolvers._exports.exported_tree_root`, which
    reads the package's ``pyproject.toml`` and picks the directory
    the build backend would actually ship. Members with no
    ``pyproject.toml`` and no ``src/`` directory have no exported
    portion at all (every file is internal).

    The workspace root itself (``virtual = "."`` in a multi-member
    workspace) is skipped -- it's a container that holds
    ``[tool.uv.workspace]``, not a package to walk.

    Direct first-party dependencies come from the lockfile's
    per-package ``dependencies`` array; they translate into ``deps``
    against the corresponding members' names. Transitive deps are
    reachable through the chain of returned packages and don't need
    to be re-listed per member.

    Run ``dead-cst`` with the project's venv active (``uv run
    dead-cst ...``) so third-party imports resolve against the
    workspace's installed distributions via the running Python's
    ``sys.path``.
    """

    lock_path: Path | None = None
    name: str = "uv"
    version: int = 1778025600

    def resolve(self, project_root: Path) -> list[Package]:
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

        out: list[Package] = []
        known_names = set(member_dirs)
        for name, member_dir in member_dirs.items():
            exported_root = exported_tree_root(member_dir)
            exported: tuple[Path, ...]
            if exported_root is not None:
                exported = (exported_root,)
            elif (member_dir / "src").is_dir():
                exported = ((member_dir / "src").resolve(),)
            else:
                exported = ()
            deps = tuple(d for d in member_deps[name] if d in known_names)
            out.append(
                Package(
                    path=member_dir,
                    name=name,
                    exported=exported,
                    deps=deps,
                )
            )
        return out

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
