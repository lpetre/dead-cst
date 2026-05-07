"""Per-project export discovery from ``pyproject.toml``.

:func:`exported_roots` returns the subdirs of a project root that the
build backend would ship as the wheel's importable surface. The
``uv``-aware contrib resolver uses it to pick each member's exported
:class:`~dead_cst.resolvers.Package` subdir; custom resolvers can
call it the same way to stay consistent with the rest of the
ecosystem.
"""

from __future__ import annotations

from pathlib import Path

from ._core import load_toml


def exported_roots(project_dir: Path) -> list[Path] | None:
    """Subdirs of ``project_dir`` that the build backend would ship.

    Mirrors what a Python build backend would actually package. For a
    project with a non-``src/`` flat layout this lets the analyzer
    pin the exported root to the directory the wheel ships, so files
    in adjacent ``tests/`` / ``scripts/`` / ``examples/`` directories
    don't accidentally claim "exported" status.

    Returns ``None`` when no restriction can be inferred (no
    ``pyproject.toml``, unknown backend, no ``[project].name``). The
    caller usually treats ``None`` as "fall back to a layout
    heuristic" -- e.g. ``src/`` if it exists, else ``project_dir``
    itself.

    Discovery order, first match wins:

    1. ``<project_dir>/src/`` exists -> ``[<project_dir>/src]``. The
       universal src-layout convention; no backend introspection
       needed.
    2. ``[build-system].build-backend`` is read and dispatched:

       - ``hatchling.build`` -> ``[tool.hatch.build.targets.wheel].packages``
       - ``setuptools.build_meta`` (and ``__legacy__``) ->
         ``[tool.setuptools.packages]``
       - ``poetry.core.masonry.api`` -> ``[tool.poetry].packages``
         (with ``include`` / ``from``)
       - ``pdm.backend`` -> ``[tool.pdm.build].includes`` (literal
         entries; globs skipped)
       - ``flit_core.buildapi`` -> ``[tool.flit.module].name``

    3. Fallback: a top-level dir named after the normalized
       ``[project].name`` containing ``__init__.py``. Covers the
       auto-discovery shape used by hatchling and setuptools when no
       explicit ``packages`` list is given.

    Setuptools' ``packages.find`` (with ``include`` / ``exclude``
    patterns) is not interpreted; falls through to the name-match
    fallback. Backends not listed above also fall through.
    """
    project_dir = project_dir.resolve()
    data = load_toml(project_dir / "pyproject.toml")
    if data is None:
        return None

    if (project_dir / "src").is_dir():
        return [(project_dir / "src").resolve()]

    backend = (data.get("build-system") or {}).get("build-backend")
    tool = data.get("tool") or {}

    roots: list[Path] | None = None
    if backend == "hatchling.build":
        wheel = (tool.get("hatch") or {}).get("build", {}).get("targets", {}).get("wheel", {})
        packages = wheel.get("packages")
        if isinstance(packages, list):
            roots = [(project_dir / p).resolve() for p in packages if isinstance(p, str)]
    elif backend in ("setuptools.build_meta", "setuptools.build_meta:__legacy__"):
        st = tool.get("setuptools") or {}
        packages = st.get("packages")
        if isinstance(packages, list):
            roots = [
                (project_dir / p.replace(".", "/")).resolve()
                for p in packages
                if isinstance(p, str)
            ]
    elif backend in ("poetry.core.masonry.api", "poetry_core.masonry.api"):
        packages = (tool.get("poetry") or {}).get("packages")
        if isinstance(packages, list):
            roots = []
            for p in packages:
                if isinstance(p, dict):
                    inc = p.get("include")
                    if not isinstance(inc, str):
                        continue
                    frm = p.get("from", "")
                    roots.append((project_dir / frm / inc).resolve())
                elif isinstance(p, str):
                    roots.append((project_dir / p).resolve())
    elif backend == "pdm.backend":
        includes = (tool.get("pdm") or {}).get("build", {}).get("includes")
        if isinstance(includes, list):
            roots = [
                (project_dir / inc).resolve()
                for inc in includes
                if isinstance(inc, str) and "*" not in inc
            ]
    elif backend == "flit_core.buildapi":
        mod = (tool.get("flit") or {}).get("module", {}).get("name")
        if isinstance(mod, str):
            roots = [(project_dir / mod).resolve()]

    if roots:
        return roots

    project_name = (data.get("project") or {}).get("name")
    if isinstance(project_name, str) and project_name:
        normalized = project_name.replace("-", "_").replace(".", "_")
        candidate = project_dir / normalized
        if (candidate / "__init__.py").is_file():
            return [candidate.resolve()]

    return None


def exported_tree_root(project_dir: Path) -> Path | None:
    """Single directory to use as a :class:`Package`'s exported subdir.

    Two cases:

    * ``<project_dir>/src/`` exists, or any of :func:`exported_roots`
      lives under ``<project_dir>/src/`` -> ``<project_dir>/src``.
      The src-layout convention: ``src/`` is the importable root
      (its contents become first-party modules) and the analyzer
      walks everything under it.
    * Otherwise -> ``<project_dir>`` itself. The flat layout: every
      package the build backend ships (``foo``, ``foo/a``, ``a/b``,
      ...) lives under ``<project_dir>``, so walking the project
      root reaches them all via longest-prefix-match. Walking the
      common parent of multiple package dirs would be wrong for
      nested packages like ``packages = ["foo/a"]`` -- ``foo.a``
      needs ``<project_dir>`` (not ``<project_dir>/foo``) on its
      sys.path to resolve.

    Returns ``None`` when the project has no ``pyproject.toml`` at all.
    """
    project_dir = project_dir.resolve()
    if not (project_dir / "pyproject.toml").is_file():
        return None
    src = project_dir / "src"
    if src.is_dir():
        return src.resolve()
    roots = exported_roots(project_dir)
    if roots and any(r.is_relative_to(src) for r in roots):
        return src.resolve()
    return project_dir
