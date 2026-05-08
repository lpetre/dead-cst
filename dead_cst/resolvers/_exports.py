"""Per-package export discovery from ``pyproject.toml``.

:func:`exported_roots` returns the subdirs of a package that should be
visible to *other* packages at import time. The analyzer uses this to
hide internal directories (like ``tests/`` in flat-layout workspace
members) from cross-package import resolution while still analyzing
them under their owning package.
"""

from __future__ import annotations

from pathlib import Path

from ._core import load_toml


def exported_roots(package_path: Path) -> list[Path] | None:
    """Subdirs of ``package_path`` visible to *other* packages at import time.

    Mirrors what a Python build backend would actually ship. For a workspace
    member with a non-``src/`` layout, this lets the analyzer hide internal
    directories like ``tests/`` from dependents -- they exist in the graph
    (the member analyzes its own files) but they don't pollute consumers'
    import-resolution lookups.

    Returns ``None`` when no restriction can be inferred (no ``pyproject.toml``,
    unknown backend, no ``[project].name``). The analyzer treats ``None`` as
    "no restriction; the whole package is exported", preserving today's behavior
    for projects that don't fit the workspace pattern.

    Discovery order, first match wins:

    1. ``<package_path>/src/`` exists -> ``[<package_path>/src]``. Universal
       src-layout convention; no backend introspection needed.
    2. ``[build-system].build-backend`` is read and dispatched:

       - ``hatchling.build`` -> ``[tool.hatch.build.targets.wheel].packages``
       - ``setuptools.build_meta`` (and ``__legacy__``) -> ``[tool.setuptools.packages]``
       - ``poetry.core.masonry.api`` -> ``[tool.poetry].packages`` (with ``include``/``from``)
       - ``pdm.backend`` -> ``[tool.pdm.build].includes`` (literal entries; globs skipped)
       - ``flit_core.buildapi`` -> ``[tool.flit.module].name``

    3. Fallback: a top-level dir named after the normalized ``[project].name``
       containing ``__init__.py``. Covers the auto-discovery shape used by
       hatchling and setuptools when no explicit ``packages`` list is given.

    Setuptools' ``packages.find`` (with ``include``/``exclude`` patterns) is
    not interpreted; falls through to the name-match fallback. Backends not
    listed above also fall through.
    """
    package_path = package_path.resolve()
    data = load_toml(package_path / "pyproject.toml")
    if data is None:
        return None

    if (package_path / "src").is_dir():
        return [(package_path / "src").resolve()]

    backend = (data.get("build-system") or {}).get("build-backend")
    tool = data.get("tool") or {}

    roots: list[Path] | None = None
    if backend == "hatchling.build":
        wheel = (tool.get("hatch") or {}).get("build", {}).get("targets", {}).get("wheel", {})
        packages = wheel.get("packages")
        if isinstance(packages, list):
            roots = [(package_path / p).resolve() for p in packages if isinstance(p, str)]
    elif backend in ("setuptools.build_meta", "setuptools.build_meta:__legacy__"):
        st = tool.get("setuptools") or {}
        packages = st.get("packages")
        if isinstance(packages, list):
            # Flat list of dotted module names; map ``foo.bar`` ->
            # ``package_path/foo/bar``.
            roots = [
                (package_path / p.replace(".", "/")).resolve()
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
                    roots.append((package_path / frm / inc).resolve())
                elif isinstance(p, str):
                    roots.append((package_path / p).resolve())
    elif backend == "pdm.backend":
        includes = (tool.get("pdm") or {}).get("build", {}).get("includes")
        if isinstance(includes, list):
            roots = [
                (package_path / inc).resolve()
                for inc in includes
                if isinstance(inc, str) and "*" not in inc
            ]
    elif backend == "flit_core.buildapi":
        mod = (tool.get("flit") or {}).get("module", {}).get("name")
        if isinstance(mod, str):
            roots = [(package_path / mod).resolve()]

    if roots:
        return roots

    # Name-match fallback: <project.name> normalized, dir with __init__.py.
    project_name = (data.get("project") or {}).get("name")
    if isinstance(project_name, str) and project_name:
        normalized = project_name.replace("-", "_").replace(".", "_")
        candidate = package_path / normalized
        if (candidate / "__init__.py").is_file():
            return [candidate.resolve()]

    return None
