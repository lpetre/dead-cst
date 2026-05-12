"""Tests for :mod:`dead_cst.resolvers._imports` path classification.

The bugs these guard against:

1. A system Python install (no venv) nests ``site-packages`` inside the
   stdlib root, so a naive ``path.is_relative_to(STDLIB)`` swallows every
   third-party package into ``[stdlib] <name>``.
2. ``pip install -e`` / ``uv pip install -e`` records only a ``.pth`` shim
   plus dist-info in ``RECORD``; the actual source files never appear in
   :func:`distribution_lookup`. Without help, an importer of an editable
   third-party package raised ``Module ... resolved to an unexpected path``.
"""

from __future__ import annotations

import json
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch

import pytest

from dead_cst.resolvers import _imports
from dead_cst.resolvers._imports import (
    _editable_source_roots,
    _is_site_packages_path,
    _is_stdlib_path,
    default_resolve_import,
)


def test_is_stdlib_path_excludes_nested_site_packages(tmp_path: Path):
    stdlib = tmp_path / "lib" / "python3.13"
    site_packages = stdlib / "site-packages"
    site_packages.mkdir(parents=True)

    with patch.object(_imports, "STDLIB", stdlib.resolve()):
        with patch.object(_imports, "PURELIB", site_packages.resolve()):
            with patch.object(_imports, "PLATLIB", site_packages.resolve()):
                # Genuine stdlib module
                assert _is_stdlib_path(stdlib / "json" / "__init__.py")
                # Third-party package nested under stdlib root
                assert not _is_stdlib_path(site_packages / "requests" / "__init__.py")


def test_is_stdlib_path_with_dist_packages_sibling(tmp_path: Path):
    """Debian-style layout: dist-packages lives alongside stdlib, not under it."""
    stdlib = tmp_path / "lib" / "python3.13"
    dist_packages = tmp_path / "lib" / "python3" / "dist-packages"
    stdlib.mkdir(parents=True)
    dist_packages.mkdir(parents=True)

    with patch.object(_imports, "STDLIB", stdlib.resolve()):
        with patch.object(_imports, "PURELIB", None):
            with patch.object(_imports, "PLATLIB", None):
                assert _is_stdlib_path(stdlib / "json" / "__init__.py")
                # dist-packages isn't under STDLIB so the check returns False;
                # the markers fallback also catches the directory name.
                assert not _is_stdlib_path(dist_packages / "yaml" / "__init__.py")


def test_is_site_packages_path_uses_purelib(tmp_path: Path):
    purelib = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    purelib.mkdir(parents=True)

    with patch.object(_imports, "PURELIB", purelib.resolve()):
        with patch.object(_imports, "PLATLIB", None):
            assert _is_site_packages_path(purelib / "click" / "__init__.py")
            assert not _is_site_packages_path(tmp_path / "src" / "myproj" / "__init__.py")


def test_is_site_packages_path_falls_back_to_markers(tmp_path: Path):
    """When sysconfig disagrees with the actual layout, the dir-name fallback
    still catches site-packages / dist-packages."""
    weird_sp = tmp_path / "weirdpython" / "lib" / "site-packages"
    weird_sp.mkdir(parents=True)

    with patch.object(_imports, "PURELIB", None):
        with patch.object(_imports, "PLATLIB", None):
            assert _is_site_packages_path(weird_sp / "pkg" / "__init__.py")


def _fake_dist(
    files: list[str] | None = None,
    direct_url: dict | None = None,
    pth_contents: dict[str, str] | None = None,
    base: Path | None = None,
) -> types.SimpleNamespace:
    """Build a duck-typed ``importlib.metadata.Distribution`` stand-in.

    Only the surface :func:`_editable_source_roots` actually touches:
    ``files`` (relative paths), ``locate_file`` (resolves against ``base``)
    and ``read_text`` (returns ``direct_url.json`` content).
    """
    base = base or Path("/nonexistent")
    pth_contents = pth_contents or {}

    class _PathLike(str):
        pass

    file_objs = [_PathLike(f) for f in (files or ())]

    def locate_file(rel: str) -> Path:
        # ``.pth`` shims live in site-packages, but for unit tests we
        # write them somewhere accessible (``pth_contents`` keys map to
        # absolute on-disk locations).
        s = str(rel)
        if s in pth_contents:
            return Path(pth_contents[s])
        return base / s

    def read_text(name: str) -> str | None:
        if name == "direct_url.json" and direct_url is not None:
            return json.dumps(direct_url)
        return None

    return types.SimpleNamespace(
        files=file_objs,
        locate_file=locate_file,
        read_text=read_text,
    )


def test_editable_source_roots_from_direct_url(tmp_path: Path):
    src = tmp_path / "src-project"
    src.mkdir()

    dist = _fake_dist(
        direct_url={"url": f"file://{src}", "dir_info": {"editable": True}},
    )
    roots = _editable_source_roots(dist)
    assert roots == [src.resolve()]


def test_editable_source_roots_ignores_non_editable_direct_url(tmp_path: Path):
    src = tmp_path / "src-project"
    src.mkdir()

    dist = _fake_dist(
        direct_url={"url": f"file://{src}", "dir_info": {"editable": False}},
    )
    assert _editable_source_roots(dist) == []


def test_editable_source_roots_from_pth_shim(tmp_path: Path):
    src = tmp_path / "my-editable"
    src.mkdir()
    pth = tmp_path / "site" / "_editable_impl_my_editable.pth"
    pth.parent.mkdir()
    pth.write_text(str(src))

    dist = _fake_dist(
        files=["_editable_impl_my_editable.pth"],
        pth_contents={"_editable_impl_my_editable.pth": str(pth)},
    )
    assert _editable_source_roots(dist) == [src.resolve()]


def test_editable_source_roots_skips_pth_import_lines(tmp_path: Path):
    """``.pth`` files can also contain ``import foo`` activation hooks; those
    aren't directories and must not be treated as editable roots."""
    pth = tmp_path / "site" / "activate.pth"
    pth.parent.mkdir()
    pth.write_text("import sitecustomize\n# a comment\n")

    dist = _fake_dist(
        files=["activate.pth"],
        pth_contents={"activate.pth": str(pth)},
    )
    assert _editable_source_roots(dist) == []


def test_editable_source_roots_handles_malformed_direct_url(tmp_path: Path):
    dist = types.SimpleNamespace(
        files=[],
        locate_file=lambda rel: tmp_path / rel,
        read_text=lambda name: "{not valid json" if name == "direct_url.json" else None,
    )
    assert _editable_source_roots(dist) == []


def _make_spec(name: str, origin: str) -> ModuleSpec:
    return ModuleSpec(name, loader=None, origin=origin)


def test_default_resolve_import_third_party_under_stdlib_root(tmp_path: Path):
    """Reproduces the real bug: stdlib at ``<prefix>/lib/python3.13`` with
    site-packages nested at ``<prefix>/lib/python3.13/site-packages``.

    Before the fix, ``requests`` resolved to ``[stdlib] requests`` because
    ``is_relative_to(STDLIB)`` was True. After the fix, the dist lookup
    classifies it as ``[external dist] requests``."""
    stdlib = tmp_path / "lib" / "python3.13"
    site_packages = stdlib / "site-packages"
    site_packages.mkdir(parents=True)
    requests_init = site_packages / "requests" / "__init__.py"
    requests_init.parent.mkdir()
    requests_init.write_text("")

    spec = _make_spec("requests", str(requests_init))

    with patch.object(_imports, "STDLIB", stdlib.resolve()):
        with patch.object(_imports, "PURELIB", site_packages.resolve()):
            with patch.object(_imports, "PLATLIB", site_packages.resolve()):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(
                        _imports,
                        "distribution_lookup",
                        return_value={requests_init.resolve(): "requests"},
                    ):
                        with patch.object(_imports, "editable_distribution_roots", return_value=()):
                            result = default_resolve_import("requests", [tmp_path])
    assert result == "[external dist] requests"


def test_default_resolve_import_unrecorded_thirdparty_under_stdlib_root(tmp_path: Path):
    """Even when the dist lookup misses a file (rare but possible -- vendored
    bundles, broken RECORD, ...), a third-party package nested under the
    stdlib root must NOT be classified as stdlib."""
    stdlib = tmp_path / "lib" / "python3.13"
    site_packages = stdlib / "site-packages"
    site_packages.mkdir(parents=True)
    pkg_init = site_packages / "vendored" / "__init__.py"
    pkg_init.parent.mkdir()
    pkg_init.write_text("")

    spec = _make_spec("vendored", str(pkg_init))

    with patch.object(_imports, "STDLIB", stdlib.resolve()):
        with patch.object(_imports, "PURELIB", site_packages.resolve()):
            with patch.object(_imports, "PLATLIB", site_packages.resolve()):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(_imports, "distribution_lookup", return_value={}):
                        with patch.object(_imports, "editable_distribution_roots", return_value=()):
                            result = default_resolve_import("vendored", [tmp_path])
    assert result == "[external file] vendored"


def test_default_resolve_import_editable_dist_outside_search_paths(tmp_path: Path):
    """An editable third-party install whose source dir is outside the
    project's search paths must resolve to ``[external dist]`` rather than
    raising ``Module ... resolved to an unexpected path``."""
    editable_root = tmp_path / "external" / "dead-cst"
    pkg_init = editable_root / "dead_cst" / "__init__.py"
    pkg_init.parent.mkdir(parents=True)
    pkg_init.write_text("")

    project_root = tmp_path / "consumer"
    project_root.mkdir()

    spec = _make_spec("dead_cst", str(pkg_init))

    with patch.object(_imports, "STDLIB", tmp_path / "stdlib"):
        with patch.object(_imports, "PURELIB", None):
            with patch.object(_imports, "PLATLIB", None):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(_imports, "distribution_lookup", return_value={}):
                        with patch.object(
                            _imports,
                            "editable_distribution_roots",
                            return_value=((editable_root.resolve(), "dead-cst"),),
                        ):
                            result = default_resolve_import("dead_cst", [project_root])
    assert result == "[external dist] dead-cst"


def test_default_resolve_import_first_party_wins_over_editable_root(tmp_path: Path):
    """Regression: a first-party path that happens to nest under an editable
    distribution's source root must classify as first-party, not as that
    dist. Reproduces the e2e flux0 failure where ``.pytest_cache/d/e2e-clones/``
    sat under the editable ``dead-cst`` root and every cloned module
    misclassified as ``[external dist] dead-cst``."""
    editable_root = tmp_path / "host"
    project_src = editable_root / "fixtures" / "clones" / "pkg" / "src"
    init_py = project_src / "myproj" / "__init__.py"
    init_py.parent.mkdir(parents=True)
    init_py.write_text("")

    spec = _make_spec("myproj", str(init_py))

    with patch.object(_imports, "STDLIB", tmp_path / "stdlib"):
        with patch.object(_imports, "PURELIB", None):
            with patch.object(_imports, "PLATLIB", None):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(_imports, "distribution_lookup", return_value={}):
                        with patch.object(
                            _imports,
                            "editable_distribution_roots",
                            return_value=((editable_root.resolve(), "host-pkg"),),
                        ):
                            result = default_resolve_import("myproj", [project_src])
    assert result == init_py.resolve()


def test_default_resolve_import_first_party_still_returns_path(tmp_path: Path):
    """Regression: the reordered checks must not steal first-party files
    that happen to live near common install roots."""
    project_src = tmp_path / "src" / "myproj"
    init_py = project_src / "__init__.py"
    init_py.parent.mkdir(parents=True)
    init_py.write_text("")

    spec = _make_spec("myproj", str(init_py))

    with patch.object(_imports, "STDLIB", tmp_path / "stdlib"):
        with patch.object(_imports, "PURELIB", None):
            with patch.object(_imports, "PLATLIB", None):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(_imports, "distribution_lookup", return_value={}):
                        with patch.object(_imports, "editable_distribution_roots", return_value=()):
                            result = default_resolve_import("myproj", [tmp_path / "src"])
    assert result == init_py.resolve()


def test_default_resolve_import_genuine_stdlib_still_classified(tmp_path: Path):
    stdlib = tmp_path / "lib" / "python3.13"
    json_init = stdlib / "json" / "__init__.py"
    json_init.parent.mkdir(parents=True)
    json_init.write_text("")

    spec = _make_spec("json", str(json_init))

    with patch.object(_imports, "STDLIB", stdlib.resolve()):
        with patch.object(_imports, "PURELIB", stdlib.resolve() / "site-packages"):
            with patch.object(_imports, "PLATLIB", stdlib.resolve() / "site-packages"):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(_imports, "distribution_lookup", return_value={}):
                        with patch.object(_imports, "editable_distribution_roots", return_value=()):
                            result = default_resolve_import("json", [tmp_path])
    assert result == "[stdlib] json"


def test_default_resolve_import_builtin_module():
    spec = ModuleSpec("sys", loader=None, origin="built-in")
    with patch.object(_imports, "safe_resolve_module", return_value=spec):
        assert default_resolve_import("sys", []) == "[stdlib] sys"


def test_default_resolve_import_unresolvable_returns_none():
    with patch.object(_imports, "safe_resolve_module", return_value=None):
        assert default_resolve_import("nonexistent_zzz", []) is None


def test_default_resolve_import_unexpected_path_still_raises(tmp_path: Path):
    """When the resolved path falls outside *every* category we recognize
    (stdlib / site-packages / dist lookup / editable roots / search paths),
    the analyzer should still surface the surprise rather than silently
    returning a wrong classification."""
    stray = tmp_path / "stray" / "weird.py"
    stray.parent.mkdir()
    stray.write_text("")
    spec = _make_spec("weird", str(stray))

    with patch.object(_imports, "STDLIB", tmp_path / "stdlib"):
        with patch.object(_imports, "PURELIB", None):
            with patch.object(_imports, "PLATLIB", None):
                with patch.object(_imports, "safe_resolve_module", return_value=spec):
                    with patch.object(_imports, "distribution_lookup", return_value={}):
                        with patch.object(_imports, "editable_distribution_roots", return_value=()):
                            with pytest.raises(Exception, match="unexpected path"):
                                default_resolve_import("weird", [tmp_path / "project"])


def test_editable_distribution_roots_self_install():
    """Smoke test against the live process: dead-cst itself is installed
    editably in the test venv, so its source dir should be discoverable."""
    roots = _imports.editable_distribution_roots()
    # Look up dead-cst's actual source root via its module file.
    import dead_cst

    expected = Path(dead_cst.__file__).resolve().parent.parent
    matching = [r for r, name in roots if name == "dead-cst"]
    assert any(expected == r or expected.is_relative_to(r) for r in matching), (
        f"expected dead-cst source root near {expected}, got {matching}"
    )


def test_distribution_lookup_survives_first_party_sys_path_change(tmp_path: Path, monkeypatch):
    """A first-party path prepended to ``sys.path`` doesn't change which
    distributions are visible -- so the dist cache must survive it without
    a rebuild. Regression for the per-package transition hot path in
    :meth:`Analysis._materialize`, which used to flush this on every
    package and burn ~10s/transition on large venvs.
    """
    from importlib import metadata

    real_distributions = metadata.distributions
    calls = 0

    def counting_distributions(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_distributions(*args, **kwargs)

    monkeypatch.setattr(metadata, "distributions", counting_distributions)

    _imports.clear_path_caches()
    _imports.distribution_lookup()
    assert calls == 1

    project_src = tmp_path / "project" / "src"
    project_src.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(project_src))
    _imports.clear_module_specs_cache()
    _imports.distribution_lookup()
    # The dist-bearing slice of sys.path didn't change, so no rebuild.
    assert calls == 1

    # ``clear_path_caches`` remains the heavy hammer for explicit resets.
    _imports.clear_path_caches()
    _imports.distribution_lookup()
    assert calls == 2
