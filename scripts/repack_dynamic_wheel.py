#!/usr/bin/env python3
"""Repack the static `dead-cst` base wheel into a dynamic-runtime one.

The default (dev) build links the runtime `rlib` statically into one
self-contained `_native.{abi3.so,pyd}`. The *published* wheel instead ships the
shared-runtime layout so end users can load native plugins with no in-place
`_native` swap: a thin dynamic `_native` shim + the runtime as a dylib +
`libstd`, all resolving each other via `$ORIGIN` / `@loader_path`.

This takes maturin's static wheel (for its correct metadata + Python sources)
and swaps the binary payload:

  * replace `dead_cst/_native.<tag>.so` with the prefer-dynamic shim,
  * add `dead_cst/libdead_cst_runtime.<ext>` (the shared runtime dylib) and
    `dead_cst/libstd-<hash>.<ext>` next to it,
  * point the shim + runtime at their siblings via a loader-relative rpath
    (`$ORIGIN` / `@loader_path`), preserving the runtime's soname so a plugin
    compiled against it resolves the already-loaded instance,
  * strip the dylibs, then regenerate `RECORD` and rezip under the same tag.

The shim + runtime dylib must come from the *same* prefer-dynamic build (so the
dylib's SVH matches the rlib closure the `dead-cst-plugin-host` extra ships and
plugins compile against). `dead-cst bundle-plugin-host` / `_build_runtime_from
_source` produce that build; pass its `deps` dir as ``--deps-dir``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _dylib_suffix() -> str:
    return ".dylib" if _is_macos() else ".so"


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _set_rpath_origin(lib: Path) -> None:
    """Point ``lib`` at its own directory for sibling resolution, dropping the
    builder's absolute rpaths so the wheel relocates. No-op (with a warning) if
    the platform tool is missing — the prefer-dynamic build already bakes the
    loader-relative entry, so siblings still resolve; this is hygiene."""
    if _is_macos():
        tool = "install_name_tool"
        if shutil.which(tool) is None:
            print(f"warning: {tool} not found; leaving rpaths as built", file=sys.stderr)
            return
        # Best-effort: ensure @loader_path is present (the build already adds it).
        _run(tool, "-add_rpath", "@loader_path", str(lib))
    else:
        if shutil.which("patchelf") is None:
            print("warning: patchelf not found; leaving rpaths as built", file=sys.stderr)
            return
        _run("patchelf", "--set-rpath", "$ORIGIN", str(lib))


def _strip(lib: Path) -> None:
    if shutil.which("strip") is None:
        return
    _run("strip", *(["-x"] if _is_macos() else ["--strip-unneeded"]), str(lib))


def _resign(lib: Path) -> None:
    # install_name_tool / strip invalidate the macOS signature; ad-hoc re-sign.
    if _is_macos() and shutil.which("codesign"):
        _run("codesign", "-s", "-", "-f", str(lib))


def repack(static_wheel: Path, deps_dir: Path, std_lib: Path, out_dir: Path) -> Path:
    suffix = _dylib_suffix()
    runtime_src = deps_dir / f"libdead_cst_runtime{suffix}"
    shim_src = deps_dir / f"libdead_cst_native{suffix}"
    libstds = sorted(std_lib.glob(f"libstd-*{suffix}"))
    for needed in (runtime_src, shim_src):
        if not needed.is_file():
            raise SystemExit(f"missing prefer-dynamic artifact: {needed}")
    if not libstds:
        raise SystemExit(f"no libstd-*{suffix} under {std_lib}")
    libstd_src = libstds[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _run(sys.executable, "-m", "wheel", "unpack", str(static_wheel), "-d", str(tmp))
        unpacked = next(p for p in tmp.iterdir() if p.is_dir())
        pkg = unpacked / "dead_cst"

        # The compiled module maturin shipped, e.g. _native.abi3.so / .pyd.
        modules = list(pkg.glob("_native.*so")) + list(pkg.glob("_native.*pyd"))
        if len(modules) != 1:
            raise SystemExit(f"expected one _native module in the wheel, found {modules}")
        module = modules[0]

        # Swap the static module for the dynamic shim (keep the maturin name so
        # Python's import machinery still finds it), add the runtime + libstd.
        shutil.copyfile(shim_src, module)
        runtime_dst = pkg / runtime_src.name
        libstd_dst = pkg / libstd_src.name
        shutil.copyfile(runtime_src, runtime_dst)
        shutil.copyfile(libstd_src, libstd_dst)

        for lib in (module, runtime_dst):
            _set_rpath_origin(lib)
        for lib in (module, runtime_dst, libstd_dst):
            _strip(lib)
            _resign(lib)

        _run(sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(out_dir))

    built = sorted(out_dir.glob(f"{static_wheel.stem.split('-')[0]}*-*.whl"))
    # `wheel pack` keeps the original tag, so the filename is unchanged.
    result = out_dir / static_wheel.name
    if not result.is_file():
        raise SystemExit(f"repacked wheel not found at {result} (saw {built})")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("static_wheel", type=Path, help="the maturin-built static base wheel")
    ap.add_argument(
        "--deps-dir",
        type=Path,
        required=True,
        help="prefer-dynamic build deps dir (libdead_cst_native + libdead_cst_runtime)",
    )
    ap.add_argument(
        "--std-lib", type=Path, required=True, help="toolchain dir holding libstd-<hash>"
    )
    ap.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    args = ap.parse_args()
    result = repack(args.static_wheel, args.deps_dir, args.std_lib, args.output)
    print(result)


if __name__ == "__main__":
    main()
