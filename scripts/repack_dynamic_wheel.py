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


def _host_std_lib() -> Path:
    """The toolchain dir holding the shared ``libstd-<hash>`` (prefer-dynamic
    artifacts rpath here)."""

    def rustc(*a: str) -> str:
        return subprocess.run(
            ["rustc", *a], capture_output=True, text=True, check=True
        ).stdout.strip()

    sysroot = Path(rustc("--print", "sysroot"))
    host = next(
        line.split("host: ", 1)[1] for line in rustc("-vV").splitlines() if "host: " in line
    )
    return sysroot / "lib" / "rustlib" / host / "lib"


def _otool_load_paths(lib: Path) -> list[str]:
    out = subprocess.run(
        ["otool", "-L", str(lib)], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip().split(" ", 1)[0] for line in out.splitlines()[1:] if line.strip()]


def _otool_rpaths(lib: Path) -> set[str]:
    out = subprocess.run(
        ["otool", "-l", str(lib)], capture_output=True, text=True, check=True
    ).stdout
    lines = out.splitlines()
    found: set[str] = set()
    for i, line in enumerate(lines):
        if "LC_RPATH" in line:
            for follow in lines[i : i + 4]:
                s = follow.strip()
                if s.startswith("path "):
                    found.add(s[len("path ") :].split(" (offset")[0].strip())
    return found


def _relocate_linux(lib: Path) -> None:
    """Set ``lib``'s run path to ``$ORIGIN`` (sibling resolution), dropping the
    builder's absolute rpaths. ``patchelf --set-rpath`` overwrites wholesale.
    No-op (warning) without patchelf — the build bakes ``$ORIGIN`` too."""
    if shutil.which("patchelf") is None:
        print("warning: patchelf not found; leaving rpaths as built", file=sys.stderr)
        return
    _run("patchelf", "--set-rpath", "$ORIGIN", str(lib))


def _relocate_macos(module: Path, runtime: Path, std_lib: Path) -> None:
    """Rewrite install names so the shim + runtime load each other (and libstd)
    relocatably via ``@rpath`` / ``@loader_path``. The prefer-dynamic build
    already bakes ``@loader_path`` (so only add it when missing) and the absolute
    sysroot rpath (drop it)."""
    if shutil.which("install_name_tool") is None:
        print(
            "warning: install_name_tool not found; leaving install names as built", file=sys.stderr
        )
        return
    runtime_id = f"@rpath/{runtime.name}"

    def ensure_loader_path(lib: Path) -> None:
        if "@loader_path" not in _otool_rpaths(lib):
            _run("install_name_tool", "-add_rpath", "@loader_path", str(lib))

    def drop_abs_std(lib: Path) -> None:
        if str(std_lib) in _otool_rpaths(lib):
            _run("install_name_tool", "-delete_rpath", str(std_lib), str(lib))

    # runtime: relocatable id + @loader_path (to find its sibling libstd).
    _run("install_name_tool", "-id", runtime_id, str(runtime))
    ensure_loader_path(runtime)
    drop_abs_std(runtime)

    # shim: point its load command for the runtime at @rpath; @loader_path rpath.
    for path in _otool_load_paths(module):
        if "libdead_cst_runtime" in path and path != runtime_id:
            _run("install_name_tool", "-change", path, runtime_id, str(module))
            break
    ensure_loader_path(module)
    drop_abs_std(module)


def _strip(lib: Path) -> None:
    # Strip debug info only: keeps the dynamic symbol table AND the embedded
    # rust metadata section, so `build-plugin`'s `rustc --extern` against the
    # shipped runtime dylib still reads its crate metadata.
    if shutil.which("strip") is None:
        return
    _run("strip", *(["-S"] if _is_macos() else ["--strip-debug"]), str(lib))


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

        if _is_macos():
            _relocate_macos(module, runtime_dst, std_lib)
        else:
            for lib in (module, runtime_dst):
                _relocate_linux(lib)
        for lib in (module, runtime_dst, libstd_dst):
            _strip(lib)
        for lib in (module, runtime_dst, libstd_dst):
            _resign(lib)  # no-op off macOS

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
        "--std-lib",
        type=Path,
        default=None,
        help="toolchain dir holding libstd-<hash> (default: derived from rustc)",
    )
    ap.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    args = ap.parse_args()
    std_lib = args.std_lib if args.std_lib is not None else _host_std_lib()
    result = repack(args.static_wheel, args.deps_dir, std_lib, args.output)
    print(result)


if __name__ == "__main__":
    main()
