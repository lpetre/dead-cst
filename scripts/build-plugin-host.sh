#!/usr/bin/env bash
#
# Build the *plugin-host* variant of dead-cst and an example external plugin,
# then run the gated external-plugin test.
#
# Unlike the default (static) wheel build, this links the `dead-cst-runtime`
# *dylib* into both `_native` and the plugin via `-C prefer-dynamic`, so they
# share one runtime instance (one salsa db / type space). That sharing is
# what makes a separately-built native plugin sound.
#
# Notes:
#   * Uses a dedicated target dir (target/plugin-host) so it never clobbers
#     the default static build that `maturin develop` manages.
#   * Builds the runtime dylib-ONLY: under -C prefer-dynamic a dep available
#     as both rlib and dylib confuses the linker (the cdylib binds the rlib's
#     SVH while the loader resolves the dylib -> "symbol not found"). The trap
#     restores the committed rlib+dylib manifest on exit.
#   * Everything dynamically links the toolchain's libstd, so an rpath to the
#     sysroot lib dir is baked in. `_native` references the runtime dylib by
#     absolute path, so artifacts must stay in the target dir (this is a
#     dev/source build, not a redistributable wheel — see the PR for the
#     wheel-packaging follow-up).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SYSROOT="$(rustc --print sysroot)"
HOST_TRIPLE="$(rustc -vV | sed -n 's/host: //p')"
STD_LIB="$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib"

export CARGO_TARGET_DIR="$ROOT/target/plugin-host"
export RUSTFLAGS="-C prefer-dynamic \
  -C link-arg=-Wl,-undefined,dynamic_lookup \
  -C link-arg=-Wl,-rpath,$STD_LIB \
  -C link-arg=-Wl,-rpath,@loader_path"

_CARGO_BAK="$(mktemp)"
cp runtime/Cargo.toml "$_CARGO_BAK"
trap 'cp "$_CARGO_BAK" runtime/Cargo.toml; rm -f "$_CARGO_BAK"' EXIT
sed -i '' 's/crate-type = \["rlib", "dylib"\]/crate-type = ["dylib"]/' runtime/Cargo.toml

echo ">> building dynamic _native + example plugin in ONE invocation"
echo "   (so dead-cst-runtime is compiled once -> _native and the plugin"
echo "    reference the same dylib SVH; prefer-dynamic, dylib-only)"
cargo build -p dead-cst-native -p main_block_plugin

echo ">> installing the dynamic _native over the editable extension"
cp "$CARGO_TARGET_DIR/debug/libdead_cst_native.dylib" "python/dead_cst/_native.abi3.so"

PLUGIN="$CARGO_TARGET_DIR/debug/libmain_block_plugin.dylib"
echo ">> running gated external-plugin test against $PLUGIN"
DEAD_CST_PLUGIN_HOST=1 PLUGIN_DYLIB="$PLUGIN" \
  uv run --no-sync pytest tests/test_plugins/test_external_dylib_plugin.py -v

echo
echo ">> NOTE: python/dead_cst/_native.abi3.so is now the dynamic build."
echo ">> Run 'uv run maturin develop --uv' to restore the default static build."
