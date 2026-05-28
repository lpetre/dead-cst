#!/usr/bin/env bash
#
# Build the example external plugin against a shared dead-cst runtime and run
# the gated external-plugin test. Thin wrapper over the `dead-cst build-plugin`
# CLI command, which does the actual prefer-dynamic / dylib-only build and
# installs the matching dynamic `_native` over the editable extension.
#
# After this runs, python/dead_cst/_native.abi3.so is the dynamic build; run
# `uv run maturin develop --uv` to restore the default static build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PLUGIN="$(uv run --no-sync dead-cst build-plugin main_block_plugin)"

echo ">> running gated external-plugin test against $PLUGIN"
DEAD_CST_PLUGIN_HOST=1 PLUGIN_DYLIB="$PLUGIN" \
  uv run --no-sync pytest tests/test_plugins/test_external_dylib_plugin.py -v
