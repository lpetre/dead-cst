# Native plugins

> **Status: experimental, macOS only.** The runtime split and in-tree native
> plugins are stable; *external* native plugin authoring is a preview — the
> API, the build flow, and the distribution are still moving, and only
> macOS (arm64) is wired today. See [Limitations](#limitations).

dead-cst has two ways to extend the reachability graph:

| | Python plugins | Native plugins |
|---|---|---|
| Language | Python (the [`Plugin`](python/dead_cst/plugins/_base.py) protocol) | Rust |
| Toolchain to author | none | rust, pinned to the build version |
| Ships in | the default wheel | a separate `[build-plugin]` extra |
| Per-op cost | a `Py` alloc + `extract` per emitted op | none — pure rust |
| View of the graph | the full `ProjectContext` Python API | the full `ProjectContext` **by rust reference** |
| Own salsa-cached queries | no | yes |

**Most plugins should be Python plugins** — see
[CONTRIBUTING.md → Adding a plugin](CONTRIBUTING.md#adding-a-plugin). Reach for
a native plugin only when the work is hot enough that the per-op Python
crossing matters, or when you want to define your own salsa-cached queries over
ty's database. This document is about **external** native plugins: ones you
compile in your own crate and load at runtime, without forking dead-cst.

(There are also *in-tree* native plugins — e.g. `MainBlockPlugin`, exposed as
`native.NativePlugin.main_block()` — compiled straight into the shipped
extension. Those aren't covered here; they're an internal fast-path, not an
extension mechanism.)

---

## How it works: one shared runtime

The native crate is split in two:

- **`dead-cst-runtime`** — the entire implementation (the graph builder, the
  query DSL, the pyclasses, the plugin API). Built as both an **`rlib`** and a
  **`dylib`**.
- **`dead-cst-native`** — a thin pyo3 `#[pymodule]` shim that becomes
  `dead_cst._native`.

The **default wheel** links the runtime `rlib` *statically* into `_native`, so
it's a single self-contained `.so` (~10 MB compressed) that needs no rust at
install time. Nothing about the everyday experience changes.

An **external plugin** is a separate `cdylib` that links the runtime **`dylib`**
(under `-C prefer-dynamic`). When both the extension module and the plugin
dynamically link the *same* `libdead_cst_runtime.dylib`, they share **one
runtime instance** — one salsa database, one `ProjectContext`, one set of
types. That's what lets a plugin receive a live `&ProjectContext` and define
its own salsa-tracked queries that cache against ty's database. (This is the
[dylint](https://github.com/trailofbits/dylint) model: link the host's runtime
dynamically, pinned to the exact compiler.)

The price of that fidelity is that a plugin is **ABI-coupled to the exact
dead-cst build** it was compiled against (Rust has no stable ABI). dead-cst
makes that safe rather than crashy — see [The ABI airlock](#the-abi-airlock).

---

## Writing a plugin

A plugin is a Rust `cdylib` that depends on `dead-cst-runtime` and implements
the [`ExternalPlugin`](runtime/src/native_plugins.rs) trait from the curated
`plugin_api`:

```rust
use dead_cst_runtime::native_plugins::plugin_api::{ExternalPlugin, PluginCtx, PluginOps};

struct KeepMainBlocksAlive;

impl ExternalPlugin for KeepMainBlocksAlive {
    fn name(&self) -> &str {
        "KeepMainBlocksAlive"
    }

    fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps) {
        // `ctx` is a restricted, mostly-stable view of the frozen graph.
        for (module_idx, decl_indices) in ctx.main_blocks() {
            // Emit ops the host folds in after every plugin has run.
            ops.keep_alive(module_idx, "<__main__>".to_string());
            for decl_idx in decl_indices {
                ops.keep_alive(decl_idx, "<__main__>".to_string());
            }
        }
    }
}
```

`PluginCtx` deliberately exposes a small, stable surface (e.g. `main_blocks()`)
rather than the whole internal `ProjectContext`, and `PluginOps` emits through
named methods (`keep_alive(...)`) instead of internal op types. The contract is
the same frozen-graph one as Python plugins: the plugin observes the base graph
only; its emissions are applied in a single batch after every plugin returns.

Each plugin `cdylib` also exports a **manifest** — a self-contained
`#[repr(C)]` table listing the plugins it provides and the ABI fingerprint it
was built against. It's deliberately boring boilerplate; copy it from the
worked example:

- **[`examples/main_block_plugin/`](examples/main_block_plugin/)** — a complete
  external plugin (a `MainBlockPlugin` equivalent) including the manifest. Start
  here.

---

## Building a plugin

You need:

1. **The prebuilt runtime**, installed via the extra:

   ```bash
   pip install dead-cst[build-plugin]
   ```

   This pulls a separate `dead-cst-plugin-host` package (large — see
   [Distribution](#distribution)) carrying the runtime dylib + the dependency
   metadata `rustc` needs to compile against it.

2. **The Rust toolchain at dead-cst's pinned version** (see
   [`rust-toolchain.toml`](rust-toolchain.toml)). A different `rustc` produces a
   different ABI fingerprint and the plugin will be rejected at load (cleanly).

Then build:

```bash
# Compiles plugin.rs against the prebuilt runtime via `rustc --extern`
# (no Cargo project, no dead-cst source, no ruff recompile), installs the
# matching dynamic _native, and prints the built .so path on stdout.
PLUGIN=$(dead-cst build-plugin path/to/plugin.rs)
```

Load it from Python and use it like any other plugin:

```python
from dead_cst import Analysis, _native as native

plugins = native.load_native_plugins(PLUGIN)
ctx = Analysis(root, plugins=plugins).materialize_all()
```

`build-plugin` resolves the runtime artifacts from, in order: `--runtime-dir`,
the installed `dead-cst-plugin-host` package, or a dead-cst **source checkout**
(building them on demand). Run `dead-cst build-plugin --help` for the options.

---

## The ABI airlock

A plugin is compiled against one exact dead-cst build, so it must be
**recompiled for each release**. dead-cst enforces that without ever crashing:

- Every plugin bakes an **ABI fingerprint** at compile time —
  `epoch | rustc commit | runtime version | target` — taken from the runtime it
  built against.
- `native.load_native_plugins(path)` opens the dylib lazily and reads a
  **self-contained manifest** (plain data + the baked fingerprint) *before*
  touching any version-hashed runtime symbol.
- If the manifest is missing, the magic is wrong, or the fingerprint doesn't
  match the running runtime, the load is **rejected with a clear error**:

  ```
  ABI mismatch — plugin built against '1|commit-hash: …|v0.13.0|aarch64-apple-darwin',
  this runtime is '1|…|v0.14.0|…'. Rebuild the plugin against this release.
  ```

So a stale `.so` is refused, not segfaulted. Rebuild with `build-plugin` and
you're back in business.

---

## Distribution

The split keeps the cost where it belongs:

| install | download | installed | for |
|---|---|---|---|
| `dead-cst` | ~10 MB | ~24 MB | everyone — analysis, the CLI, Python plugins |
| `dead-cst[build-plugin]` | +~130 MB | +~350 MB | authoring native plugins |

The `[build-plugin]` extra pulls **`dead-cst-plugin-host`**, a data-only,
platform-specific package shipping the runtime dylib + its `.rlib` /
proc-macro-dylib dependency closure + `libstd`, all relocatable. `build-plugin`
finds it via `import dead_cst_plugin_host`. It's large because `rustc` needs the
full dependency closure to compile a plugin against the runtime; the default
wheel ships only the final linked, stripped extension.

Maintainers produce that payload with **`dead-cst bundle-plugin-host`**, which
builds the runtime, gathers the closure, rewrites install names / rpaths to
`@rpath` / `@loader_path`, strips + ad-hoc re-signs the dylibs, and drops it
into the `dead_cst_plugin_host` package. (CI wheels for it are still
[in progress](#limitations).)

---

## Limitations

This is a preview. Known gaps:

- **macOS (arm64) only.** Linux/Windows loader plumbing isn't wired yet.
- **Single-`.rs` plugins.** `build-plugin` compiles one source file; multi-crate
  plugins with their own dependencies are a follow-up.
- **No published `dead-cst-plugin-host` wheels yet.** Today the bundle is built
  locally (`bundle-plugin-host`) or from a source checkout; the cross-platform
  CI that publishes the `[build-plugin]` payload is still to come.
- **Recompile per release**, by design (full Rust fidelity has no stable ABI).
  The airlock makes a mismatch a clean error, not a crash.
- **Python plugins remain the supported extension path** for anything that
  doesn't specifically need native speed or plugin-defined salsa queries.
