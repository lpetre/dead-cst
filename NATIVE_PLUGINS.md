# Native plugins

> **Status: experimental, macOS + Linux.** The runtime split and in-tree native
> plugins are stable; *external* native plugin authoring is a preview — the
> API, the build flow, and the distribution are still moving, and only
> macOS (arm64) and Linux (x86_64 / aarch64) are wired today. See
> [Limitations](#limitations).

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

The runtime is linked one of two ways:

- The **dev build** (`maturin develop` / `uv sync`) and the **Windows wheel**
  link the runtime `rlib` *statically* into `_native` — one self-contained `.so`
  / `.pyd` that needs no rust at install time. This is the simplest, most
  robust layout, and it's what contributors and Windows users get. It cannot
  load external plugins (the runtime isn't shared).
- The **shipped macOS + Linux wheel** links the runtime **`dylib`** instead: a
  thin `_native` shim + `libdead_cst_runtime` + `libstd` ride in the package,
  resolving each other via `$ORIGIN` / `@loader_path`. The host therefore runs
  the *shared* runtime out of the box.

An **external plugin** is a separate `cdylib` that links the runtime **`dylib`**
(under `-C prefer-dynamic`). When both the extension module and the plugin
dynamically link the *same* `libdead_cst_runtime`, they share **one runtime
instance** — one salsa database, one `ProjectContext`, one set of types. That's
what lets a plugin receive a live `&ProjectContext` and define its own
salsa-tracked queries that cache against ty's database. (This is the
[dylint](https://github.com/trailofbits/dylint) model: link the host's runtime
dynamically, pinned to the exact compiler.) Because the shipped wheel already
runs the dylib, loading a plugin needs **no `_native` swap** — just the plugin's
compile closure, which the `[build-plugin]` extra provides.

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

You need (on macOS or Linux — the shipped dynamic wheel, not a dev checkout):

1. **The compile closure**, installed via the extra:

   ```bash
   pip install dead-cst[build-plugin]
   ```

   This pulls a separate `dead-cst-plugin-host` package (large — see
   [Distribution](#distribution)) carrying the `.rlib` dependency closure +
   proc-macro dylibs `rustc` needs to compile against the runtime. (The runtime
   dylib itself already ships inside the `dead_cst` wheel.)

2. **The Rust toolchain at dead-cst's pinned version** (see
   [`rust-toolchain.toml`](rust-toolchain.toml)). A different `rustc` produces a
   different ABI fingerprint and the plugin will be rejected at load (cleanly).

Then build:

```bash
# Compiles plugin.rs against the in-package runtime dylib (via `rustc --extern`)
# + the extra's rlib closure — no Cargo project, no dead-cst source, no swap —
# and prints the built plugin path on stdout.
PLUGIN=$(dead-cst build-plugin path/to/plugin.rs)
```

Load it from Python and use it like any other plugin:

```python
from dead_cst import Analysis, _native as native

plugins = native.load_native_plugins(PLUGIN)
ctx = Analysis(root, plugins=plugins).materialize_all()
```

The shipped wheel already runs the shared runtime, so the built plugin loads
straight away. `build-plugin` finds the runtime dylib inside the installed
`dead_cst` package and the rlib closure via `import dead_cst_plugin_host`;
`--runtime-dir` overrides both. Run `dead-cst build-plugin --help` for the
options. (In a dev/static checkout there's no in-package runtime dylib, so
`build-plugin` needs a dynamic wheel installed first.)

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
| `dead-cst` | ~25 MB | ~60 MB | everyone — analysis, the CLI, Python plugins |
| `dead-cst[build-plugin]` | +~130 MB | +~320 MB | authoring native plugins |

The base `dead-cst` macOS/Linux wheel carries the shared runtime dylib + libstd
alongside the `_native` shim (so it's a bit larger than a pure static build, and
the host runs the shared runtime). The `[build-plugin]` extra then pulls
**`dead-cst-plugin-host`**, a data-only, platform-specific package shipping only
the **compile closure**: the `.rlib` dependency archives + proc-macro dylibs
`rustc` needs to validate the crate graph. `build-plugin` finds it via `import
dead_cst_plugin_host`. It's large because `rustc` needs the full closure to
compile against the runtime; nothing in it is needed at *runtime*.

Maintainers produce both halves from **one** prefer-dynamic build, so the
runtime dylib's SVH matches the rlib closure plugins compile against. **`dead-cst
bundle-plugin-host`** does that build and gathers the rlib closure into the
`dead_cst_plugin_host` package; the publish workflow then repacks the runtime
dylib + libstd into the base wheel (`$ORIGIN` / `@loader_path`, stripped) and
builds one `py3-none-<plat>` `dead-cst-plugin-host` wheel per target (macOS
arm64 + Linux x86_64/aarch64). Both wheels are version-locked by
`scripts/stamp_version.py` and shipped to TestPyPI on every push to `main`, to
PyPI on release.

---

## Limitations

This is a preview. Known gaps:

- **macOS (arm64) and Linux only.** The Windows wheel is static — it runs
  analysis fine but can't load native plugins (no shared runtime); the loader
  plumbing isn't wired there. Plugins need the shipped *dynamic* macOS/Linux
  wheel: a dev/static checkout has no in-package runtime dylib, so
  `build-plugin` there asks you to install a dynamic wheel first.
- **Single-`.rs` plugins.** `build-plugin` compiles one source file; multi-crate
  plugins with their own dependencies are a follow-up.
- **`dead-cst-plugin-host` wheels are published to TestPyPI** (per push to
  `main`) **and PyPI** (per release), one per macOS arm64 / Linux x86_64 /
  Linux aarch64; their version is kept in lockstep with `dead-cst` by
  `scripts/stamp_version.py`. The base wheel's dynamic-runtime repack uses
  [`patchelf`](https://github.com/NixOS/patchelf) on Linux (present in the
  manylinux build image).
- **Recompile per release**, by design (full Rust fidelity has no stable ABI).
  The airlock makes a mismatch a clean error, not a crash.
- **Python plugins remain the supported extension path** for anything that
  doesn't specifically need native speed or plugin-defined salsa queries.
