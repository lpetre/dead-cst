# Changelog

All notable changes to `dead-cst` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first stable release the public API and CLI may change between any
two versions.

## [Unreleased]

### Added

- **`plugin_api` epoch in the ABI fingerprint.** The native-plugin ABI
  fingerprint gains a dedicated `api<N>` segment
  (`native_plugins::plugin_api::PLUGIN_API_EPOCH`, bumped in
  `runtime/build.rs`) tracking the curated `plugin_api` surface. Bumping it
  rejects plugins compiled against an older API at load — distinct from a
  rustc / runtime-version / target change — so an incompatible `plugin_api`
  edit can invalidate stale plugins on its own.
- **`flag_decl` plugin op.** `PluginOps::flag_decl(decl_idx, flags)` and the
  per-file `FileOps::flag_decl(decl_local_idx, flags)` OR a node-flag bitset
  onto an existing decl (mapping to the new `PreparedOp::FlagDecl` /
  `FileLocalOp::FlagDecl`). Plugins use it to stamp a registered flag — the bit
  from `ctx.node_flag(name)` — directly on a discovered decl instead of routing
  it through a synthetic seed marker node. Bumps `PLUGIN_API_EPOCH` to 4.
- **External native plugins (experimental).** The native crate is split
  into a `dead-cst-runtime` library (built as both `rlib` and `dylib`)
  and a thin `dead-cst-native` cdylib shim, so a plugin can dynamically
  link the *same* runtime as the extension module and share one salsa/ty
  instance. Plugins implement
  `dead_cst_runtime::native_plugins::plugin_api::ExternalPlugin` (see
  `examples/main_block_plugin/`) and load via
  `native.load_native_plugins(path)`, which reads a self-contained
  `repr(C)` manifest and rejects — cleanly, before touching any
  version-hashed symbol — any plugin whose baked ABI fingerprint
  (`rustc` commit + runtime version + target) differs from the running
  runtime's.
  - The **shipped macOS + Linux wheel is dynamic**: a thin `_native` shim
    + `libdead_cst_runtime` + `libstd` ride in the package and resolve
    each other via `$ORIGIN` / `@loader_path`, so the host runs the
    shared runtime out of the box and plugins load with **no `_native`
    swap**. The **dev build and the Windows wheel stay static**
    (self-contained `_native`, no plugin loading).
  - `dead-cst build-plugin PLUGIN.rs` compiles a plugin with `rustc
    --extern` against the in-package runtime dylib + the `.rlib`
    dependency closure from the `dead-cst[build-plugin]` extra, and
    prints the plugin path — no Cargo project, no runtime source, no
    swap. Needs the pinned toolchain (the fingerprint enforces it).
  - The `[build-plugin]` extra ships **`dead-cst-plugin-host`**, a
    separate platform-specific package carrying only the compile closure
    (the `.rlib` archives + proc-macro dylibs). `dead-cst
    bundle-plugin-host` produces it; the publish workflow builds the
    dynamic base wheel (repacking the runtime dylib + libstd into
    maturin's static wheel via `scripts/repack_dynamic_wheel.py`) and the
    `dead-cst-plugin-host` wheel from **one** prefer-dynamic build per
    target (macOS arm64 + Linux x86_64/aarch64), so the runtime dylib's
    SVH matches the closure. Both are stamped to the **same** version by
    `scripts/stamp_version.py` (which also pins the extra to `==
    <version>`) and shipped to TestPyPI on every push to `main` and PyPI
    on release. The lockstep is mandatory — the runtime version is part
    of the ABI fingerprint. No Windows plugin support (static wheel).
  - The `dead-cst-plugin-host` closure is now stored **xz-compressed**
    inside the wheel (each `.rlib` / proc-macro dylib as `<name>.xz`,
    `ZIP_STORED`). The raw closure is ~320 MB and a deflate wheel lands at
    ~107 MB — over PyPI's 100 MB/file cap; xz packs it to ~70 MB.
    `build-plugin` decompresses to a temp dir (freed on exit) before
    invoking `rustc`. Uses the stdlib `lzma` module, so the
    `[build-plugin]` extra gains no dependency.
- **Fleshed-out external native plugin API.** The curated
  `dead_cst_runtime::native_plugins::plugin_api` an external plugin
  compiles against grew from a single `PluginCtx::main_blocks()` /
  `PluginOps::keep_alive(...)` pair into a usable index-based surface.
  `PluginCtx` now also exposes `node(idx)` (returning an owned
  `NodeView`), `node_count()`, the structural lookups `find_module`,
  `find_declarations`, `module_for`, `resolve`, `decls_under`,
  `find_subclasses_of`, and the reachability walks `descendants`,
  `ancestors`, `direct_predecessors`. `PluginOps` gained `add_edge(...)`
  and `add_synthetic_node(...)` alongside `keep_alive(...)`, plus a
  `FLAG_ENTRYPOINT` constant. Every query is index-based and GIL-free;
  no `Python<'_>` token is exposed.
- **Per-file external native plugins.** An external native plugin can
  now opt into the salsa-cached *per-file* path (previously in-tree
  only) by implementing the new `PerFilePlugin` trait and returning
  `Some(self)` from `ExternalPlugin::per_file()`. When it does, the host
  ignores the project-wide `run` and instead invokes
  `run_on_file(file, ops)` once per project file through the same
  salsa-cached query the in-tree `MainBlockPlugin` uses — so an
  unchanged file's ops are reused across a `re_materialize` with zero
  re-run. `run_on_file` gets a restricted single-file `PluginFileCtx`
  (file-local nodes, parsed AST, `line_span`, `main_block_range`) and
  emits file-local ops through `FileOps::add_synthetic_node`. The
  per-file output must be a pure function of the file (the documented
  cache-soundness contract). See `examples/per_file_main_block/`.
- **Ready-made per-file query API + richer file ops + a pre-graph
  hook for native plugins.** A per-file plugin no longer has to
  hand-roll import / decorator / call matching from the raw AST. The
  restricted `PluginFileCtx` (and the in-tree `FileContext` it wraps)
  now exposes file-local queries that reuse the *same* matcher cores
  the project-wide native plugins are built on, so a per-file plugin and
  a project-wide twin agree on what matches:
  - `imports_any_module(&["click"])` — a cheap presence guard.
  - `decorated_decls(&["click"], &["command", "group"])` — file-local
    indices of decls decorated by the named imports (resolving direct
    imports + aliases).
  - `constructions(&["flask"], &["Flask"])` / `calls(&["typer"],
    &["Typer"])` — file-local owners of matched constructor calls /
    call targets.
  `FileOps` gained `keep_alive(local_idx)` and
  `add_edge(src_local, dst_local)` alongside `add_synthetic_node`, so
  the common "keep these decls alive" shape is one call. And
  `ExternalPlugin::prepare(project_root)` is now a real pre-graph hook
  — `NativePlugin.prepare(...)` forwards to it (and to in-tree
  project-wide impls), so a plugin can read config under the project
  root before the graph is built. See `examples/per_file_decorated/`.
- **Project-wide matcher + write parity for the native plugin airlock.**
  The project-wide `PluginCtx` now exposes the same ready-made matchers
  the per-file `PluginFileCtx` got — `decorated_decls(modules, names)`
  and `constructions(modules, names)` (sharing the matcher cores the
  in-tree native plugins use, so the two surfaces agree) — plus the targeted reads
  `module_surface`, `dunder_all_exports`, `literal_list_entries`, and
  `decls_matching_name`. On the write side, `PluginOps`/`FileOps`
  `add_edge` now takes a `flags` argument (with `plugin_api::FLAG_DEAD_BRANCH`
  / `FLAG_DYNAMIC_IMPORT` re-exported) and `add_synthetic_node` takes an
  `edges_from` (in-edge) list — so the airlock's write surface can stamp
  dead-branch / dynamic-import edges and wire a synthetic node's full
  in/out edges.
- **Unified plugin API — every built-in dogfoods the curated surface.**
  All in-tree built-ins now implement the *same*
  `plugin_api::ExternalPlugin` / `PerFilePlugin` traits an external plugin
  compiles against; the parallel internal plugin traits are gone, so the
  curated surface is the only plugin surface and can no longer drift from
  what the shipped plugins use. To make that possible:
  - `ExternalPlugin::run` is now **fallible** —
    `run(&PluginCtx, &mut PluginOps) -> Result<(), PluginError>`. A
    returned `Err` aborts the materialize and surfaces to Python as an
    exception (`PluginError::value(..)` → `ValueError`,
    `PluginError::runtime(..)` → `RuntimeError`); previously an external
    plugin's failure was silently swallowed. `PluginError` is pyo3-free;
    per-file `run_on_file` stays infallible. (Breaking for the
    experimental external API: a plugin's `run` must now return
    `Ok(())`.)
  - `PluginCtx` grew to cover every query the built-ins use: bulk reads
    (`nodes_at`, `node_paths`, `modules_for_paths`, `module_surfaces`),
    more matchers (`calls_on_var`, `handler_decorators_via`,
    `decorated_decls_with_args`, `factory_decls`, `classes_defining_method`,
    `module_top_level_decls`, `has_imports_of` / `imports_of`,
    `function_parameters` / `class_method_parameters`), a structured
    `nodes_matching(&NodeFilter)` filter, and the borrowing `nodes()` /
    `edges()` iterators (`NodeRef` / `EdgeRef`) for whole-graph scans.
    `CallArgs` / `ArgValue` are re-exported from `plugin_api`.
  - The internal `FrozenView` query layer was de-Pythoned: its methods
    return plain `Option` / `Vec` / `Result<_, PluginError>`, so `pyo3`
    survives only at the true host boundary (the `NativePlugin` pyclass +
    `materialize`), never threading Python types through GIL-free Rust.
- `Analysis.re_materialize(events)` — incrementally rebuild the
  project graph against the existing `native.ProjectContext`. The
  caller supplies the change events: typically
  `ctx.detect_changes()` (which today returns a single
  `ChangeEvent.rescan()`), or an explicit list of
  `native.ChangeEvent`s for LSP integrations and file-watchers.
  `native.ProjectContext.apply_changes(events)` forwards to
  `ty_project::ProjectDatabase::apply_changes`, which handles each
  variant correctly — `Changed` bumps the file's salsa revision only
  if its mtime / size actually differ; `Created` registers brand-new
  paths with the project file set so they're discovered on the next
  rebuild; `Deleted` removes them; `Rescan` triggers a full
  `Files::sync_all` + project re-walk + metadata rediscovery.
  Configuration files (`pyproject.toml`, ignore files, custom-stdlib
  `VERSIONS`) trigger a project reload automatically. Salsa's
  per-file cache for content-unchanged files survives across calls,
  so the assemble pass and plugin pass run on a warm cache. Plugin
  `prepare` is a one-shot owned by `materialize_all`, not re-run on
  re_materialize.
- `native.ChangeEvent` Python class with `changed(path)`,
  `created(path)`, `deleted(path)`, and `rescan()` classmethods plus
  `.kind` / `.path` accessors, exposed for LSP-style integrations
  that want precise control over what to invalidate.
- `native.ProjectContext.clear_plugins()` and `reset_progress()` —
  internal helpers `re_materialize` uses to keep plugin registrations
  and progress counters from leaking across calls.
- **External plugin read/write API reaches `query(ctx)` parity, plus a
  configurable `dispatch_app` factory.** The remaining gaps between the
  external Rust `plugin_api` surface and the Python `query(ctx)` DSL are
  closed: `PluginCtx` gains `find_subclasses_of_fqn(base_fqn, transitive)`
  (subclass search by string FQN, not just an in-graph index),
  `handler_decorators(&[attr])` (functions decorated `@<owner>.<attr>(...)`,
  paired with their owner), `calls_with_string_arg(modules, name,
  arg_index)` (calls to an imported callable paired with a string-literal
  positional argument), and `calls_on_attr(attr, arg_index)` (its attr-method
  twin — `<recv>.<attr>(...)` calls of any receiver shape paired with a
  string-literal positional argument, mirroring
  `query(ctx).calls().where_attr(...).string_arg_at(...)`);
  `PluginOps::add_synthetic_node` now takes a source
  `path` (empty for a placeless marker) so a project-wide synthetic node can
  be attributed to a file. On the Python side, `NativePlugin.dispatch_app(name,
  marker_prefix, app_classes, registration_decorators, seed_as_entrypoint)`
  exposes the generic engine behind `flask()` … `celery()`, so a custom
  framework can be wired without a bespoke plugin (the celery `@shared_task`
  fan-out stays internal to `celery()`).
- **`serde_json` and `regex` are available to external plugin authors.** Both
  are pinned as direct runtime dependencies so their `.rlib`s are always in the
  plugin-compile closure, and `dead-cst build-plugin` wires `--extern serde_json`
  / `--extern regex` from a curated allowlist so a plugin can `use
  serde_json::Value;` / `use regex::Regex;` out of the box. The rest of the
  runtime's private dependency tree is intentionally not exposed.
- **`Analysis.set_stack_size(bytes)` for deeply-nested ASTs.** Overrides the
  rayon worker-thread stack size for *both* the per-file populate fan-out and
  the project-wide plugin pass (the class-hierarchy re-walks subclass resolution
  drives recurse as deep as file ingest). Call it before `materialize_all()`;
  with no override both phases use rayon's default stack (2 MiB unless
  `RAYON_STACK_SIZE` / `RUST_MIN_STACK` is set process-wide), which suffices for
  typical code. Use it on projects with deeply-nested generated code (protobuf
  modules, ML-generated ASTs, large nested literal dicts) that stack-overflow at
  the default — the size is virtual address space, so a large value costs no
  resident memory unless actually used.
- **Extensible node/edge flag registry (`FlagSpec`).** Flags are no longer a
  fixed catalog of compile-time bits. Every flag — engine built-ins included —
  is now described by a `FlagSpec` (`name` as `owner/name`, `seed`, `default_on`,
  `description`) and lives in one of two registries (a 32-bit node space, an
  8-bit edge space). The engine registers its `engine/…` flags with explicit
  masks (so hot-path reads still constant-fold); a native plugin contributes its
  own through the new `ExternalPlugin::declare_node_flags()` /
  `declare_edge_flags()` hooks and reads the host-assigned bit back at run time
  with `PluginCtx::node_flag(name) -> Option<u32>` / `edge_flag(name) ->
  Option<u8>`. Plugin bits are allocated above the engine masks in registration
  order (deterministic); the `engine/` namespace is reserved (a plugin using it
  fails the materialize); declaring the same spec from two plugins is idempotent
  and shares one bit (this is how the built-in `pytest` and `unittest` plugins
  now share a `test/testcase` flag), while a conflicting re-declaration or
  exhausting the bit width fails loudly. Both registries are **serialized into
  the `.dcg` graph file** so an external reader can decode any bit by name;
  `native.ProjectContext` / a loaded graph expose `node_flag_registry()`,
  `edge_flag_registry()`, `default_seed_mask()`, `node_flag(name)`, and
  `edge_flag(name)`, and `GraphMetadata` carries both tables. (`PLUGIN_API_EPOCH`
  bumped 1→2; `FORMAT_VERSION` bumped 1→2 — old graph files are rejected.)

### Changed

- **Module-level dunder / `__future__` keep-alive moved into the engine.**
  Module-scope dunders (`__all__`, `__version__`, PEP 562
  `__getattr__` / `__dir__`, …) and `__future__` imports are now kept alive by
  an edge *from their module node*, emitted at edge-collection time. They are
  not standalone reachability seeds: a dunder survives only while its module is
  reachable, so an unreachable module's dunders die with it. This replaces the
  per-file `ModuleDundersPlugin` (and its synthetic `<dunder>:` seed node) with
  a `module -> dunder` edge; the CLI no longer appends it to the default plugin
  set because the behavior is now unconditional.
- **`unittest` plugin no longer mints a `<unittest>:` synthetic seed.**
  `TestCase` subclasses (unconditional test roots) now carry the
  `test/testcase` flag stamped directly on the decl via the new
  `PluginOps::flag_decl`, and module lifecycle hooks (`setUpModule`,
  `tearDownModule`, `load_tests`) are kept alive by a `module -> hook` edge
  instead of the seed marker. Behavior change: a lifecycle hook in a module
  with no live `TestCase` now dies with its (unreachable) module, matching the
  module-dunder model — previously the per-module seed kept it alive
  unconditionally.
- **Resolved site-packages imports are now real `kind="external"` graph
  nodes.** A site-packages import (`[external dist] requests`) mints a real
  external node carrying the resolved site-packages path, replacing the old
  path-less synthetic sink. The real path lets the codemod exclude them by
  the same `is_relative_to(package)` test it uses for everything else.
  **Stdlib imports mint no node** — an unused `import os` is still caught
  dead via its alias's zero in-edge count, so a stdlib endpoint would be
  pure noise. An import that genuinely doesn't resolve also mints **no**
  sink node — its local alias is flagged `NodeFlags.UNRESOLVED` instead. The
  `[unresolved]` / unparseable synthetic-node prefix constants
  (`UNRESOLVED_PREFIX`, `UNPARSEABLE_PREFIX`, `SYNTHETIC_PATH_PREFIXES`, and
  the now-unused `STDLIB_PREFIX`) are dropped from `dead_cst.plugins`.
- **`__init_subclass__` now emits direct `parent -> subclass` edges.** A base
  class that defines `__init_subclass__` previously kept its subclasses alive
  through a `<__init_subclass__>:Parent` synthetic anchor node; it now links
  the parent straight to each subclass with an edge flagged
  `EdgeFlags.INIT_SUBCLASS`, so `why-alive` chains read `Foo <- Parent`
  without the intermediate hop.
- **1:1 keep-alive plugins flag the target decl directly.** A plugin that
  keeps a single decl alive (explicit entrypoints, discord.py apps) now sets
  `NodeFlags.ENTRYPOINT` on that decl instead of minting a `{marker}:{fqname}`
  synthetic seed plus an edge. The `PluginOps`/`FileOps` `keep_alive` op
  dropped its `marker: String` parameter (`plugin_api` epoch 2 → 3).
- **Graph node indices are now deterministic across runs.** ty's `files()`
  set has no stable iteration order, so the global index assigned to each node
  (and therefore the order of `Analysis.dead()` / `reachable()` results and the
  CLI's reports) could vary run-to-run. The build now sorts files into
  canonical path order and assigns each node a position-derived index
  (`offset[file] + local_index`) — the per-file offsets come from prefix-summing
  node counts captured during the parallel fan-out — so output is reproducible.
- **`from X import *` now mints a node per re-exported name.** Previously a
  star import collapsed to a single `<mod>.*<source>` statement node. It now
  also mints one `NodeFlags.STAR_REEXPORT` per-name `kind="import"` node for
  each name `X` exports (`mod.g`, `mod.h`, …), each keyed on ty's per-name
  `StarImport` definition so a use of a star-bound name resolves *straight to
  its own node* rather than to the shared statement node. Every per-name node
  edges to the kept statement node, which retains the single upstream-module
  edge and stays the unit the codemod removes; cross-module `from X import g`
  still chases the star chain to the real upstream decl. The codemod skips
  `STAR_REEXPORT` nodes — they share the `*` token's source range and have no
  removable span of their own — so a dead star import is handled exactly as
  before. This grows the graph but lets shadowed/duplicate star re-exports and
  per-name reachability be expressed precisely.
- **Per-file parsed ASTs are freed mid-build to cut the memory peak.** The
  project-wide plugin queries that used to re-walk each file's AST (parameters,
  class-defining-method, decorators, constructions, factories, and the
  call-site family) now read precomputed per-file facts from a salsa-cached
  `FileExtraction`, warmed during the per-file fan-out alongside the existing
  node/edge payloads. Once a file's payloads, `FileExtraction`, and per-file
  plugins are computed, its parsed AST is dropped (`ParsedModule::clear`)
  instead of staying resident through assembly and the project-wide plugin
  pass. On a 250-file synthetic project this takes the parsed-AST salsa heap
  from ~5 MB to ~0 at build-end (~30% of total salsa memory). Subclass
  resolution no longer re-parses **and no longer runs ty's `find_references`
  walk at all**: during the per-file fan-out each top-level class base is
  decomposed — via ty's use-def chain, not type inference — into a lightweight
  symbolic spec: a same-file class (keyed by its name-range) or a
  `module` + `name` member reference for anything imported. These specs are
  deliberately *not* resolved across files at store time, so a file's cached
  payload stays salsa-invalidation-local. Cross-file resolution happens later,
  at both assemble and query time, through one shared member resolver
  (`resolve_member_def`) built on ty's module resolver plus the same use-def
  decomposition the store side runs (ty's `global_scope`/`place_table`/
  `use_def_map` primitives), which follows re-export and alias chains.
  Because the observed bases (at assemble) and the subclass query both funnel
  every member reference through that same resolver, every spelling of one base
  collapses to a single `(file, name-range)` key by construction: an external
  base's sibling spellings (`unittest.TestCase` and `unittest.case.TestCase`
  both name the single `class TestCase`), relative re-exports
  (`from .bases import TestCase`), `from … import *` names, and module-level
  aliases (`Base = Imported` *or* the attribute form `Base = mod.Imported`) all
  resolve to the same decl as the original spelling. At assemble time each
  resolved base is a single hashmap probe against the class index — a hit is a
  project parent, a miss is an external base, keyed by that same
  `(file, name-range)` so the subclass query is an O(1) lookup with no scan. No
  per-file name-binding table, no eager cross-file resolution, no hand-rolled
  fqn chase, no env-var fallback. The memory and no-re-parse mechanics are
  results-neutral; the one behavioral change is that subclasses reached through
  a star import, a module-level alias, or an absolute *or* relative re-export
  are kept alive where the default path previously dropped them.
- **Built-in plugins are migrating to native (Rust) implementations.**
  The `main_block`, `module_dunders`, `init_subclass`, `server_config`,
  and `unittest` built-ins now resolve to native `NativePlugin`
  instances through a Rust registry
  (`native._builtin_native_plugin(name)`) that the CLI's `_load_plugin`
  consults before the Python builtin map. Behaviour is identical;
  `module_dunders` and `server_config` are *per-file* (salsa-cached)
  plugins, so an unchanged file's entrypoints are reused across
  `re_materialize` with zero re-run. `NativePlugin.server_config(filenames=…)`
  is the first *configured* per-file plugin: the matched filename set is
  carried as config and the per-file cache key is hash-interned on it, so
  identical filename sets share one salsa cache entry (`filenames=None` bakes
  the default Gunicorn/Hypercorn set). The Python `ModuleDundersPlugin` /
  `InitSubclassPlugin` / `MainBlockPlugin` / `UnittestPlugin` classes remain
  available for now; they will be removed once every built-in is ported.
- **Dispatch-app frameworks ported to a native engine.** The `flask`,
  `fastapi`, `typer`, `cyclopts`, `slack_bolt`, `fastmcp`, and `celery`
  built-ins are now native `NativePlugin` instances
  (`NativePlugin.flask()` … `NativePlugin.celery()`), resolved through the
  same Rust registry. One `DispatchAppPluginImpl` carries each framework's
  config — app classes, per-instance registration decorators, and whether
  the app seeds itself as an entrypoint (Celery additionally fans
  `@shared_task` out module-wide) — and runs project-wide over the existing
  `find_*` queries. Behaviour is identical. Because each framework is now an
  independent project-wide native plugin, the Python-side automatic batching
  of multiple `DispatchAppPlugin` instances is retired.
- **`click` ported to a native plugin.** The `click` built-in is now a native
  `NativePlugin.click()`, resolved through the same Rust registry. Unlike the
  dispatch-app frameworks it keeps its own implementation: Click groups never
  seed themselves as entrypoints (reach them via `[project.scripts]` /
  `__main__` / `add_command`), and a `@<group>.group()` handler is promoted to
  a group through a fixpoint so nested sub-commands wire transitively.
  Behaviour is identical.
- **Remaining built-in plugins ported to native.** The `mock_patch`,
  `discordpy`, `pytest`, `project_scripts`, and `dynamic_import_fallback`
  built-ins are now native `NativePlugin` instances resolved through the same
  Rust registry (`NativePlugin.mock_patch()` …
  `NativePlugin.dynamic_import_fallback()`, or the matching `--plugin` keys).
  The explicit-entrypoint plugin is likewise native —
  `NativePlugin.explicit(regexes, str_specs, abs_paths)`, which the CLI drives
  from `-e` / `--entrypoint-regex` (it is not a `--plugin` key). Behaviour is
  identical. With this, every built-in plugin is native; the CLI resolves
  `--plugin` names through the native registry (`_builtin_native_plugin`).
- **Per-file native plugins now run inside the build pipeline.** Their
  salsa-cached file-local ops are warmed during the parallel (GIL-released)
  file fan-out and folded into the graph during serial assembly, instead of a
  separate post-build plugin pass. Project-wide plugins still run after the
  build. Behaviour is identical; per-file plugins no longer emit ops in the
  post-build pass (they would double-apply), so a graph with only per-file
  native plugins skips that pass entirely.
- **Project-wide plugin orchestration moved into Rust.** The project-wide
  plugin pass now fans out across a GIL-free `rayon` scope inside
  `ProjectContext.materialize()` (one `py.allow_threads` wrapping the scope, one
  worker per plugin), each plugin running against a `Send` `FrozenView` snapshot
  of the build outputs and pushing a `Vec<PreparedOp>` that folds into the graph
  in registration order in one atomic apply — mirroring the existing per-file
  fan-out. This replaces the Python `ThreadPoolExecutor` that previously drove
  the pass, so the now-unused `_native` methods that only served it are gone
  (`build_only`, `run_plugin`, `run_plugin_collect`, `apply_ops_batched`,
  `snapshot_graph`, the per-plugin progress-driver methods, the `CollectedOps`
  type, and the leftover `query()`-DSL query wrappers). Behaviour is identical.
- **The last native plugins that re-entered Python at build time are now pure
  Rust.** `project_scripts` parses `[project.scripts]` with the `toml` crate
  instead of stdlib `tomllib`, and `dynamic_import_fallback` matches its
  include/exclude globs with the `regex` crate instead of `fnmatch.fnmatchcase`
  / `pathlib.PurePosixPath.match` (componentwise, from the right, case-sensitive
  — same match set). Behaviour is identical, but no built-in plugin re-acquires
  the GIL inside the GIL-free `rayon` fan-out anymore.
- **Edge flags narrowed to `u8`** (node flags stay `u32`). `EdgeFlags` and every
  edge-flag carrier across the runtime are now 8-bit; Python still sees edge-tuple
  flags and node `.flags` as plain `int`, so the change is transparent across the
  FFI boundary (the bincode layout change rides the `FORMAT_VERSION` 2 bump).
- **The default keepalive mask is now registry-derived.** Reachability defaults
  to `ctx.default_seed_mask()` — the OR of every registered flag that is `seed &&
  default_on` — instead of the hand-maintained `KEEPALIVE_DEFAULT` constant. A
  consequence: `test/testcase` is in the default mask **iff** a test plugin
  (`pytest` / `unittest`) is registered, which is exactly the right behaviour.
- **Overload status is now internal build-time metadata.** The `OVERLOAD` node
  flag is replaced by an internal `is_overload` field on the build-time node
  payload; it is no longer a reachability flag, is not serialized into the graph
  file, and is not Python-visible. Overload behaviour is unchanged (the
  `impl → stub` anchor edges are still emitted and cross-module imports still
  resolve to the implementation, not a stub).

### Removed

- **`NativePlugin.module_dunders()` / `ModuleDundersPlugin` (breaking).** The
  per-file module-dunder plugin is gone; its keep-alive behavior is now
  always-on engine policy (see Changed). The CLI `--plugin module_dunders`
  name and the `_builtin_native_plugin("module_dunders")` lookup are removed.
  The `ProjectContext.find_module_dunders_indices()` introspection query is
  retained.
- **`NodeFlags` roster surgery (breaking).** `NodeFlags.SHADOWED` and
  `NodeFlags.EXPORTED` (both dead — no set/read site) are deleted;
  `NodeFlags.OVERLOAD` and `NodeFlags.TESTCASE` are removed from the public flag
  surface. Overload status became internal build-time metadata (see Changed),
  and `TESTCASE` became the registered `test/testcase` plugin flag — resolve it
  with `ctx.node_flag("test/testcase")` (returns `None` when no test plugin is
  registered). A new `NodeFlags.DEAD_BRANCH` (engine metadata for decls in a
  statically-dead region; not a seed) is added. The surviving roster is `NONE`,
  `ENTRYPOINT`, `NOQA`, `NOTEBOOK`, `DEAD_BRANCH`, `STAR_REEXPORT`.
- **`dead_cst.graph.KEEPALIVE_DEFAULT` (breaking).** The hand-maintained
  keepalive constant is gone; callers default `seed_flags` to the registry-derived
  `default_seed_mask()` instead (available on `Analysis`, `native.ProjectContext`,
  and a loaded graph). `Analysis` also gains a `node_flag(name)` passthrough.
- `ProjectContext.find_comment_patterns(pattern)` (the regex-over-comments
  query) and its `_native.pyi` stub. It had no in-tree callers and was the last
  graph query that re-parsed files on demand after the build; removing it drops
  that re-parse path entirely.
- The Python `ServerConfigPlugin` (`dead_cst.contrib.ServerConfigPlugin`).
  `server_config` is now native-only: use `NativePlugin.server_config()` (the
  CLI's `server_config` key already resolves to it), passing `filenames=[…]`
  for custom server-config basenames. This is the first built-in whose Python
  class is removed as part of the native migration.
- The Python dispatch-app framework plugins — `dead_cst.contrib.flask_plugin`,
  `fastapi_plugin`, `typer_plugin`, `cyclopts_plugin`, `slack_bolt_plugin`,
  `fastmcp_plugin`, and `CeleryPlugin` — and the reusable `DispatchAppPlugin` /
  `DispatchAppSpec` / `DispatchAppGather` shapes in `dead_cst.plugins`. Use the
  native factories (`NativePlugin.flask()` … `NativePlugin.celery()`); the CLI
  keys (`flask`, `fastapi`, `typer`, `cyclopts`, `slack_bolt`, `fastmcp`,
  `celery`) already resolve to them.
- The Python `ClickPlugin` (`dead_cst.contrib.ClickPlugin`). `click` is now
  native-only: use `NativePlugin.click()` (the CLI's `click` key already
  resolves to it).
- The Python standalone plugins — `dead_cst.plugins.DynamicImportFallbackPlugin`,
  `ExplicitEntrypointPlugin`, and `ProjectScriptsPlugin`, plus
  `dead_cst.contrib.MockPatchPlugin`, `PytestPlugin`, and `DiscordPyPlugin`. Use
  the native factories (`NativePlugin.dynamic_import_fallback()`,
  `NativePlugin.explicit(…)`, `NativePlugin.project_scripts()`,
  `NativePlugin.mock_patch()`, `NativePlugin.pytest()`, `NativePlugin.discordpy()`);
  the CLI keys already resolve to them (and `-e` / `--entrypoint-regex` drive
  `explicit`).
- **The Python `Plugin` protocol is removed.** With every built-in now native,
  the `Plugin` ABC (`dead_cst.plugins.Plugin`), the reusable
  `DecoratedDeclPlugin` / `LiteralListPlugin` base shapes, and the
  `dead_cst.plugins.decl_shapes` module are gone, along with the four
  parity-twin Python classes (`MainBlockPlugin`, `ModuleDundersPlugin`,
  `InitSubclassPlugin`, `UnittestPlugin`). The rust harness no longer runs
  Python `plugin.run(ctx)` callbacks, so the `AddNode` / `AddEdge` /
  `AddEntrypoint` graph-op pyclasses and their `AddNodeByIdx` / `AddEdgeByIdx` /
  `AddEntrypointByIdx` index-keyed siblings are removed too. Built-in plugins
  are the `NativePlugin.<name>()` factories; author out-of-tree plugins as
  external native plugins (see `NATIVE_PLUGINS.md`). `Analysis(...,
  plugins=[…])` accepts only `dead_cst._native.NativePlugin` instances.
  `dead_cst.plugins.__all__` is now just the external-node prefix constants
  plus `simple_name`; `dead_cst.contrib.__all__` is empty. `CollectedOps` and
  `PreparedOp` are unaffected.

#### The chainable `query(ctx)` DSL is removed
- `native.query(ctx)` and the whole chainable builder — `QueryBuilder` and
  every `*Query` class (`DeclQuery`, `DecoratorQuery`, `ConstructionQuery`,
  `CallQuery`, `ImportQuery`, `ClassQuery`, `SubclassQuery`, `EdgeQuery`,
  `ModuleQuery`, `MainBlockQuery`, `FactoryQuery`), their `*IdxRef` result
  rows, and the `CallArg` / `ArgLiteral` / `ArgNodeRef` / `ArgOpaque` argument
  union. Every built-in plugin is now a native Rust `NativePlugin` that calls
  the rust query cores directly, so the Python-facing DSL had no remaining
  in-tree consumers.
- The query surface that stays on `native.ProjectContext` is the `*_indices`
  methods (`find_declarations_indices`, `module_surface_indices`,
  `decls_matching_indices`, `indices_where`, …) — each returns positional
  indices into `ctx.nodes()`, materialized in bulk via `ctx.nodes_at(idxs)` /
  `ctx.node_attrs(idxs)` — plus the direct accessors (`find_module_idx`,
  `module_for_indices`, `find_main_blocks_indices`, `find_factory_decls`, …).
  The decorator / construction / call walks the native plugins drive are now
  rust-internal.

### Fixed

- `dead-cst bundle-plugin-host` no longer ships duplicate dependency archives.
  Cargo's `deps/` directory accumulates multiple SVH-suffixed artifacts per
  crate across incremental rebuilds, and the closure copy previously shipped all
  of them (e.g. two `regex` rlibs), bloating the `dead-cst-plugin-host` payload.
  The closure is now deduplicated to exactly one artifact per `(crate, kind)` —
  the newest, which is the set this build's runtime dylib binds against.
- Uses inside an assignment whose target is a subscript or slice
  (`os.environ["k"] = v`, `f[:] = [SomeClass()]`) are no longer
  dropped. Such a target binds no name, so ty mints no Definition and
  the per-decl pass skips it; the module-level walk previously skipped
  it too because the statement was classified as a definition. The
  subscripted object on the left and every name on the right are now
  walked, so the imports / decls they reference keep their in-edges
  instead of looking dead.
- Sibling submodule imports that share a root binding (`import a.foo`
  then `import a.bar`) no longer leave the earlier statement looking
  dead. ty's flow-sensitive use-def chain attributes a use of the
  shared root name to the last rebind only, so `a.foo.x()` failed to
  keep `import a.foo` alive. A chained access now also resolves against
  the scope-wide reachable bindings and keeps every `import <root>.<…>`
  whose submodule suffix matches the access chain.
- A binding introduced only under `if TYPE_CHECKING:` no longer hides
  the runtime binding of the same name. ty narrows `TYPE_CHECKING` to
  `True`, so its use-def chain resolves a use to the type-checking-only
  binding and the branch that actually runs (`else: from b import X`,
  `else: X = …`, or a later rebind) was left with zero in-edges. When
  the flow-resolved binding falls inside a `TYPE_CHECKING` block, the
  resolver now also keeps every reachable binding outside such blocks,
  so the runtime import/decl survives.

## [0.13.0] - 2026-05-26

### Added

#### Plugin harness
- `Analysis(progress_callback=fn)` keyword argument. `fn(event,
  **kwargs)` receives structured progress events
  (`phase_start(phase, total)`, `phase_progress(phase, current,
  total)`, `phase_end(phase, elapsed_ms)`, `plugin_start(name,
  index, total)`, `plugin_end(name, elapsed_ms)`) from a daemon
  thread polling rust-side atomic counters every ~100 ms. Mutually
  exclusive with `show_progress=True`; the latter now installs a
  default stderr-text callback. Exports `PROGRESS_PHASES`,
  `PROGRESS_POLL_INTERVAL_S`, and `ProgressCallback` from
  `dead_cst.analyze`.
- Frozen-graph plugin execution: every plugin's `run(ctx)` observes
  the base graph only — never another plugin's emissions or its
  own earlier yields. Ops are collected per-plugin into a
  `CollectedOps` handle and batch-applied after every plugin
  returns. Same contract on the rust-side serial loop and the
  `ThreadPoolExecutor` parallel loop.
- `Analysis` auto-batches `DispatchAppPlugin` instances. Registering
  multiple dispatch plugins triggers a single fused gather
  (shared subclass-walk cache, per-distinct-module construction /
  factory queries, single project-wide variable scan) on the main
  thread between the build pass and the plugin fan-out;
  per-plugin `policy(ctx, gathered)` calls fan out through the
  same `ThreadPoolExecutor` the harness uses for every other
  plugin. Replaces the explicit `BatchDispatchAppPlugin` wrapper.
- `DispatchAppPlugin` split into a frozen `DispatchAppSpec` (the
  gather config) and a `policy(ctx, gathered)` emission method.
  `DispatchAppGather` carries the pre-walked data. Subclass
  overrides of `policy()` are honored uniformly by both the
  standalone run path and the auto-batched gather. `CeleryPlugin`'s
  `@shared_task` fan-out moved to a `policy()` override and now
  fires correctly under batching.

#### Query DSL
- Chainable query API via `native.query(ctx)` replacing the legacy
  `ctx.find_*` family. Streams: `decorators()` / `constructions()` /
  `calls()` / `subclasses()` / `imports()` / `modules()` /
  `classes()` / `factories()` / `edges()` / `decls()` /
  `declarations()` / `main_blocks()` / `literal_lists()`. Seeded
  closure walks via `from_idx(seed).descendants() / .ancestors() /
  .direct_predecessors()`. Top-level terminals:
  `reachable(seed_flags=..., skip_flags=...)` and
  `matching_specs(project_root, regexes=..., str_specs=...,
  abs_paths=...)`.
- Uniform terminal set on every Tier-1 query (`DeclQuery`,
  `ModuleQuery`, `SubclassQuery`, `ImportQuery`, `ClassQuery`,
  `DeclarationsQuery`):
  - `.indices() -> list[int]`.
  - `.attrs() -> list[NodeAttrs]` — folds
    `ctx.node_attrs(q.indices())` into a chainable call.
  - `.first_idx() -> int | None`.
  - `.indices_by_path() -> dict[str, list[int]]` — group matched
    indices by their owning file path; lets plugins fan out
    per-file work without re-querying.
  - `.count()` / `__iter__`.
- `.indices_by_path()` on every Tier-2 row query (`DecoratorQuery`,
  `ConstructionQuery`, `CallQuery`, `FactoryQuery`) — reads `path`
  straight off each row.
- `DeclQuery` predicate vocabulary: `with_kind` / `with_kinds`,
  `with_filename` / `with_filenames`, `with_simple_name` /
  `with_simple_names`, `with_simple_name_regex`, `with_paths`,
  `with_path_regex`, `with_path_prefix`, `with_path_contains`,
  `with_flags` / `with_any_flag`, `with_fqname_prefix`,
  `with_fqname_under` (segment-bounded), `where_fqname` (literal /
  regex / mixed).
- Idx-form result rows: `DecoratorIdxRef`, `ConstructionIdxRef`,
  `CallIdxRef`, `FactoryIdxRef`. Each carries a primary idx
  (`decorated_idx` / `var_idx` / `owner_idx` / `decl_idx`),
  `path`, and query-shape metadata strings (`decorator_owner`,
  `class_name`, `kinds`, …). The lazy `args` / `kwargs` getters
  surface entries as a `CallArg` discriminated union of
  `ArgLiteral(value)` / `ArgNodeRef(idx)` / `ArgOpaque()` — tagged
  dispatch via `isinstance` or `match`, with embedded decl refs as
  `ArgNodeRef` rather than `Py<SymbolNode>`.
- `NodeAttrs` — tuple-like pyclass returned by
  `ProjectContext.node_attrs` and every `.attrs()` terminal.
  Supports both attribute access (`attr.fqname`) and tuple
  semantics (`kind, path, fqname, flags = attr`; `attr[2]`;
  `len(attr) == 4`).

#### Graph ops
- `AddNodeByIdx`, `AddEdgeByIdx`, `AddEntrypointByIdx` —
  index-keyed siblings of `AddNode` / `AddEdge` / `AddEntrypoint`.
  Take positional indices into `ctx.nodes()`; the apply pass
  resolves them rust-side instead of round-tripping through
  `Py<SymbolNode>`. Bounds-checked at apply time; out-of-range
  indices raise `IndexError` before any new node is interned, so a
  bad endpoint never leaves an unconnected synthetic behind.

#### `ProjectContext` idx helpers
- `ctx.nodes_at(indices)`, `ctx.node_attrs(indices)` (returns
  `list[NodeAttrs]`), `ctx.node_paths(indices)`,
  `ctx.indices_where(*, kind=, fqname_prefix=, ...)`,
  `ctx.reachable_indices`, `ctx.descendants_indices`,
  `ctx.ancestors_indices`, `ctx.direct_predecessors_idx` — pair
  with the `.indices()` terminals to stay in idx-space end-to-end.
- `ctx.modules_for_paths(paths)` and
  `ctx.module_surfaces_indices(fqns)` — bulk forms that fuse N
  point lookups into one.

### Changed
- `with_args` flipped from opt-out to opt-in on `DecoratorQuery` /
  `ConstructionQuery` / `CallQuery`. Default is now `False` — the
  per-row `extract_call_args_kwargs` walk is skipped and row
  `args` / `kwargs` getters surface empty containers. Plugins
  reading args/kwargs off rows must call `.with_args(True)`.
  `.where_kwarg(...)` still forces extraction back on, so
  kwarg-filtered queries don't need the explicit opt-in.
- `AddNode`'s apply pass pre-resolves every `edges_from` /
  `edges_to` key to its builder index *before* minting the
  synthetic node, matching `AddNodeByIdx`. A missing key now
  surfaces as `ValueError` without leaving an orphan node
  (previously the synthetic was interned first, then the failing
  key lookup aborted, stranding the new node).
- Every bundled plugin (`plugins/*` and `contrib/*`) ported to the
  idx-form APIs — no plugin in the tree allocates or reads a
  `Py<SymbolNode>` anymore. Plugins fan out through `.indices()` /
  `.attrs()` terminals, batch attr fetches via `ctx.node_attrs`,
  and yield `AddEdgeByIdx` / `AddNodeByIdx` /
  `AddEntrypointByIdx` exclusively.

### Removed
- `BatchDispatchAppPlugin`. Migrate by replacing
  `[BatchDispatchAppPlugin(plugins=[flask_plugin(),
  fastapi_plugin()])]` with `[flask_plugin(), fastapi_plugin()]`
  in the `Analysis(plugins=...)` argument; `Analysis` now
  auto-batches every registered dispatch plugin.
- Legacy node-form `ctx.find_*` and `ctx.decls_*` helpers (all
  returning `list[SymbolNode]`): `find_declarations`,
  `find_module`, `find_module_dunders`,
  `find_module_top_level_decls`,
  `find_module_dunder_all_exports`, `find_main_blocks`,
  `find_subclasses`, `find_subclasses_of`, `subclasses_of_fqn`,
  `subclasses_of_node`, `find_classes_defining_method`,
  `find_imports_of`, `find_nodes_matching_specs`, `module_for`,
  `module_surface`, `module_surfaces`, `resolve`, `decls_under`,
  `decls_matching`, `decls_matching_name`, `direct_predecessors`,
  `has_imports_of`, `imports_of_count`. Plugin authors migrate to
  the chainable form
  (`native.query(ctx).<stream>().<predicates>().indices()` /
  `.attrs()`); the `_indices` siblings that back the DSL remain on
  `ctx` as low-level escape hatches.
- Node-form ref types `DecoratorRef`, `ConstructionRef`, `CallRef`,
  `FactoryRef` and the `.row_indices()` terminal. `.collect()` on
  the four ref queries returns the `IdxRef` siblings directly.
  Plugins migrate via `s/row_indices/collect/` plus
  `ref.decorated.fqname` → `ctx.nodes()[ref.decorated_idx].fqname`
  (or `ctx.node_attrs(...)`).

## [0.12.2] - 2026-05-25

### Added
- `dead_cst.contrib.slack_bolt_plugin` keeps `slack_bolt.App` and
  `slack_bolt.async_app.AsyncApp` instances alive, treats every
  `@app.event` / `@app.message` / `@app.command` / `@app.action` /
  `@app.shortcut` / `@app.view` / `@app.options` / `@app.error` /
  `@app.step` / `@app.function` handler as an entrypoint, and
  follows factory-style construction. Registered under the
  `slack_bolt` CLI plugin key.

## [0.12.1] - 2026-05-24

### Changed
- `ProjectContext.find_comment_patterns` no longer rescans the
  project-wide `global_index` once per file with a matching comment.
  The per-file `(start, decl_idx)` list is now bucketed lazily on the
  first match anywhere in the project (O(N) once instead of
  O(matched_files × N)). The internal `file_decl_sites` helper is
  removed; it had only one caller and was the source of the quadratic
  scan.
- `ProjectContext.find_main_blocks` (powering `MainBlockPlugin`) has
  the same O(matched_files × all_decls) anti-pattern fixed. A cheap
  text + AST prefilter identifies the matched-file set first, then
  one sweep of `global_index` buckets only those files. On a
  synthetic 1000-file project where every file carries a `__main__`
  block, the plugin's cold delta drops from +3.3 ms to noise; at
  2000 files it drops from +13.8 ms to +5.1 ms.
- The `assemble_graph` pass now pre-counts the total node population
  from the salsa-memoized `file_to_nodes` payloads and uses the sum
  to `with_capacity_and_hasher` its five FxHashMaps + the `GraphBuilder`
  Vec-backed fields, eliminating rehash work as the maps grow.
- `build_class_children` resolves Attribute-base modules
  (`module.Cls`) via the `path_to_file` index instead of a linear
  `project_files.iter().find(...)` scan keyed on path-string
  equality.
- Pass 2 (edge translation) in `assemble_graph` now runs
  GIL-free under `Python::allow_threads` + rayon
  `par_iter().flat_map_iter().filter_map()` + `par_sort_unstable() +
  dedup()`. Externals are pre-minted serially first so the parallel
  section is pure FxHashMap probing — no `unsafe`, no salsa
  snapshotting. On a 5000-file synthetic the `pass2` phase drops
  from ~5.6 ms to ~3.4 ms (~40 %) and its stdev compresses ~2.5×.
  Small-N (~200 files) pays a ~150 µs dispatch tax that's
  invisible in wall-clock; serial path retained behind
  `DEAD_CST_PASS2_SERIAL=1` as an A/B knob + kill switch.
- Subclass queries (`find_subclasses`, `find_subclasses_of`, powering
  `UnittestPlugin`, `InitSubclassPlugin`, and `DispatchAppPlugin`'s
  `include_subclasses=True` path) build a project-wide
  parent→children index once at the end of `assemble_graph` and BFS
  the index, instead of calling `ty_ide::find_references` per BFS
  seed. On an 800-file / 3,200-class synthetic project,
  `InitSubclassPlugin`'s cold delta drops from +1129 ms to +4 ms
  (~280×); `UnittestPlugin` drops from +302 ms to +22 ms (~14×).
  Index build cost is ~125 µs at 200 files / ~700 µs at 800 files
  (~0.5 % of cold materialize).
- External-seed subclass queries (`DispatchAppPlugin(typer.Typer)`,
  `DispatchAppPlugin(fastapi.FastAPI)`, …) now do a parallel
  AST-scan over project files for direct subclasses and skip the
  expensive `find_references` walk entirely when no project file
  imports the seed module. On `flux0_server` (no typer imports)
  `DispatchAppPlugin(typer.Typer)` drops from +117.9 ms to
  +26.2 ms; on `dead_cst` self (no fastapi imports)
  `DispatchAppPlugin(fastapi.FastAPI)` drops from +16.7 ms to
  noise. Synthetic 100-file project with every file importing
  `typer`: +605 ms → +540 ms.
- `DispatchAppPlugin`'s framework-subclass discovery is inverted:
  instead of asking ty "what subclasses framework class F" (which
  loads F's module from the venv to run `find_references`), the
  plugin now uses a new `ctx.find_subclasses_via_bases(base_fqns)`
  query that walks project files in parallel, resolves each class's
  base list against local imports, and builds a `base_fqn →
  children` index — never touching the venv. On a 333-file
  synthetic with `flask`: +171 ms → +19 ms; 663-file with
  `typer`: +369 ms → +33 ms (11×); the subclass walk itself
  collapses 154 ms → 0.4 ms (~385×).

### Fixed
- `where_module` / `of_module` now accept `str | list[str]` on every
  chainable that has them (`DecoratorQuery`, `ConstructionQuery`,
  `CallQuery`, `FactoryQuery`, `SubclassQuery`). List semantics is
  OR — match if the row's module is any element. Empty list silently
  matches nothing.
- Syntactic matchers (`query.decorators` / `.constructions` /
  `.calls` / `.factories`) now resolve relative imports. Decorators
  / constructions / calls imported via `from .foo import N` or
  `from ..pkg import N` no longer slip through `where_module(...)
  .where_name(N)` filters.
- `query.constructions.where_name(N)` now matches subscripted
  constructors (`Generic[T]()`, `Logger[Self]("name")`, …). The
  callee classifier strips one level of `Expr::Subscript` before
  matching, mirroring how Python evaluates them.

### Added
- `Plugin.prepare(self, repo_root: Path) -> None` pre-graph hook.
  Called once per plugin per `Analysis.materialize_all` invocation,
  before any graph construction. Default is a no-op; plugins that
  need to scan the repo for config files (`pyproject.toml`,
  framework manifests, etc.) or compute setup state independent of
  the graph should override. Exceptions raised inside propagate
  before the `ProjectContext` is constructed.
- `ctx.query().edges()` chainable query with `.with_flags(mask)`,
  `.with_src_kind(kind)`, `.with_dst_kind(kind)` predicates and
  `.collect()` / `.first()` / `.count()` terminals. Returns
  `EdgeRef` rows with `src` / `dst` `SymbolNode`s already resolved.
  `DynamicImportFallbackPlugin` migrated to use it — on a synthetic
  project with 1,800 dynamic-import edges, the filter loop is 3.06×
  faster (1.235 ms → 0.403 ms best-of-12).
- `ctx.query().decls()` chainable query with `.with_kind` /
  `.with_kinds`, `.with_filename` / `.with_filenames`,
  `.with_simple_name` / `.with_simple_names`, `.with_paths`,
  `.with_path_regex`, `.with_flags` / `.with_any_flag`,
  `.with_fqname_prefix`, and `.where_fqname` predicates. Returns
  `SymbolNode` rows directly. `where_fqname` accepts `str`,
  `list[str]`, `re.Pattern`, `list[re.Pattern]`, or any mixed
  sequence — literal equality and regex search OR together,
  empty list is the matches-nothing sentinel.
  Five plugins migrated off Python-side `ctx.nodes()`
  filter loops with the following warm-best-of-12 wins on the
  flux0 workspace (1,933 nodes):
  * `ServerConfigPlugin`: 36.55 ms → 0.24 ms (~153×, the
    `pathlib.Path(...).name` Python hop was the killer).
  * `DiscordPyPlugin`: 1.32 ms → 0.07 ms (~18.8×).
  * `UnittestPlugin`: 0.87 ms → 0.09 ms (~9.9×).
  * `DispatchAppPlugin` vars-by-file scan: 0.68 ms → 0.27 ms (~2.5×).
  * `PytestPlugin`: 1.00 ms → 0.46 ms (~2.2×).

## [0.12.0] - 2026-05-24

### Changed
- Bumped vendored `ruff` (ty) submodule to `18448938c8` and switched
  the upstream URL to the `lpetre/ruff` fork. Picks up 159 ty commits
  including a search-path cache keyed by top-level module-name
  component, plus a long tail of upstream ty resolution fixes
  (enum/class-decorator handling, TypedDict union fallbacks,
  fall-through narrowing, cycle-recovery panics).

### Fixed
- `Analysis(..., venv=...)` no longer drops `project_root` from ty's
  static search paths when a venv is given. The previous suppression
  relied on `.pth`-derived dynamic paths to cover every first-party
  package, which failed silently for setuptools' default PEP 660
  editable install (`__editable__.<dist>.pth` is a finder stub, not a
  flat path). Every cross-file first-party import resolved to an
  `[unresolved]` synthetic and the whole graph fell apart — see #222
  for the `FastAPIPlugin` symptom. `project_root` is now always on
  the search list; a new specificity-aware reverse module lookup
  (`helpers::canonical_module_name_for_file`) makes a deeper `.pth`
  path still win the fqname for files it covers, so monorepo layouts
  (`packages/lib_a/src/lib_a/`) keep their short `lib_a` fqname
  instead of regressing to `packages.lib_a.src.lib_a`.
- `build_scope_table` (dead-branch detector) no longer livelocks on
  scopes that rebind an inherited name to a flipped version of itself
  — e.g. a function whose body is `global foo; foo = not foo` when
  the enclosing module has `foo = False`. The fixed-point loop would
  oscillate the table forever between `True` and `False`; it now
  poisons such names so they drop out of the table on the first flip.

## [0.11.0] - 2026-05-20

### Added
- `dead-cst build ROOT -o PATH` persists the materialized graph to a
  bincode file; `analyze` / `remove --graph PATH` reuse it. Plugins
  don't round-trip; rebuild to re-run them.
- `--meta key=value` on `build` stashes user-supplied metadata in the
  graph file alongside auto-recorded counts.
- `--query {dead,test-only}` on `analyze` / `remove`. `test-only` runs
  the `NodeFlags.TESTCASE` blast-radius query.
- `--entrypoint-regex REGEX` replaces the old `-e re:<pattern>` magic
  prefix.
- `--exit-zero` on `analyze` always exits 0.
- Public graph persistence API: `dead_cst.graph.write_graph`,
  `read_graph`, `GraphMetadata`, `LoadedGraph`. Hard-versioned header;
  version mismatch fails fast.
- `EdgeFlags.DEAD_BRANCH` edges (from `if False:`, post-`return`,
  etc.) via ty's reachability constraints.
- `EdgeFlags.DYNAMIC_IMPORT` edges for string-literal `__import__()` /
  `importlib.import_module()` calls.
- `.pyi` stub ingestion for compiled-extension layouts.

### Changed (breaking)
- Rust-native graph builder. libcst visitor, SQLite per-file cache,
  and networkx are replaced by a pyo3 extension on ty's
  `SemanticIndex`. libcst is only used by the codemod now.
- Build system: hatchling → maturin. Crate in `src/`, Python in
  `python/dead_cst/`, wheel ships `_native.{abi3.so,pyd}`.
- CLI trimmed to `build` / `analyze` / `remove`. `why-alive`,
  `dependencies`, `unused-exports` removed (still in the Python API).
- `Analysis.materialize_all()` returns a live
  `native.ProjectContext`; the Python `SymbolGraph` facade is gone.
  Iterate `ctx.nodes()` / `ctx.edges()`; walk via `ctx.reachable()` /
  `descendants()` / `ancestors()`.
- `PackageView` removed. Filter `analysis.dead()` by
  `Path(n.path).is_relative_to(pkg)` for per-package slices.
- `codemod.remove_code` / `generate_patch` take an iterable of dead
  `SymbolNode`s (not a graph).
- Plugin protocol: subclass `Plugin`, implement `run(ctx)` yielding
  `AddNode` / `AddEdge` / `AddEntrypoint`. `observe()` / `finalize()`
  and per-file caching are gone.
- `Plugin.name` / `version` fields dropped. Progress label is
  `type(plugin).__qualname__`. The three declarative bases
  (`DecoratedDeclPlugin`, `DispatchAppPlugin`, `LiteralListPlugin`)
  gained a required `marker_prefix` field.
- Framework plugins are now factory functions (`fastapi_plugin()`,
  `flask_plugin()`, `typer_plugin()`, `cyclopts_plugin()`,
  `fastmcp_plugin()`). Plugins with custom `run()` or per-instance
  config remain classes.
- Plugin query API: chainable builder
  (`query(ctx).decorators().where_module(...).collect()`) replaces
  `ctx.find_*` helpers.
- Plugin / resolver registries live in `dead_cst.cli`. Contrib plugin
  re-exports from `dead_cst.plugins` are gone, as is `UvResolver` from
  `dead_cst.resolvers` — import from `dead_cst.contrib`.
- `BUILTIN_PLUGINS` / `BUILTIN_RESOLVERS` / `load_plugin` /
  `load_resolver` removed.
- `-e re:<pattern>` magic prefix dropped (use `--entrypoint-regex`);
  plain `-e` is now path-or-FQN only.

### Fixed
- `ModuleDundersPlugin` pins module-level dunder *functions*
  (`__getattr__` / `__dir__` per PEP 562), not just variables.
- Quoted type annotations (`def f(x: "Helper")`) contribute use edges
  via ty's `enter_string_annotation`.
- `from .submod import X` in `__init__.py` no longer reports the
  submodule attribute as dead; sibling aliases keep it alive.

### Removed
- `tqdm` runtime dep (rust uses `indicatif`).
- `PackageView`, `Analysis.count_nodes`, `Analysis.materialize_closure`,
  `Analysis.package`, `Analysis.views`.
- `dead_cst._notebooks` helper (notebook ingestion is in rust).
- libcst graph builder, SQLite cache, `TruthinessResolver`,
  `networkx` / `rustworkx` deps.

## [0.10.0] - 2026-05-15

### Added
- :class:`dead_cst.plugins.DynamicImportFallbackPlugin` — a plugin
  that reads :attr:`EdgeFlags.DYNAMIC_IMPORT` edges and fans each
  flagged ``src -> module`` edge out to the module's exports. Gives
  projects that prefer the conservative "import-by-name keeps every
  export alive" semantic an opt-in path without baking the fan-out
  into the visitor — pass ``DynamicImportFallbackPlugin()`` to
  :class:`Analysis` or to ``native.ProjectContext.add_plugin`` to
  enable. Six options spanning the three intended rollout stages:

  * **Catch-all shape** (stage 1): ``include_underscore=False``
    (default) skips ``_private`` names matching
    ``from X import *`` runtime semantics, and
    ``respect_dunder_all=True`` (default) uses the target module's
    ``__all__`` list as the export set when present.
  * **Targeted excludes** (stage 2, after focused per-feature
    plugins land): ``exclude_sources`` (path globs matched against
    each call site's source path) and ``exclude_targets`` (fnmatch
    patterns matched against the target module fqname) opt specific
    files / module trees *out* of the catch-all so the focused
    plugin owns them.
  * **Targeted includes** (stage 3, when the exclude list gets
    unwieldy): ``include_sources`` / ``include_targets`` flip the
    semantics — when non-empty, only matching call sites
    participate. Composes with the exclude lists as
    ``include AND NOT exclude``.

  The plugin implements both the libcst-side ``finalize`` and the
  rust-backend ``run(ctx)`` protocols. On the libcst pipeline (which
  inlines fan-out at visit time without setting the
  ``DYNAMIC_IMPORT`` flag) the plugin sees no flagged edges and is a
  no-op. On the rust backend it walks ``ctx.edges()`` and uses the
  new :meth:`ProjectContext.find_module_top_level_decls` /
  :meth:`find_module_dunder_all_exports` queries to enumerate
  exports.

- ``from <pkg> import <name>`` now resolves through ``from <other> import *``
  re-exports. ``build_contribution`` runs a second pass over each
  package after its per-file payloads are applied: for every
  module-level ``from X import *`` in a file, it looks ``X`` up against
  the package's own trie plus each dep's exported trie (deps are built
  first thanks to the dep-ordered ``refresh`` loop) and synthesizes one
  ``"import"``-typed :class:`SymbolNode` in the importing module for
  every name the target exposes. Cross-module ``from <importer> import
  <name>`` resolves through these synthetics like any other re-export
  decl; a downstream ``from <importer> import *`` fan-out picks them up
  for free. Star chains within a package converge via fixed-point
  iteration and cycles terminate after one trip via a
  ``(importer, target, name)`` ``seen`` set. The synthetics carry a
  new :data:`NodeFlags.STAR_REEXPORT` flag so :mod:`dead_cst.codemod`
  skips them (the file has no per-name ``from <target> import <decl>``
  line to remove), and inherit :data:`NodeFlags.EXPORTED` from their
  importing module so the visibility story flows through
  :meth:`SymbolTrie.merge_exported` to downstream packages. The
  pre-existing module-level fan-out in
  :func:`dead_cst._edges.resolve_edges` is preserved alongside
  materialization so non-module-level stars
  (``def a(): __import__('p.functions')``, which the materializer
  skips because synthetics need a module home) still produce
  pessimistic ``<enclosing_decl> -> target.<name>`` keep-alive edges.
- New :func:`dead_cst._package.build_contribution` keyword argument
  ``dep_contributions: Sequence[PackageContribution] = ()``; callers
  must pass already-built dep contributions for cross-package star
  resolution to work. :meth:`Analysis.refresh` walks ``self.packages``
  (dep-first order) rather than the caller's ``packages`` iterable so
  this contract is upheld regardless of how ``refresh`` is invoked.

### Changed
- Rust backend: ``resolve_from_imported`` now matches CPython's
  ``_handle_fromlist`` semantics — namespace lookup first, submodule
  fallback only when nothing's bound. Previously the submodule was
  tried first, which gave the wrong answer for ``from p import q``
  when ``p/__init__.py`` bound ``q`` (e.g. ``q = 42``) *and* a
  ``p/q.py`` file also existed; CPython binds ``q`` to the int and
  the submodule never runs, but the old order linked the consumer to
  the submodule and would have kept dead code in ``p/q.py`` alive.
  No test in the suite exercises that exact shape today, so the
  pytest delta is zero failures either way; the change is for
  correctness on real-world ``__init__.py`` aliasing patterns. Perf
  is a wash — the work shifts between Phase 2 and Phase 3 but Salsa
  caching keeps the total cost identical (≈ ±5 ms on flux0 workspace
  cold, inside measurement noise).

- ``tests/prototype/_bridge.materialize`` (and the inlined copy in
  ``scripts/profile_backends.py``) interns ``pathlib.Path`` objects per
  build and reuses the ``NodeFlags(0)`` / ``EdgeFlags(0)`` singletons
  instead of constructing fresh enum instances per node/edge. Halves
  the warm-path bridge cost (``dead_cst`` self: 15.0 ms → 7.5 ms;
  flux0 server: 2.2 ms → 1.0 ms), which is the dominant cost on warm
  rust runs now that ty's Salsa db keeps ``Project.build()`` at 10 ms.

- Swapped the graph backend from `networkx.MultiDiGraph` to a minimal
  `rustworkx.PyDiGraph` wrapper (`dead_cst._graphstore.SymbolGraph`).
  The wrapper exposes the `SymbolNode <-> int` index bookkeeping
  (`add`, `add_edge`, `index`, `node`, `subgraph`, `nodes`,
  `__contains__`, `__iter__`, `__len__`) plus the raw rustworkx graph
  on `SymbolGraph.raw`. Everything beyond that -- edge-payload-aware
  iteration, `has_edge`, `in_degree`, algorithm calls, and
  `successor_indices` / `predecessor_indices` traversal -- goes
  through `.raw` directly. Plugin extension points
  (`PluginContext.graph`, `Analysis.materialize_*` return types) are
  now `SymbolGraph` instead of `networkx.MultiDiGraph`. `networkx` is
  no longer a runtime dependency; `rustworkx>=0.15` replaces it.

- **Breaking for plugin authors and downstream callers.** The wrapper
  intentionally does not provide `SymbolNode`-yielding traversal
  sugar. Plugins that previously called `ctx.graph.successors(node)`
  / `predecessors(node)` now write
  `ctx.graph.raw.successor_indices(ctx.graph.index(node))` and resolve
  back to a `SymbolNode` via `ctx.graph.node(i)` only when the body
  needs the payload. Index-keyed visited sets (`set[int]`) replace
  `set[SymbolNode]` in BFS-shaped code.

- `_find_reachable` / `_entrypoint_seeds` operate in index space.
  `_entrypoint_seeds(graph)` now returns `list[int]`;
  `_find_reachable(graph, seeds, ...)` takes `Iterable[int]` for
  seeds. The pre-merge `SymbolNode -> int` round-trip on every BFS
  invocation is gone. Tests that built explicit seed lists build them
  as `[graph.index(n) for n in ...]`.

- `PluginContext.importers` removed. The synthetic-prefix walk +
  predecessor filter is inlined in `PackageView.importers_of`, which
  was the only production caller; the public CLI command and tests
  continue to drive it via that method. Out-of-tree plugins that
  called `ctx.importers(...)` need to switch to
  `analysis.package(path).importers_of(...)` or use
  `ctx.find_module(...)` / `require_resolved_dep(ctx, package)` for
  the in-plugin equivalent.

- Edge deduplication is centralized in the compose pass. One
  `emitted: set[(src, dst, EdgeFlags)]` owned by `_materialize` is
  shared across the three edge sources (contribution edges,
  `resolve_edges` import resolution, plugin `AddEdge` ops), so
  cross-source and cross-package duplicates collapse to one edge
  instead of accumulating as parallel `MultiDiGraph` edges.
  `resolve_edges` and `apply_ops` both take `emitted` as a required
  argument (breaking).

- Dead-suite positions moved off the materialized graph onto the
  analysis itself: `Analysis.dead_suites()` returns the merged
  `{file: tuple[CodeRange, ...]}` mapping across every package's
  contribution, and `PackageView.dead_suites()` returns the per-package
  slice. The previous `graph.graph["dead_suites"]` attribute is gone.
  Reads off `_contributions` instead of duplicating onto the graph; the
  CLI's `dead-cst report` is the only first-party consumer.

- New `NodeFlags.EXPORTED` tags every node from a file under
  `Package.exported`, set via the visitor's `default_flags` mechanism
  (same pattern as `NOTEBOOK`). `Package.exported` now participates in
  the per-package cache fingerprint via the new `package=` argument on
  `compute_fingerprint`; editing exported subdirs invalidates that
  package's cache (siblings are unaffected). `Package.path` / `name` /
  `deps`, the resolver, and `search_paths` remain outside the
  fingerprint. Schema bumped to 4; existing caches rebuild on first run.

- `PackageContribution` is now a raw record (`frozenset[SymbolNode]` /
  `frozenset[(src, dst, EdgeFlags)]` / `Mapping[Path, tuple[CodeRange, ...]]`
  plus the trie and import-edges); no more `nx.MultiDiGraph` wrapper.
  The two-trie design (`current_trie` + `export_trie`) collapses into
  one `trie`; consumer-side merges call the new
  `SymbolTrie.merge_exported`, which filters by `NodeFlags.EXPORTED`
  while walking through unexported intermediates so exported
  descendants stay reachable.

- The file-vs-package precedence case (`foo.py` next to
  `foo/__init__.py`) is now called **eclipsed** to disambiguate from
  `NodeFlags.SHADOWED` (intra-file decl rebinding, unchanged). The
  helper is `eclipsed_paths`; the warning text says "eclipsed by
  sibling package".

### Removed (breaking, plugin API)
- `PluginContext.package` and `PluginContext.package_nodes` are folded
  into a single `PluginContext.contribution: PackageContribution`
  field. Read `ctx.contribution.package` / `ctx.contribution.nodes`
  instead. `PackageContribution` is now re-exported from
  `dead_cst.plugins` for plugin authors. `contribution` also exposes
  the package-local `trie`, raw `edges`, `dead_suites`, and
  `import_edges` — fields plugins previously couldn't reach.
- `PluginContext.package_modules()` and `PackageView.modules()` are
  removed. Neither had any in-tree consumer (only tests and a
  benchmark referenced them); callers that want module nodes can
  filter `ctx.contribution.nodes` / `view.declarations()` by `n.type == "module"`.
- `PluginContext.package_graph` and `PluginContext.module_nodes` are
  gone. `contribution.nodes` is a `frozenset[SymbolNode]` (was a
  method with internal caching on the old `package_nodes`); call sites
  change from `ctx.package_nodes()` to `ctx.contribution.nodes`.
  `package_modules()` derives modules by filter.
- `AddNode` drops its `entrypoint: bool` / `testcase: bool` fields.
  Plugins that need an entrypoint synthetic stamp the flag at
  construction: `synthetic_node(..., flags=NodeFlags.ENTRYPOINT)`.
- `NodeFlags.ENTRYPOINT` / `TESTCASE` are the only source of truth
  for reachability seeds. The `graph.nodes[n]["entrypoint"]` /
  `"testcase"` attr-dict mirror is no longer set or read anywhere;
  `_find_reachable` reads flags directly off `SymbolNode`.
- `SymbolTrie.add_module_hierarchy_edges(graph)` (mutator) replaced
  by `SymbolTrie.module_hierarchy_edges()` (iterator yielding
  `(child_module, parent_module)` pairs).

### Refactored
- The per-package apply layer (`PackageContribution`, `build_contribution`,
  `_apply_payload`, `eclipsed_paths`) moved from `_refresh.py` into a new
  `_package.py` module. `_refresh.py` now hosts the per-file pipeline
  exclusively (enumerate, parse, observe, cache).
- `Analysis._materialize` renamed `scope` -> `included` and dropped
  the `None`-means-everything case. `Analysis._build_symbol_lookup`
  lost its `scope` parameter — `_interesting_set` is closed under
  transitive deps, so the filter could never trigger.

### Removed
- The overlay / what-if API on `Analysis` is gone (breaking):
  `Analysis.preview_payloads`, `Analysis.materialize_with`,
  `Analysis.preview`, and the `GraphView` class (with its
  `dead_cst.analyze` re-export) have been deleted. The design
  didn't pay its keep — callers comparing baseline vs. perturbed
  reachability can construct a second `Analysis` with a substitute
  detector or modified sources and diff `dead()` directly.
  `TruthinessResolver.resolve_constant` and the
  `DefaultUnreachableRegionDetector.resolve(expr, resolver)` hook
  stay — they're independently useful for custom detectors.

### Changed
- `DispatchAppPlugin` (in `dead_cst.plugins.decl_shapes`) is now the
  shared base for `FlaskPlugin`, `FastAPIPlugin`, and `CeleryPlugin` in
  addition to `TyperPlugin` / `CycloptsPlugin`. The base learned an
  opt-in factory-aware mode driven by a new
  `instance_kinds: Mapping[str, bool]` field: when set, the plugin
  emits `<{name}-app>:` / `<{name}-pending>:` / `<{name}-factory>:`
  synthetics and runs a per-package finalize pass that walks pending
  variables forward to a discriminating import node or factory marker
  before promoting the matching kinds to entrypoints. When
  `instance_kinds` is empty (Typer / Cyclopts) the plugin behaves as
  before: pure observe, no entrypoint promotion. Flask / FastAPI /
  Celery now consist almost entirely of their handler / kind config,
  with Celery's `@shared_task` channel as its only override. No
  user-visible behavior change.

### Added
- `FastMCPPlugin` ships in `dead_cst.contrib.fastmcp` and is registered
  under the `fastmcp` builtin name. Marks top-level
  `X = FastMCP(...)` server instances as entrypoints (the `fastmcp` CLI
  loads `module:mcp` by import path the same way `uvicorn` loads a
  FastAPI `module:app`, so every FastMCP server is framework-visible
  the moment it's constructed) and wires `@mcp.tool` / `@mcp.resource`
  / `@mcp.prompt` / `@mcp.completion` decorators on top-level functions
  through the owning server. Supports the
  `def create_server() -> FastMCP: ...` factory shape across packages
  via the shared `DispatchAppPlugin` factory-marker mechanism. Only the
  `fastmcp` import path is recognized; the Anthropic MCP SDK's
  compatibility re-export (`mcp.server.fastmcp.FastMCP`) is not
  detected -- users on that path can keep their handlers alive with
  explicit `-e` entrypoints or a project-local plugin.
- Jupyter notebook (`.ipynb`) support. Every `.ipynb` file under a
  package root is ingested by concatenating its code cells into a
  single libcst-parseable module; IPython line magics (`%foo`),
  cell magics (`%%bash`), shell escapes (`!ls`), and trailing-help
  forms (`obj?`, `obj.attr??`) are neutralized to `pass  # <line>`
  so libcst accepts the source. Notebooks aren't importable, so their
  decls are deliberately kept out of the cross-module lookup trie.
- `NodeFlags.NOTEBOOK` stamps every `SymbolNode` sourced from a
  notebook. `SymbolVisitor` now takes a `default_flags` kwarg that
  `_refresh._process_one_file` sets to `NOTEBOOK | ENTRYPOINT` for
  notebooks, so a notebook's contents are reachability seeds and any
  `.py` code the notebook imports stays alive. Malformed notebooks
  fall through to the same `[unparseable] <module>` placeholder used
  for `.py` files that fail to parse.
- The codemod (`generate_patch` / `remove_code`) skips
  `NodeFlags.NOTEBOOK` nodes; cell-aware writeback into the notebook
  JSON envelope is out of scope.
- `CeleryPlugin` ships in `dead_cst.contrib.celery` and is registered
  under the `celery` builtin name. Marks top-level `X = Celery(...)`
  app instances as entrypoints (the Celery worker process loads
  `module:app` by import path via `celery -A`, mirroring how
  `FastAPIPlugin` / `FlaskPlugin` treat their app instances), wires
  `@app.task` / `@app.task(...)` decorators on top-level functions
  through the owning app (so a task callable lives as long as the app
  does), supports the `def make_celery(): return Celery(...)` factory
  shape across packages via a factory marker, and seeds module-level
  `@shared_task` / `@shared_task(...)` decorated functions (with
  `shared_task` imported from `celery`) as entrypoints directly --
  `shared_task` registers into Celery's global registry and is invoked
  by name with no owning app variable to wire through.

## [0.9.4] - 2026-05-12

### Changed
- `PluginContext` now requires `package_graph` (the per-package
  contribution graph) and `module_nodes` (its module-typed entries).
  `package_nodes` and `package_modules` source from these directly,
  dropping the `Path.is_relative_to` filter and `type == "module"`
  scan they used to run on the merged cross-package `graph`. The
  analyzer wires both through automatically; custom callers that
  construct `PluginContext` directly must pass them.
- `PackageView.modules()` now reads `PackageContribution.module_nodes`
  instead of refiltering the contribution graph.

## [0.9.3] - 2026-05-12

### Added
- `ServerConfigPlugin` ships in `dead_cst.contrib.server_config` and
  is registered under the `server_config` builtin name. Matches
  Gunicorn / Hypercorn config files by basename (`gunicorn.conf.py`,
  `gunicorn_conf.py`, `hypercorn.conf.py`, `hypercorn_conf.py` by
  default; override the `filenames` tuple for non-standard layouts)
  and marks the module plus every top-level decl (functions for hook
  callbacks like `on_starting` / `post_fork` / `when_ready`,
  variables for settings like `bind` / `workers`, classes for inline
  custom logger / worker definitions, imports for helpers used only
  to build config values) as an entrypoint. These files are loaded
  by the server process at startup (Docker, Cloud Run, systemd) and
  not imported anywhere in the project, so without this plugin their
  whole surface looks dead.
- `find_factory_decls(module, imports, valid_targets)` is exported from
  `dead_cst.plugins`. Third-party framework plugins that follow the
  same instance-construction shape can use it together with
  `walk_to_instance_kind(..., factory_marker_prefix=...)` to get
  cross-package factory support for free.

### Fixed
- `FastAPIPlugin` and `FlaskPlugin` now classify the factory pattern
  across packages when the factory uses `import fastapi; fastapi.FastAPI()`
  / `import flask; flask.Flask()` (module-prefixed form). The
  external-edge classifier drops the `decl=` half of the access, so the
  downstream walk had no discriminator to tell `FastAPI` from
  `APIRouter` (or `Flask` from `Blueprint`). `observe` now tags every
  top-level decl whose body constructs one of those classes with a
  `<{fastapi,flask}-factory>:<kind>:<owner>` synthetic, and
  `walk_to_instance_kind` accepts a `factory_marker_prefix=` kwarg so
  the per-package finalize walk picks the marker up regardless of which
  file the factory lives in. The named-import shape
  (`from fastapi import FastAPI`) was already covered by the
  import-node discriminator and is unaffected.
- Attribute access on a runtime module dunder (`some_pkg.__file__`,
  `some_pkg.__name__`, `some_pkg.__spec__`, etc.) no longer surfaces a
  "Failed to resolve import edge" warning. The import machinery injects
  these attributes on every module object at runtime, so the chain past
  them is a path / string op, not a symbol reference. The visitor now
  truncates the access chain at the dunder and emits a clean
  `Import(module=X, decl=None)` instead of a speculative
  `Import(module=X, decl="__file__")`. Reachability is unchanged -- the
  module-level edge that previously rode alongside the failed lookup is
  the same edge that's now emitted directly. Recognised dunders:
  `__file__`, `__name__`, `__doc__`, `__loader__`, `__spec__`,
  `__package__`, `__path__`, `__builtins__`, `__cached__`. Visitor
  `version` is bumped so cached payloads rebuild.

## [0.9.2] - 2026-05-12

### Added
- `DiscordPyPlugin` ships in `dead_cst.contrib.discordpy` and is
  registered under the `discordpy` builtin name. The plugin recognizes
  top-level `commands.Bot` / `discord.Client` (and the `AutoSharded*`
  variants) constructions and seeds them as entrypoints, wires
  `@bot.command()` / `@bot.event` / `@bot.listen()` / `@bot.tree.command()`
  / `@bot.tree.context_menu()` decorators (and their group / hybrid /
  invoke-hook siblings) to their bot variable, marks any module that
  defines a `commands.Cog` subclass as alive together with its module-
  level `setup` / `teardown` hooks, and resolves
  `<expr>.load_extension("dotted.path")` / `load_extensions([...])`
  string-literal targets onto the captured module's surface
  (matching `importlib.import_module` semantics).

### Changed
- `distribution_lookup` and `editable_distribution_roots` are now keyed
  on the dist-bearing slice of `sys.path` (site-packages /
  dist-packages / purelib / platlib entries) instead of an empty
  tuple, so they survive the analyzer's per-package `sys.path` rebind
  for free -- only the first-party prefix moves during a transition,
  and that prefix never enters the key. `Analysis._materialize` now
  uses the narrower `clear_module_specs_cache()` (also newly exported
  from `dead_cst.resolvers`) on every package transition instead of
  the full `clear_path_caches()`, dropping a ~10s/package
  `importlib.metadata` walk that dominated large-workspace runs. A
  real venv change (uv splicing in a workspace `.venv`) still flips
  the key automatically and triggers a single rebuild. Pure
  performance change.
- `_count_nodes_by_prefix` batches the per-package node-counting that
  the CLI text/JSON report does. Previously
  `_output_text` / `_output_json` called `_count_nodes(graph, prefix)`
  twice per package, each walking the entire graph; on large
  workspaces this dominated report formatting. The bucketed helper
  walks the graph once and aggregates by prefix in a second pass keyed
  on unique file paths.
- `is_from_module` (exported from `dead_cst.plugins`) now recognizes
  dotted module names — `is_from_module(node, "discord.ext.commands")`
  matches `from discord.ext.commands import ...`. Previously only
  single-segment module names worked, because the helper bottomed out
  in `is_name` (bare `cst.Name` only). Backward-compatible: every
  existing single-segment caller still matches. `collect_module_imports`
  inherits the change, so plugins can now scan dotted source modules
  without rolling their own import-walker.
- `SymbolVisitor` now hoists the `_descendant_ids` cache used by
  `live_referents` / `live_at_exit` onto the visitor instance, so a
  single shared cache covers every flow-analysis call the visitor
  makes for a file. Previously each multi-referent access in
  `on_leave` triggered a fresh cache allocation, so large files with
  many reassignments re-walked the same statement subtrees from
  scratch on every access. Pure performance change — output and
  payload-cache fingerprint are unchanged.

### Fixed
- A `foo.py` sibling of a `foo/__init__.py` package no longer asserts
  out of `SymbolTrie.add_declaration`. The new
  `dead_cst._refresh.shadowed_paths` pre-pass mirrors CPython's
  `FileFinder` precedence (regular package wins over a same-named
  module file), so the trie holds the package and cross-module imports
  of `pkg.foo` route there. The shadowed `.py` is still parsed and its
  nodes still appear in the package graph -- observe-time entrypoints
  (`__main__` blocks, plugin synthetics) keep working -- but consumer
  imports never see its decls. A WARNING is logged per shadowed file
  so the layout (almost always a bug) surfaces during analysis.

## [0.9.1] - 2026-05-11

### Fixed
- `resolve_edges` no longer spins forever on cyclic re-exports. The
  worklist DFS now carries a per-walk `visited` set keyed on
  `(id(SymbolTrie), parts_tail)`, so a pathological pair like
  `A.x: from B import x` / `B.x: from A import x` terminates after one
  trip around the cycle instead of repeatedly chaining back to its
  starting state. The decls actually encountered along the cycle are
  still emitted, so first-party reachability through the chain is
  preserved.

### Changed
- `resolve_edges` now memoizes the full per-import resolution at three
  layers: ``_resolve_targets`` keyed by ``Import`` value (so equal
  spellings across files share the precomputed dst list — the visitor
  builds fresh ``Import`` objects per file, but they hash equal because
  ``Import`` is frozen with an eager ``__hash__``); ``_walk`` keyed by
  ``(start_node, decl_parts)`` (so different ``Import`` shapes that
  canonicalize to the same trie state share the re-export DFS); and
  ``_classify`` keyed by ``(import.module, import.speculative)`` (so
  the resolver runs once per unique external name). The per-src loop
  collapses to ``for dst in cached_targets: emit(...)``. On large
  multi-package workspaces where one big package's edge contribution
  dominates composition (the case `compute_fingerprint` / hash
  precomputation helped on but did not change the algorithmic shape
  of), this turns the per-package compose loop's growth in importer
  count from multiplicative to additive.

## [0.9.0] - 2026-05-11

### Changed
- The parallel refresh pool (``--workers >= 2``) now drains worker
  results via ``concurrent.futures.as_completed`` instead of
  ``pool.map``. Cache writes and progress ticks land in completion
  order, so a single slow file no longer blocks the cache from warming
  with the fast files behind it. Tasks are still submitted in
  ``(package_path, file)`` order so same-package work stays contiguous.
- Refresh now collects per-task failures and raises a single
  ``ExceptionGroup`` after every other task has finished, instead of
  aborting on the first error. Successfully-parsed files are still
  cache-warmed before the group is raised, so a re-run after fixing
  the bad file only re-parses what failed.
- The pool branch installs SIGTERM / SIGINT handlers for the lifetime
  of the run; on signal it cancels every still-pending future, calls
  ``pool.shutdown(wait=False, cancel_futures=True)``, restores the
  prior handlers, and raises ``KeyboardInterrupt``. Files that
  completed before the signal stay cache-warmed.
- `SymbolNode` and `Import` now pre-compute their hash in
  `__post_init__` and store it in a private `_hash` slot; `__hash__`
  becomes a single attribute read. The instances are frozen so the
  result is stable. Cuts edge-stitching time on large multi-package
  workspaces where `resolve_edges._emit` re-hashes the same
  `(src, dst, flags)` tuples into its dedup set, and pays off again
  every time a `SymbolNode` is hashed by networkx (graph insertion,
  BFS traversal). `SCHEMA_VERSION` is bumped to 3 so cache rows
  pickled before the slot existed are invalidated on first use.
- Progress reporting is fully logger-driven and controlled by the root
  logger level. Per-file refresh status ``[i/N] ok|FAILED <file>`` goes
  through ``logger.debug`` on ``dead_cst._refresh``; off-TTY decile
  checkpoints go through ``logger.info`` on ``dead_cst._progress``.
  The on-TTY tqdm bar is preserved and wraps its iteration in
  ``logging_redirect_tqdm`` so concurrent log records print above the
  bar without shattering it. ``dead-cst -v`` keeps its meaning (the
  CLI's ``setup_logging`` flips the root level to ``DEBUG``); library
  users get the same firehose by configuring their root logger.

### Fixed
- Stdlib imports no longer emit a spurious ``Failed to resolve import
  module: <name>`` warning during edge stitching. The orphaned warning
  in ``_edges._emit_external`` fired for every successfully-classified
  stdlib import (``import datetime`` / ``from pathlib import Path`` / …)
  because the silent-drop and speculative-miss branches both returned
  ``None``; the warning is removed since no legitimate path reached it
  -- truly-unresolved non-speculative imports already surface as
  ``[unresolved] <top-level>`` synthetic nodes.
- ``default_resolve_import`` now falls back to the parent module when a
  dotted name can't be resolved directly, so ``collections.abc``,
  ``importlib.resources.abc``, and similar synthesized-in-``__init__``
  submodules classify as ``[stdlib] <name>`` instead of being misfiled
  as ``[unresolved] <top>``.

## [0.8.0] - 2026-05-09

### Added
- The visitor now honors ruff/pyflakes ``# noqa`` directives that
  silence F401 (unused-import). A per-line ``# noqa``,
  ``# noqa: F401``, multi-rule ``# noqa: E501, F401``, or case-variant
  ``# NOQA`` on the same source line as an import alias pins the
  resulting import node alive (``NodeFlags.ENTRYPOINT |
  NodeFlags.NOQA``) so it is no longer reported as dead. File-level
  directives -- ``# ruff: noqa``, ``# ruff: noqa: F401``, and the
  ``# flake8: noqa`` aliases -- pin every import in the file. The
  ``ruff:`` / ``flake8:`` prefix is matched case-sensitively per
  ruff's documented behavior; the ``noqa`` keyword is case-insensitive.
  Per-alias directives inside a parenthesized ``from x import (a, b)``
  pin only the alias on that line. This brings dead-cst's unused-import
  semantics in line with ruff: an import you have explicitly marked as
  intentionally preserved (re-exports, side-effect imports,
  ``TYPE_CHECKING`` shims guarded by F401) is no longer surfaced or
  removed.
- New ``NodeFlags.NOQA`` flag, layered on ``NodeFlags.ENTRYPOINT``
  (parallel to ``NodeFlags.TESTCASE``). Read ``n.flags & NodeFlags.NOQA``
  off the ``SymbolNode`` directly; there is no graph attr-dict mirror.
- `dead_cst.codemod.generate_patch(G, root)` returns the same removal
  as `remove_code` as a `git apply`-compatible unified diff (with
  `diff --git` headers and `deleted file mode 100644` for module
  deletions) instead of writing in place. Selection is driven entirely
  by `G.nodes`, so callers can slice the unreachable graph (e.g.
  `G.subgraph(scc)` for one SCC at a time) to review a big codebase as
  a series of focused patches.
- `TruthinessResolver.resolve_constant(expr) -> Const | None` returns
  the underlying literal *value* (`str` / `int` / `bool` / `None`,
  wrapped in the new `Const` dataclass) over the same flow-sensitive
  Name walk `evaluate` uses for truthiness. Intended for custom
  detectors that pattern-match against a flag *name* -- e.g.
  `check_flag(FEATURE_A)` where `FEATURE_A = "feature_a"` is bound at
  module level. `Const(None)` (the `None` literal was proved) stays
  distinct from a bare `None` return ("unknown"). Re-exported from
  `dead_cst.branches`.
- `Analysis.preview_payloads(files, *, detector=None) -> dict[Path, VisitorPayload]`
  regenerates per-file payloads for a hand-picked file set, bypassing
  the on-disk cache (no read, no write) and accepting a one-shot
  `UnreachableRegionDetector` override. Pairs with the new
  `Analysis.materialize_with(payloads)` to splice the substitute
  payloads into a fresh graph without mutating any baseline state.
  `Analysis.preview(files, *, detector=None)` chains the two and
  returns a new `GraphView` (also new) wrapping the overlay graph
  with the same reachability surface as `Analysis` (`reachable`,
  `dead`, `kept_alive_by_dead_branches`, `kept_alive_by_flags_only`,
  `count_nodes`). The combined API enables what-if graph surgery:
  bake a flag's truthiness into a small set of files and ask "what
  becomes dead if we land this rollout?" without forking the
  analysis-wide fingerprint or polluting the cache.

### Changed
- **Breaking:** The two ``kept_alive_by_*_only`` methods on
  ``Analysis`` and ``PackageView`` have been collapsed into a single
  ``kept_alive_by_flags_only(flags: NodeFlags)``. Pass
  ``NodeFlags.TESTCASE`` for the old ``kept_alive_by_tests_only``
  behavior ("blast radius of dropping the test suite"),
  ``NodeFlags.NOQA`` for "blast radius of removing every F401 pin",
  or both ORed together. ``dead_cst.analyze._find_reachable(graph,
  exclude_flags=NodeFlags.NONE)`` is the matching private helper
  shape; ``_find_reachable_excluding_tests`` and
  ``_find_kept_alive_by_tests_only`` are gone.
- **Breaking:** `dead-cst remove` no longer modifies files in place. It
  now emits a unified diff to stdout (or to `--output PATH`) and exits;
  apply with `dead-cst remove . | git apply` (or `git apply <path>` if
  you used `--output`). The `--dry-run` flag and the
  `Proceed with removal? [y/N]` confirmation prompt are gone -- the
  command is non-destructive by construction. The "Building symbol
  graph" banner, the "No dead code found." message, and the apply hint
  go to stderr so the patch on stdout stays clean.
- **Breaking:** `DefaultUnreachableRegionDetector.resolve(expr)` is now
  `resolve(expr, resolver)`. The active `TruthinessResolver` is the
  second argument, so overrides can call
  `resolver.resolve_constant(...)` to fold a flag-name `Name` to its
  literal value before pattern-matching. Subclasses with a one-arg
  `def resolve(self, expr): ...` need to add the `resolver` parameter;
  ignore it if you don't use it.

### Fixed
- `DefaultUnreachableRegionDetector` now recognizes compound statements
  as terminators when every reachable branch itself terminates. An
  ``if`` whose taken branch always ``return``s (e.g.
  ``if True: return``, ``if FLAG: return`` with ``FLAG = True``, or
  ``if cond: return; else: return``) kills statements that follow it
  in the enclosing suite, so constant-folded early returns now flag
  trailing dead code as expected. Same handling applies to ``with``
  whose body terminates, and to ``try``/``except``/``finally`` where
  every path terminates (or a ``finally`` itself terminates).
- `TruthinessResolver` no longer folds a ``Name`` whose binding's RHS
  is a mutable container literal (``[]``, ``{}``, ``set()``-shaped
  comprehensions, etc.). The binding-only flow walk is invisible to
  ``.append`` / item assignment / ``.update`` mutations, so an
  ``edges = []; edges.append(x); if not edges:`` chain used to fold
  ``edges`` to ``False`` and incorrectly mark the trailing code dead.
  Tuples and primitive literals are immutable and stay safe to fold.
  Bumps the detector's `version` so cached payloads rebuild.

## [0.7.0] - 2026-05-09

### Added
- New `[unparseable] <module>` synthetic node, exposed via the
  `UNPARSEABLE_PREFIX` constant re-exported from `dead_cst.plugins`.
  When `libcst` cannot parse a file, the analyser now logs a warning
  and emits a minimal payload pairing the real module node with one of
  these synthetics flagged `ENTRYPOINT`, instead of aborting the whole
  run with `ParserSyntaxError`. The file stays alive in reachability
  and importers can still target the module; decls inside the file
  remain invisible until parsing succeeds. The placeholder rides the
  per-file cache, so a fresh source SHA (i.e. the user fixing the
  syntax) invalidates the entry automatically.

### Changed
- **Breaking:** `PathResolver` no longer satisfies the `Cacheable`
  protocol -- the `name` / `version` attributes have been removed from
  the protocol and from the shipped `ManualResolver` and `UvResolver`.
  Resolver output flows through the (uncached) edge-stitching pass, so
  swapping a resolver re-stitches edges without invalidating any
  cached `VisitorPayload` blob; there was nothing for the
  `(name, version)` pair to gate. Out-of-tree resolvers should drop
  their `name` and `version` fields. `BUILTIN_RESOLVERS` now keys the
  builtin entry by the literal string `"uv"` rather than reading
  `UvResolver.name`.
- **Breaking:** `dead_cst.branches.fold_constants` is gone, replaced
  by `dead_cst.branches.TruthinessResolver`, a goal-directed
  flow-sensitive truthiness object. From-scratch
  `UnreachableRegionDetector` implementations should construct one
  resolver per file and pass `resolver.evaluate` as the `resolve_expr`
  argument to `unreachable_suites` / `evaluate_truthiness`.
  `DefaultUnreachableRegionDetector`'s `resolve(self, expr)` subclass
  hook is unchanged. The internal `_const_fold` module is gone.
- `DefaultUnreachableRegionDetector.find_regions` is now a two-pass
  design: one CST visit collects every `If` / `While` / suite-bearing
  site, and the resolver answers truthiness queries on demand. Files
  with no conditional tests now skip `ScopeProvider` /
  `ParentNodeProvider` resolution entirely. Self-analysis benchmark
  drops `find_regions` from 24.2 s to 1.8 s over `dead_cst/` (13×).
- Bumped the `libcst` floor to `>=1.8.6` so PEP 750 template strings
  (`t"..."`, Python 3.14+) parse rather than aborting with
  `ParserSyntaxError`. Names referenced inside t-string interpolations
  and format specs flow through the visitor's existing scope
  resolution -- a `t"hello {NAME}"` inside a function now produces
  the same `func -> NAME` edge an f-string would, and the
  documented limitation has been removed.

### Fixed
- `enumerate_files` now skips directories whose name happens to end in
  `.py` / `.pyi`. `Path.rglob` matches by name only, so an oddly-named
  directory (e.g. `something.py/`) used to slip through and crash the
  visitor on `read_text`.
- Fixed an `AttributeError` crash in the visitor when a lambda body
  contained an access with multiple potential referents (e.g. a
  walrus that rebinds the lambda's own parameter). The flow-analysis
  helper now skips lambdas, whose body is a single expression rather
  than an `IndentedBlock`.

## [0.6.0] - 2026-05-08

### Changed
- **Breaking:** the analyzer's "base" terminology has been renamed to
  "package" everywhere it referred to a `Package` (the unit of
  workspace membership). On `Analysis`, `bases` is now `packages`
  (returning `tuple[Package, ...]` in BFS order; the previous
  `packages` resolver-order tuple is gone), `refresh(bases=)` is now
  `refresh(packages=)`, `reverse_closure(base)` /
  `materialize_closure(base)` rename their parameter to `package`,
  and `Analysis.package(base)` takes `path`. `PackageView.base` is
  now `PackageView.package` (a `Package`) with a `.path` convenience
  property. On `PluginContext` and `ObserveContext`, the `base: Path`
  field is now `package: Package` (use `ctx.package.path` for the
  directory); `PluginContext.base_modules()` /
  `PluginContext.base_nodes()` are now `package_modules()` /
  `package_nodes()`. `remove_code(G, base)` is now
  `remove_code(G, package_path)`, and
  `dead_cst.resolvers.exported_roots(base)` is now
  `exported_roots(package_path)`. Out-of-tree plugins and resolver
  consumers must update accordingly.
- The CLI's `-p` / `--path` spec is now described as
  `'package:dep1,dep2' or 'package'` (formerly
  `'base:dep1,dep2' or 'base'`). The parsing rules are unchanged --
  same syntax, clearer name -- so existing scripts keep working.
- Per-file refresh logic moved from `dead_cst/analyze.py` into a new
  `dead_cst/_refresh.py` (file enumeration, stale detection, the
  worker pool, payload application, and per-package contribution
  build). `analyze.py` keeps cross-package composition, reachability,
  and the public `Analysis` / `PackageView` classes. Tests that
  monkey-patched `analyze.SymbolVisitor` /
  `analyze.ProcessPoolExecutor` should now patch the same names on
  `dead_cst._refresh`.

### Added
- `.pyi` stub files are now ingested by the analyzer and rewritten by
  the codemod, targeting the **compiled-extension layout**: a binary
  module (e.g. `mypkg/_native.so`) shipping next to its hand-written
  type stub (`mypkg/_native.pyi`) with no `.py` twin. The stub is
  parsed through the same visitor as a regular module under its
  natural FQN (`mypkg._native`), so `from mypkg._native import
  compute` resolves to the stub's `compute` decl through the normal
  cross-module import path, and reachability + the codemod work the
  same as for any first-party module. A `.pyi` shipped alongside a
  real `.py` is dropped during file enumeration -- the runtime
  module is the canonical declaration of those names, and ingesting
  the stub on top would collide with it in the symbol trie. Peer-mode
  stubs are therefore invisible to dead-cst.
- New `NodeFlags.OVERLOAD` flag plus visitor support for
  `typing.overload`. `@overload`-decorated functions (recognized
  syntactically -- bare `overload`, `typing.overload`, and
  `typing_extensions.overload`) are flagged and excluded from the
  cross-module lookup trie just like `SHADOWED` decls, so `from mod
  import f` continues to resolve to the impl rather than a typing
  stub. The visitor also wires `impl -> overload` edges so an
  overload's lifetime is anchored to its same-name impl: the codemod
  removes the overloads alongside the impl when the impl is dead, and
  preserves them as long as the impl is alive.
- New `NodeFlags.TESTCASE` flag tags entrypoints created by test
  plugins (pytest / unittest discovery, fixture seeds). The default
  `Analysis.reachable()` traversal still treats those seeds as
  ordinary entrypoints. The opt-in `Analysis.kept_alive_by_tests_only()`
  / `PackageView.kept_alive_by_tests_only()` returns the "blast
  radius" of dropping the test suite -- production code currently
  kept alive only because tests still touch it. `PytestPlugin` and
  `UnittestPlugin` stamp `ENTRYPOINT | TESTCASE` on every synthetic
  seed they create; their `version` epochs were bumped accordingly.
  `AddNode` plugin ops grew an analogous `testcase: bool = False`
  field for plugins that emit entrypoints from `finalize`.
- Progress reporting around the per-file visitor pass ("Parsing
  files") and the cross-package composition pass ("Reconciling
  packages")
  in `Analysis.refresh` / `Analysis._materialize`. On a TTY the user
  sees a live `tqdm` bar; off a TTY (pytest capture, pipes, agent
  harnesses) the same wrapper emits one newline-terminated checkpoint
  at 0%, every ~10%, and at 100%, so CI logs and LLM tool consumers
  can track long runs without `tqdm`'s `\r`-overwriting frames going
  to mush. `tqdm>=4.66` is now a hard runtime dependency.
- New plugin helpers re-exported from `dead_cst.plugins`: `module_node`,
  `dotted_parts`, `dotted_name`, `string_value`,
  `payload_imports_module`. They consolidate boilerplate every contrib
  plugin used to inline (lookup the per-file module node, walk
  attribute chains, evaluate string literals, scan payload imports).
- New `DispatchAppPlugin` base in `dead_cst.plugins.decl_shapes` for
  CLI-style dispatch apps (`X = App(); @X.command(...)`). `TyperPlugin`
  and `CycloptsPlugin` are now thin subclasses configuring the module,
  constructor, and registration-decorator names.

### Fixed
- `UnittestPlugin` now resolves transitive `TestCase` subclasses.
  Refactored into a two-phase plugin: `observe` emits
  `<unittest:base-of>:<base_fqname>` bucket markers for every class
  def, and `finalize` walks the bucket chain from `unittest.TestCase`
  / `IsolatedAsyncioTestCase` (plus every import alias of those) to
  find the transitive subclass closure. This closes two real-world
  gaps the previous single-pass implementation missed: test classes
  that inherit from a project-local `TestCase` mixin (`class
  ProjectTC(unittest.TestCase); class MyTest(ProjectTC)`), and test
  classes that inherit from a re-exported `TestCase` (`from
  pkg.bases import TestCase` where `pkg.bases` does `from unittest
  import TestCase`). Alias expansion uses each import node's raw
  `Import` metadata rather than graph successors, so it works
  across the stdlib boundary that `unittest.TestCase` sits on. The
  plugin's `version` was bumped, invalidating cached payloads.
- `dead-cst unused-exports` no longer matches variables whose names
  merely end with the literal `__all__` (e.g. `pkg.foo__all__`); only
  variables actually named `__all__` are considered.

### Changed
- **Breaking:** `compute_fingerprint` no longer takes a `base: Path`
  argument. The visitor's output is purely a function of the file's
  source plus the plugin / detector chain, so the analysis fingerprint
  is now a single value shared across every base. Callers should drop
  the `base=` keyword from any `compute_fingerprint(...)` invocation;
  per-file cache rows continue to gate on the analysis-wide fingerprint
  the same way they previously gated on the per-base one.
- File parsing is now flow-based rather than partitioned per base.
  `Analysis.refresh` walks each requested base's tree, collapses every
  base's cache misses into one global stale-file list, and runs the
  visitor + observe pass once across the whole batch. Multi-base
  refreshes that previously paid for one worker pool startup per base
  now pay for one total.
- The package dependency graph is no longer represented as a
  `networkx.DiGraph`, and `Package.deps` may now contain cycles.
  `Analysis.bases` BFS-walks forward from no-dep packages through
  the precomputed consumer reverse map (so dependencies precede
  their consumers when the graph is acyclic, and any cycle-trapped
  packages get appended in path order); `reverse_closure` walks the
  same consumer map from a single seed; `_interesting_set` walks
  the dep map from `reverse_closure`'s result. All three share one
  `_bfs_order` helper whose visited-set guard makes them cycle-safe,
  and their results memoize on `Analysis` so repeated `PackageView`
  queries share the cost. Tolerating cycles lets a package with an
  acyclic exported subset list cyclic dev/test deps without
  hand-splitting them out.
- **Breaking:** `Analysis` no longer accepts a pre-built `paths` mapping.
  The constructor now takes `project_root` as the first argument and a
  required `resolver=` keyword argument (singular -- there is no
  resolver chain). Callers that used to write
  `Analysis({base: deps}, ...)` should switch to
  `Analysis(base, resolver=ManualResolver(specs=["."]), ...)` (or
  whichever resolver describes their layout). The CLI's `-p` / `--path`
  and `--resolver` flags are mutually exclusive, and `--resolver` takes
  a single value.
- **Breaking:** `PathResolver.resolve()` now returns
  `tuple[Package, ...]` instead of a `dict[Path, list[Path]]`
  (`PathMap`). `Package` is a frozen dataclass carrying `path`,
  `name` (unique within an analysis), `exported` (subdirs visible to
  consumers; empty means "no restriction"), and `deps` (other
  packages by name). The `PathMap` type alias and `merge_paths`
  re-export are gone; `Analysis` validates a single resolver's output
  internally. Resolvers no longer represent non-package search paths
  (workspace `.venv/site-packages`, vendored bundles) in
  `Package.deps`; `UvResolver` splices the workspace venv onto
  `sys.path` lazily inside its own `resolve_import` instead.
- **Breaking:** `Analysis.paths` is replaced by `Analysis.packages`,
  which returns the `tuple[Package, ...]` the analysis was built
  with. The previous `Analysis.packages()` iterator (yielding one
  `PackageView` per base) is renamed to `Analysis.views()` to free
  the name for the new data attribute.
- **Breaking:** `UvWorkspaceResolver` is renamed to `UvResolver`, the
  CLI resolver name `uv_workspace` is renamed to `uv`, and the module
  moved from `dead_cst/contrib/uv_workspace.py` to
  `dead_cst/contrib/uv.py`. The `name` field bumps to `"uv"` so cached
  per-base fingerprints rebuild automatically.
- **Breaking:** the CLI helper `resolve_paths` is replaced by
  `build_resolver`, which returns a single `PathResolver`. Callers
  (and the CLI itself) read the package list back from
  `Analysis.packages` after construction.

### Removed
- **Breaking:** `VenvResolver` and `PyprojectResolver` are removed,
  along with the `--resolver venv` and `--resolver pyproject` CLI
  names. `MissingVenvError` is also no longer part of the resolver
  surface; `UvResolver` raises its own `dead_cst.contrib.uv.MissingVenvError`
  when the workspace's shared `.venv` is missing. Use `-p src` (the
  CLI's `ManualResolver`) for the old `pyproject` `src/` shortcut.

### Added
- `dead_cst.resolvers.Package`, the frozen dataclass every resolver
  now emits.
- `dead_cst.resolvers.clear_path_caches`, a one-call helper that
  drops the three `sys.path`-derived LRU caches
  (`safe_resolve_module`, `distribution_lookup`,
  `editable_distribution_roots`). Custom resolvers that mutate
  `sys.path` should call it instead of clearing each cache by hand.
- New primary API: `dead_cst.Analysis` and `dead_cst.PackageView`,
  the lazy entry point that callers should reach for on large repos.
  Construction is cheap (no filesystem walk, no parsing). `refresh()`
  is base-scoped and idempotent. `package(base)` returns a
  `PackageView` whose `modules` / `declarations` / `count_nodes`
  queries are local to that base, while `dead` / `reachable` /
  `kept_alive_by_dead_branches` / `importers_of` / `graph` /
  `remove_dead_code` materialize only the "interesting set" -- the
  forward closure of the base's reverse (consumer) closure -- which
  is the smallest scope that gives correct reachability answers for
  decls in that base.
- `CycloptsPlugin` (`dead_cst.contrib.cyclopts`, re-exported from
  `dead_cst.plugins` and `dead_cst.contrib`) wires
  `@<app>.command` and `@<app>.default` handlers through their owning
  `cyclopts.App` instance, mirroring the Typer/Click plugins.
  Registered in `BUILTIN_PLUGINS` under the name `cyclopts` and
  loadable via `--plugin cyclopts`.
- `MockPatchPlugin` (`dead_cst.contrib.mock_patch`, re-exported from
  `dead_cst.plugins` and `dead_cst.contrib`) resolves string-fqname
  targets passed to `unittest.mock.patch` / `mock.patch` /
  `mocker.patch` (decorator and context-manager forms) plus pytest's
  `monkeypatch.setattr("X.Y", value)` and `monkeypatch.delattr("X.Y")`
  so symbols whose only consumers are tests patching them by string
  aren't flagged as dead. `patch.object`, `patch.dict`,
  `patch.multiple`, and the object form of `monkeypatch.setattr` /
  `delattr` are intentionally not handled -- their targets are real
  references the analyzer already sees. `monkeypatch.setitem` /
  `setenv` / `syspath_prepend` / etc. are not handled either -- their
  string args name dict keys, env vars, or paths, not symbols.
  Registered in `BUILTIN_PLUGINS` under the name `mock_patch` and
  loadable via `--plugin mock_patch`.

### Changed
- **Breaking (visitor / cache / `Import` shape):** Cross-file import
  resolution moved out of the per-file visitor and into the edge
  stitcher (`dead_cst._edges.resolve_edges`). Consequences:
  - `dead_cst.graph.Import` drops `path: Path | str` and gains
    `speculative: bool = False` (set on the synthetic star imports
    the visitor produces for `__import__(name, fromlist=[...])`
    fromlist entries; the stitcher silently drops a speculative
    entry when neither the trie nor the resolver places it).
    Plugins that consumed `Import.path` should switch to
    `Import.module` (and pair it with `ctx.find_module` /
    `ctx.importers` if they need a resolved target).
  - `compute_fingerprint` no longer takes `search_paths=` /
    `resolvers=`. The per-file fingerprint covers only the visitor /
    plugin / detector versions plus the base path; resolver swaps
    and search-path changes re-stitch edges on the next analysis
    without invalidating any cached `VisitorPayload` blobs.
  - `SymbolVisitor.__init__` no longer takes `search_paths` or
    `import_resolver`; the visitor is purely a function of the
    file's source. `resolve_edges` gained
    `import_resolver=` / `search_paths=` keyword arguments for the
    trie-miss classification path.
  - `Analysis._materialize` now rebinds `sys.path` to each base's
    `(base, *deps)` view before composing it (and clears the
    resolver LRUs at every transition), restoring the original
    `sys.path` on the way out. Workers in the parallel visitor pass
    no longer touch `sys.path` at all -- the cost moves from O(files)
    to O(bases).
  - `from p import functions` (where `functions` is a submodule of
    `p`) now produces edges to `p.functions` only when the access
    path canonicalizes to that module; the previous shape sometimes
    pointed at the parent package instead. Reachability is
    unchanged; the only observable difference is which intermediate
    module appears in the edge set.
  - Visitor `version` bumped (cache wipes on first run).
- **Breaking (cache API):** `compute_fingerprint` is now per-base
  (`base=`) rather than per-project (`paths=`). Each cache row
  carries its own fingerprint, so changing one base's plugins no
  longer invalidates sibling bases' cached payloads.
  `GraphCache(db_path)` no longer takes a fingerprint at open time;
  `get` and `put` take it per call. Schema version bumped to 2;
  older databases auto-wipe on first open. The CLI is unaffected --
  `Analysis` computes per-base fingerprints internally.
- `_order_packages` (formerly `_order_paths`) returns only the
  paths of `Package` records the resolvers emit. Search paths that
  aren't themselves packages (e.g. a workspace's
  `.venv/site-packages`) are never walked; resolvers that need them
  during classification handle that inside their `resolve_import`.
- `PluginContext`, `ObserveContext`, and the `AddNode` / `AddEdge` /
  `RemoveEdge` graph-op value objects now use `__slots__`, as do the
  analyzer-internal `_BaseSpec` / `_Task` records, shaving a
  per-instance `__dict__` off objects allocated per file and per
  emitted op.

### Removed
- **Breaking (top-level API):** `build_symbol_graph`, `find_reachable`,
  `find_kept_alive_by_dead_branches`, `count_nodes`, `order_paths`,
  and the top-level `remove_code` re-export are gone. Replace with
  the equivalent `Analysis` / `PackageView` methods:
  - `build_symbol_graph(...)` -> `Analysis(...).materialize_all()`
    (returns the same `nx.MultiDiGraph` for callers that want raw
    access).
  - `find_reachable(graph)` -> `Analysis.reachable()` /
    `PackageView.reachable()`.
  - `find_kept_alive_by_dead_branches(graph)` ->
    `Analysis.kept_alive_by_dead_branches()` /
    `PackageView.kept_alive_by_dead_branches()`.
  - `count_nodes(graph, prefix)` -> `Analysis.count_nodes(prefix)` /
    `PackageView.count_nodes()`.
  - `order_paths(paths)` -> `Analysis(...).bases`.
  - `remove_code(graph, base)` -> `PackageView.remove_dead_code()`
    for the high-level entry point. The standalone function is still
    available at `dead_cst.codemod.remove_code` for power users.
- `PluginContext.prime_module`, the public method for inserting an
  already-parsed `cst.Module` into the request-scope `parse` memo.
  The analyzer never called it (warm cache hits skip parsing entirely
  and the visitor's parsed module is consumed in-process), so it was
  dead code in the public surface. Plugins that need a `cst.Module`
  during `observe` already get it for free as `ObserveContext.module`;
  during `finalize` they go through `ctx.parse(path)` and that path
  still memoizes within the analysis.

### Fixed
- Documentation no longer claims that `PluginContext.parse` is primed
  with the modules the analyzer walked during the visitor pass: that
  hasn't been true since per-file results moved into the SQLite cache
  (warm hits skip parsing entirely). The `parse` cache is now
  documented as a request-scope memo, populated lazily on first
  access.
- `from mod import MyAlias` where `MyAlias` is a PEP 695 `type` statement
  no longer raises `AssertionError` during edge resolution. `type_alias`
  declarations are now treated as concrete termination points in the
  re-export follower, the same as `function`, `class`, and `variable`.

## [0.5.0] - 2026-05-05

### Added
- New public surface: `GraphCache`, `compute_fingerprint`,
  `clear_cache`, `default_cache_path`, and `SCHEMA_VERSION` from
  `dead_cst.cache`; `VisitorPayload` from `dead_cst.graph`;
  `evaluate_truthiness`, `unreachable_suites`, `unreachable_bodies`,
  `ResolveExpr`, and `fold_constants` from `dead_cst.branches`;
  the synthetic-node prefix constants (`STDLIB_PREFIX`,
  `EXTERNAL_DIST_PREFIX`, `EXTERNAL_FILE_PREFIX`, `UNRESOLVED_PREFIX`,
  `EXTERNAL_PREFIXES`, `SYNTHETIC_PATH_PREFIXES`, `SYNTHETIC_POSITION`)
  from `dead_cst.plugins`; `safe_resolve_module`,
  `distribution_lookup`, `editable_distribution_roots`, `STDLIB`,
  `SITE_PACKAGES_MARKERS`, and `load_toml` from `dead_cst.resolvers`.
  All previously reachable only via private (underscored) modules.
- `tests/test_public_api.py` pins each public module's `__all__`
  against a snapshot so accidental drops fail loudly in CI.

- Dynamic-import calls with a string-literal argument
  (`__import__('pkg.mod')` and `importlib.import_module('pkg.mod')`)
  are now treated like `from pkg.mod import *`: every top-level decl
  in the target module is fanned out as a successor of the enclosing
  top-level decl, so `getattr(__import__('pkg.mod'), 'name')()` keeps
  `pkg.mod.name` reachable instead of being silently dropped.
  Relative names are resolved against the file's enclosing package
  the same way `from .x import *` is:
  `importlib.import_module('.sub')` from `pkg/x.py` resolves to
  `pkg.sub`, and `__import__('sub', ..., level=1)` does the same;
  an explicit `package=` literal overrides the inferred anchor.
  `__import__(name, fromlist=[...])` with a literal list/tuple is
  parsed: every entry that resolves as a submodule of `name`
  (e.g. `__import__('pkg', fromlist=['mod'])` imports `pkg.mod`
  as a side effect) is fanned out the same way, while non-resolving
  entries are silently treated as plain attributes (already covered
  by the fan-out from `name`). Non-literal arguments (name,
  `level`, `package`, `fromlist`) skip with a warning. Bumps
  `SymbolVisitor.version` to invalidate cached payloads.
- `build_symbol_graph(workers=N)` (and matching `--workers` / `-j` CLI
  flag) dispatches per-file visitor + observe passes to a
  `ProcessPoolExecutor` when at least two cache-miss files exist
  across all bases. Workers return `VisitorPayload` blobs to the
  main process, which still owns SQLite cache writes, trie
  stitching, and edge resolution; serial behaviour and graph output
  are unchanged. A single persistent pool spans the whole run with
  tasks sorted by `search_paths`, so any one worker tends to see
  contiguous miss runs from the same base; on each transition the
  worker rebinds `sys.path` and clears `safe_resolve_module` plus
  `distribution_lookup` so cross-venv uv-workspace members don't
  inherit a sibling base's resolution state. The FQN provider's
  per-base cache is now built once in the parent over miss files
  only and shipped per-task to workers, so workers no longer rebuild
  a `FullRepoManager` and the analyzer skips FQN computation for
  cache-hit files entirely. `workers=None` (default) and
  `workers=1` keep the in-process path.

### Changed
- `build_symbol_graph` runs as three phases: collect per-base specs
  (cache hits + miss files + per-base FQN cache), compute every miss
  payload (in-process or via the pool), then per-base apply + edge
  stitch + plugin finalize. The graph and cache contents are
  unchanged. The in-process and worker paths share a single
  `_process_task` body — the only difference is whether the runner
  state lives on the main process or in worker globals.

### Removed
- The internal `temp_sys_path` context manager
  (`dead_cst.resolvers._imports`). The runner now manages
  `sys.path` directly, restoring it from a baseline snapshot when
  the in-process path finishes. Not part of the public API.

### Changed (breaking)
- Public modules dropped their leading underscore: `_analyze` →
  `analyze`, `_branches` → `branches`, `_cache` → `cache`, `_codemod`
  → `codemod`, `_plugins` → `plugins`, `_resolvers` → `resolvers`,
  and `_symbols` was renamed to `graph` (with `VisitorPayload` moved
  in from `_visitor`). The `explicit` plugin module was renamed to
  `explicit_entrypoint` to match its class name.
- New top-level `dead_cst.contrib` package collects every
  third-party-aware extension: framework plugins (`FastAPIPlugin`,
  `FlaskPlugin`, `ClickPlugin`, `TyperPlugin`, `PytestPlugin`,
  `UnittestPlugin`) and `UvWorkspaceResolver`. They are re-exported
  from `dead_cst.plugins` and `dead_cst.resolvers` for ergonomics, so
  `from dead_cst.plugins import FastAPIPlugin` and
  `from dead_cst.resolvers import UvWorkspaceResolver` keep working.
- The top-level `dead_cst` package no longer re-exports every plugin
  and resolver class. The curated highlights remain importable from
  `dead_cst` directly (`build_symbol_graph`, `find_reachable`,
  `find_kept_alive_by_dead_branches`, `count_nodes`, `order_paths`,
  `remove_code`, `Cacheable`, `SymbolNode`, `Import`, `NodeFlags`,
  `EdgeFlags`, `__version__`). Plugin and resolver classes must now
  be imported from `dead_cst.plugins`, `dead_cst.resolvers`, or
  `dead_cst.contrib`.
- Modules still prefixed with `_` (`_visitor`, `_edges`, `_flow`,
  `_fqn`, `_const_fold`, `_cacheable`, `_version`) are internal and
  not part of the supported surface.

### Fixed
- Path classification in `default_resolve_import` no longer
  misclassifies third-party packages as stdlib when running against
  a Python install whose `site-packages` is nested *inside* the
  stdlib root (the typical layout for a system Python with no venv,
  e.g. `/usr/local/lib/python3.13/site-packages` under
  `/usr/local/lib/python3.13`). The stdlib check now excludes paths
  under `purelib` / `platlib` and any directory named
  `site-packages` / `dist-packages`.
- Editably-installed third-party packages (`pip install -e`,
  `uv pip install -e`) are now resolved to `[external dist] <name>`
  instead of raising `Module ... resolved to an unexpected path`.
  Distribution metadata is consulted via PEP 610
  `direct_url.json` and any `.pth` shims in the dist's
  `RECORD`, so editable source dirs that live outside the project's
  search paths still get attributed to their owning distribution.
  The new cache (`editable_distribution_roots`) is cleared alongside
  `distribution_lookup` on worker venv transitions. All four
  shipping resolvers (`venv`, `pyproject`, `uv_workspace`,
  `manual`) bump their `version` so cached `VisitorPayload` blobs
  rebuild against the corrected classification.
- First-party search paths now win over editable distribution roots
  in `default_resolve_import`. Previously a project whose source
  happened to live inside another editable install's root (e.g. an
  e2e fixture cloned into `.pytest_cache/` of an editable
  `dead-cst` checkout) had every first-party file misclassified as
  `[external dist] <host-pkg>`, severing cross-module edges and
  reporting the entire surface as dead. The four shipping resolvers
  bump their `version` again so cached payloads rebuild.

## [0.4.0] - 2026-05-03

### Added
- PEP 695 `type` statements are now tracked. `type Foo = list[int]`
  surfaces `mod.Foo` as a top-level declaration of kind `"type_alias"`,
  and refs in the RHS are attributed to the alias so removing a dead
  alias releases its references rather than holding them through the
  enclosing module. The generic form (`type Pair[T] = tuple[T, T]`)
  is captured the same way. Cross-module users of the alias (e.g.
  `def f(x: Foo) -> Foo`) get a normal edge into the alias decl.
  `dead-cst remove` deletes unreachable aliases through a new
  `leave_TypeAlias` pass on `RemoveDeadSymbols`. Bumps
  `SymbolVisitor.version` to invalidate cached payloads.

## [0.3.0] - 2026-05-03

### Added
- PEP 572 walrus (`:=`) bindings at module scope are now surfaced as
  top-level declarations. `if (Y := src()): ...` registers `mod.Y`
  with an outgoing edge to whatever `src` resolves to, and downstream
  references like `def use(): return Y` get a `mod.use -> mod.Y`
  edge. Walruses leaked from a module-level comprehension (e.g.
  `result = [last := n for n in nums]`) are also captured: libcst's
  `ScopeProvider` keeps the binding inside the comprehension scope,
  so `SymbolVisitor` patches the gap by routing any unresolved Name
  access whose `.value` matches a leaked walrus target back to the
  matching decl. The default unreachable-region detector folds
  walrus bindings the same way it folds `Assign` / `AnnAssign` --
  `(DEBUG := False)` and `if (DEBUG := False):` both flag dead
  branches. Walruses inside a function / class / lambda body still
  bind locally, matching Python's runtime semantics.
- `SymbolVisitor` now satisfies the `Cacheable` protocol with
  class-level `name: str = "default"` and an epoch `version: int`,
  and `compute_fingerprint` includes the pair in the cache key.
  Bump `SymbolVisitor.version` on any change to the visitor's
  per-file output (new node kinds surfaced as decls, edge-attribution
  rules, flow-analysis fixes, etc.) so stale `VisitorPayload` blobs
  invalidate even between releases. Concurrent bumps on different
  branches merge with `max()` semantics. The walrus-support change
  bumps the visitor version accordingly.
- `UnreachableRegionDetector`: pluggable Protocol for module-level
  dead-region detection. Implementers provide
  `find_regions(wrapper) -> list[CodeRange]` and a `(name, version)`
  pair; consumers pass an instance via the new
  `build_symbol_graph(unreachable_detector=...)` keyword. Lets a
  company fold domain knowledge (e.g. "`settings.IS_PROD` is always
  `True` in production") into the analysis without forking the
  package. The shipped `DefaultUnreachableRegionDetector` preserves
  existing behavior — literal-only truthiness on `if` / `while`
  tests. The detector runs from inside `SymbolVisitor.visit_Module`
  reusing the analyzer's already-resolved `PositionProvider`, so the
  abstraction is free for the default path.
- `dead_cst._const_fold.fold_constants(wrapper, resolve_expr=None)`:
  fixpoint constant-folding pass that returns a `dict[id(Name), bool]`
  of every access whose binding ties back to a simple `Name = literal`
  (or `Name: T = literal`) assignment. Iteration is the point: chained
  forms like `foo = False; bar = foo or False; if bar: ...` resolve
  fully because each pass propagates one more level of indirection.
  Flow-sensitive (a later rebinding shadows an earlier one) and
  conservative (mixed-value bindings, non-literal RHS, unsupported
  shapes, and cycles all stay unknown). The optional `resolve_expr`
  callback gets first crack at any expression encountered during RHS
  evaluation, so domain-specific truthiness composes with the literal
  fold automatically — `flag = check_flag("x"); if flag:` resolves
  when the resolver answers for the call.
- `DefaultUnreachableRegionDetector` now runs three passes per file:
  the literal-only `unreachable_suites` walk, the new `fold_constants`
  pre-pass, and a post-terminator scan over every statement-bearing
  suite. Patterns like `DEBUG = False; if DEBUG: ...`, `return`
  followed by dead code, and `assert False` followed by dead code are
  all flagged out of the box. Post-terminator detection is purely
  suite-relative, so a `raise` inside a `try` body kills the rest of
  the try body without touching the `except` handler, which runs on
  its own path.
- `DefaultUnreachableRegionDetector.resolve(self, expr) -> bool | None`:
  overridable hook for subclasses to fold non-literal expressions to a
  known truthiness — e.g. `check_flag("migration-abc")` is always
  `True` in production. The default returns `None` (defer to literal
  handling). The override is consulted recursively for every
  non-keyword expression in every `if` / `while` / `assert` test and
  every foldable assignment RHS; folded values flow through the
  fixpoint loop alongside `Name = literal` bindings, so a single
  high-level decision propagates through chains. Subclasses bump
  `version` (epoch-int) for cache invalidation.
- `evaluate_truthiness` and `unreachable_suites` / `unreachable_bodies`
  now accept `resolve_expr: Callable[[cst.BaseExpression], bool | None]`
  in place of the previous `Name`-only `resolve_name` callback. The
  resolver is consulted for any non-keyword expression and short-
  circuits the built-in literal handling when it returns a `bool`;
  language keywords (`True` / `False` / `None`) always resolve to
  their language-defined truthiness and are never passed through.
  Detector `version` set to the current Unix epoch (`1777795837`),
  matching the convention used by every other shipped `Cacheable`,
  so any cached `VisitorPayload` from the prior detector is
  automatically invalidated and concurrent bumps merge with `max()`
  semantics.
- `Cacheable` Protocol (`name: str`, `version: int`): the shared
  cache-fingerprint contract that `EdgePlugin`, `PathResolver`, and
  `UnreachableRegionDetector` now all inherit from. Bumping a
  component's epoch `version` invalidates the per-file cache the same
  way it does for plugins. `compute_fingerprint` reads the attributes
  directly instead of falling back to `getattr` defaults.
- `DecoratedDeclPlugin`: abstract `EdgePlugin` base for the "find decls
  decorated by `@<module>.<name>(...)` or assigned via
  `X = <module>.<ctor>(...)` in files matching a search path" idiom.
  Subclasses set `package_prefix`, `decorator_module`, `decorator_names`,
  `constructor_names`. Pure observe-time, so matches turn directly into
  cached entrypoint payloads. `ClickPlugin` is now a subclass that adds
  the nested-group fixpoint pass and overrides `observe` to emit
  `instance -> handler` edges instead of seeding entrypoints.
- `LiteralListPlugin`: abstract `EdgePlugin` base for the "read
  `<owner>.<var> = ["a.b.c", ...]` and revive every fqname inside" idiom.
  observe parses the literal once and emits ENTRYPOINT-flagged synthetic
  decls (one per entry, positioned at the literal's site); finalize is a
  graph-only pass that adds the cross-file edges. Owner-file-only CST
  work is cached alongside the per-file payload, so warm runs do zero
  parsing for this plugin. Both bases are abstract -- subclasses must
  set `name` and `version`.
- `PluginContext.module_surface(fqname)`: returns the module's
  `SymbolNode` plus every top-level decl plus every transitive
  submodule's surface, walked via the symbol trie in
  `O(decls_in_subtree)`. Replaces hand-rolled scans of
  `ctx.base_nodes()` for the common "this module is loaded
  dynamically -- keep its surface alive" pattern.
- `synthetic_node(..., position=...)`: optional `CodeRange` to anchor
  a synthetic at a specific source location so `why-alive` and the
  codemod report the right line. Defaults to `SYNTHETIC_POSITION` for
  backwards compatibility.
- New public re-exports under `dead_cst._plugins` (and the most-used
  ones at the top level): `ObserveContext`, `make_payload` (renamed
  from the private `_payload_from`), `mark_entrypoints`,
  `decls_by_simple_name`, `simple_name`, `collect_module_imports`,
  `matched_attr_call`, `single_target_assignment`, `find_handlers`,
  `find_call_assignments`, `decorator_owner`, `is_name`,
  `is_from_module`. These are the helpers user-written plugins reach
  for; previously they lived in `_core` and required private imports.
- E2E test suite under `tests/e2e/`, deselected by default
  (`addopts = "-m 'not e2e'"`); run with `uv run pytest -m e2e`. The
  first target is `flux0-ai/flux0` pinned at SHA `8d04176`. Tests
  cover three levels: analyze runs to completion, `why-alive` chains
  for known-alive symbols, and project-specific plugins that close
  the repo's `importlib`-driven blind spots. `tests/e2e/conftest.py`
  exposes a `clone_repo(name, url, sha)` fixture that shallow-clones
  into pytest's cache dir (or `DEAD_CST_E2E_CACHE`) with a SHA
  marker for idempotent reuse.

### Changed
- The package `__version__` is no longer folded into the cache
  fingerprint. Every component whose output can shift between
  releases (visitor, resolvers, plugins, detector) already carries
  its own `Cacheable` `(name, version)` knob; mixing `__version__`
  in on top let unbumped components ride for free on a release bump
  and masked cases where the granular versions weren't being
  maintained. The discipline now is to bump the relevant component's
  `version`. Schema version and Python version still participate.
- `PathResolver` is now a `Cacheable` Protocol: shipped resolvers
  (`ManualResolver`, `PyprojectResolver`, `UvWorkspaceResolver`,
  `VenvResolver`) all carry an epoch `version: int` matching the
  plugin convention, and the per-file cache fingerprint includes
  each resolver's `(name, version)` pair. Bump a resolver's
  `version` when its layout-discovery or `resolve_import` logic
  changes; stale `VisitorPayload` blobs rebuild automatically.
- `EdgePlugin.version` is now `int` (Unix epoch by convention) rather
  than `str`. Bump `version` to the current epoch on any change to a
  plugin's `observe` shape that should not be served from older
  caches; concurrent bumps on different branches merge with `max()`
  semantics rather than colliding on a re-used label like `"2"`. All
  builtin plugins migrated; the cache fingerprint format follows.
- `_decls_by_simple_name` -- a four-line helper duplicated across
  `ClickPlugin`, `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin` -- is
  hoisted into `dead_cst._plugins._core.decls_by_simple_name` and
  re-exported. Same behaviour, one definition.
- `ClickPlugin` now subclasses `DecoratedDeclPlugin`, dropping ~70
  lines of duplicated decorator-finding code. Behaviour is preserved
  -- the plugin still emits `instance -> handler` edges and does not
  seed Click groups as entrypoints.

## [0.2.0] - 2026-05-01

### Added
- SQLite-backed `GraphCache` keyed by per-file SHA-256 hashes,
  storing pickled `VisitorPayload` blobs under
  `<root>/.dead-cst-cache/cache.db`. Cache hits skip the per-file
  visitor pass (the dominant cost in `build_symbol_graph`); the
  per-base `resolve_edges` step and plugin pass run unconditionally,
  so a graph built from a warm cache is identical to one built from
  scratch. The cache is keyed by a fingerprint over
  `(__version__, python version, PathMap, resolver chain)`; a
  fingerprint mismatch wipes `file_cache` and rebuilds. Plugins are
  intentionally **not** part of the fingerprint -- swapping plugins
  reuses cached payloads. New `--no-cache` flag on `analyze`,
  `why-alive`, `unused-exports`, `dependencies`, and `remove`; new
  `dead-cst cache clear` subcommand. `build_symbol_graph` accepts a
  new `cache=` keyword.
- `ManualResolver`: a `PathResolver` built from explicit
  ``base:dep1,dep2`` specs. The CLI's ``-p`` flag now flows through
  this resolver, so explicit specs sit in the same chain as named
  resolvers and participate in `resolve_import` lookups too.
- `NodeFlags.ENTRYPOINT`: a node flag that `_apply_payload` reads to
  set `graph.nodes[node]["entrypoint"] = True`. Plugin observe passes
  emit synthetic nodes flagged `ENTRYPOINT` to declare reachability
  seeds without a separate API surface.

### Changed
- `EdgePlugin` is now a two-pass protocol:
  - `observe(ctx) -> VisitorPayload | None` runs in the per-file
    analyzer loop with the file's parsed CST and just-built
    `VisitorPayload`. The plugin returns the same payload shape
    (`nodes` + `edges`) and the analyzer concatenates that with the
    visitor's payload before applying. Plugin contributions are
    cached alongside the visitor's output, so a warm cache hit
    returns the combined payload in one read.
  - `finalize(ctx) -> Iterable[GraphOp]` runs once per base after
    `resolve_edges`. It is graph-only -- no CST access -- and is
    where cross-file work belongs (FastAPI's factory walk,
    InitSubclass's transitive subclass closure, `[project.scripts]`
    lookups). Plugins use synthetic markers from their `observe`
    pass to communicate state forward into `finalize` (e.g.
    `<fastapi-pending>:<X.fqname>` for variables that need a graph
    walk to classify).
  - `EdgePlugin.contribute` is replaced by `observe` + `finalize`.
    Every builtin plugin -- `MainBlockPlugin`,
    `ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`,
    `ModuleDundersPlugin`, `PytestPlugin`, `UnittestPlugin`,
    `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin`, `ClickPlugin`,
    `InitSubclassPlugin` -- migrates accordingly.
  - Plugins declare a `version: str` attribute. The cache
    fingerprint includes each plugin's `(name, version)` pair, so
    bumping a plugin's version invalidates the file_cache (its
    observe contributions are baked into cached payloads).
- Warm cache runs with the full builtin plugin set now parse **zero**
  files: the visitor and every plugin's observe contributions are
  baked into the cached payloads, and `finalize` runs purely off the
  graph. Pinned by `test_warm_run_with_plugins_parses_zero_files`.
- `PathResolver` protocol now includes a `resolve_import(name, search_paths)`
  method, folding `name -> path` lookup into the resolver alongside
  search-path discovery. The shipped resolvers (`ManualResolver`,
  `PyprojectResolver`, `UvWorkspaceResolver`, `VenvResolver`) all
  delegate to the new `dead_cst._resolvers.default_resolve_import`, the
  `sys.path` + `importlib` implementation. Custom resolvers can now
  override import lookups for their own layout (vendored deps, `.pyi`
  siblings, ...) without monkey-patching internals.
- `build_symbol_graph` accepts a new `resolvers=` keyword whose entries'
  `resolve_import` methods are tried in order. With no resolvers it
  falls back to `default_resolve_import`, preserving today's behavior.
- The CLI threads loaded resolvers through to the analyzer so
  `--resolver` (and ``-p``) selections govern import lookup, not just
  search paths.
- Renamed `dead_cst._resolve` to `dead_cst._edges`. The remaining
  module is purely about edge construction in the symbol trie;
  resolution now lives under `dead_cst._resolvers`.

### Removed
- `dead_cst.cli.parse_paths` -- callers should construct a
  `ManualResolver` and call `.resolve(root)`.

### Fixed
- `MainBlockPlugin` now keeps decls bound inside the
  `if __name__ == "__main__":` block alive, not just the containing
  module. Previously a top-level decl introduced by an assignment in
  the block (e.g. `app = Foo(fn=main).cli()`) had no incoming edge --
  the visitor's value-frame produced `app -> Foo` / `app -> main`, but
  nothing pointed at `app` itself -- so the chain was unreachable and
  `Foo` / `main` were reported dead. The plugin now resolves the
  block's `CodeRange` via `PositionProvider` and emits `synth -> decl`
  edges for every top-level decl whose binding site falls inside the
  block.

## [0.1.0] - 2026-04-28

Initial alpha release. `dead-cst` is pre-1.0 software: the public Python API,
CLI flags, and output formats may change without notice between any two
versions until the first stable release.

### Added
- Symbol-level reachability analysis built on LibCST's
  `FullyQualifiedNameProvider` and `ScopeProvider`.
- Resolution of relative imports, aliased imports, and re-export chains
  through `__init__.py`.
- `dead-cst analyze` CLI for reporting unreachable symbols and unreachable
  branches (`if False:`, raise-only suites, etc.), with `text` and `json`
  output formats.
- `dead-cst why-alive` CLI for explaining why a symbol is kept alive.
- `dead-cst remove` CLI that rewrites files in place via a LibCST codemod,
  with import pruning when the last local user of an import is deleted and
  position-aware shadowing so a shadowed dead binding no longer drags its
  live sibling out with it.
- `dead-cst unused-exports` CLI command: report `__all__` entries whose
  targets are kept alive only because they are listed in `__all__`.
- `dead-cst dependencies` CLI command: list third-party distributions and
  files imported by the codebase, surfaced as synthetic
  `[external dist] <name>` / `[external file] <name>` graph nodes.
- Multi-package / monorepo support via the `-p base:dep1,dep2` search-path
  spec, with topological ordering of bases.
- Edge plugin architecture (`EdgePlugin`, `CSTAwareEdgePlugin`,
  `PluginContext`, `GraphOp`/`AddNode`/`AddEdge`/`RemoveEdge`, `apply_ops`,
  `synthetic_node`). Built-in plugins: `MainBlockPlugin`,
  `ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`, `ModuleDundersPlugin`,
  `PytestPlugin`, `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin`,
  `ClickPlugin`, `UnittestPlugin`, `InitSubclassPlugin`. Third-party
  plugins register under the `dead_cst.plugins` entry-point group and load
  via `load_plugin`.
- `PytestPlugin` (`--plugin pytest`): keep pytest-discovered tests,
  `conftest.py` decls, and `@pytest.fixture` functions alive.
- `FastAPIPlugin` (`--plugin fastapi`): detect top-level `FastAPI()` and
  `APIRouter()` instances (including factory-style apps), mark `FastAPI`
  apps as entrypoints, and emit `instance -> handler` edges for
  `@app.get(...)`-style decorators (HTTP methods, websockets, middleware,
  exception handlers, `on_event`). Routers stay pass-through, so an
  `APIRouter` that's never `include_router`'d remains dead.
- `FlaskPlugin` (`--plugin flask`): detect top-level `Flask()` /
  `Blueprint()` instances (including factory-style apps) and emit
  `instance -> handler` edges for `@app.route(...)`, HTTP-verb shortcuts,
  request-lifecycle hooks (`before_request`, `after_request`,
  `teardown_*`), error handlers, template helpers (`context_processor`,
  `template_filter`, ...), and URL processors. `Flask` apps are seeded as
  entrypoints (WSGI servers load `module:app`); `Blueprint`s stay
  pass-through, so a blueprint never `register_blueprint`'d remains dead,
  mirroring the `APIRouter` behavior in `FastAPIPlugin`.
- `TyperPlugin` (`--plugin typer`): detect top-level `Typer()` instances
  and emit `instance -> handler` edges for `@app.command(...)` and
  `@app.callback(...)` decorators. Typer apps are pass-through;
  reachability is expected through `[project.scripts]` or a `__main__`
  block, after which every registered command and callback stays alive.
  Sub-typers that are never `add_typer`'d remain dead.
- `ClickPlugin` (`--plugin click`): detect top-level Click `Group`
  instances (functions decorated `@click.group(...)` / `@click.Group(...)`,
  `X = click.Group(...)` constructor calls, and inline sub-groups
  registered via `@<group>.group(...)`, all resolved via fixpoint so a
  chain of nested groups is fully discovered) and emit
  `instance -> handler` edges for `@<group>.command(...)`,
  `@<group>.group(...)`, and `@<group>.result_callback(...)` decorators.
  Click groups stay pass-through; reachability is expected through
  `[project.scripts]` or a `__main__` block, mirroring `TyperPlugin`.
- `UnittestPlugin` (`--plugin unittest`): mark stdlib `unittest.TestCase`
  and `unittest.IsolatedAsyncioTestCase` subclasses, plus module-level
  `setUpModule` / `tearDownModule` / `load_tests` hooks, as entrypoints.
  Discovery is CST-based and prefiltered to files whose import nodes
  reference `unittest`. Supports `import unittest` (with alias),
  `from unittest import TestCase` (with alias), and module-prefixed base
  references. Only direct base-class matches are recognised; transitive
  subclasses through a project-local mixin need an explicit `-e`
  entrypoint or coverage from `PytestPlugin`'s filename heuristics.
  `from unittest import *`-only files are skipped (the resolver doesn't
  surface stdlib star imports as graph nodes); use a non-star import.
- `InitSubclassPlugin` (`--plugin init_subclass`): detect classes that
  define `__init_subclass__` and route reachability through a synthetic
  marker node `<__init_subclass__>:<parent.fqname>` with edges
  `parent -> marker -> subclass` for every transitive first-party
  subclass. Registry-pattern subclasses stay alive whenever the parent
  class does; the marker shows up in `why-alive` chains as a labeled
  breadcrumb. Parents are pass-through, so a registry base nobody else
  uses still surfaces as dead code.
- `ModuleDundersPlugin`: keep module-level dunder variables (`__all__`,
  `__version__`, `__future__` imports, etc.) alive. Always registered by
  the CLI.
- Path resolver architecture (`PathResolver`, `merge_paths`). Built-in
  resolvers: `VenvResolver`, `PyprojectResolver`, `UvWorkspaceResolver`
  (parses `uv.lock` to discover workspace members and inter-member
  edges, including virtual workspace members that don't ship as wheels).
  Third-party resolvers register under `dead_cst.resolvers` and load via
  `load_resolver`.
- `exported_roots(base)` in `dead_cst._resolvers`: inspect a base's
  `pyproject.toml` (src-layout, hatchling/setuptools/poetry/pdm/flit
  backends, name-match fallback) to determine which subdirs the build
  backend would actually ship, so internal dirs like `tests/` stay scoped
  to their owning workspace member during cross-member import resolution.
- `--resolver` and `--plugin` flags on `analyze`, `why-alive`,
  `unused-exports`, and `remove` for selecting path resolvers and edge
  plugins.
- Public Python API: `build_symbol_graph`, `find_reachable`,
  `count_nodes`, `order_paths`, `remove_code`, plus a `position` field
  on `SymbolNode`.
- `py.typed` marker for downstream type-checking.
- `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and `ROADMAP.md` with a
  stack-ranked plan from alpha to 1.0.

[Unreleased]: https://github.com/lpetre/dead-cst/compare/v0.12.2...HEAD
[0.12.2]: https://github.com/lpetre/dead-cst/compare/v0.12.1...v0.12.2
[0.12.1]: https://github.com/lpetre/dead-cst/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/lpetre/dead-cst/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/lpetre/dead-cst/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/lpetre/dead-cst/compare/v0.9.4...v0.10.0
[0.9.4]: https://github.com/lpetre/dead-cst/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/lpetre/dead-cst/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/lpetre/dead-cst/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/lpetre/dead-cst/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/lpetre/dead-cst/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/lpetre/dead-cst/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/lpetre/dead-cst/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/lpetre/dead-cst/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/lpetre/dead-cst/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/lpetre/dead-cst/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/lpetre/dead-cst/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lpetre/dead-cst/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lpetre/dead-cst/releases/tag/v0.1.0
