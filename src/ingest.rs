//! Three-phase graph-construction pipeline:
//!
//! 1. `ingest_decls` walks every project file's global scope via ty's
//!    `SemanticIndex` and mints one graph node per binding.
//! 2. `emit_module_hierarchy` + `emit_import_edges` wire up the
//!    submodule edges and the cross-file alias/upstream chain for
//!    every import binding.
//! 3. `emit_reference_edges` (driven by [`RefCollector`]) walks each
//!    owned expression and emits the `use -> alias` and parallel
//!    `use -> upstream` edges that fall out of ty's use-def chain.
//!
//! `detect_dynamic_call` / `parse_dynamic_args` / `resolve_dynamic_target`
//! plus the per-file helpers (`file_package_name`,
//! `collapse_attribute_chain`, `module_name_resolves`,
//! `emit_visitor_warning`) round out the dynamic-import lane.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use pyo3::prelude::*;
use ruff_db::files::{File, FilePath};
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_db::source::{line_index, source_text};
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Expr, ExprName, Stmt};
use ruff_text_size::{Ranged, TextRange};
use ty_module_resolver::{
    file_to_module, resolve_module, search_paths, ModuleName, ModuleResolveMode,
};
use ty_project::ProjectDatabase;
use ty_python_core::ast_ids::HasScopedUseId;
use ty_python_core::definition::{DefinitionKind, DefinitionState, TargetKind};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;
use ty_python_core::SemanticIndex;
use ty_python_semantic::SemanticModel;

use crate::builder::GraphBuilder;
use crate::graph::{
    DeclIndex, GlobalsByName, Import, ImportSpec, LiveDeclIndex, Resolution, StarReexports,
    SymbolNode,
};
use crate::helpers::{
    detect_dead_ranges, file_default_flags, file_path_string, module_fqname_for_file, position,
    range_key, scan_noqa_directives, EDGE_FLAG_DEAD_BRANCH, EDGE_FLAG_DYNAMIC_IMPORT,
    NODE_FLAGS_NOQA_PIN, NODE_FLAG_ENTRYPOINT,
};

// ---------------------------------------------------------------------------
// Phase 1: decl enumeration via ty's SemanticIndex
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn ingest_decls(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
    alias_imports: &mut HashMap<usize, ImportSpec>,
    live_decls: &mut LiveDeclIndex,
    globals_by_name: &mut GlobalsByName,
    star_reexports: &mut StarReexports,
    class_by_selection: &mut HashMap<(File, (u32, u32)), usize>,
    decl_by_name_range: &mut HashMap<(File, (u32, u32)), usize>,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = module_fqname_for_file(db, file);
    let default_flags = file_default_flags(db, file);
    // Per-decl stub flagging context. For .pyi files we OR in
    // ``NODE_FLAG_ENTRYPOINT`` for any decl whose name has no
    // matching runtime decl in the .py twin (or has no twin at all
    // — native-extension and protobuf-style stubs). Decls that DO
    // have a runtime counterpart stay un-flagged; reachability flows
    // through them via the stub-runtime edge emitted in
    // ``emit_stub_runtime_edges`` after both files have ingested.
    let is_stub = path_str.ends_with(".pyi");
    let stub_py_twin: Option<File> = if is_stub {
        builder.peer_pyi_to_py.get(&file).copied()
    } else {
        None
    };
    let (file_pinned_by_noqa, per_line_noqa_pins) =
        scan_noqa_directives(&parsed, &source, &line_index);

    let (msl, msc, mel, mec) = position(&line_index, &source, parsed.syntax().range);
    let module_idx = builder.intern_node(
        py,
        SymbolNode {
            fqname: module_fqname.clone(),
            kind: "module",
            path: path_str.clone(),
            start_line: msl,
            start_column: msc,
            end_line: mel,
            end_column: mec,
            flags: default_flags,
            imports: None,
            cached_hash: OnceLock::new(),
        },
    )?;
    module_nodes.insert(file, module_idx);

    // Iterate every binding (including shadowed siblings) — Principle 3.
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);
    // ty emits one `StarImport` definition per imported name, but
    // every per-name def from the same `from X import *` statement
    // shares the `*` token's range — so the `*<src>` local name we
    // synthesize is identical across all of them. Cache by that
    // shared range to avoid re-resolving `from_module_string` and
    // re-allocating the format string N times for a star statement
    // that brings in N names.
    let mut star_local_name_cache: HashMap<TextRange, String> = HashMap::new();

    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }

        let kind = def.kind(db);
        let Some(node_kind) = decl_kind_str(kind) else {
            continue;
        };

        // Bound place must be a simple symbol — Member places (e.g.
        // `x.y = ...` style attribute defs) aren't top-level decls in
        // our model.
        let place_id = def.place(db);
        let PlaceExprRef::Symbol(symbol) = place_table.place(place_id) else {
            continue;
        };
        // The per-name binding (e.g. `f`, `g`) ty sees. We keep this
        // for the `globals_by_name` / `star_reexports` maps that
        // Phase 2's cross-module chain walk uses to chase a
        // `from A import g` through A's `from B import *`.
        let per_name = symbol.name().as_str().to_string();

        // `from X import *` — ty produces one `StarImport` definition
        // per name brought in, but the graph holds *one* node per
        // statement: the import statement itself is the local thing
        // that should be kept alive by use sites, and ty's name
        // resolution still routes each use to its specific upstream.
        // The libcst per-name synthetic alias was a workaround for
        // libcst's inability to resolve uses through star imports —
        // we don't need it here.
        //
        // Collapse by giving every per-name `StarImport` from the
        // same statement a shared `*<source>` local name; combined
        // with the shared `target_range` (the `*` token), they all
        // intern to the same `NodeKey` and therefore the same
        // `node_idx`. The per-name lookup maps (`globals_by_name`,
        // `star_reexports`) keep using each per-name as their key,
        // all pointing at this single node — so a downstream
        // `from A import g` still finds the star alias by name and
        // can chase through it to B.
        let target_range = kind.target_range(&parsed);
        let local_name = match kind {
            DefinitionKind::StarImport(k) => star_local_name_cache
                .entry(target_range)
                .or_insert_with(|| {
                    let src = from_module_string(db, file, k.import(&parsed));
                    format!("*{src}")
                })
                .clone(),
            _ => per_name.clone(),
        };

        let (mut sl, mut sc, el, ec) = position(&line_index, &source, target_range);

        // For Function / Class / TypeAlias, ty's `target_range` is the
        // bound *name* (e.g. `f` in `def f(): ...`). The libcst
        // pipeline reports the position of the introducing keyword
        // (`def` / `class` / `type`) — which is the first non-space
        // character on the same line as the name (decorators sit on
        // earlier lines and don't count). Align to libcst by snapping
        // the start column to the line's indent.
        if matches!(
            kind,
            DefinitionKind::Function(_) | DefinitionKind::Class(_) | DefinitionKind::TypeAlias(_)
        ) {
            let name_start = line_index.line_column(target_range.start(), &source);
            let line_text = &source[line_index.line_range(name_start.line, &source)];
            let indent = line_text
                .bytes()
                .take_while(|b| matches!(*b, b' ' | b'\t'))
                .count();
            sl = name_start.line.get();
            sc = indent;
        }

        let import_spec = if node_kind == "import" {
            Some(import_payload_for(kind, db, file, &parsed))
        } else {
            None
        };
        let imports = if let Some(spec) = &import_spec {
            Some(Py::new(
                py,
                Import {
                    module: spec.module.clone(),
                    decl: spec.decl.clone(),
                    star: spec.star,
                },
            )?)
        } else {
            None
        };

        // Pin imports preserved by a `# noqa[: …F401…]` (per-alias)
        // or by a file-level `# ruff: noqa` / `# flake8: noqa` so
        // reachability keeps them alive — matching ruff's own
        // semantics for explicitly-preserved unused-import lines.
        // We tag both `ENTRYPOINT` (the live-set seed) and `NOQA`
        // (so the blast-radius query can subtract noqa-only liveness).
        let mut flags: u32 = default_flags;
        if node_kind == "import" && (file_pinned_by_noqa || per_line_noqa_pins.contains(&sl)) {
            flags |= NODE_FLAGS_NOQA_PIN;
        }
        if is_stub {
            let has_runtime = stub_py_twin
                .map(|py| globals_by_name.contains_key(&(py, local_name.clone())))
                .unwrap_or(false);
            if !has_runtime {
                flags |= NODE_FLAG_ENTRYPOINT;
            }
        }

        let node_idx = builder.intern_node(
            py,
            SymbolNode {
                fqname: format!("{module_fqname}.{local_name}"),
                kind: node_kind,
                path: path_str.clone(),
                start_line: sl,
                start_column: sc,
                end_line: el,
                end_column: ec,
                flags,
                imports,
                cached_hash: OnceLock::new(),
            },
        )?;
        builder.add_edge(node_idx, module_idx, 0);
        global_index.insert((file, place_id, range_key(target_range)), node_idx);
        if node_kind == "class" {
            class_by_selection.insert((file, range_key(target_range)), node_idx);
        }
        // Last-write-wins matches ``live_decls``: a single (file, range)
        // can host multiple bindings (try/except rebind, star imports),
        // and the query callers care about the end-of-scope live binding.
        decl_by_name_range.insert((file, range_key(target_range)), node_idx);

        // Lookup maps key by the *per-name* (the actual bound symbol
        // each ty `StarImport` corresponds to), not the node's
        // `local_name` — the star node's collapsed `*<src>` fqname
        // would otherwise miss `from A import g` chains that probe
        // for the per-name `"g"`. For non-star imports `per_name`
        // and `local_name` are identical so this is a no-op there.
        let name_key = (file, per_name);
        if let Some(spec) = import_spec {
            // Star-reexport synthetics need their upstream tracked so
            // Phase 2 can walk a `from A import g` resolution through
            // an `A.g` star-reexport alias all the way to its real
            // def. Non-star imports clear the entry so a shadowing
            // import correctly disables chain walking.
            if spec.star {
                star_reexports.insert(name_key.clone(), spec.module.clone());
            } else {
                star_reexports.remove(&name_key);
            }
            alias_imports.insert(node_idx, spec);
        } else if node_kind != "module" {
            // Star-reexport gets killed by any later real decl; the
            // multi-binding `live_decls` is populated by the post-pass
            // below from ty's end-of-scope view, which already encodes
            // sequential-rebind / branch-bind semantics correctly.
            star_reexports.remove(&name_key);
        }
    }

    // Post-pass: populate `globals_by_name` (every binding kind) and
    // `live_decls` (real decls only) from ty's end-of-scope live
    // bindings per symbol. We can't do this inside the loop above
    // because that loop iterates `all_definitions_with_usage`, which
    // includes *every* binding — including dead ones superseded by a
    // later sequential rebind (`def f; def f` keeps only the second
    // as the live one). The end-of-scope query already encodes ty's
    // flow analysis: it preserves both branches of `if/else` and
    // `try/except` when each is a real bind, and collapses sequential
    // rebinds to the latest. Walking it once here keeps cross-module
    // `from lib import f` resolution and `emit_upstream`'s decl probe
    // multi-binding-aware without having to re-derive shadowing rules
    // at lookup time.
    for (symbol_id, bindings) in use_def_map.all_end_of_scope_symbol_bindings() {
        let PlaceExprRef::Symbol(sym) = place_table.place(ScopedPlaceId::Symbol(symbol_id)) else {
            continue;
        };
        let name = sym.name().as_str().to_string();
        let mut live: Vec<usize> = Vec::new();
        let mut live_real_decls: Vec<usize> = Vec::new();
        for binding in bindings {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != file || def.file_scope(db) != global {
                continue;
            }
            let kind = def.kind(db);
            let key = (file, def.place(db), range_key(kind.target_range(&parsed)));
            if let Some(&idx) = global_index.get(&key) {
                live.push(idx);
                // ``live_decls`` mirrors what's reachable as a decl-like
                // target in the module's namespace — an import alias is
                // still a decl from the consumer's standpoint, so it
                // belongs here too. Filtering on ``!kind.is_import()``
                // would skip ``mod -> lib.f@2:18`` when ``lib.f`` is
                // ``from a import f`` (Principle 2's parallel-upstream
                // edge from the use site to the decl in ``lib``).
                live_real_decls.push(idx);
            }
        }
        let key = (file, name);
        if !live.is_empty() {
            globals_by_name.insert(key.clone(), live);
        }
        if !live_real_decls.is_empty() {
            live_decls.insert(key, live_real_decls);
        }
    }

    Ok(())
}

pub(crate) fn decl_kind_str(kind: &DefinitionKind<'_>) -> Option<&'static str> {
    if kind.is_import() {
        return Some("import");
    }
    Some(match kind {
        DefinitionKind::Function(_) => "function",
        DefinitionKind::Class(_) => "class",
        DefinitionKind::TypeAlias(_) => "type_alias",
        // Walrus's bound name (PEP 572) leaks to the enclosing scope
        // — at module level that's a top-level decl. The other
        // binding-statement targets (`for`, `with ... as`, `except as`,
        // structural-`match` captures) are local loop / context /
        // pattern bindings that the libcst pipeline does *not* model
        // as top-level nodes; skip them so we don't mint phantom
        // module-scope decls.
        DefinitionKind::Assignment(_)
        | DefinitionKind::AnnotatedAssignment(_)
        | DefinitionKind::NamedExpression(_) => "variable",
        _ => return None,
    })
}

pub(crate) fn import_payload_for<'db>(
    kind: &DefinitionKind<'db>,
    db: &'db dyn ty_python_semantic::Db,
    file: File,
    parsed: &ParsedModuleRef,
) -> ImportSpec {
    match kind {
        DefinitionKind::Import(k) => {
            let alias = k.alias(parsed);
            ImportSpec {
                module: alias.name.id.as_str().to_string(),
                decl: None,
                star: false,
            }
        }
        DefinitionKind::ImportFrom(k) => {
            let alias = k.alias(parsed);
            ImportSpec {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(alias.name.id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::ImportFromSubmodule(k) => {
            // Bound name is one of the dotted submodule segments. The
            // module string is the parent of that segment; ``decl`` is
            // the segment itself, mirroring the libcst convention for
            // ``from a.b import c`` where ``c`` is a submodule.
            ImportSpec {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(k.module(parsed).id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::StarImport(k) => ImportSpec {
            module: from_module_string(db, file, k.import(parsed)),
            decl: None,
            star: true,
        },
        _ => unreachable!("import_payload_for called with non-import kind"),
    }
}

/// Resolved absolute module name for a `from ... import ...` clause.
///
/// Returns the empty string when ty's `from_import_statement` fails to
/// resolve (invalid syntax or too many leading dots) — downstream
/// classification can treat that as an unresolved target.
pub(crate) fn from_module_string(
    db: &dyn ty_python_semantic::Db,
    file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
) -> String {
    ModuleName::from_import_statement(db, file, stmt)
        .map(|n| n.as_str().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Phase 2: module-hierarchy + cross-file alias edges
// ---------------------------------------------------------------------------

pub(crate) fn emit_module_hierarchy(
    db: &ProjectDatabase,
    file: File,
    module_nodes: &HashMap<File, usize>,
    builder: &mut GraphBuilder,
) {
    if let Some((self_idx, parent_idx)) = parent_module_edge(db, file, module_nodes) {
        builder.add_edge(self_idx, parent_idx, 0);
    }
}

pub(crate) fn parent_module_edge(
    db: &ProjectDatabase,
    file: File,
    module_nodes: &HashMap<File, usize>,
) -> Option<(usize, usize)> {
    let parent_name = file_to_module(db, file)?.name(db).parent()?;
    let parent_file = resolve_module(db, file, &parent_name)?.file(db)?;
    let self_idx = *module_nodes.get(&file)?;
    let parent_idx = *module_nodes.get(&parent_file)?;
    Some((self_idx, parent_idx))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_import_edges(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
    globals_by_name: &GlobalsByName,
    star_reexports: &StarReexports,
    dist_lookup: &DistLookup,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);

    // The "is this target idx a module node?" check below used to
    // rebuild this set on every iteration. Hoist it; the resolve_*
    // calls below can add new module nodes, so refresh only when
    // module_nodes actually grew.
    let mut module_idx_set: HashSet<usize> = module_nodes.values().copied().collect();
    let mut module_nodes_len = module_nodes.len();

    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let kind = def.kind(db);
        let place_id = def.place(db);
        let PlaceExprRef::Symbol(_) = place_table.place(place_id) else {
            continue;
        };

        let alias_idx =
            match global_index.get(&(file, place_id, range_key(kind.target_range(&parsed)))) {
                Some(&idx) => idx,
                None => continue,
            };

        let from_stmt = match kind {
            DefinitionKind::ImportFrom(k) => Some(k.import(&parsed)),
            DefinitionKind::ImportFromSubmodule(k) => Some(k.import(&parsed)),
            DefinitionKind::StarImport(k) => Some(k.import(&parsed)),
            _ => None,
        };

        let targets: Vec<usize> = match kind {
            DefinitionKind::Import(k) => resolve_import_target(
                py,
                db,
                k.alias(&parsed).name.id.as_str(),
                file,
                builder,
                module_nodes,
                dist_lookup,
            )?
            .into_iter()
            .collect(),
            DefinitionKind::ImportFrom(k) => resolve_from_imported(
                py,
                db,
                file,
                k.import(&parsed),
                k.alias(&parsed).name.id.as_str(),
                builder,
                module_nodes,
                globals_by_name,
                star_reexports,
                dist_lookup,
            )?,
            DefinitionKind::ImportFromSubmodule(k) => resolve_from_imported(
                py,
                db,
                file,
                k.import(&parsed),
                k.module(&parsed).id.as_str(),
                builder,
                module_nodes,
                globals_by_name,
                star_reexports,
                dist_lookup,
            )?,
            // `from X import *` — no per-name fan-out edge. ty's
            // `definitions_for_imported_symbol` would resolve each
            // per-name `StarImport` to its specific upstream decl,
            // but we collapse the per-name aliases to one node per
            // statement (see `ingest_decls`) and don't want N edges
            // pointing at the same alias. The from-style parallel
            // path below still emits `alias → upstream module` once,
            // which is exactly what carries reachability under the
            // new model — uses of star-bound names emit their own
            // parallel `use → upstream module / upstream decl` edges
            // via `emit_upstream` and ty's name resolution.
            DefinitionKind::StarImport(_) => Vec::new(),
            _ => continue,
        };
        if module_nodes.len() != module_nodes_len {
            module_idx_set.extend(module_nodes.values().copied());
            module_nodes_len = module_nodes.len();
        }
        let all_targets_are_modules =
            !targets.is_empty() && targets.iter().all(|idx| module_idx_set.contains(idx));
        for target_idx in &targets {
            builder.add_edge(alias_idx, *target_idx, 0);
        }

        // Parallel reachability edge: when `from X import Y` resolved
        // to a *decl* (Y in module X), also link the alias to the
        // upstream module X so reachability can see the file's
        // module-level dependency. Skip when *every* resolved target
        // is itself a module (the `from X import sub` case where
        // `sub` is a submodule of X) — the submodule's parent-module
        // edge already keeps X alive. When at least one target is a
        // decl, the upstream-module edge is still useful for the
        // others' reachability.
        if let Some(stmt) = from_stmt {
            if !all_targets_are_modules {
                if let Some(upstream_module_idx) =
                    from_import_module_node(py, db, file, stmt, builder, module_nodes, dist_lookup)?
                {
                    if !targets.contains(&upstream_module_idx) && upstream_module_idx != alias_idx {
                        builder.add_edge(alias_idx, upstream_module_idx, 0);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Resolve a dotted module name to its target module node.
///
/// For `import a.b.c`, the alias binds the local name "a" to the
/// deepest module (`a.b.c`) — that's what this returns. Submodule
/// hierarchy edges are only emitted for project modules via
/// [`emit_module_hierarchy`]; external chains are not modeled today.
/// Classification of an import target into one of four graph shapes.
///
/// Mirrors ``dead_cst.resolvers._imports.default_resolve_import``'s
/// stdlib / external-dist / external-file / unresolved buckets from
/// the libcst pipeline. The rust path only needs three of those
/// outcomes — ``[stdlib]`` is silent (no node minted), ``FirstParty``
/// mints a real ``"module"`` node, and ``External`` / ``Unresolved``
/// mint deduplicated ``"synthetic"`` nodes. The ``[external file]``
/// libcst distinction (editable installs) collapses into ``External``
/// here; ty's public ``SearchPath`` predicates don't separate them.
pub(crate) enum ImportTarget {
    Stdlib,
    FirstParty(File),
    /// A site-packages file owned by an installed distribution's
    /// ``RECORD``. Carries the PEP 503-canonical dist name (e.g.
    /// ``"pillow"`` for an ``import PIL``).
    ExternalDist(String),
    /// A site-packages file (or editable / extra search-path file)
    /// not claimed by any installed distribution's ``RECORD``.
    /// Carries the top-level module name. Mirrors libcst's
    /// ``[external file]`` bucket for editable installs and orphan
    /// files in ``site-packages``.
    ExternalFile(String),
    Unresolved(String),
}

impl ImportTarget {
    /// Synthetic node fqname for the non-first-party variants, or
    /// `None` for `Stdlib` / `FirstParty`. Single source of truth for
    /// the ``[external dist] X`` / ``[external file] X`` /
    /// ``[unresolved] X`` synthetic prefixes — both mint
    /// (``target_to_node``) and lookup (``emit_upstream``) go through
    /// this so the format strings can't drift apart.
    pub(crate) fn synthetic_fqname(&self) -> Option<String> {
        match self {
            ImportTarget::Stdlib | ImportTarget::FirstParty(_) => None,
            ImportTarget::ExternalDist(name) => Some(format!("[external dist] {name}")),
            ImportTarget::ExternalFile(name) => Some(format!("[external file] {name}")),
            ImportTarget::Unresolved(name) => Some(format!("[unresolved] {name}")),
        }
    }
}

/// ``abs_file_path -> PEP 503-canonical dist name``. Populated once
/// per ``materialize`` call by walking every ``*.dist-info/`` directory
/// under each ``is_site_packages()`` search path that ty knows about
/// and indexing the files listed in the distribution's ``RECORD``.
///
/// Mirrors ``dead_cst.resolvers._imports.distribution_lookup`` from
/// the libcst pipeline but reads dist-info directly off the
/// filesystem — no Python callback, so the same lookup works in
/// pyo3 contexts where issuing ``importlib.metadata.distributions()``
/// would round-trip through the runtime.
pub(crate) type DistLookup = HashMap<PathBuf, String>;

/// PEP 503 normalization. Replaces every run of ``[-_.]`` with a
/// single ``-`` and lowercases the rest. Equivalent to Python's
/// ``re.sub(r"[-_.]+", "-", name).lower()`` from libcst's
/// ``_canonical_dist_name``.
pub(crate) fn pep503_canonicalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for ch in name.chars() {
        if matches!(ch, '-' | '_' | '.') {
            if !prev_sep {
                out.push('-');
                prev_sep = true;
            }
        } else {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        }
    }
    out
}

/// Site-packages roots ty's resolver is configured with, canonicalised
/// upfront so the worker thread that runs ``build_dist_lookup`` doesn't
/// need to borrow ``db`` (Salsa's ``ProjectDatabase`` is !Sync).
pub(crate) fn site_packages_roots(db: &dyn ty_python_semantic::Db) -> Vec<PathBuf> {
    search_paths(db, ModuleResolveMode::StubsAllowed)
        .filter(|sp| sp.is_site_packages())
        .filter_map(|sp| sp.as_system_path().map(|p| p.as_str()))
        .map(|s| std::fs::canonicalize(s).unwrap_or_else(|_| PathBuf::from(s)))
        .collect()
}

/// Build the dist-file lookup by walking ``*.dist-info/`` under every
/// site-packages search path.
///
/// For each dist-info directory:
///
/// 1. Read ``METADATA`` for the ``Name:`` header (the project's PyPI
///    name pre-canonicalization). Stop at the blank line that
///    separates headers from the long description.
/// 2. Apply PEP 503 normalization so casing / separators don't matter
///    at lookup time.
/// 3. Read ``RECORD`` for the comma-separated ``path,hash,size``
///    entries, take the path (relative to the site-packages dir),
///    join + canonicalize against the dist-info parent, and map each
///    resolved absolute path to the canonical name.
///
/// Failures (missing METADATA, malformed RECORD, unreadable site-
/// packages) are silently skipped — the file just won't appear in
/// the lookup and the caller falls through to ``[external file]``.
///
/// Per-dist work is dispatched across a rayon pool: a typical venv
/// has dozens of dists and the per-RECORD ``canonicalize`` syscalls
/// (~2.5k in dead-cst's own dev venv) dominate everything else.
pub(crate) fn build_dist_lookup(site_packages: &[PathBuf]) -> DistLookup {
    use rayon::prelude::*;

    site_packages
        .iter()
        .flat_map(|sp_root| {
            std::fs::read_dir(sp_root)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "dist-info"))
                .map(move |dist_info| (dist_info, sp_root.as_path()))
        })
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|(dist_info, sp_root)| process_dist_info(dist_info, sp_root))
        .flatten()
        .collect()
}

/// One ``*.dist-info/`` directory's ``(absolute file path, canonical
/// dist name)`` pairs. ``None`` when METADATA / RECORD are missing or
/// the ``Name:`` header isn't found.
fn process_dist_info(dist_info: &Path, sp_root: &Path) -> Option<Vec<(PathBuf, String)>> {
    let metadata = std::fs::read_to_string(dist_info.join("METADATA")).ok()?;
    let canonical = metadata
        .lines()
        .take_while(|l| !l.is_empty())
        .find_map(|line| {
            line.strip_prefix("Name:")
                .map(|v| pep503_canonicalize(v.trim()))
        })?;
    let record = std::fs::read_to_string(dist_info.join("RECORD")).ok()?;
    Some(
        record
            .lines()
            .filter_map(|line| {
                let (rel, _) = line.split_once(',')?;
                let joined = sp_root.join(rel);
                let abs = std::fs::canonicalize(&joined).unwrap_or(joined);
                Some((abs, canonical.clone()))
            })
            .collect(),
    )
}

/// Classify a dotted module name relative to the file that imports it.
///
/// * Resolved + stdlib → ``Stdlib``
/// * Resolved + first-party + file present → ``FirstParty(file)``
/// * Resolved + non-first-party, file in some dist's ``RECORD`` →
///   ``ExternalDist(canonical-dist-name)``
/// * Resolved + non-first-party, file not in any ``RECORD`` →
///   ``ExternalFile(top-level-module-name)``
/// * Resolved + namespace package (no file) → ``Unresolved(name)``
/// * Unresolved → ``Unresolved(top-level-module-name)``, with a fix-up
///   for dotted-child stdlib misses (``collections.abc`` shouldn't
///   surface as ``[unresolved] collections`` when ``collections`` itself
///   is stdlib but ty's resolver returned None for the child).
pub(crate) fn classify_import_target(
    db: &dyn ty_python_semantic::Db,
    importing_file: File,
    module_name: &ModuleName,
    dist_lookup: &DistLookup,
) -> ImportTarget {
    let dotted = module_name.as_str();
    let top_level = dotted.split('.').next().unwrap_or(dotted);
    let Some(module) = resolve_module(db, importing_file, module_name) else {
        // Dotted-child miss: if the top-level resolves as stdlib,
        // inherit (matches libcst's parent-fallback behavior for
        // ``collections.abc``-shaped names).
        if dotted != top_level {
            if let Some(top_mn) = ModuleName::new(top_level) {
                if resolve_module(db, importing_file, &top_mn)
                    .and_then(|m| m.search_path(db))
                    .is_some_and(|sp| sp.is_standard_library())
                {
                    return ImportTarget::Stdlib;
                }
            }
        }
        return ImportTarget::Unresolved(top_level.to_string());
    };
    let search_path = module.search_path(db);
    if search_path.is_some_and(|sp| sp.is_standard_library()) {
        return ImportTarget::Stdlib;
    }
    if search_path.is_some_and(|sp| sp.is_first_party()) {
        return match module.file(db) {
            Some(f) => ImportTarget::FirstParty(f),
            None => ImportTarget::Unresolved(top_level.to_string()),
        };
    }
    // Non-first-party, non-stdlib: site-packages, editable, extra,
    // or namespace package. Probe the dist-RECORD lookup for the
    // resolved file's canonical name; fall back to ``[external file]``
    // when the file isn't owned by any installed distribution.
    let Some(file) = module.file(db) else {
        return ImportTarget::Unresolved(top_level.to_string());
    };
    let path_str = match file.path(db) {
        FilePath::System(p) => p.to_string(),
        _ => return ImportTarget::ExternalFile(top_level.to_string()),
    };
    let path = std::fs::canonicalize(&path_str).unwrap_or_else(|_| PathBuf::from(&path_str));
    if let Some(canonical) = dist_lookup.get(&path) {
        ImportTarget::ExternalDist(canonical.clone())
    } else {
        ImportTarget::ExternalFile(top_level.to_string())
    }
}

/// Mint (or look up) the graph node a target classification should
/// resolve to. ``Stdlib`` returns ``None`` (silent drop); the other
/// four return a node index.
pub(crate) fn target_to_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    target: ImportTarget,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<Option<usize>> {
    match target {
        ImportTarget::Stdlib => Ok(None),
        ImportTarget::FirstParty(file) => {
            Ok(Some(mint_module_node(py, db, file, builder, module_nodes)?))
        }
        ref t @ (ImportTarget::ExternalDist(_)
        | ImportTarget::ExternalFile(_)
        | ImportTarget::Unresolved(_)) => {
            let fqname = t.synthetic_fqname().expect("non-stdlib non-first-party");
            Ok(Some(builder.intern_synthetic(py, fqname)?))
        }
    }
}

pub(crate) fn resolve_import_target(
    py: Python<'_>,
    db: &ProjectDatabase,
    dotted: &str,
    importing_file: File,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    dist_lookup: &DistLookup,
) -> PyResult<Option<usize>> {
    let Some(module_name) = ModuleName::new(dotted) else {
        return Ok(None);
    };
    let target = classify_import_target(db, importing_file, &module_name, dist_lookup);
    target_to_node(py, db, target, builder, module_nodes)
}

/// Resolve `from <stmt> import <symbol>` to its upstream target node.
///
/// Mirrors CPython's `_handle_fromlist` (`Lib/importlib/_bootstrap.py`):
/// check the package's namespace first, fall back to a submodule only
/// when nothing's bound. Concretely:
///
/// 1. **Namespace lookup** — walks `globals_by_name` for the target
///    module, through any star-reexport chain (`A → B → C`), until it
///    hits a non-reexport binding (a decl or non-star import alias).
///    This is CPython's `hasattr(module, name)` step: if a name is
///    bound in `p/__init__.py` (whether by `q = 42` or by
///    `from . import q`), that binding wins — *even* if a submodule
///    `p/q.py` also exists. The shadow case
///    (`from other import *; def g():` in mod) also lands here via
///    last-write-wins `globals_by_name`.
/// 2. **Submodule fallback** — `<module>.<symbol>` resolves as a
///    module (`from p import q` where `p/q.py` exists and nothing is
///    bound to `q` in `p/__init__.py`). This is CPython's
///    `__import__(f"{p}.{name}")` branch.
/// 3. **Module fallback** — alias still gets an out-edge to the
///    upstream module so reachability propagates.
///
/// Why namespace-first (the CPython order) matters: for `from p
/// import q` where `p/__init__.py` does `q = 42` *and* `p/q.py`
/// exists, CPython binds `q` to the int — the submodule never
/// executes. A submodule-first analyzer would wrongly keep `p/q.py`
/// alive and miss the real binding. The reorder fixes this case and
/// agrees with CPython semantics on every other case (including
/// `from . import q` aliases that bind the submodule to the package
/// namespace explicitly).
///
/// Deliberately does NOT use ty's `definitions_for_imported_symbol`,
/// which recursively chases alias chains across files. Per Principle 2
/// every alias is its own graph node with an outgoing edge, so the
/// transitive walk is already encoded in the graph — replicating it
/// here cost ~100µs per from-import (94% of Phase 2 on flux0 workspace)
/// for no extra reachability information.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_from_imported(
    py: Python<'_>,
    db: &ProjectDatabase,
    importing_file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
    symbol_name: &str,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    globals_by_name: &GlobalsByName,
    star_reexports: &StarReexports,
    dist_lookup: &DistLookup,
) -> PyResult<Vec<usize>> {
    let Ok(module_name) = ModuleName::from_import_statement(db, importing_file, stmt) else {
        return Ok(Vec::new());
    };

    let target = classify_import_target(db, importing_file, &module_name, dist_lookup);
    // External / unresolved targets short-circuit at the module level —
    // we don't have file-level namespace info, so the alias edges to
    // the ``[external dist] X`` / ``[external file] X`` / ``[unresolved] X``
    // synthetic. Stdlib drops silently.
    let target_file = match target {
        ImportTarget::Stdlib => return Ok(Vec::new()),
        ImportTarget::FirstParty(f) => f,
        ImportTarget::ExternalDist(_)
        | ImportTarget::ExternalFile(_)
        | ImportTarget::Unresolved(_) => {
            return Ok(target_to_node(py, db, target, builder, module_nodes)?
                .into_iter()
                .collect());
        }
    };

    // 1. Probe the first-party target's namespace — CPython's
    //    ``hasattr(module, name)``. May return multiple bindings when
    //    the name is branch-bound (try/except, if/else with both
    //    branches assigning).
    let chain = walk_globals_chain(
        db,
        target_file,
        symbol_name,
        globals_by_name,
        star_reexports,
    );
    if !chain.is_empty() {
        return Ok(chain);
    }

    // 2. Namespace miss — fall back to importing `<module>.<symbol>`
    //    as a submodule. CPython's `__import__(f"{p}.{name}")`.
    if let Some(submodule_name) = ModuleName::new(symbol_name) {
        let mut combined = module_name.clone();
        combined.extend(&submodule_name);
        let sub_target = classify_import_target(db, importing_file, &combined, dist_lookup);
        if !matches!(sub_target, ImportTarget::Unresolved(_)) {
            // Stdlib silent-drop / first-party submodule / external
            // submodule all land here. ``Unresolved`` falls through to
            // step 3 so we link to the original module rather than
            // minting ``[unresolved] symbol`` for what's really just
            // "X has no attribute symbol".
            return Ok(target_to_node(py, db, sub_target, builder, module_nodes)?
                .into_iter()
                .collect());
        }
    }

    // 3. Fallback: link the alias to the upstream module node.
    Ok(vec![mint_module_node(
        py,
        db,
        target_file,
        builder,
        module_nodes,
    )?])
}

/// Walk `(file, name)` through star-reexport chains. Returns every
/// non-star-reexport binding reachable from `(target_file, symbol_name)`
/// — a decl, a non-star import alias, or several of either when the
/// name has multiple live bindings (try/except, if/else where both
/// branches assign).
///
/// `from A import g` where A has `from B import *` lands on A's star
/// alias for `g`; we resolve B → file, look up `g` there, and recurse.
/// Stops on a decl, on a non-star import, on a missed lookup (yields
/// nothing past it), or on a cycle (revisit of an already-seen key).
pub(crate) fn walk_globals_chain(
    db: &ProjectDatabase,
    target_file: File,
    symbol_name: &str,
    globals_by_name: &GlobalsByName,
    star_reexports: &StarReexports,
) -> Vec<usize> {
    let mut seen: HashSet<(File, String)> = HashSet::new();
    let mut out: Vec<usize> = Vec::new();
    let mut stack: Vec<(File, String)> = vec![(target_file, symbol_name.to_string())];
    while let Some(key) = stack.pop() {
        if !seen.insert(key.clone()) {
            continue;
        }
        let Some(idxs) = globals_by_name.get(&key) else {
            continue;
        };
        // If `key` is a `from <upstream> import *` reexport, step
        // into the upstream file's same-name lookup. Star reexports
        // carry the name unchanged. Otherwise the binding(s) are the
        // terminal answer.
        if let Some(upstream_module) = star_reexports.get(&key) {
            if let Some(mn) = ModuleName::new(upstream_module) {
                if let Some(upstream) = resolve_module(db, key.0, &mn) {
                    if let Some(upstream_file) = upstream.file(db) {
                        stack.push((upstream_file, key.1.clone()));
                        // The star-alias node itself isn't a useful
                        // target — uses should land on the upstream
                        // decl. Skip emitting it.
                        continue;
                    }
                }
            }
        }
        for &idx in idxs {
            out.push(idx);
        }
    }
    out
}

/// Resolve the upstream module of a `from <stmt> import ...` and
/// return (or mint) its module node.
///
/// Returns `Ok(None)` when ty's `from_import_statement` cannot resolve
/// the target (invalid syntax, too many leading dots, missing file).
pub(crate) fn from_import_module_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    importing_file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    dist_lookup: &DistLookup,
) -> PyResult<Option<usize>> {
    let Ok(module_name) = ModuleName::from_import_statement(db, importing_file, stmt) else {
        return Ok(None);
    };
    let target = classify_import_target(db, importing_file, &module_name, dist_lookup);
    target_to_node(py, db, target, builder, module_nodes)
}

pub(crate) fn mint_module_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<usize> {
    if let Some(&idx) = module_nodes.get(&file) {
        return Ok(idx);
    }
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let (sl, sc, el, ec) = position(&line_index, &source, parsed.syntax().range);
    let fqname = module_fqname_for_file(db, file);
    let path_str = file_path_string(db, file);
    let flags = file_default_flags(db, file);
    let idx = builder.intern_node(
        py,
        SymbolNode {
            fqname,
            kind: "module",
            path: path_str,
            start_line: sl,
            start_column: sc,
            end_line: el,
            end_column: ec,
            flags,
            imports: None,
            cached_hash: OnceLock::new(),
        },
    )?;
    module_nodes.insert(file, idx);
    Ok(idx)
}

// ---------------------------------------------------------------------------
// Phase 3: same-file Name→decl reference edges
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_reference_edges(
    db: &ProjectDatabase,
    file: File,
    global_index: &DeclIndex,
    module_nodes: &HashMap<File, usize>,
    alias_imports: &HashMap<usize, ImportSpec>,
    live_decls: &LiveDeclIndex,
    dist_lookup: &DistLookup,
    builder: &mut GraphBuilder,
) {
    let Some(&module_idx) = module_nodes.get(&file) else {
        return;
    };
    let parsed = parsed_module(db, file).load(db);
    let dead_ranges = detect_dead_ranges(&parsed);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let use_def_map = index.use_def_map(global);
    let model = SemanticModel::new(db, file);
    // Move ``synthetic_nodes`` out of the builder for the duration of
    // this pass: the per-statement walks need a long-lived immutable
    // borrow while ``coll.flush(builder)`` takes ``&mut builder``.
    // The synthetic map is populated by ``emit_import_edges`` ahead of
    // this phase and isn't mutated here, so swap-out / swap-in is safe.
    let synthetic_nodes = std::mem::take(&mut builder.synthetic_nodes);

    // (a) Definitions that own an expression / body — attribute their
    //     contained Names to the owning decl.
    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let kind = def.kind(db);
        let target_range = kind.target_range(&parsed);
        let Some(&owner_idx) = global_index.get(&(file, def.place(db), range_key(target_range)))
        else {
            continue;
        };

        let mut coll = RefCollector::new(
            owner_idx,
            &model,
            file,
            &parsed,
            index,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            &synthetic_nodes,
            dist_lookup,
            &dead_ranges,
        );
        walk_owned(kind, &parsed, &mut coll);
        coll.flush(builder);
    }

    // (b) Module-level statements that don't carry a Definition (and
    //     so didn't get covered by (a)) attribute to the module node.
    for stmt in &parsed.syntax().body {
        if stmt_creates_top_level_definition(stmt) {
            continue;
        }
        let mut coll = RefCollector::new(
            module_idx,
            &model,
            file,
            &parsed,
            index,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            &synthetic_nodes,
            dist_lookup,
            &dead_ranges,
        );
        coll.visit_stmt(stmt);
        coll.flush(builder);
    }
    builder.synthetic_nodes = synthetic_nodes;
}

/// True iff this top-level statement is a binding form whose Names
/// have already been attributed by the per-definition walk in (a).
///
/// Compound non-scope statements (``if`` / ``while`` / ``for`` / ...)
/// return ``false`` here: their *bodies* contain definitions that (a)
/// covers, but their *test/iter/etc. expressions* belong to the module
/// and (b) needs to walk them.
pub(crate) fn stmt_creates_top_level_definition(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::FunctionDef(_)
            | Stmt::ClassDef(_)
            | Stmt::Assign(_)
            | Stmt::AnnAssign(_)
            | Stmt::TypeAlias(_)
            | Stmt::Import(_)
            | Stmt::ImportFrom(_)
    )
}

/// Walk every value-bearing AST node a Definition owns.
///
/// Functions and classes own their body statements; assignments own
/// the RHS expression; annotated assignments own annotation + value;
/// `for x in iter:` owns the iterable; `with X as y:` owns the
/// context expression; walrus owns its value; type aliases own their
/// value expression. Other Definition kinds (imports, parameters, …)
/// own no walk-worthy expression.
pub(crate) fn walk_owned(
    kind: &DefinitionKind<'_>,
    parsed: &ParsedModuleRef,
    v: &mut RefCollector<'_, '_>,
) {
    match kind {
        DefinitionKind::Function(func) => {
            let node = func.node(parsed);
            // Header parts evaluate at the *definition* site (module
            // scope for top-level defs), not inside the body — leave
            // `nested_context` false so a stray `import X` in a
            // decorator expression doesn't get re-attributed as a
            // body-local nested import.
            walk_decorators(&node.decorator_list, v);
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            walk_parameters(&node.parameters, v);
            if let Some(returns) = &node.returns {
                v.visit_expr(returns);
            }
            v.nested_context = true;
            for s in &node.body {
                v.visit_stmt(s);
            }
            v.nested_context = false;
        }
        DefinitionKind::Class(cls) => {
            let node = cls.node(parsed);
            walk_decorators(&node.decorator_list, v);
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            if let Some(args) = &node.arguments {
                for base in &args.args {
                    v.visit_expr(base);
                }
                for kw in &args.keywords {
                    v.visit_expr(&kw.value);
                }
            }
            v.nested_context = true;
            for s in &node.body {
                v.visit_stmt(s);
            }
            v.nested_context = false;
        }
        DefinitionKind::Assignment(a) => {
            let value = a.value(parsed);
            if target_is_dunder_all(a.target(parsed)) {
                emit_dunder_all_edges(v, value);
            } else if let TargetKind::Sequence(_, unpack) = a.target_kind() {
                // ``c, d = a, b`` produces one Definition per LHS
                // name, each with ``value`` set to the whole RHS
                // ``(a, b)``. Walking the full RHS for both ``c`` and
                // ``d`` over-approximates (``c -> b``, ``d -> a``);
                // when both sides are flat sequences of matching
                // arity, pair index-by-index instead.
                let db = v.model.db();
                let lhs = unpack.target(db, parsed);
                if let Some(paired) = paired_unpack_rhs(lhs, a.target(parsed), value) {
                    v.visit_expr(paired);
                } else {
                    v.visit_expr(value);
                }
            } else {
                v.visit_expr(value);
            }
        }
        DefinitionKind::AnnotatedAssignment(a) => {
            v.visit_expr(a.annotation(parsed));
            if let Some(val) = a.value(parsed) {
                if target_is_dunder_all(a.target(parsed)) {
                    emit_dunder_all_edges(v, val);
                } else {
                    v.visit_expr(val);
                }
            }
        }
        DefinitionKind::NamedExpression(named) => v.visit_expr(named.node(parsed).value.as_ref()),
        DefinitionKind::TypeAlias(alias) => {
            let node = alias.node(parsed);
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            v.visit_expr(node.value.as_ref());
        }
        // For / WithItem / ExceptHandler / MatchPattern bindings are
        // not modeled as top-level decls (see `decl_kind_str`), so
        // their definitions never appear in `global_index` and
        // `walk_owned` never runs for them. Their value-bearing
        // sub-expressions (loop iterables, context managers, etc.)
        // are walked instead from the module-level non-definition
        // pass, where `Stmt::For` / `Stmt::With` / `Stmt::Try` get
        // their normal `walk_stmt` recursion.
        _ => {}
    }
}

pub(crate) fn walk_decorators(
    decorators: &[ruff_python_ast::Decorator],
    v: &mut RefCollector<'_, '_>,
) {
    for d in decorators {
        v.visit_expr(&d.expression);
    }
}

pub(crate) fn walk_parameters(
    parameters: &ruff_python_ast::Parameters,
    v: &mut RefCollector<'_, '_>,
) {
    for p in parameters
        .posonlyargs
        .iter()
        .chain(&parameters.args)
        .chain(&parameters.kwonlyargs)
    {
        if let Some(annotation) = &p.parameter.annotation {
            v.visit_expr(annotation);
        }
        if let Some(default) = &p.default {
            v.visit_expr(default);
        }
    }
    if let Some(vararg) = &parameters.vararg {
        if let Some(annotation) = &vararg.annotation {
            v.visit_expr(annotation);
        }
    }
    if let Some(kwarg) = &parameters.kwarg {
        if let Some(annotation) = &kwarg.annotation {
            v.visit_expr(annotation);
        }
    }
}

pub(crate) fn walk_type_params(
    type_params: &ruff_python_ast::TypeParams,
    v: &mut RefCollector<'_, '_>,
) {
    for tp in &type_params.type_params {
        match tp {
            ruff_python_ast::TypeParam::TypeVar(t) => {
                if let Some(bound) = &t.bound {
                    v.visit_expr(bound);
                }
                if let Some(default) = &t.default {
                    v.visit_expr(default);
                }
            }
            ruff_python_ast::TypeParam::TypeVarTuple(t) => {
                if let Some(default) = &t.default {
                    v.visit_expr(default);
                }
            }
            ruff_python_ast::TypeParam::ParamSpec(t) => {
                if let Some(default) = &t.default {
                    v.visit_expr(default);
                }
            }
        }
    }
}

/// Pair a tuple-unpack target with the RHS element at the same
/// position, so each LHS name only collects edges from the value it
/// actually receives.
///
/// Returns the paired RHS element when:
/// * the full LHS (the parent unpack target) is a flat `Tuple` /
///   `List` literal with no starred element, and
/// * the RHS is a flat `Tuple` / `List` literal with the same arity,
///   and
/// * the bare `target` name appears as one of the LHS elements.
///
/// Returns `None` otherwise so the caller falls back to walking the
/// whole RHS (the safe over-approximation for non-literal RHS values
/// like `c, d = f()` and starred-target patterns).
pub(crate) fn paired_unpack_rhs<'ast>(
    lhs: &Expr,
    target: &Expr,
    rhs: &'ast Expr,
) -> Option<&'ast Expr> {
    let lhs_elts = match lhs {
        Expr::Tuple(t) => &t.elts,
        Expr::List(l) => &l.elts,
        _ => return None,
    };
    let rhs_elts = match rhs {
        Expr::Tuple(t) => &t.elts,
        Expr::List(l) => &l.elts,
        _ => return None,
    };
    if lhs_elts.len() != rhs_elts.len() {
        return None;
    }
    if lhs_elts.iter().any(|e| matches!(e, Expr::Starred(_))) {
        return None;
    }
    let target_range = target.range();
    let pos = lhs_elts.iter().position(|e| e.range() == target_range)?;
    Some(&rhs_elts[pos])
}

/// True iff `target` is the bare `Name("__all__")` LHS of an
/// assignment (`__all__ = ...` or `__all__: list[str] = ...`).
///
/// Subscript / attribute / tuple-unpack targets (e.g.
/// `__all__[0] = "f"`, `mod.__all__ = [...]`) are excluded — those
/// don't redefine the binding in a way the libcst pipeline picks up,
/// and treating them as `__all__` would emit edges from unrelated
/// nodes.
pub(crate) fn target_is_dunder_all(target: &Expr) -> bool {
    matches!(target, Expr::Name(n) if n.id.as_str() == "__all__")
}

/// Walk the value of an `__all__` assignment and emit one edge from
/// the owner (the `__all__` variable node) to each module-scope
/// binding whose name appears in the list/tuple.
///
/// Only string-literal elements are followed; computed entries (e.g.
/// `__all__ = [*BASE, "extra"]`, `__all__ = list(...)`) are silently
/// skipped — matching the libcst pipeline, which folds `__all__` only
/// when it's assigned a list or tuple of string literals. Names that
/// don't resolve in the file's global scope are skipped without a
/// warning (`__all__ = ["missing"]` is a runtime error at import
/// time, not a static dep).
pub(crate) fn emit_dunder_all_edges(v: &mut RefCollector<'_, '_>, value: &Expr) {
    let elements = match value {
        Expr::List(l) => &l.elts,
        Expr::Tuple(t) => &t.elts,
        _ => return,
    };
    for elem in elements {
        if let Expr::StringLiteral(s) = elem {
            if let Some(idx) = v.lookup_module_scope_name(s.value.to_str()) {
                v.emit_edge(idx);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference collector
// ---------------------------------------------------------------------------

/// Walks an expression / body and records every Name reference,
/// attributing to a single owner decl.
///
/// Per Principle 2, every use of an imported name emits an edge to
/// the local alias *and* parallel reachability edges to whatever the
/// alias resolves to upstream. Bare-Name uses (`f()`) emit edges to
/// the upstream module/decl directly. Attribute chains on aliased
/// modules (`foo.bar.f()`) are walked segment by segment, emitting
/// edges to each module / decl reached. Shadow handling falls out of
/// ty's flow-sensitive use-def chain (Principle 3).
pub(crate) struct RefCollector<'a, 'db> {
    owner: usize,
    model: &'a SemanticModel<'db>,
    file: File,
    parsed: &'a ParsedModuleRef,
    /// Cached `&SemanticIndex` for `self.file`. Avoids re-doing the
    /// Salsa `semantic_index(db, file)` lookup per Name reference —
    /// every call to `find_local_bindings` / `lookup_module_scope_name`
    /// used to issue one. ~10k saved lookups on a 100-file workspace.
    index: &'a SemanticIndex<'db>,
    global_index: &'a DeclIndex,
    module_nodes: &'a HashMap<File, usize>,
    alias_imports: &'a HashMap<usize, ImportSpec>,
    live_decls: &'a LiveDeclIndex,
    /// Ranges of statically-dead source regions in the current file
    /// (`if False:` bodies, statements after a `return`/`raise`/`break`/
    /// `continue`/`assert <falsy>`, etc.). A use whose source range
    /// is `contains_range`-covered by any of these gets
    /// `EdgeFlags::DEAD_BRANCH` stamped on every edge it emits.
    dead_ranges: &'a [TextRange],
    /// `{ synthetic_fqname -> node idx }` mirrored from
    /// ``GraphBuilder::synthetic_nodes`` so the collector can route
    /// upstream edges through pre-minted ``[external dist] X``,
    /// ``[external file] X``, and ``[unresolved] X`` synthetics
    /// without holding the mutable builder. Populated by
    /// ``emit_import_edges`` before the collector pass runs.
    synthetic_nodes: &'a HashMap<String, usize>,
    /// ``abs_file_path -> PEP 503-canonical dist name`` built once
    /// per ``materialize`` call. The collector re-classifies
    /// ``emit_upstream``'s loading target to compute the same
    /// synthetic fqname ``emit_import_edges`` used at mint time.
    dist_lookup: &'a DistLookup,
    /// Edges accumulated for this collector pass. The value is the
    /// AND of the flags contributed by each reference that produced
    /// this `(src, dst)` pair — so a `(src, dst)` reachable from both
    /// a live and a dead reference loses `DEAD_BRANCH` (the live ref
    /// wins), matching the libcst pipeline's parallel-edge semantics.
    edges: HashMap<(usize, usize), u32>,
    /// `true` while walking a function- or class-body subtree. In
    /// that context, nested `Stmt::Import` / `Stmt::ImportFrom`
    /// statements emit parallel upstream edges from `owner` (no
    /// alias node is minted, since the binding lives in a non-global
    /// scope). At module level the flag stays false, so we don't
    /// double-emit for imports that `emit_import_edges` already
    /// processed via their proper alias nodes.
    nested_context: bool,
    /// Flags stamped on each edge emitted by the current reference
    /// (set by `emit_name_use` / nested-import handlers based on the
    /// reference's source position; cleared afterward).
    current_flags: u32,
}

impl<'a, 'db> RefCollector<'a, 'db> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        owner: usize,
        model: &'a SemanticModel<'db>,
        file: File,
        parsed: &'a ParsedModuleRef,
        index: &'a SemanticIndex<'db>,
        global_index: &'a DeclIndex,
        module_nodes: &'a HashMap<File, usize>,
        alias_imports: &'a HashMap<usize, ImportSpec>,
        live_decls: &'a LiveDeclIndex,
        synthetic_nodes: &'a HashMap<String, usize>,
        dist_lookup: &'a DistLookup,
        dead_ranges: &'a [TextRange],
    ) -> Self {
        Self {
            owner,
            model,
            file,
            parsed,
            index,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            synthetic_nodes,
            dist_lookup,
            dead_ranges,
            edges: HashMap::new(),
            nested_context: false,
            current_flags: 0,
        }
    }

    fn flush(self, builder: &mut GraphBuilder) {
        for ((src, dst), flags) in self.edges {
            builder.add_edge(src, dst, flags);
        }
    }

    fn emit_edge(&mut self, dst: usize) {
        if dst != self.owner {
            let flags = self.current_flags;
            self.edges
                .entry((self.owner, dst))
                .and_modify(|f| *f &= flags)
                .or_insert(flags);
        }
    }

    /// Returns `EDGE_FLAG_DEAD_BRANCH` if `range` is contained in any
    /// dead-region recorded for this file, else `0`.
    fn flags_for_range(&self, range: TextRange) -> u32 {
        if self.dead_ranges.iter().any(|r| r.contains_range(range)) {
            EDGE_FLAG_DEAD_BRANCH
        } else {
            0
        }
    }

    /// Look up a name in the current file's global scope and return
    /// its end-of-scope live binding's graph node.
    ///
    /// Used by the `__all__` walk: each string literal listed there
    /// should resolve to a top-level decl (or import alias) bound in
    /// the same file. Names that don't resolve are silently skipped —
    /// `__all__ = ["missing"]` is a runtime error at import time but
    /// doesn't influence static dep tracking.
    fn lookup_module_scope_name(&self, name: &str) -> Option<usize> {
        let db = self.model.db();
        let global = FileScopeId::global();
        let place_table = self.index.place_table(global);
        let symbol_id = place_table.symbol_id(name)?;
        let use_def_map = self.index.use_def_map(global);
        for binding in use_def_map.end_of_scope_symbol_bindings(symbol_id) {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != self.file {
                continue;
            }
            let key = (
                self.file,
                def.place(db),
                range_key(def.kind(db).target_range(self.parsed)),
            );
            if let Some(&idx) = self.global_index.get(&key) {
                return Some(idx);
            }
        }
        None
    }

    /// Walk the use's scope chain looking for the reaching definitions
    /// of `name`. Returns one entry per reaching def — typically a
    /// single binding, but `try`/`except` branches that both bind the
    /// name leave multiple defs reaching the use and every one gets
    /// edges (Principle 3).
    ///
    /// The use's own scope is queried with
    /// :meth:`UseDefMap::bindings_at_use`, which is position-sensitive
    /// — for `a = 1; a = a + 1`, the RHS `a` resolves to the line-1
    /// def even though the line-2 def has already been registered in
    /// the scope's place table. For free variables (used in a scope
    /// that doesn't bind the name), walks outward through
    /// `visible_ancestor_scopes` and falls back to that scope's
    /// end-of-scope bindings.
    ///
    /// Resolutions split on whether the binding has a graph node:
    /// module-scope decls and import aliases give `Alias(idx)`;
    /// imports nested in a function/class scope (no graph node minted)
    /// give `NestedImport { spec, bound_name }` so the caller can fan
    /// out parallel upstream edges from the enclosing top-level owner.
    ///
    /// Deliberately does NOT use `definitions_for_name`: that walks
    /// past `from X import y` to the upstream definition in `X`,
    /// flattening the local alias edge Principle 2 requires.
    fn find_local_bindings(&self, name: &ExprName) -> Vec<Resolution> {
        let db = self.model.db();
        let Some(file_scope) = self.model.scope(name.into()) else {
            return Vec::new();
        };
        let mut first = true;
        for (scope_id, _scope) in self.index.visible_ancestor_scopes(file_scope) {
            let place_table = self.index.place_table(scope_id);
            let Some(symbol_id) = place_table.symbol_id(name.id.as_str()) else {
                first = false;
                continue;
            };
            let use_def_map = self.index.use_def_map(scope_id);
            // Position-sensitive query for the use's own scope; fall
            // back to end-of-scope bindings for enclosing scopes (where
            // the use isn't recorded under any specific position).
            let bindings = if first {
                let use_id = name.scoped_use_id(db, scope_id.to_scope_id(db, self.file));
                use_def_map.bindings_at_use(use_id)
            } else {
                use_def_map.end_of_scope_symbol_bindings(symbol_id)
            };
            let mut saw_binding = false;
            let mut results: Vec<Resolution> = Vec::new();
            for binding in bindings {
                let Some(def) = binding.binding.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                saw_binding = true;
                let kind = def.kind(db);
                let place_id = def.place(db);
                let key = (
                    self.file,
                    place_id,
                    range_key(kind.target_range(self.parsed)),
                );
                if let Some(&idx) = self.global_index.get(&key) {
                    results.push(Resolution::Alias(idx));
                    continue;
                }
                if kind.is_import() {
                    let PlaceExprRef::Symbol(sym) = place_table.place(place_id) else {
                        continue;
                    };
                    let bound_name = sym.name().as_str().to_string();
                    let spec = import_payload_for(kind, db, self.file, self.parsed);
                    results.push(Resolution::NestedImport { spec, bound_name });
                }
            }
            if saw_binding {
                return results;
            }
            // ``X: T`` (annotation-only assignment) is a *declaration*
            // in ty's model, not a runtime binding, so it doesn't
            // appear in ``bindings_at_use`` /
            // ``end_of_scope_symbol_bindings``. For dead-code purposes
            // an annotation-only decl is still the only "this name
            // exists" signal for that symbol, so when no bindings
            // reach the use, fall back to end-of-scope declarations.
            // Declarations aren't flow-sensitive the way bindings are,
            // so we don't need a position-keyed variant here.
            for declaration in use_def_map.end_of_scope_symbol_declarations(symbol_id) {
                let Some(def) = declaration.declaration.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                let kind = def.kind(db);
                let place_id = def.place(db);
                let key = (
                    self.file,
                    place_id,
                    range_key(kind.target_range(self.parsed)),
                );
                if let Some(&idx) = self.global_index.get(&key) {
                    results.push(Resolution::Alias(idx));
                }
            }
            if !results.is_empty() {
                return results;
            }
            first = false;
        }
        Vec::new()
    }

    /// Emit edges implied by a use of `name`.
    ///
    /// `extra_chain` is the list of attribute segments past the bare
    /// name (`[]` for a bare-name use, `["bar", "f"]` for
    /// `name.bar.f`).
    ///
    /// For module-scope bindings: emit `owner → alias` (codemod
    /// invariant), then parallel upstream reachability edges from
    /// `alias_imports[idx]` if the binding is an import. For nested
    /// imports there's no alias node — go straight to the parallel
    /// edges, using the spec ty handed us. When ty reports multiple
    /// reaching defs (if/else branches, try/except, …), we emit for
    /// each one.
    fn emit_name_use(&mut self, name: &ExprName, extra_chain: &[&str]) {
        // `Name`s in non-`Load` context are binding sites, not uses:
        // the `x` in `for x in data`, `with y as x`, `except E as x`,
        // and the LHS of `x = ...` / `del x`. Skipping them keeps the
        // graph free of spurious target → binding-source edges (and
        // mirrors the libcst pipeline, which only emits edges for
        // reads).
        if !matches!(name.ctx, ruff_python_ast::ExprContext::Load) {
            return;
        }
        self.current_flags = self.flags_for_range(name.range());
        for resolution in self.find_local_bindings(name) {
            match resolution {
                Resolution::Alias(alias_idx) => {
                    self.emit_edge(alias_idx);
                    if let Some(spec) = self.alias_imports.get(&alias_idx).cloned() {
                        self.emit_upstream(&spec, name.id.as_str(), extra_chain);
                    }
                }
                Resolution::NestedImport { spec, bound_name } => {
                    self.emit_upstream(&spec, &bound_name, extra_chain);
                }
            }
        }
        self.current_flags = 0;
    }

    fn emit_upstream(&mut self, spec: &ImportSpec, bound_name: &str, extra_chain: &[&str]) {
        if spec.module.is_empty() {
            return;
        }

        // Decide the "loading target" — the absolute dotted module
        // the alias makes available — plus any extra prefix segments
        // the import statement walked deeper than the bound name.
        //
        // `import M[.Y.Z]` no-asname: bound name matches `M` (first
        //   segment). The runtime object of `M` is just the package
        //   `M`, but the import statement *loaded* `M.Y.Z`. The
        //   user's attribute chain typically re-traverses `.Y.Z`
        //   before walking beyond; we peel that prefix off so the
        //   walk continues from the deepest loaded module.
        // `import M[.Y.Z] as A`: bound name is `A`; loading target
        //   is `M.Y.Z` and there's no prefix to strip.
        // `from P import D[as A]`: try `P.D` as submodule; if it
        //   resolves the alias represents that submodule, else it
        //   binds the decl `D` in `P`.
        // `from P import *` (per-name synthetic): bound name is the
        //   name brought in; alias binds either `P.<name>` (submodule)
        //   or decl `<name>` in `P`.
        let db = self.model.db();
        let module_first_seg = spec.module.split('.').next().unwrap_or("").to_string();

        let mut adjusted_chain: Vec<&str> = extra_chain.to_vec();
        let loading_target: String;
        let mut decl_tail: Option<String> = None;

        if spec.star {
            // Star reexport: alias's `module` is the source package and
            // `bound_name` is one of the names it exports. Resolve as
            // either submodule `module.bound_name` or decl `bound_name`
            // in `module`.
            let candidate = format!("{}.{}", spec.module, bound_name);
            if module_name_resolves(&candidate, self.file, db) {
                loading_target = candidate;
            } else {
                loading_target = spec.module.clone();
                decl_tail = Some(bound_name.to_string());
            }
        } else {
            match &spec.decl {
                Some(decl) => {
                    let candidate = format!("{}.{}", spec.module, decl);
                    if module_name_resolves(&candidate, self.file, db) {
                        loading_target = candidate;
                    } else {
                        loading_target = spec.module.clone();
                        decl_tail = Some(decl.clone());
                    }
                }
                None => {
                    let no_asname = bound_name == module_first_seg;
                    if no_asname && spec.module != module_first_seg {
                        // `import M.Y.Z` no-asname: peel the loading
                        // prefix off the chain before walking past.
                        let loading_extras: Vec<&str> = spec.module.split('.').skip(1).collect();
                        let n = loading_extras.len();
                        let prefix_matches = adjusted_chain.len() >= n
                            && adjusted_chain
                                .iter()
                                .take(n)
                                .zip(&loading_extras)
                                .all(|(a, b)| *a == *b);
                        if prefix_matches {
                            adjusted_chain.drain(..n);
                            loading_target = spec.module.clone();
                        } else {
                            // User reached for something off the bare
                            // `M` module (`import M.Y.Z; M.other`).
                            // Walk from `M`, not `M.Y.Z`.
                            loading_target = module_first_seg;
                        }
                    } else {
                        loading_target = spec.module.clone();
                    }
                }
            }
        }

        // Classify the loading target. Stdlib drops silently;
        // external / unresolved fan out to the pre-minted synthetic
        // and stop (no submodule chain walk through them).
        let Some(start_mn) = ModuleName::new(&loading_target) else {
            return;
        };
        let target = classify_import_target(db, self.file, &start_mn, self.dist_lookup);
        let start_file = match target {
            ImportTarget::Stdlib => return,
            ImportTarget::FirstParty(f) => f,
            t @ (ImportTarget::ExternalDist(_)
            | ImportTarget::ExternalFile(_)
            | ImportTarget::Unresolved(_)) => {
                if let Some(fqname) = t.synthetic_fqname() {
                    if let Some(&idx) = self.synthetic_nodes.get(&fqname) {
                        self.emit_edge(idx);
                    }
                }
                return;
            }
        };

        // Decl-style alias: emit edges to the upstream module and the
        // decl inside it, then stop. Attribute access past a decl is
        // field access on the decl's value, which we don't model.
        if let Some(decl_name) = decl_tail {
            if let Some(&idx) = self.module_nodes.get(&start_file) {
                self.emit_edge(idx);
            }
            if let Some(idxs) = self.live_decls.get(&(start_file, decl_name)) {
                for &idx in idxs {
                    self.emit_edge(idx);
                }
            }
            return;
        }

        // Module-style alias: walk the chain submodule-by-submodule
        // and record the *deepest* module reached plus any decl the
        // chain ends on. We emit exactly one module edge (the deepest)
        // and at most one decl edge — mirroring the libcst stitcher's
        // canonicalization rule that pushes decl parts into the module
        // as long as they resolve as submodules.
        let mut current_file = start_file;
        let mut current_path = loading_target.clone();
        let mut terminal_decl_idxs: Vec<usize> = Vec::new();
        for seg in &adjusted_chain {
            let candidate = format!("{current_path}.{seg}");
            let submodule_file = ModuleName::new(&candidate)
                .and_then(|mn| resolve_module(db, self.file, &mn))
                .and_then(|m| m.file(db));
            if let Some(sub_file) = submodule_file {
                current_file = sub_file;
                current_path = candidate;
                continue;
            }
            terminal_decl_idxs = self
                .live_decls
                .get(&(current_file, (*seg).to_string()))
                .cloned()
                .unwrap_or_default();
            break;
        }

        if let Some(&idx) = self.module_nodes.get(&current_file) {
            self.emit_edge(idx);
        }
        for idx in terminal_decl_idxs {
            self.emit_edge(idx);
        }
    }

    /// Handle `import X[.Y.Z][ as A]` inside a function/class body.
    ///
    /// No alias node is minted — the binding lives in a non-global
    /// scope. We emit parallel upstream edges directly from `self.owner`
    /// for each alias in the statement, simulating a `synthetic chain`
    /// that matches the loading prefix so `emit_upstream` walks all
    /// the way to the deepest loaded module rather than stopping at
    /// the bound name's first segment (the bare-name use shortcut
    /// that's correct for use sites but wrong for the import
    /// statement itself).
    fn emit_nested_import(&mut self, stmt: &ruff_python_ast::StmtImport) {
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let dotted = alias.name.id.as_str();
            let first_seg = dotted.split('.').next().unwrap_or(dotted);
            let (bound_name, synthetic_chain): (&str, Vec<&str>) = match &alias.asname {
                Some(asname) => (asname.id.as_str(), Vec::new()),
                None => (first_seg, dotted.split('.').skip(1).collect()),
            };
            let spec = ImportSpec {
                module: dotted.to_string(),
                decl: None,
                star: false,
            };
            self.emit_upstream(&spec, bound_name, &synthetic_chain);
        }
        self.current_flags = 0;
    }

    /// Handle `from X import ...` inside a function/class body.
    ///
    /// Resolves the from-clause via ty (so relative imports get
    /// their level dots converted to an absolute name), then walks
    /// every alias. `*` imports fan out to every non-underscore decl
    /// in the upstream module via `emit_nested_star`; explicit names
    /// (with or without `as`) emit upstream edges through the same
    /// `emit_upstream` path module-scope from-imports use.
    fn emit_nested_import_from(&mut self, stmt: &ruff_python_ast::StmtImportFrom) {
        let db = self.model.db();
        let Ok(module_name) = ModuleName::from_import_statement(db, self.file, stmt) else {
            return;
        };
        let module_str = module_name.as_str().to_string();
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let name = alias.name.id.as_str();
            if name == "*" {
                self.emit_nested_star(&module_name);
                continue;
            }
            let bound_name = match &alias.asname {
                Some(asname) => asname.id.as_str(),
                None => name,
            };
            let spec = ImportSpec {
                module: module_str.clone(),
                decl: Some(name.to_string()),
                star: false,
            };
            self.emit_upstream(&spec, bound_name, &[]);
        }
        self.current_flags = 0;
    }

    /// Fan a nested `from X import *` out to every non-underscore
    /// decl in the upstream module, plus the module itself.
    ///
    /// Module-scope star imports use ty's `StarImport` definitions
    /// (one per resolved name) and route through the per-alias-node
    /// path; nested star imports have no per-name graph nodes to
    /// mint, so we go straight to the upstream `live_decls`. The
    /// underscore filter approximates the libcst pipeline's star
    /// expansion, which respects PEP 8's "names starting with `_`
    /// are not star-exported" rule (and `__all__` when present —
    /// not modeled here yet, but no current test exercises that).
    fn emit_nested_star(&mut self, module_name: &ModuleName) {
        let db = self.model.db();
        let Some(module) = resolve_module(db, self.file, module_name) else {
            return;
        };
        let Some(target_file) = module.file(db) else {
            return;
        };
        if let Some(&idx) = self.module_nodes.get(&target_file) {
            self.emit_edge(idx);
        }
        let targets: Vec<usize> = self
            .live_decls
            .iter()
            .filter(|((file, name), _)| *file == target_file && !name.starts_with('_'))
            .flat_map(|(_, idxs)| idxs.iter().copied())
            .collect();
        for idx in targets {
            self.emit_edge(idx);
        }
    }

    /// Handle a call expression that may be a dynamic-import shape:
    /// `__import__('name', …)` or `importlib.import_module('name', …)`.
    ///
    /// Returns `true` if the call was recognized (recognized but
    /// rejected — e.g. non-literal first argument — still returns
    /// `true`, since the visitor shouldn't fall through to walk the
    /// arguments looking for normal name references). Returns `false`
    /// when the call doesn't match either shape.
    fn try_emit_dynamic_import(&mut self, call: &ruff_python_ast::ExprCall) -> bool {
        let Some(kind) = detect_dynamic_call(&call.func) else {
            return false;
        };
        let py = unsafe { Python::assume_gil_acquired() };
        match parse_dynamic_args(kind, call) {
            DynamicParseResult::Ok {
                name,
                fromlist,
                explicit_package,
                explicit_level,
            } => {
                let db = self.model.db();
                let file_pkg = file_package_name(db, self.file);
                let pkg = explicit_package.or(file_pkg.as_deref());
                match resolve_dynamic_target(kind, &name, explicit_level, pkg) {
                    Ok(target) => self.emit_dynamic_edges(&target, &fromlist),
                    Err(message) => emit_visitor_warning(py, &message),
                }
                true
            }
            DynamicParseResult::Warn(message) => {
                emit_visitor_warning(py, &message);
                true
            }
            DynamicParseResult::NotApplicable => false,
        }
    }

    /// Emit `owner → target` (and `owner → target.entry` for each
    /// fromlist entry that resolves) for a dynamic import, each
    /// tagged with `EDGE_FLAG_DYNAMIC_IMPORT`. The visitor stays
    /// minimal: one edge per literal symbol the call mentioned. A
    /// contrib plugin can read the flag and fan out further.
    fn emit_dynamic_edges(&mut self, target: &str, fromlist: &[&str]) {
        let db = self.model.db();
        let saved = self.current_flags;
        self.current_flags |= EDGE_FLAG_DYNAMIC_IMPORT;

        // Edge to the literal name's module — but only when there's
        // no fromlist, since with a fromlist the base module is just
        // a stepping stone to the named entries.
        if fromlist.is_empty() {
            self.emit_resolved_module(target);
        } else {
            // With a non-empty fromlist Python still loads the base
            // module (`__import__('p', fromlist=[…])` returns `p`),
            // so emit the base edge and then resolve each entry as
            // either a submodule or a global-scope decl in that
            // module.
            self.emit_resolved_module(target);
            for entry in fromlist {
                if entry.is_empty() {
                    continue;
                }
                let candidate = format!("{target}.{entry}");
                if module_name_resolves(&candidate, self.file, db) {
                    self.emit_resolved_module(&candidate);
                    continue;
                }
                let target_file = ModuleName::new(target)
                    .and_then(|n| resolve_module(db, self.file, &n))
                    .and_then(|m| m.file(db));
                if let Some(target_file) = target_file {
                    if let Some(idxs) = self.live_decls.get(&(target_file, (*entry).to_string())) {
                        for &decl_idx in idxs {
                            self.emit_edge(decl_idx);
                        }
                    }
                }
                // Entries that don't resolve as either submodule or
                // decl are dropped silently — the libcst pipeline
                // does the same.
            }
        }

        self.current_flags = saved;
    }

    fn emit_resolved_module(&mut self, dotted: &str) {
        let db = self.model.db();
        let Some(mn) = ModuleName::new(dotted) else {
            return;
        };
        let Some(module) = resolve_module(db, self.file, &mn) else {
            return;
        };
        if module
            .search_path(db)
            .is_some_and(|sp| sp.is_standard_library())
        {
            return;
        }
        let Some(target_file) = module.file(db) else {
            return;
        };
        if let Some(&idx) = self.module_nodes.get(&target_file) {
            self.emit_edge(idx);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DynamicKind {
    DunderImport,
    ImportlibImportModule,
}

impl DynamicKind {
    fn label(self) -> &'static str {
        match self {
            DynamicKind::DunderImport => "__import__(...)",
            DynamicKind::ImportlibImportModule => "importlib.import_module(...)",
        }
    }
}

/// Match `call.func` against the two dynamic-import call shapes we
/// recognize: a bare `Name("__import__")` or
/// `Attribute(Name("importlib"), "import_module")`. The `importlib`
/// receiver is matched textually — a local `class importlib` would
/// be a false positive, but that pattern is vanishingly rare in
/// practice and the libcst pipeline does the same syntactic match.
pub(crate) fn detect_dynamic_call(func: &Expr) -> Option<DynamicKind> {
    match func {
        Expr::Name(n) if n.id.as_str() == "__import__" => Some(DynamicKind::DunderImport),
        Expr::Attribute(attr) => {
            if let Expr::Name(receiver) = &*attr.value {
                if receiver.id.as_str() == "importlib" && attr.attr.as_str() == "import_module" {
                    return Some(DynamicKind::ImportlibImportModule);
                }
            }
            None
        }
        _ => None,
    }
}

pub(crate) enum DynamicParseResult<'a> {
    /// Successfully parsed.
    Ok {
        name: String,
        fromlist: Vec<&'a str>,
        explicit_package: Option<&'a str>,
        explicit_level: Option<u32>,
    },
    /// Recognized as a dynamic import but rejected with this warning
    /// message. The visitor still skips the call's argument walk —
    /// the call site itself doesn't carry normal name references
    /// once we've identified it as a dynamic import.
    Warn(String),
    /// Not a dynamic import (e.g. the first argument is missing
    /// entirely, so it isn't really one of these call shapes). Caller
    /// should fall through to a normal expression walk.
    NotApplicable,
}

pub(crate) fn parse_dynamic_args<'a>(
    kind: DynamicKind,
    call: &'a ruff_python_ast::ExprCall,
) -> DynamicParseResult<'a> {
    let Some(first_arg) = first_positional(call) else {
        return DynamicParseResult::NotApplicable;
    };
    let Some(name) = string_literal(first_arg) else {
        return DynamicParseResult::Warn(format!(
            "Skipping dynamic import '{}': name is not a string literal",
            kind.label()
        ));
    };

    match kind {
        DynamicKind::DunderImport => {
            // `__import__(name, globals=None, locals=None, fromlist=[], level=0)`
            let fromlist_expr = positional_or_kwarg(call, 3, "fromlist");
            let level_expr = positional_or_kwarg(call, 4, "level");
            let fromlist = match fromlist_expr {
                None => Vec::new(),
                Some(expr) => match string_literal_list(expr) {
                    Some(list) => list,
                    None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': fromlist is not a literal \
                             list/tuple of strings (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            let explicit_level = match level_expr {
                None => None,
                Some(expr) => match int_literal(expr) {
                    Some(n) if n >= 0 => Some(n as u32),
                    Some(_) | None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': level is not an int literal \
                             (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            DynamicParseResult::Ok {
                name: name.to_string(),
                fromlist,
                explicit_package: None,
                explicit_level,
            }
        }
        DynamicKind::ImportlibImportModule => {
            // `importlib.import_module(name, package=None)`
            let pkg_expr = positional_or_kwarg(call, 1, "package");
            let explicit_package = match pkg_expr {
                None => None,
                Some(expr) => match string_literal(expr) {
                    Some(s) => Some(s),
                    None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': package is not a string literal \
                             (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            DynamicParseResult::Ok {
                name: name.to_string(),
                fromlist: Vec::new(),
                explicit_package,
                explicit_level: None,
            }
        }
    }
}

pub(crate) fn first_positional(call: &ruff_python_ast::ExprCall) -> Option<&Expr> {
    call.arguments.args.first()
}

pub(crate) fn positional_or_kwarg<'a>(
    call: &'a ruff_python_ast::ExprCall,
    idx: usize,
    kwarg_name: &str,
) -> Option<&'a Expr> {
    if let Some(expr) = call.arguments.args.get(idx) {
        return Some(expr);
    }
    call.arguments
        .keywords
        .iter()
        .find(|k| k.arg.as_ref().is_some_and(|id| id.as_str() == kwarg_name))
        .map(|k| &k.value)
}

pub(crate) fn string_literal(expr: &Expr) -> Option<&str> {
    if let Expr::StringLiteral(s) = expr {
        Some(s.value.to_str())
    } else {
        None
    }
}

pub(crate) fn int_literal(expr: &Expr) -> Option<i64> {
    let Expr::NumberLiteral(n) = expr else {
        return None;
    };
    if let ruff_python_ast::Number::Int(i) = &n.value {
        i.as_i64()
    } else {
        None
    }
}

pub(crate) fn string_literal_list(expr: &Expr) -> Option<Vec<&str>> {
    let elements: &[Expr] = match expr {
        Expr::List(l) => &l.elts,
        Expr::Tuple(t) => &t.elts,
        _ => return None,
    };
    let mut out = Vec::with_capacity(elements.len());
    for elem in elements {
        out.push(string_literal(elem)?);
    }
    Some(out)
}

/// Resolve a dynamic-import `name` against its level / package
/// context to an absolute module name, or return an error message
/// suitable for logging.
///
/// Per CPython semantics:
/// * `__import__`: leading dots in `name` are invalid (level is the
///   `level=` keyword/positional arg); `level=N` means "start from
///   the current package and go up N-1 levels."
/// * `importlib.import_module`: leading dots in `name` are the level
///   (count them); `level=1` means current package, `level=2` parent,
///   etc. `package=` is optional and defaults to the caller's
///   package — that's what `pkg` is here.
pub(crate) fn resolve_dynamic_target(
    kind: DynamicKind,
    name: &str,
    explicit_level: Option<u32>,
    pkg: Option<&str>,
) -> Result<String, String> {
    let (name_no_dots, level): (&str, u32) = match kind {
        DynamicKind::DunderImport => {
            if name.starts_with('.') {
                return Err(format!(
                    "Skipping dynamic import '{}': leading dots are invalid for \
                     __import__ (name: '{name}')",
                    kind.label()
                ));
            }
            (name, explicit_level.unwrap_or(0))
        }
        DynamicKind::ImportlibImportModule => {
            let dots = name.chars().take_while(|c| *c == '.').count();
            (&name[dots..], dots as u32)
        }
    };

    if level == 0 {
        return Ok(name_no_dots.to_string());
    }
    let Some(pkg) = pkg else {
        return Err(format!(
            "Skipping dynamic import '{}': relative import '{name}' has no package context",
            kind.label()
        ));
    };
    let segments: Vec<&str> = pkg.split('.').filter(|s| !s.is_empty()).collect();
    let levels_up = (level - 1) as usize;
    if levels_up > segments.len() {
        return Err(format!(
            "Skipping dynamic import '{}': relative import '{name}' goes beyond top-level \
             package '{pkg}'",
            kind.label()
        ));
    }
    let base = &segments[..segments.len() - levels_up];
    let mut parts: Vec<&str> = base.to_vec();
    if !name_no_dots.is_empty() {
        parts.push(name_no_dots);
    }
    Ok(parts.join("."))
}

/// Package name (i.e. enclosing package) of `file`. For
/// `pkg/__init__.py` this is `"pkg"`; for `pkg/sub.py` this is
/// `"pkg"`; for a top-level `mod.py` this is `None`.
pub(crate) fn file_package_name(db: &dyn ty_python_semantic::Db, file: File) -> Option<String> {
    let module = file_to_module(db, file)?;
    let name = module.name(db);
    let path_str = match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    };
    let is_init = path_str.ends_with("/__init__.py") || path_str.ends_with("\\__init__.py");
    if is_init {
        Some(name.as_str().to_string())
    } else {
        name.parent().map(|n| n.as_str().to_string())
    }
}

/// Send a warning to the `dead_cst._visitor` logger so the
/// `visitor_warnings` test fixture (pytest caplog scoped to that
/// logger) can observe it.
pub(crate) fn emit_visitor_warning(py: Python<'_>, message: &str) {
    let _ = (|| -> PyResult<()> {
        let logging = py.import_bound("logging")?;
        let logger = logging.call_method1("getLogger", ("dead_cst._visitor",))?;
        logger.call_method1("warning", (message,))?;
        Ok(())
    })();
}

/// Peel an attribute chain back to its `Name` root.
///
/// Returns `Some((root, segments))` when `expr` is `Name`, `Name.s1`,
/// `Name.s1.s2`, ... — i.e. one or more attribute accesses on a bare
/// name. Returns `None` when the chain bottoms out at anything else
/// (a call result, subscript, attribute of attribute of a non-Name, …).
pub(crate) fn collapse_attribute_chain(expr: &Expr) -> Option<(&ExprName, Vec<&str>)> {
    let mut segments: Vec<&str> = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Attribute(attr) => {
                segments.push(attr.attr.as_str());
                current = &attr.value;
            }
            Expr::Name(n) => {
                segments.reverse();
                return Some((n, segments));
            }
            _ => return None,
        }
    }
}

/// True iff `dotted` resolves to *some* module (project, stdlib, or
/// third-party) as seen from `anchor`. Used to disambiguate
/// "submodule" vs "decl in module" for `from X import Y`.
pub(crate) fn module_name_resolves(
    dotted: &str,
    anchor: File,
    db: &dyn ty_python_semantic::Db,
) -> bool {
    ModuleName::new(dotted)
        .and_then(|n| resolve_module(db, anchor, &n))
        .is_some()
}

impl<'ast, 'db> Visitor<'ast> for RefCollector<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Name(n) = expr {
            self.emit_name_use(n, &[]);
            return;
        }
        if let Expr::Named(named) = expr {
            // Walrus `(y := expr)` at module scope has its own
            // ``DefinitionKind::NamedExpression`` entry in ty's global
            // scope. ``walk_owned`` walks the inner expression and
            // attributes uses to `y`; walking it again here would
            // double-attribute every reference (once to the walrus
            // target, once to whatever owns the enclosing expression).
            // Skip into the walrus and leave its body to its own walk.
            //
            // Inside function / class bodies (``nested_context``), the
            // walrus's Definition lives in the nested scope and isn't
            // covered by ``ingest_top_level``'s per-def loop, so we
            // still need to walk the inner value here — attributing to
            // the enclosing top-level decl.
            if !self.nested_context {
                return;
            }
            self.visit_expr(&named.value);
            return;
        }
        if matches!(expr, Expr::Attribute(_)) {
            if let Some((root, segments)) = collapse_attribute_chain(expr) {
                self.emit_name_use(root, &segments);
                // Don't recurse into the chain — every Name in it has
                // been handled and an attribute access has no other
                // walk-worthy children.
                return;
            }
        }
        if let Expr::Call(call) = expr {
            if self.try_emit_dynamic_import(call) {
                // For `importlib.import_module(...)`, attribute the
                // `importlib` receiver to the call's owner so the
                // alias keeps a use edge — without it the
                // module-level call would leave `p.x.importlib`
                // orphaned from the call site. The chain segment
                // (`import_module`) isn't a walkable target, so we
                // pass an empty extra-chain rather than walking the
                // attribute via `collapse_attribute_chain`.
                if let Expr::Attribute(attr) = &*call.func {
                    if let Expr::Name(receiver) = &*attr.value {
                        self.emit_name_use(receiver, &[]);
                    }
                }
                // Walk arguments for any nested non-string Names
                // (e.g. `__import__(name, fromlist=names)` where
                // `name` and `names` should still emit normal use
                // edges that attribute the *receiver* of those
                // values to the owner).
                for arg in &call.arguments.args {
                    self.visit_expr(arg);
                }
                for kw in &call.arguments.keywords {
                    self.visit_expr(&kw.value);
                }
                return;
            }
        }
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        if self.nested_context {
            match stmt {
                Stmt::Import(s) => {
                    self.emit_nested_import(s);
                    return;
                }
                Stmt::ImportFrom(s) => {
                    self.emit_nested_import_from(s);
                    return;
                }
                _ => {}
            }
        } else if stmt_creates_top_level_definition(stmt) {
            // At module-level walks (or per-def walks of value
            // expressions, where Stmt nodes wouldn't appear anyway),
            // nested top-level-definition statements have already been
            // processed by the per-def pass with their own owner.
            // Skipping them here prevents double-attribution when a
            // compound statement (`if` / `for` / `try` / `with` /
            // `match`) at module scope contains a definition in its
            // body — e.g. `if True: f = g` would otherwise emit
            // `mod -> g` *and* the proper `mod.f -> g`.
            return;
        }
        walk_stmt(self, stmt);
    }
}
