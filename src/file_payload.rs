//! Per-file payload returned by the salsa-tracked `file_to_nodes` query.
//!
//! This is the first step of the three-phase → fan-out/fan-in
//! refactor described in the project conversation. `file_to_nodes` is
//! a `#[salsa::tracked]` function so that:
//!
//! * the per-file walk runs in parallel via salsa's lock-free worker
//!   coordination (the same machinery that already gives `prewarm_file`
//!   16k files/sec);
//! * cross-file calls — e.g. `file_to_edges(A)` looking up a target
//!   `Definition` in `B` via `file_to_nodes(B)` — are memoized for free
//!   (`returns(ref)` returns a borrow straight into salsa's storage,
//!   no clone).
//!
//! Pattern cribbed from `function_known_decorators` in
//! `vendor/ruff/crates/ty_python_semantic/src/types/infer.rs`, which is
//! the closest analogue (tracked fn, `'db`-lifetimed return, salsa::Update
//! derive on the carrier struct).
//!
//! This module DOES NOT yet replace `ingest_decls` — the existing
//! pipeline still runs. The function is wired into the crate but has
//! no callers; landing it first lets us validate the salsa macro
//! pattern + the `NodeData` shape before the driver swap.

use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::{line_index, source_text};
use ruff_text_size::TextRange;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::HashMap;
use ty_project::Db as ProjectDb;
use ty_python_core::definition::{Definition, DefinitionKind, DefinitionState};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;

use crate::helpers::{
    file_default_flags, file_path_string, module_fqname_for_file, position, scan_noqa_directives,
    NODE_FLAGS_NOQA_PIN,
};
use crate::ingest::{decl_kind_str, from_module_string};

/// One graph node's data without the `Py<SymbolNode>` envelope.
///
/// Pure rust so it can live inside a salsa-tracked function's return
/// value. The assembly pass converts these to `Py<SymbolNode>` in a
/// single GIL acquisition at the end of the build.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct NodeData {
    pub(crate) fqname: String,
    pub(crate) kind: NodeKind,
    pub(crate) path: String,
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
    pub(crate) flags: u32,
    pub(crate) imports: Option<ImportPayload>,
}

/// Compact 1-byte discriminant for the graph's `kind` field. Mirrors
/// `graph::VALID_KINDS` exactly and converts to the static `&str` at
/// `Py<SymbolNode>` materialization time.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, salsa::Update, get_size2::GetSize)]
#[repr(u8)]
pub(crate) enum NodeKind {
    Function = 0,
    Class = 1,
    Variable = 2,
    Import = 3,
    TypeAlias = 4,
    Module = 5,
    Synthetic = 6,
}

impl NodeKind {
    // Used by the assembly pass that converts `NodeData → Py<SymbolNode>`
    // once `file_to_edges` lands and the driver is swapped.
    #[allow(dead_code)]
    pub(crate) fn as_static_str(self) -> &'static str {
        match self {
            NodeKind::Function => "function",
            NodeKind::Class => "class",
            NodeKind::Variable => "variable",
            NodeKind::Import => "import",
            NodeKind::TypeAlias => "type_alias",
            NodeKind::Module => "module",
            NodeKind::Synthetic => "synthetic",
        }
    }

    fn from_decl_kind_str(s: &str) -> Option<Self> {
        Some(match s {
            "function" => NodeKind::Function,
            "class" => NodeKind::Class,
            "variable" => NodeKind::Variable,
            "import" => NodeKind::Import,
            "type_alias" => NodeKind::TypeAlias,
            "module" => NodeKind::Module,
            "synthetic" => NodeKind::Synthetic,
            _ => return None,
        })
    }
}

/// Per-alias import metadata. Same fields as the python-facing
/// `Import` pyclass, but without the `Py` envelope so it can live
/// inside `NodeData`.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct ImportPayload {
    pub(crate) module: String,
    pub(crate) decl: Option<String>,
    pub(crate) star: bool,
}

/// Salsa-tracked output of [`file_to_nodes`]. All data is pure rust;
/// the only salsa references are the `Definition<'db>` keys, which
/// require the `'db` lifetime parameter and the `salsa::Update` derive.
///
/// Field shapes mirror the seven HashMaps `build_project_graph`
/// threads through `ingest_decls` today, restricted to the per-file
/// slice:
///
/// * `module` — the kind="module" `NodeData` for this file. Reachability
///   anchors the file via it (every per-file decl gets a `decl → module`
///   edge in `file_to_edges`).
/// * `defs` — every global-scope `Definition` with its `NodeData`, in
///   `use_def_map.all_definitions_with_usage()` iteration order.
///   `def_to_local[def] = i` matches `defs[i].0 == def`.
/// * `def_to_local` — `Definition → local_idx` for cross-file lookups
///   (`file_to_edges` needs this to translate a `from M import bar`
///   alias's target into a NodeRef on `bar`).
/// * `exports_by_name` — per-name → list of live local indices, computed
///   from `use_def_map.all_end_of_scope_symbol_bindings()`. This is the
///   `globals_by_name` shortcut a consumer's `from <thisfile> import x`
///   uses to find the live binding(s).
/// * `star_reexports` — per-name → upstream absolute module name when
///   `name` in this file is bound by a `from <upstream> import *`.
///   Lets a downstream chain walk hop through this file.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileNodes<'db> {
    pub(crate) module: NodeData,
    pub(crate) defs: Box<[(Definition<'db>, NodeData)]>,
    pub(crate) def_to_local: FxHashMap<Definition<'db>, u32>,
    pub(crate) exports_by_name: FxHashMap<String, SmallVec<[u32; 2]>>,
    pub(crate) star_reexports: FxHashMap<String, String>,
}

/// Build the per-file node payload. Salsa-tracked so it's memoized and
/// safe to call concurrently from rayon workers; `returns(ref)` so
/// callers get a borrow into salsa's storage instead of a clone.
///
/// Stub flag handling (`is_stub` / `stub_py_twin`) is intentionally
/// omitted here — it depends on the project-wide `peer_pyi_to_py` map,
/// which we'll feed into the assembly pass once the file-level work is
/// in place. The flag set is otherwise the same as today's
/// `ingest_decls`.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_nodes<'db>(db: &'db dyn ProjectDb, file: File) -> FileNodes<'db> {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = module_fqname_for_file(db, file);
    let default_flags = file_default_flags(db, file);
    let (file_pinned_by_noqa, per_line_noqa_pins) =
        scan_noqa_directives(&parsed, &source, &line_index);

    let (msl, msc, mel, mec) = position(&line_index, &source, parsed.syntax().range);
    let module = NodeData {
        fqname: module_fqname.clone(),
        kind: NodeKind::Module,
        path: path_str.clone(),
        start_line: msl,
        start_column: msc,
        end_line: mel,
        end_column: mec,
        flags: default_flags,
        imports: None,
    };

    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);

    // Star imports: ty mints one Definition per name brought in by
    // `from X import *`, but every per-name def from the same statement
    // shares the `*` token's range. Cache the synthesized `*<src>`
    // local name so we don't re-format / re-resolve it N times for
    // a star with N names. Identical names → identical NodeData →
    // dedup happens naturally at assembly time.
    let mut star_local_name_cache: HashMap<TextRange, String> = HashMap::new();

    // (Definition, NodeData, per_name) collected so we can populate
    // exports_by_name from end_of_scope bindings in a single post-pass
    // using the def-keyed local indices.
    let mut defs: Vec<(Definition<'db>, NodeData)> = Vec::new();
    let mut def_to_local: FxHashMap<Definition<'db>, u32> = FxHashMap::default();
    let mut star_reexports: FxHashMap<String, String> = FxHashMap::default();
    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }

        let kind = def.kind(db);
        let Some(node_kind_str) = decl_kind_str(kind) else {
            continue;
        };
        let Some(node_kind) = NodeKind::from_decl_kind_str(node_kind_str) else {
            continue;
        };

        let place_id = def.place(db);
        let PlaceExprRef::Symbol(symbol) = place_table.place(place_id) else {
            continue;
        };
        let per_name = symbol.name().as_str().to_string();

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

        // Snap `def`/`class`/`type` decls to the keyword position so
        // positions line up with the libcst pipeline (see comment in
        // `ingest_decls` for the rationale).
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

        let import_spec: Option<ImportPayload> = if matches!(node_kind, NodeKind::Import) {
            Some(import_payload_for_pure(kind, db, file, &parsed))
        } else {
            None
        };

        let mut flags: u32 = default_flags;
        if matches!(node_kind, NodeKind::Import)
            && (file_pinned_by_noqa || per_line_noqa_pins.contains(&sl))
        {
            flags |= NODE_FLAGS_NOQA_PIN;
        }
        // Stub-only ENTRYPOINT flagging is deferred to the assembly
        // pass; it needs the project-wide peer-stub map and the .py
        // twin's `exports_by_name`, which aren't visible to this
        // per-file query.

        let node = NodeData {
            fqname: format!("{module_fqname}.{local_name}"),
            kind: node_kind,
            path: path_str.clone(),
            start_line: sl,
            start_column: sc,
            end_line: el,
            end_column: ec,
            flags,
            imports: import_spec.clone(),
        };
        let local_idx = defs.len() as u32;
        def_to_local.insert(def, local_idx);
        defs.push((def, node));

        // Star-reexport tracking: mirror `ingest_decls`'s logic — a
        // later non-star binding for the same name clears the star
        // entry, so a shadowing import correctly disables chain walks
        // through this file.
        if let Some(spec) = &import_spec {
            if spec.star {
                star_reexports.insert(per_name.clone(), spec.module.clone());
            } else {
                star_reexports.remove(&per_name);
            }
        } else {
            star_reexports.remove(&per_name);
        }
    }

    // Post-pass: populate `exports_by_name` from ty's end-of-scope
    // symbol bindings. This collapses sequential rebinds to the latest
    // while preserving branch-bound multi-binding cases (try/except,
    // if/else where each branch assigns). Mirrors `globals_by_name` in
    // today's pipeline.
    let mut exports_by_name: FxHashMap<String, SmallVec<[u32; 2]>> = FxHashMap::default();
    for (symbol_id, bindings) in use_def_map.all_end_of_scope_symbol_bindings() {
        let PlaceExprRef::Symbol(sym) = place_table.place(ScopedPlaceId::Symbol(symbol_id)) else {
            continue;
        };
        let name = sym.name().as_str().to_string();
        let mut live: SmallVec<[u32; 2]> = SmallVec::new();
        for binding in bindings {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != file || def.file_scope(db) != global {
                continue;
            }
            if let Some(&local_idx) = def_to_local.get(&def) {
                live.push(local_idx);
            }
        }
        if !live.is_empty() {
            exports_by_name.insert(name, live);
        }
    }

    FileNodes {
        module,
        defs: defs.into_boxed_slice(),
        def_to_local,
        exports_by_name,
        star_reexports,
    }
}

/// Pure-rust variant of `ingest::import_payload_for`. Returns the same
/// three fields but as the per-file payload's `ImportPayload`, no
/// `ImportSpec` allocation.
fn import_payload_for_pure<'db>(
    kind: &DefinitionKind<'db>,
    db: &'db dyn ty_python_semantic::Db,
    file: File,
    parsed: &ruff_db::parsed::ParsedModuleRef,
) -> ImportPayload {
    match kind {
        DefinitionKind::Import(k) => {
            let alias = k.alias(parsed);
            ImportPayload {
                module: alias.name.id.as_str().to_string(),
                decl: None,
                star: false,
            }
        }
        DefinitionKind::ImportFrom(k) => {
            let alias = k.alias(parsed);
            ImportPayload {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(alias.name.id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::ImportFromSubmodule(k) => ImportPayload {
            module: from_module_string(db, file, k.import(parsed)),
            decl: Some(k.module(parsed).id.as_str().to_string()),
            star: false,
        },
        DefinitionKind::StarImport(k) => ImportPayload {
            module: from_module_string(db, file, k.import(parsed)),
            decl: None,
            star: true,
        },
        _ => unreachable!("import_payload_for_pure called with non-import kind"),
    }
}
