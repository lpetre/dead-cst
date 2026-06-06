//! Per-file reference-spec collection: the combined walk that replaced
//! the old `file_to_edges` / `file_to_ref_edges` pair.
//!
//! [`file_to_refspecs`] walks the AST inside every owned expression
//! plus the file's import bindings, and emits a flat, sorted
//! [`RefSpec`] list expressed entirely in local terms (local node
//! indices, symbolic [`MemberRef`] / [`DynamicRef`] descriptors). The
//! query is **file-local by construction**: its salsa dependencies are
//! this file's parse, semantic index, and `file_to_nodes` payload plus
//! search-path configuration — never another file's contents. Editing
//! an imported file no longer invalidates the importer's walk; only
//! the (much cheaper) assembly-time resolution re-runs.
//!
//! Cross-file resolution — module resolver probes, upstream
//! `exports_by_name` lookups, star-reexport chain walks, external
//! classification — happens in the assembly pass via
//! [`crate::refspec::resolve_member`] / [`crate::refspec::resolve_dynamic`],
//! memoized project-wide.
//!
//! Structural edges that are fully derivable from `file_to_nodes`
//! payloads (decl → module anchors, overload impl → stub anchors, the
//! submodule → parent-package hierarchy edge) are *not* emitted here;
//! the assembly pass synthesizes them directly so the payload carries
//! only real information.

use compact_str::{CompactString, ToCompactString};
use ruff_db::files::File;
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Expr, ExprName, ExprStringLiteral, Stmt};
use ruff_text_size::{Ranged, TextRange};
use ty_project::Db as ProjectDb;
use ty_python_core::ast_ids::HasScopedUseId;
use ty_python_core::definition::{DefinitionKind, DefinitionState, TargetKind};
use ty_python_core::place::PlaceExprRef;
use ty_python_core::scope::FileScopeId;
use ty_python_core::{semantic_index, SemanticIndex};
use ty_python_semantic::SemanticModel;

use crate::file_payload::{
    def_key, file_to_nodes, import_payload_for_pure as import_payload_for, FileNodes,
    ImportPayload, NodeData, NodeKind, NodeRef,
};
use crate::refspec::{DynamicRef, MemberRef, MemberRole, RefSpec, Target};

/// Outcome of resolving a `Name` use to its reaching definition.
///
/// `Alias` is the module-scope path: the use has a local graph node
/// (an import alias or a top-level decl) that takes the in-edge.
/// `NestedImport` is the function-/class-scope path: ty saw an
/// import binding in a non-global scope, so no graph node was
/// minted, and the use's parallel upstream edges flow from the
/// enclosing top-level owner instead.
enum Resolution {
    Alias(u32),
    NestedImport {
        spec: ImportPayload,
        bound_name: CompactString,
    },
}
use crate::helpers::{
    detect_dead_ranges, detect_type_checking_ranges, is_dunder_name, range_key,
    EDGE_FLAG_DEAD_BRANCH, EDGE_FLAG_DYNAMIC_IMPORT,
};
use crate::ingest::{
    collapse_attribute_chain, detect_dynamic_call, file_package_name, from_module_string,
    paired_unpack_rhs, parse_dynamic_args, resolve_dynamic_target,
    stmt_creates_top_level_definition, target_is_dunder_all, DynamicParseResult,
};

/// Salsa-tracked output of [`file_to_refspecs`]. `specs` is sorted +
/// deduped, so the payload is deterministic and `Eq`-backdating
/// compares plain content. The whole struct is `'static` — no
/// `NodeRef` endpoints, no interned keys — so salsa id reshuffles
/// across revisions can't defeat backdating.
///
/// `warnings` are visitor messages (e.g. "Skipping dynamic import …")
/// buffered per-file in pure rust. The driver flushes them to the
/// `dead_cst._visitor` Python logger from the main thread once all
/// per-file workers have finished — keeps `file_to_refspecs` itself
/// GIL-free so workers run inside `py.allow_threads` cleanly.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileRefSpecs {
    pub(crate) specs: Box<[RefSpec]>,
    pub(crate) warnings: Box<[String]>,
}

/// Per-file reference-spec collection. Salsa-tracked so the AST walk
/// parallelizes via salsa's worker coordination. See the module docs
/// for the file-locality contract.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_refspecs(db: &dyn ProjectDb, file: File) -> FileRefSpecs {
    let self_nodes = file_to_nodes(db, file);
    let mut specs: Vec<RefSpec> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let parsed = parsed_module(db, file).load(db);
    let dead_ranges = detect_dead_ranges(&parsed);
    let tc_ranges = detect_type_checking_ranges(&parsed);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let use_def_map = index.use_def_map(global);
    let model = SemanticModel::new(db, file);

    // (a) Per-Definition: walk every value-bearing AST owned by each
    //     global-scope decl, attribute name uses to that decl.
    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let owner_ref = NodeRef::Def(def_key(db, def, &parsed));
        let Some(&owner_local) = self_nodes.ref_to_local.get(&owner_ref) else {
            continue;
        };

        let kind = def.kind(db);
        let mut walker = RefWalker {
            owner_local,
            file,
            db,
            parsed: &parsed,
            index,
            self_nodes,
            model: &model,
            dead_ranges: &dead_ranges,
            tc_ranges: &tc_ranges,
            specs: &mut specs,
            warnings: &mut warnings,
            nested_context: false,
            current_flags: 0,
            in_annotation: 0,
            in_string_annotation: false,
        };
        walk_owned(kind, &parsed, &mut walker);
    }

    // (b) Module-level non-definition statements (top-level expression
    //     statements, `for`/`while`/`if` at module scope, …) attribute
    //     to the module node (local index 0). The per-decl pass above
    //     intentionally skips these via `stmt_creates_top_level_definition`.
    for stmt in &parsed.syntax().body {
        if stmt_creates_top_level_definition(stmt) {
            continue;
        }
        let mut walker = RefWalker {
            owner_local: 0,
            file,
            db,
            parsed: &parsed,
            index,
            self_nodes,
            model: &model,
            dead_ranges: &dead_ranges,
            tc_ranges: &tc_ranges,
            specs: &mut specs,
            warnings: &mut warnings,
            nested_context: false,
            current_flags: 0,
            in_annotation: 0,
            in_string_annotation: false,
        };
        walker.visit_stmt(stmt);
    }

    // (c) Module-level dunders (`__all__`, `__version__`, PEP 562
    //     `__getattr__`/`__dir__`, …) and `__future__` imports are kept
    //     alive by an edge *from the module node*, not by a seed flag of
    //     their own: removing one changes module semantics, but only
    //     while the module itself is live. A module nothing reaches dies,
    //     and its dunders die with it. This is the node-level analogue of
    //     the `__all__ -> listed-name` edges emitted in pass (a).
    for (local_idx, node) in self_nodes.nodes.iter().enumerate() {
        let is_dunder_decl = matches!(node.kind, NodeKind::Variable | NodeKind::Function)
            && is_dunder_name(&node.fqname);
        let is_future_import = node
            .imports
            .as_ref()
            .is_some_and(|imp| imp.module == "__future__");
        if (is_dunder_decl || is_future_import) && local_idx != 0 {
            specs.push(RefSpec {
                src: 0,
                target: Target::Local(local_idx as u32),
                flags: 0,
            });
        }
    }

    // (d) Import-alias binding specs. Walk each kind="import"
    //     Definition in this file's global scope and emit one
    //     `Member(Binding)` spec carrying the per-kind `(module, decl)`
    //     resolution input. The assembly pass resolves it to
    //     `alias → Module(target)` (+ `alias → target decl` for
    //     from-imports), classifies stdlib / external / unresolved
    //     targets, and applies the `_handle_fromlist`
    //     namespace-vs-submodule disambiguation — all cross-file work
    //     lives there, not here.
    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let alias_ref = NodeRef::Def(def_key(db, def, &parsed));
        // Skip Definitions that file_to_nodes didn't mint (non-symbol
        // places, unsupported kinds). Keeps the spec set internally
        // consistent with the node set.
        let Some(&alias_local) = self_nodes.ref_to_local.get(&alias_ref) else {
            continue;
        };

        let kind = def.kind(db);
        if !kind.is_import() {
            continue;
        }

        let mut src_local = alias_local;
        let (target_module_str, decl_name, is_star): (CompactString, Option<CompactString>, bool) =
            match kind {
                DefinitionKind::Import(k) => {
                    // `import a.b.c` binds `a` locally, but the *statement*
                    // loads the deepest module; the binding spec resolves
                    // the full dotted path.
                    let alias = k.alias(&parsed);
                    (alias.name.id.as_str().to_compact_string(), None, false)
                }
                DefinitionKind::ImportFrom(k) => {
                    let module = from_module_string(db, file, k.import(&parsed));
                    let alias = k.alias(&parsed);
                    (
                        module,
                        Some(alias.name.id.as_str().to_compact_string()),
                        false,
                    )
                }
                DefinitionKind::ImportFromSubmodule(k) => {
                    // ImportFromSubmodule binds a submodule attribute on
                    // the containing package as a side effect of a
                    // `from X import …` statement inside __init__.py.
                    // The bound NAME (k.module) is the attribute, but the
                    // TARGET module is the from-clause itself — for
                    // `from pkg._internal import *` in pkg/__init__.py
                    // the side-effect attribute `_internal` on pkg
                    // points at the file `pkg._internal`, not at
                    // `pkg._internal._internal`. The from-clause module
                    // is what resolves.
                    let target = from_module_string(db, file, k.import(&parsed));
                    (target, None, false)
                }
                DefinitionKind::StarImport(k) => {
                    // `from X import *`. Each per-name `STAR_REEXPORT` node
                    // edges into the kept statement-level `*X` node, which in
                    // turn carries the module-level upstream spec. Rebind
                    // `src_local` to that `*X` node so the binding spec
                    // (stdlib / external / unresolved handling included)
                    // resolves from `*X`, not from the per-name node. Uses
                    // of star-bound names land on the per-name node (ty's
                    // use-def chain distinguishes them); `from <here>
                    // import name` lookups resolve per-name via
                    // walk_exports_chain.
                    let module = from_module_string(db, file, k.import(&parsed));
                    let star_ref = NodeRef::StarStmt(file, range_key(kind.target_range(&parsed)));
                    if let Some(&star_local) = self_nodes.ref_to_local.get(&star_ref) {
                        if star_local != alias_local {
                            specs.push(RefSpec {
                                src: alias_local,
                                target: Target::Local(star_local),
                                flags: 0,
                            });
                        }
                        src_local = star_local;
                    }
                    (module, None, true)
                }
                _ => continue,
            };

        if target_module_str.is_empty() {
            continue;
        }

        specs.push(RefSpec {
            src: src_local,
            target: Target::Member(MemberRef {
                role: MemberRole::Binding,
                spec: ImportPayload {
                    module: target_module_str,
                    decl: decl_name,
                    star: is_star,
                },
                bound_name: CompactString::default(),
                chain: Vec::new(),
            }),
            flags: 0,
        });
    }

    // Sorted + deduped so the payload is deterministic regardless of
    // emission order, identical re-emissions collapse (a name used
    // twice resolving to the same target), and `Eq` backdating is a
    // cheap content compare.
    specs.sort_unstable();
    specs.dedup();

    FileRefSpecs {
        specs: specs.into_boxed_slice(),
        warnings: warnings.into_boxed_slice(),
    }
}

/// Per-owner walker. Mirrors the libcst pipeline's `RefCollector` but
/// emits local-terms [`RefSpec`]s instead of resolved edges.
struct RefWalker<'a, 'db> {
    /// Local index (into `self_nodes.refs`) of the owning node.
    owner_local: u32,
    file: File,
    /// The project db. We can't go through `self.model.db()` for
    /// `file_to_nodes` calls because that returns `&dyn
    /// ty_python_semantic::Db` and our salsa-tracked queries are
    /// defined over `&dyn ty_project::Db` (a super-trait).
    db: &'db dyn ProjectDb,
    #[allow(dead_code)]
    parsed: &'a ParsedModuleRef,
    index: &'a SemanticIndex<'db>,
    self_nodes: &'a FileNodes,
    model: &'a SemanticModel<'db>,
    /// Statically-dead source regions for this file. Uses originating
    /// inside any of these get `EDGE_FLAG_DEAD_BRANCH` stamped on
    /// every spec they emit.
    dead_ranges: &'a [TextRange],
    /// Statement ranges inside `if TYPE_CHECKING:` blocks. A use whose
    /// flow-resolved binding falls inside one of these has had its
    /// runtime binding narrowed away by ty (which treats
    /// `TYPE_CHECKING` as `True`); `find_local_bindings` recovers it
    /// from the scope-wide reachable bindings.
    tc_ranges: &'a [TextRange],
    specs: &'a mut Vec<RefSpec>,
    /// Per-file warnings buffer. Workers push pure-rust strings; the
    /// driver flushes them to Python logging from the main thread.
    warnings: &'a mut Vec<String>,
    nested_context: bool,
    /// Recursive depth of "inside a type annotation". Bumped by
    /// `visit_annotation` around each annotation walk; gates the
    /// string-annotation parsing in `visit_expr` so we don't pay
    /// `enter_string_annotation`'s scope-type-inference cost on
    /// every dict-key / log-message / docstring literal.
    in_annotation: u32,
    /// True only inside the sub-walker spawned by
    /// `walk_string_annotation` for the parsed sub-AST. Names there
    /// aren't in the file's uses_map, so `find_local_bindings` must
    /// skip the position-sensitive `scoped_use_id` lookup that would
    /// otherwise panic.
    in_string_annotation: bool,
    /// Flags stamped on each spec emitted by the current reference
    /// (set by `emit_name_use` / nested-import handlers from
    /// `flags_for_range` on the reference's source position;
    /// reset to 0 after the reference completes).
    current_flags: u8,
}

impl<'db> RefWalker<'_, 'db> {
    /// Emit a `use → local node` spec (the codemod-invariant alias
    /// edge, `__all__` edges, …).
    fn emit_local(&mut self, dst_local: u32) {
        if dst_local != self.owner_local {
            self.specs.push(RefSpec {
                src: self.owner_local,
                target: Target::Local(dst_local),
                flags: self.current_flags,
            });
        }
    }

    /// Emit an unresolved `Use`-role member spec (the parallel
    /// upstream-reachability edges past an import alias). The
    /// assembly pass resolves it via `resolve_member`.
    fn emit_member_use(&mut self, spec: &ImportPayload, bound_name: &str, extra_chain: &[&str]) {
        if spec.module.is_empty() {
            return;
        }
        self.specs.push(RefSpec {
            src: self.owner_local,
            target: Target::Member(MemberRef {
                role: MemberRole::Use,
                spec: spec.clone(),
                bound_name: bound_name.to_compact_string(),
                chain: extra_chain.iter().map(|s| s.to_compact_string()).collect(),
            }),
            flags: self.current_flags,
        });
    }

    /// Returns `EDGE_FLAG_DEAD_BRANCH` if `range` is contained in any
    /// statically-dead region recorded for this file, else `0`.
    fn flags_for_range(&self, range: TextRange) -> u8 {
        if self.dead_ranges.iter().any(|r| r.contains_range(range)) {
            EDGE_FLAG_DEAD_BRANCH
        } else {
            0
        }
    }

    /// True when `range` falls inside a recorded type-checking block
    /// (see [`detect_type_checking_ranges`]).
    fn range_in_tc_block(&self, range: TextRange) -> bool {
        self.tc_ranges.iter().any(|tc| tc.contains_range(range))
    }

    /// Resolve a `Name` use to the reaching local Definition(s) via
    /// ty's flow-sensitive use-def chain. Returns the local index for
    /// each reaching def whose graph node lives in this file, or a
    /// `NestedImport` descriptor for import bindings in non-global
    /// scopes (which have no node of their own).
    fn find_local_bindings(&self, name: &ExprName, extra_chain: &[&str]) -> Vec<Resolution> {
        let db = self.model.db();
        // For names from a string-annotation sub-AST, ty doesn't know
        // their scope (we parsed them ourselves rather than going
        // through enter_string_annotation). Fall back to the file's
        // global scope — typing references inside annotations almost
        // always resolve via global-scope imports, and the
        // ancestor-scope walk below still climbs from global to find
        // them. Loses precision for "T defined inside def f, used in
        // f's annotation as string" which is rare.
        let file_scope = if self.in_string_annotation {
            self.model
                .scope(name.into())
                .unwrap_or(FileScopeId::global())
        } else {
            let Some(s) = self.model.scope(name.into()) else {
                return Vec::new();
            };
            s
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
            // back to end-of-scope for enclosing scopes (where the use
            // isn't recorded under any specific position). Names from
            // a string-annotation sub-AST aren't in the file's
            // uses_map, so `scoped_use_id` would panic; the
            // `in_string_annotation` flag routes them through the
            // end-of-scope fallback.
            let bindings = if first && !self.in_string_annotation {
                let use_id = name.scoped_use_id(db, scope_id.to_scope_id(db, self.file));
                use_def_map.bindings_at_use(use_id)
            } else {
                use_def_map.end_of_scope_symbol_bindings(symbol_id)
            };
            let mut saw_binding = false;
            // True when a flow-resolved binding lives inside an
            // `if TYPE_CHECKING:` block. ty narrows `TYPE_CHECKING` to
            // `True`, so in that case the *runtime* binding has been
            // narrowed away — `find_local_bindings` recovers it below.
            let mut resolved_in_tc = false;
            let mut results: Vec<Resolution> = Vec::new();
            for binding in bindings {
                let Some(def) = binding.binding.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                saw_binding = true;
                if self.range_in_tc_block(def.full_range(db, self.parsed).range()) {
                    resolved_in_tc = true;
                }
                let candidate = NodeRef::Def(def_key(self.db, def, self.parsed));
                if let Some(&local) = self.self_nodes.ref_to_local.get(&candidate) {
                    results.push(Resolution::Alias(local));
                    continue;
                }
                // Nested-context import: ty sees an import binding in
                // a non-global scope, so no graph node was minted. The
                // use's parallel upstream edges flow from the
                // enclosing top-level owner via emit_member_use; we
                // package the spec + bound name into a NestedImport
                // resolution so emit_name_use can drive that.
                let kind = def.kind(db);
                if kind.is_import() {
                    let place_id = def.place(db);
                    let PlaceExprRef::Symbol(sym) = place_table.place(place_id) else {
                        continue;
                    };
                    let bound_name = sym.name().as_str().to_compact_string();
                    let spec = import_payload_for(kind, db, self.file, self.parsed);
                    results.push(Resolution::NestedImport { spec, bound_name });
                }
            }
            if saw_binding {
                // Supplement the flow-sensitive resolution with
                // scope-wide reachable bindings in the two situations
                // ty's last-write-wins / TYPE_CHECKING-narrowed view
                // drops a binding the runtime still depends on:
                //
                // * Sibling submodule imports (`import a.foo` then
                //   `import a.bar`) rebind the same root name, so
                //   `a.foo.x()` resolves only to the last rebind even
                //   though it needs the `import a.foo` statement
                //   (importing `a.bar` doesn't import the `a.foo`
                //   submodule). Keep any plain `import <root>.<seg…>`
                //   whose submodule suffix is a prefix of the chain.
                // * `if TYPE_CHECKING: …` narrows to `True`, so the
                //   resolved binding is the type-checking-only one and
                //   the branch that runs at runtime (`else:` or a later
                //   rebind) is dropped. Keep every reachable binding
                //   *outside* a `TYPE_CHECKING` block.
                //
                // Each criterion is gated (chain present / resolved in a
                // TYPE_CHECKING block), so plain straight-line rebinds
                // keep their last-write-wins resolution.
                if !extra_chain.is_empty() || resolved_in_tc {
                    let root = name.id.as_str();
                    for binding in use_def_map.reachable_symbol_bindings(symbol_id) {
                        let Some(def) = binding.binding.definition() else {
                            continue;
                        };
                        if def.file(db) != self.file {
                            continue;
                        }
                        let candidate = NodeRef::Def(def_key(self.db, def, self.parsed));
                        let Some(&idx) = self.self_nodes.ref_to_local.get(&candidate) else {
                            continue;
                        };
                        if results
                            .iter()
                            .any(|r| matches!(r, Resolution::Alias(d) if *d == idx))
                        {
                            continue;
                        }
                        let recovers_runtime_binding = resolved_in_tc
                            && !self.range_in_tc_block(def.full_range(db, self.parsed).range());
                        let on_access_chain = !extra_chain.is_empty()
                            && plain_import_matches_chain(
                                &self.self_nodes.nodes[idx as usize],
                                root,
                                extra_chain,
                            );
                        if recovers_runtime_binding || on_access_chain {
                            results.push(Resolution::Alias(idx));
                        }
                    }
                }
                return results;
            }
            // Annotation-only declaration fallback (mirrors today's
            // pipeline at ingest.rs:1814).
            for declaration in use_def_map.end_of_scope_symbol_declarations(symbol_id) {
                let Some(def) = declaration.declaration.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                let candidate = NodeRef::Def(def_key(self.db, def, self.parsed));
                if let Some(&local) = self.self_nodes.ref_to_local.get(&candidate) {
                    results.push(Resolution::Alias(local));
                }
            }
            if !results.is_empty() {
                return results;
            }
            first = false;
        }
        Vec::new()
    }

    fn emit_name_use(&mut self, name: &ExprName, extra_chain: &[&str]) {
        // Names in non-Load context (LHS of `=`, `for x in …`, etc.)
        // are binding sites, not uses — skip to match today's pipeline.
        if !matches!(name.ctx, ruff_python_ast::ExprContext::Load) {
            return;
        }
        self.current_flags = self.flags_for_range(name.range());
        for resolution in self.find_local_bindings(name, extra_chain) {
            match resolution {
                Resolution::Alias(dst_local) => {
                    self.emit_local(dst_local);
                    // If the resolved alias is an import, emit a
                    // parallel `Use` member spec through it
                    // (Principle 2) — resolved at assembly.
                    let node_data = &self.self_nodes.nodes[dst_local as usize];
                    if matches!(node_data.kind, NodeKind::Import) {
                        if let Some(spec) = node_data.imports.clone() {
                            self.emit_member_use(&spec, name.id.as_str(), extra_chain);
                        }
                    }
                }
                Resolution::NestedImport { spec, bound_name } => {
                    self.emit_member_use(&spec, &bound_name, extra_chain);
                }
            }
        }
        self.current_flags = 0;
    }

    /// `import X[.Y.Z][ as A]` inside a function/class body. No alias
    /// node is minted (binding lives in non-global scope); emit a
    /// `Use` member spec from `self.owner_local` directly.
    fn emit_nested_import(&mut self, stmt: &ruff_python_ast::StmtImport) {
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let dotted = alias.name.id.as_str();
            let first_seg = dotted.split('.').next().unwrap_or(dotted);
            let (bound_name, synthetic_chain): (&str, Vec<&str>) = match &alias.asname {
                Some(asname) => (asname.id.as_str(), Vec::new()),
                None => (first_seg, dotted.split('.').skip(1).collect()),
            };
            let spec = ImportPayload {
                module: compact_str::ToCompactString::to_compact_string(&dotted),
                decl: None,
                star: false,
            };
            self.emit_member_use(&spec, bound_name, &synthetic_chain);
        }
        self.current_flags = 0;
    }

    /// `from X import name` inside a function/class body. Resolves the
    /// from-clause through ty (so relative imports get level dots
    /// converted to absolute — a self-property, still file-local),
    /// then walks each alias and emits the `Use` member specs. A
    /// nested `from X import *` is a SyntaxError ("import * only
    /// allowed at module level") so it cannot occur in runnable
    /// Python — it is skipped rather than fanned out.
    fn emit_nested_import_from(&mut self, stmt: &ruff_python_ast::StmtImportFrom) {
        let module_str = from_module_string(self.model.db(), self.file, stmt);
        if module_str.is_empty() {
            return;
        }
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let name = alias.name.id.as_str();
            // `from X import *` here would be a module-level-only
            // SyntaxError; skip it (no fan-out for non-runnable code).
            if name == "*" {
                continue;
            }
            let bound_name = match &alias.asname {
                Some(asname) => asname.id.as_str(),
                None => name,
            };
            let spec = ImportPayload {
                module: module_str.as_str().into(),
                decl: Some(compact_str::CompactString::from(name)),
                star: false,
            };
            self.emit_member_use(&spec, bound_name, &[]);
        }
        self.current_flags = 0;
    }

    /// Walk an annotation expression, marking the recursion so any
    /// string literal inside gets parsed as a deferred annotation
    /// via [`walk_string_annotation`].
    fn visit_annotation(&mut self, expr: &Expr) {
        self.in_annotation += 1;
        self.visit_expr(expr);
        self.in_annotation -= 1;
    }

    /// Parse a string-typed annotation as a Python expression and
    /// walk it for name uses.
    ///
    /// Uses `ruff_python_parser::parse_expression` directly instead
    /// of `SemanticModel::enter_string_annotation`. The latter calls
    /// `infer_complete_scope_types`, which triggers ty's full type
    /// inference for the enclosing scope — that fault-loads typeshed
    /// (`typing.pyi`, `builtins.pyi`, …) and on typing-heavy
    /// codebases adds tens of GB of `parsed_module` storage to
    /// salsa's cache at 200k-file scale.
    ///
    /// Trade-off: we lose ty's `is_string_annotation` syntactic
    /// gate (which would distinguish a typing-relevant string from
    /// e.g. a `Callable[["unused param doc"], int]` shape). Names
    /// inside the parsed sub-AST resolve via `in_string_annotation`'s
    /// global-scope fallback in `find_local_bindings`; sub-AST
    /// nodes aren't in the file's `uses_map`, so the
    /// position-sensitive `scoped_use_id` path is skipped and
    /// resolution falls through to `end_of_scope_symbol_bindings`.
    fn walk_string_annotation(&mut self, string_expr: &ExprStringLiteral) {
        let Some(string_literal) = string_expr.as_single_part_string() else {
            return;
        };
        let parsed = match ruff_python_parser::parse_expression(string_literal.as_str()) {
            Ok(p) => p,
            Err(_) => return,
        };
        let prev = self.in_string_annotation;
        self.in_string_annotation = true;
        self.visit_expr(parsed.expr());
        self.in_string_annotation = prev;
    }

    /// Recognize and emit a spec for a dynamic-import call. Returns
    /// `true` if the call was a dynamic-import shape (so the visitor
    /// doesn't fall through to walk its arguments as ordinary names).
    ///
    /// `resolve_dynamic_target` runs here, at walk time: it is pure
    /// string manipulation whose only context is the owning file's
    /// package name (a self-property), so the emitted [`DynamicRef`]
    /// carries an already-absolutized dotted target and its error
    /// messages stay in the file-local warnings buffer. The actual
    /// cross-file resolution happens at assembly via `resolve_dynamic`.
    fn try_emit_dynamic_import(&mut self, call: &ruff_python_ast::ExprCall) -> bool {
        let Some(kind) = detect_dynamic_call(&call.func) else {
            return false;
        };
        match parse_dynamic_args(kind, call) {
            DynamicParseResult::Ok {
                name,
                fromlist,
                explicit_package,
                explicit_level,
            } => {
                let file_pkg = file_package_name(self.model.db(), self.file);
                let pkg = explicit_package.or(file_pkg.as_deref());
                match resolve_dynamic_target(kind, &name, explicit_level, pkg) {
                    Ok(target) => self.specs.push(RefSpec {
                        src: self.owner_local,
                        target: Target::Dynamic(DynamicRef {
                            target: target.into(),
                            fromlist: fromlist
                                .iter()
                                .filter(|e| !e.is_empty())
                                .map(|e| e.to_compact_string())
                                .collect(),
                        }),
                        flags: self.current_flags | EDGE_FLAG_DYNAMIC_IMPORT,
                    }),
                    Err(message) => self.warnings.push(message),
                }
                true
            }
            DynamicParseResult::Warn(message) => {
                self.warnings.push(message);
                true
            }
            DynamicParseResult::NotApplicable => false,
        }
    }

    /// `__all__ = ["foo", "bar"]`: each string literal resolves to a
    /// module-scope binding; emit local edges to each one.
    /// Computed-`__all__` shapes (concat, `list(...)`) silently skip.
    fn emit_dunder_all_edges(&mut self, value: &Expr) {
        let elements = match value {
            Expr::List(l) => &l.elts,
            Expr::Tuple(t) => &t.elts,
            _ => return,
        };
        for elem in elements {
            if let Expr::StringLiteral(s) = elem {
                if let Some(dst_local) = self.lookup_module_scope_name(s.value.to_str()) {
                    self.emit_local(dst_local);
                }
            }
        }
    }

    /// Resolve a bare name to its module-scope live binding's local
    /// index. Used by [`emit_dunder_all_edges`].
    fn lookup_module_scope_name(&self, name: &str) -> Option<u32> {
        let global = FileScopeId::global();
        let place_table = self.index.place_table(global);
        let symbol_id = place_table.symbol_id(name)?;
        if let Some(locals) = self.self_nodes.exports_by_name.get(name) {
            if let Some(&local_idx) = locals.first() {
                return Some(local_idx);
            }
        }
        // Fallback: walk end-of-scope bindings (matches today's
        // pipeline behavior when exports_by_name misses).
        let use_def_map = self.index.use_def_map(global);
        for binding in use_def_map.end_of_scope_symbol_bindings(symbol_id) {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(self.model.db()) != self.file {
                continue;
            }
            let candidate = NodeRef::Def(def_key(self.db, def, self.parsed));
            if let Some(&local) = self.self_nodes.ref_to_local.get(&candidate) {
                return Some(local);
            }
        }
        None
    }
}

impl<'ast> Visitor<'ast> for RefWalker<'_, '_> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        // Inside an annotation context, route string literals through
        // the deferred-annotation parse so `List["Foo"]`-style refs
        // resolve. Gating on `in_annotation` avoids paying the
        // scope-type-inference cost for every dict-key / log-message /
        // docstring literal.
        if self.in_annotation > 0 {
            if let Expr::StringLiteral(s) = expr {
                self.walk_string_annotation(s);
                return;
            }
        }
        if let Expr::Name(n) = expr {
            self.emit_name_use(n, &[]);
            return;
        }
        // Collapse attribute chains rooted at a `Name`. The chain
        // segments past the root become `extra_chain` so the `Use`
        // member spec can walk submodule segments past an aliased
        // module at resolution time.
        if matches!(expr, Expr::Attribute(_)) {
            if let Some((root, segments)) = collapse_attribute_chain(expr) {
                self.emit_name_use(root, &segments);
                return;
            }
        }
        // Recognize dynamic-import calls before falling through to a
        // normal Call walk. Matches today's RefCollector::visit_expr
        // sequencing — receiver gets its own emit_name_use, args walk
        // normally, but the literal name/fromlist are handled by the
        // Dynamic spec so they don't get attributed as string refs to
        // something else.
        if let Expr::Call(call) = expr {
            if self.try_emit_dynamic_import(call) {
                if let Expr::Attribute(attr) = &*call.func {
                    if let Expr::Name(receiver) = &*attr.value {
                        self.emit_name_use(receiver, &[]);
                    }
                }
                for arg in &call.arguments.args {
                    self.visit_expr(arg);
                }
                for kw in &call.arguments.keywords {
                    self.visit_expr(&kw.value);
                }
                return;
            }
        }
        if let Expr::Named(named) = expr {
            // Walrus visibility: at module scope, the walrus has its own
            // Definition with its own owned-expression walk (covered by
            // the (a) pass). Skip to avoid double-attribution. Inside
            // nested context, fall through.
            if !self.nested_context {
                return;
            }
            self.visit_expr(&named.value);
            return;
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
            return;
        }
        walk_stmt(self, stmt);
    }
}

/// True when `node` is a plain `import <root>.<seg…>` (no `as` alias,
/// no `from`, no `*`) whose submodule segments after the root are a
/// prefix of `extra_chain` — i.e. the access chain reaches into that
/// submodule, so the importing statement must stay alive.
fn plain_import_matches_chain(node: &NodeData, root: &str, extra_chain: &[&str]) -> bool {
    if !matches!(node.kind, NodeKind::Import) {
        return false;
    }
    let Some(spec) = node.imports.as_ref() else {
        return false;
    };
    if spec.star || spec.decl.is_some() {
        return false;
    }
    // Plain `import root.seg1.seg2` (no `as`): the bound name equals the
    // module's first segment.
    let mut segs = spec.module.split('.');
    if segs.next() != Some(root) {
        return false;
    }
    let suffix: Vec<&str> = segs.collect();
    !suffix.is_empty()
        && suffix.len() <= extra_chain.len()
        && suffix.iter().zip(extra_chain).all(|(s, c)| s == c)
}

/// Walk every value-bearing AST node a Definition owns. Mirrors
/// `ingest::walk_owned` but takes the new `RefWalker` collector.
fn walk_owned<'a, 'db>(
    kind: &DefinitionKind<'db>,
    parsed: &'a ParsedModuleRef,
    v: &mut RefWalker<'a, 'db>,
) {
    match kind {
        DefinitionKind::Function(func) => {
            let node = func.node(parsed);
            for d in &node.decorator_list {
                v.visit_expr(&d.expression);
            }
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            walk_parameters(&node.parameters, v);
            if let Some(returns) = &node.returns {
                v.visit_annotation(returns);
            }
            v.nested_context = true;
            for s in &node.body {
                v.visit_stmt(s);
            }
            v.nested_context = false;
        }
        DefinitionKind::Class(cls) => {
            let node = cls.node(parsed);
            for d in &node.decorator_list {
                v.visit_expr(&d.expression);
            }
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
                v.emit_dunder_all_edges(value);
            } else if let TargetKind::Sequence(_, unpack) = a.target_kind() {
                // `c, d = a, b` produces one Definition per LHS
                // name, each with `value` set to the whole RHS
                // `(a, b)`. Walk only the matching RHS element when
                // both sides are flat sequences of the same arity;
                // otherwise fall back to walking the full RHS.
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
            v.visit_annotation(a.annotation(parsed));
            if let Some(val) = a.value(parsed) {
                if target_is_dunder_all(a.target(parsed)) {
                    v.emit_dunder_all_edges(val);
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
            v.visit_annotation(node.value.as_ref());
        }
        // Imports, parameters, loop / with / except / match bindings:
        // either no walk-worthy expression or handled at module-level.
        _ => {}
    }
}

fn walk_parameters<'a, 'db>(params: &'a ruff_python_ast::Parameters, v: &mut RefWalker<'a, 'db>) {
    let walk_one = |p: &'a ruff_python_ast::Parameter, v: &mut RefWalker<'a, 'db>| {
        if let Some(ann) = &p.annotation {
            v.visit_annotation(ann);
        }
    };
    let walk_with_default = |p: &'a ruff_python_ast::ParameterWithDefault,
                             v: &mut RefWalker<'a, 'db>| {
        walk_one(&p.parameter, v);
        if let Some(default) = &p.default {
            v.visit_expr(default);
        }
    };
    for p in &params.posonlyargs {
        walk_with_default(p, v);
    }
    for p in &params.args {
        walk_with_default(p, v);
    }
    if let Some(vararg) = &params.vararg {
        walk_one(vararg, v);
    }
    for p in &params.kwonlyargs {
        walk_with_default(p, v);
    }
    if let Some(kwarg) = &params.kwarg {
        walk_one(kwarg, v);
    }
}

fn walk_type_params<'a, 'db>(
    type_params: &'a ruff_python_ast::TypeParams,
    v: &mut RefWalker<'a, 'db>,
) {
    for tp in &type_params.type_params {
        match tp {
            ruff_python_ast::TypeParam::TypeVar(tv) => {
                if let Some(bound) = &tv.bound {
                    v.visit_expr(bound);
                }
                if let Some(default) = &tv.default {
                    v.visit_expr(default);
                }
            }
            ruff_python_ast::TypeParam::ParamSpec(ps) => {
                if let Some(default) = &ps.default {
                    v.visit_expr(default);
                }
            }
            ruff_python_ast::TypeParam::TypeVarTuple(tvt) => {
                if let Some(default) = &tvt.default {
                    v.visit_expr(default);
                }
            }
        }
    }
}
