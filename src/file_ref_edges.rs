//! Per-file reference-edge collection. The Phase 3 port: walks the
//! AST inside every owned expression and emits `use → reaching-def`
//! edges keyed on `NodeRef` instead of `usize`.
//!
//! Status: **first-cut skeleton.** Emits the codemod-invariant
//! `use → local-alias` edge for every `ExprName` reference whose
//! reaching def is a local graph node. Sufficient to validate the
//! salsa-tracked pattern + per-file ref walk shape against the
//! existing `collect_reference_edges`.
//!
//! Deliberately deferred to follow-ups, each documented in-place
//! when the corresponding code path is stubbed:
//!
//! * **Parallel upstream reachability edges** (emit_upstream). For
//!   `from foo import bar; bar()`, today's pipeline emits
//!   `owner → alias` AND `owner → foo.bar` AND `owner → foo`. This
//!   cut emits only the first.
//! * **Attribute-chain edges.** `mod.bar.f()` collapses to a
//!   bare-name use of `mod` here (the root); the chain segments
//!   (`bar`, `f`) are not walked through the alias's resolution.
//! * **Nested-context imports.** `import X` / `from X import y`
//!   inside a function or class body emit no edges in this cut
//!   (they go to dead-letter for now).
//! * **String annotations.** Skipped entirely; `enter_string_annotation`
//!   triggers ty type-inference and is the largest per-file salsa
//!   cost we want to deliberately structure around.
//! * **Dynamic imports.** `__import__(...)` / `importlib.import_module(...)`
//!   walked as regular call expressions.
//! * **Dead-branch flag.** Edges originating inside statically-dead
//!   regions (`if False:`, post-`return`, …) emit no
//!   `EdgeFlags::DEAD_BRANCH` flag.
//! * **`__all__` edges.** Skipped.
//!
//! Each of those is its own follow-up commit on PR #226.

use ruff_db::files::File;
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Expr, ExprName, ExprStringLiteral, Stmt};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::FxHashSet;
use ty_module_resolver::{resolve_module, ModuleName};
use ty_project::Db as ProjectDb;
use ty_python_core::ast_ids::HasScopedUseId;
use ty_python_core::definition::{DefinitionKind, DefinitionState, TargetKind};
use ty_python_core::place::PlaceExprRef;
use ty_python_core::scope::FileScopeId;
use ty_python_core::{semantic_index, SemanticIndex};
use ty_python_semantic::SemanticModel;

use crate::file_payload::{
    file_to_nodes, import_payload_for_pure as import_payload_for, ExternalKey, FileNodes,
    ImportPayload, NodeKind, NodeRef,
};

/// Outcome of resolving a `Name` use to its reaching definition.
///
/// `Alias` is the module-scope path: the use has a local graph node
/// (an import alias or a top-level decl) that takes the in-edge.
/// `NestedImport` is the function-/class-scope path: ty saw an
/// import binding in a non-global scope, so no graph node was
/// minted, and the use's parallel upstream edges flow from the
/// enclosing top-level owner instead.
enum Resolution<'db> {
    Alias(NodeRef<'db>),
    NestedImport {
        spec: ImportPayload,
        bound_name: String,
    },
}
use crate::helpers::{detect_dead_ranges, EDGE_FLAG_DEAD_BRANCH, EDGE_FLAG_DYNAMIC_IMPORT};
use crate::ingest::{
    collapse_attribute_chain, detect_dynamic_call, file_package_name, from_module_string,
    module_name_resolves, paired_unpack_rhs, parse_dynamic_args, resolve_dynamic_target,
    stmt_creates_top_level_definition, target_is_dunder_all, DynamicParseResult,
};

/// Salsa-tracked output of [`file_to_ref_edges`]. Set semantics so
/// duplicate emissions (a name used twice resolving to the same
/// target) collapse to a single entry. Edges are unresolved
/// (`NodeRef` endpoints, not `u32` graph indices) — the assembly
/// pass translates at the end.
///
/// `warnings` are visitor messages (e.g. "Skipping dynamic import …")
/// buffered per-file in pure rust. The driver flushes them to the
/// `dead_cst._visitor` Python logger from the main thread once all
/// per-file workers have finished — keeps `file_to_ref_edges` itself
/// GIL-free so workers run inside `py.allow_threads` cleanly.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileRefEdges<'db> {
    pub(crate) edges: FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)>,
    pub(crate) warnings: Vec<String>,
}

/// Per-file reference-edge collection. Salsa-tracked so the AST
/// walk parallelizes via salsa's worker coordination, and cross-file
/// lookups into [`file_to_nodes`] are memoized.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_ref_edges<'db>(db: &'db dyn ProjectDb, file: File) -> FileRefEdges<'db> {
    let self_nodes = file_to_nodes(db, file);
    let mut edges: FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)> = FxHashSet::default();
    let mut warnings: Vec<String> = Vec::new();

    let parsed = parsed_module(db, file).load(db);
    let dead_ranges = detect_dead_ranges(&parsed);
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
        let owner_ref = NodeRef::Def(def);
        if !self_nodes.ref_to_local.contains_key(&owner_ref) {
            continue;
        }

        let kind = def.kind(db);
        let mut walker = RefWalker {
            owner: owner_ref,
            file,
            db,
            parsed: &parsed,
            index,
            self_nodes,
            model: &model,
            dead_ranges: &dead_ranges,
            edges: &mut edges,
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
    //     to the module node. The per-decl pass above intentionally
    //     skips these via `stmt_creates_top_level_definition`.
    let module_ref = NodeRef::Module(file);
    for stmt in &parsed.syntax().body {
        if stmt_creates_top_level_definition(stmt) {
            continue;
        }
        let mut walker = RefWalker {
            owner: module_ref,
            file,
            db,
            parsed: &parsed,
            index,
            self_nodes,
            model: &model,
            dead_ranges: &dead_ranges,
            edges: &mut edges,
            warnings: &mut warnings,
            nested_context: false,
            current_flags: 0,
            in_annotation: 0,
            in_string_annotation: false,
        };
        walker.visit_stmt(stmt);
    }

    FileRefEdges { edges, warnings }
}

/// Per-owner walker. Mirrors today's `RefCollector` but with
/// `NodeRef`-typed edges and a reduced feature set (see module docs).
struct RefWalker<'a, 'db> {
    owner: NodeRef<'db>,
    file: File,
    /// The project db. We can't go through `self.model.db()` for
    /// `file_to_nodes` calls because that returns `&dyn
    /// ty_python_semantic::Db` and our salsa-tracked queries are
    /// defined over `&dyn ty_project::Db` (a super-trait).
    db: &'db dyn ProjectDb,
    #[allow(dead_code)]
    parsed: &'a ParsedModuleRef,
    index: &'a SemanticIndex<'db>,
    self_nodes: &'a FileNodes<'db>,
    model: &'a SemanticModel<'db>,
    /// Statically-dead source regions for this file. Uses originating
    /// inside any of these get `EDGE_FLAG_DEAD_BRANCH` stamped on
    /// every edge they emit.
    dead_ranges: &'a [TextRange],
    edges: &'a mut FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)>,
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
    /// Flags stamped on each edge emitted by the current reference
    /// (set by `emit_name_use` / nested-import handlers from
    /// `flags_for_range` on the reference's source position;
    /// reset to 0 after the reference completes).
    current_flags: u32,
}

impl<'a, 'db> RefWalker<'a, 'db> {
    fn emit_edge(&mut self, dst: NodeRef<'db>) {
        if dst != self.owner {
            self.edges.insert((self.owner, dst, self.current_flags));
        }
    }

    /// Returns `EDGE_FLAG_DEAD_BRANCH` if `range` is contained in any
    /// statically-dead region recorded for this file, else `0`.
    fn flags_for_range(&self, range: TextRange) -> u32 {
        if self.dead_ranges.iter().any(|r| r.contains_range(range)) {
            EDGE_FLAG_DEAD_BRANCH
        } else {
            0
        }
    }

    /// Emit an edge from `self.owner` to a synthetic External node.
    /// `target_file` Some when the import resolved to a file
    /// (drives `[external dist] X` PEP 503 canonicalisation via
    /// project_dist_lookup); None for genuinely-unresolved imports
    /// (`[unresolved] X`).
    fn emit_external(&mut self, dotted: &str, target_file: Option<File>) {
        let top_level = dotted.split('.').next().unwrap_or(dotted);
        let fqname = match target_file {
            Some(f) => crate::file_payload::external_fqname_for(self.db, f, top_level),
            None => format!("[unresolved] {top_level}"),
        };
        let key = ExternalKey::new(self.db, fqname);
        self.emit_edge(NodeRef::External(key));
    }

    /// Resolve a `Name` use to the reaching local Definition(s) via
    /// ty's flow-sensitive use-def chain. Returns `NodeRef::Def` for
    /// each reaching def whose graph node lives in this file.
    ///
    /// Differs from today's `find_local_bindings`:
    ///
    /// * No `Resolution::NestedImport` variant — nested-context
    ///   imports are deferred. Bindings without a graph node are
    ///   silently dropped.
    /// * No `in_string_annotation` plumbing — string annotations
    ///   aren't entered yet.
    fn find_local_bindings(&self, name: &ExprName) -> Vec<Resolution<'db>> {
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
            let mut results: Vec<Resolution<'db>> = Vec::new();
            for binding in bindings {
                let Some(def) = binding.binding.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                saw_binding = true;
                let candidate = NodeRef::Def(def);
                if self.self_nodes.ref_to_local.contains_key(&candidate) {
                    results.push(Resolution::Alias(candidate));
                    continue;
                }
                // Nested-context import: ty sees an import binding in
                // a non-global scope, so no graph node was minted. The
                // use's parallel upstream edges flow from the
                // enclosing top-level owner via emit_upstream; we
                // package the spec + bound name into a NestedImport
                // resolution so emit_name_use can drive that.
                let kind = def.kind(db);
                if kind.is_import() {
                    let place_id = def.place(db);
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
            // Annotation-only declaration fallback (mirrors today's
            // pipeline at ingest.rs:1814).
            for declaration in use_def_map.end_of_scope_symbol_declarations(symbol_id) {
                let Some(def) = declaration.declaration.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                let candidate = NodeRef::Def(def);
                if self.self_nodes.ref_to_local.contains_key(&candidate) {
                    results.push(Resolution::Alias(candidate));
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
        for resolution in self.find_local_bindings(name) {
            match resolution {
                Resolution::Alias(dst) => {
                    self.emit_edge(dst);
                    // If the resolved alias is an import, emit
                    // parallel reachability edges through it
                    // (Principle 2).
                    let import_spec: Option<ImportPayload> = self
                        .self_nodes
                        .ref_to_local
                        .get(&dst)
                        .copied()
                        .and_then(|idx| {
                            let node_data = &self.self_nodes.nodes[idx as usize];
                            if matches!(node_data.kind, NodeKind::Import) {
                                node_data.imports.clone()
                            } else {
                                None
                            }
                        });
                    if let Some(spec) = import_spec {
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

    /// `import X[.Y.Z][ as A]` inside a function/class body. No alias
    /// node is minted (binding lives in non-global scope); emit
    /// parallel upstream edges from `self.owner` directly. Mirrors
    /// `ingest::RefCollector::emit_nested_import` (lines 2044–2061).
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
                module: dotted.to_string(),
                decl: None,
                star: false,
            };
            self.emit_upstream(&spec, bound_name, &synthetic_chain);
        }
        self.current_flags = 0;
    }

    /// `from X import …` inside a function/class body. Resolves the
    /// from-clause through ty (so relative imports get level dots
    /// converted to absolute), then walks each alias. `*` fans out to
    /// every non-underscore export in the upstream via
    /// [`emit_nested_star`]. Mirrors today's
    /// `RefCollector::emit_nested_import_from` (lines 2071–2096).
    fn emit_nested_import_from(&mut self, stmt: &ruff_python_ast::StmtImportFrom) {
        let module_str = from_module_string(self.model.db(), self.file, stmt);
        if module_str.is_empty() {
            return;
        }
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let name = alias.name.id.as_str();
            if name == "*" {
                let Some(module_name) = ModuleName::new(&module_str) else {
                    continue;
                };
                self.emit_nested_star(&module_name);
                continue;
            }
            let bound_name = match &alias.asname {
                Some(asname) => asname.id.as_str(),
                None => name,
            };
            let spec = ImportPayload {
                module: module_str.clone(),
                decl: Some(name.to_string()),
                star: false,
            };
            self.emit_upstream(&spec, bound_name, &[]);
        }
        self.current_flags = 0;
    }

    /// Fan a nested `from X import *` out to every non-underscore
    /// export in the upstream module. Mirrors today's
    /// `RefCollector::emit_nested_star` (lines 2109–2129).
    fn emit_nested_star(&mut self, module_name: &ModuleName) {
        let Some(module) = resolve_module(self.db, self.file, module_name) else {
            return;
        };
        let Some(target_file) = module.file(self.db) else {
            return;
        };
        self.emit_edge(NodeRef::Module(target_file));
        let target_nodes = file_to_nodes(self.db, target_file);
        for (name, locals) in &target_nodes.exports_by_name {
            if name.starts_with('_') {
                continue;
            }
            for &local_idx in locals {
                self.emit_edge(target_nodes.refs[local_idx as usize]);
            }
        }
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

    /// Recognize and emit edges for a dynamic-import call. Returns
    /// `true` if the call was a dynamic-import shape (so the visitor
    /// doesn't fall through to walk its arguments as ordinary names).
    /// Mirrors today's `RefCollector::try_emit_dynamic_import` +
    /// `emit_dynamic_edges` + `emit_resolved_module`.
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
                    Ok(target) => self.emit_dynamic_edges(&target, &fromlist),
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

    /// Emit `owner → target` plus per-fromlist-entry edges for a
    /// dynamic import, each tagged with `EDGE_FLAG_DYNAMIC_IMPORT`.
    fn emit_dynamic_edges(&mut self, target: &str, fromlist: &[&str]) {
        let saved = self.current_flags;
        self.current_flags |= EDGE_FLAG_DYNAMIC_IMPORT;
        if fromlist.is_empty() {
            self.emit_resolved_module(target);
        } else {
            self.emit_resolved_module(target);
            for entry in fromlist {
                if entry.is_empty() {
                    continue;
                }
                let candidate = format!("{target}.{entry}");
                if module_name_resolves(&candidate, self.file, self.model.db()) {
                    self.emit_resolved_module(&candidate);
                    continue;
                }
                // Treat as decl-style: resolve target to file, look up
                // entry in its exports_by_name.
                let target_file = ModuleName::new(target)
                    .and_then(|n| resolve_module(self.db, self.file, &n))
                    .and_then(|m| m.file(self.db));
                if let Some(target_file) = target_file {
                    let target_nodes = file_to_nodes(self.db, target_file);
                    if let Some(locals) = target_nodes.exports_by_name.get(*entry) {
                        for &local_idx in locals {
                            self.emit_edge(target_nodes.refs[local_idx as usize]);
                        }
                    }
                }
            }
        }
        self.current_flags = saved;
    }

    /// Emit a single edge to the resolved module's NodeRef. Stdlib
    /// drops silently; otherwise routes to first-party (Module) or
    /// External as appropriate.
    fn emit_resolved_module(&mut self, dotted: &str) {
        let Some(mn) = ModuleName::new(dotted) else {
            return;
        };
        let Some(module) = resolve_module(self.db, self.file, &mn) else {
            self.emit_external(dotted, None);
            return;
        };
        if module
            .search_path(self.db)
            .is_some_and(|sp| sp.is_standard_library())
        {
            return;
        }
        let Some(target_file) = module.file(self.db) else {
            self.emit_external(dotted, None);
            return;
        };
        if module
            .search_path(self.db)
            .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
        {
            self.emit_external(dotted, Some(target_file));
            return;
        }
        self.emit_edge(NodeRef::Module(target_file));
    }

    /// `__all__ = ["foo", "bar"]`: each string literal resolves to a
    /// module-scope binding; emit edges to each one. Mirrors today's
    /// `emit_dunder_all_edges`. Computed-`__all__` shapes (concat,
    /// `list(...)`) silently skip.
    fn emit_dunder_all_edges(&mut self, value: &Expr) {
        let elements = match value {
            Expr::List(l) => &l.elts,
            Expr::Tuple(t) => &t.elts,
            _ => return,
        };
        for elem in elements {
            if let Expr::StringLiteral(s) = elem {
                if let Some(dst) = self.lookup_module_scope_name(s.value.to_str()) {
                    self.emit_edge(dst);
                }
            }
        }
    }

    /// Resolve a bare name to its module-scope live binding's NodeRef.
    /// Used by [`emit_dunder_all_edges`]; mirrors today's
    /// `RefCollector::lookup_module_scope_name`.
    fn lookup_module_scope_name(&self, name: &str) -> Option<NodeRef<'db>> {
        let global = FileScopeId::global();
        let place_table = self.index.place_table(global);
        let symbol_id = place_table.symbol_id(name)?;
        if let Some(locals) = self.self_nodes.exports_by_name.get(name) {
            if let Some(&local_idx) = locals.first() {
                return Some(self.self_nodes.refs[local_idx as usize]);
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
            let candidate = NodeRef::Def(def);
            if self.self_nodes.ref_to_local.contains_key(&candidate) {
                return Some(candidate);
            }
        }
        None
    }

    /// Emit parallel reachability edges past a local import alias.
    /// First-party path only — stdlib / external / unresolved targets
    /// silently drop. Synthetic external nodes land with
    /// `NodeRef::External` in a follow-up.
    ///
    /// Mirrors today's `ingest::RefCollector::emit_upstream` (lines
    /// 1880–2032): classify the loading target, walk the submodule
    /// chain, emit edges to the deepest module reached plus any
    /// terminal decl.
    fn emit_upstream(&mut self, spec: &ImportPayload, bound_name: &str, extra_chain: &[&str]) {
        if spec.module.is_empty() {
            return;
        }
        let db = self.model.db();
        let module_first_seg = spec.module.split('.').next().unwrap_or("").to_string();

        let mut adjusted_chain: Vec<&str> = extra_chain.to_vec();
        let loading_target: String;
        let mut decl_tail: Option<String> = None;

        if spec.star {
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
                            loading_target = module_first_seg;
                        }
                    } else {
                        loading_target = spec.module.clone();
                    }
                }
            }
        }

        // Classify the loading target. Stdlib drops silently (matches
        // existing pipeline); external/unresolved targets emit a
        // single edge to a globally-interned synthetic External node
        // and stop (no submodule chain walk through them, since they
        // don't have file_to_nodes payloads to look up against).
        let Some(start_mn) = ModuleName::new(&loading_target) else {
            self.emit_external(&loading_target, None);
            return;
        };
        let Some(start_module) = resolve_module(db, self.file, &start_mn) else {
            self.emit_external(&loading_target, None);
            return;
        };
        if start_module
            .search_path(db)
            .is_some_and(|sp| sp.is_standard_library())
        {
            // Stdlib drops silently — matches today's behavior.
            return;
        }
        let Some(start_file) = start_module.file(db) else {
            self.emit_external(&loading_target, None);
            return;
        };
        // Non-first-party (site-packages / external paths). Mint a
        // synthetic External node keyed by PEP 503 canonical dist
        // name via project_dist_lookup; falls back to
        // `[external file] X` for orphan site-packages files.
        if start_module
            .search_path(db)
            .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
        {
            self.emit_external(&loading_target, Some(start_file));
            return;
        }

        // Decl-style alias: emit edge to the upstream module and the
        // decl inside it. Attribute access past a decl is field access
        // on the decl's value, which we don't model.
        if let Some(decl_name) = decl_tail {
            // Use sites land on whatever's in exports_by_name
            // (including star-reexport aliases). Don't walk through
            // star aliases here — the from-import binding side uses
            // walk_exports_chain to skip past stars; this is the
            // *use* side, which should reach the star alias itself.
            self.emit_edge(NodeRef::Module(start_file));
            let target_nodes = file_to_nodes(self.db, start_file);
            if let Some(locals) = target_nodes.exports_by_name.get(&decl_name) {
                for &local_idx in locals {
                    self.emit_edge(target_nodes.refs[local_idx as usize]);
                }
            }
            return;
        }

        // Module-style alias: walk the chain submodule-by-submodule,
        // emit one module edge (deepest reached) plus at most one
        // terminal decl edge.
        let mut current_file = start_file;
        let mut current_path = loading_target.clone();
        let mut terminal_decl_refs: Vec<NodeRef<'db>> = Vec::new();
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
            // Not a submodule — check for decl in current_file's exports.
            let target_nodes = file_to_nodes(self.db, current_file);
            if let Some(locals) = target_nodes.exports_by_name.get(*seg) {
                for &local_idx in locals {
                    terminal_decl_refs.push(target_nodes.refs[local_idx as usize]);
                }
            }
            break;
        }
        self.emit_edge(NodeRef::Module(current_file));
        for r in terminal_decl_refs {
            self.emit_edge(r);
        }
    }
}

impl<'ast, 'db> Visitor<'ast> for RefWalker<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        // Inside an annotation context, route string literals through
        // ty's `enter_string_annotation` so `List["Foo"]`-style refs
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
        // segments past the root become `extra_chain` so emit_upstream
        // can walk submodule segments past an aliased module.
        if matches!(expr, Expr::Attribute(_)) {
            if let Some((root, segments)) = collapse_attribute_chain(expr) {
                self.emit_name_use(root, &segments);
                return;
            }
        }
        // Recognize dynamic-import calls before falling through to a
        // normal Call walk. Matches today's RefCollector::visit_expr
        // sequencing — receiver gets its own emit_name_use, args walk
        // normally, but the literal name/fromlist are handled by
        // emit_dynamic_edges so they don't get attributed as string
        // refs to something else.
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

/// Walk every value-bearing AST node a Definition owns. Mirrors
/// `ingest::walk_owned` but takes the new `RefWalker` collector.
///
/// TODO(file_to_ref_edges follow-up): port the `__all__` special case
/// (currently the assignment value is walked as a generic expression,
/// which is wrong — names listed inside `__all__` should resolve as
/// module-scope binding lookups, not the strings themselves).
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
