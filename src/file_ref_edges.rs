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
use ruff_python_ast::{Expr, ExprName, Stmt};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::FxHashSet;
use ty_module_resolver::{resolve_module, ModuleName};
use ty_project::Db as ProjectDb;
use ty_python_core::ast_ids::HasScopedUseId;
use ty_python_core::definition::{DefinitionKind, DefinitionState};
use ty_python_core::scope::FileScopeId;
use ty_python_core::{semantic_index, SemanticIndex};
use ty_python_semantic::SemanticModel;

use crate::file_payload::{
    file_to_nodes, ExternalKey, FileNodes, ImportPayload, NodeKind, NodeRef,
};
use crate::helpers::{detect_dead_ranges, EDGE_FLAG_DEAD_BRANCH};
use crate::ingest::{
    collapse_attribute_chain, from_module_string, module_name_resolves,
    stmt_creates_top_level_definition,
};

/// Salsa-tracked output of [`file_to_ref_edges`]. Set semantics so
/// duplicate emissions (a name used twice resolving to the same
/// target) collapse to a single entry. Edges are unresolved
/// (`NodeRef` endpoints, not `u32` graph indices) — the assembly
/// pass translates at the end.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileRefEdges<'db> {
    pub(crate) edges: FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)>,
}

/// Per-file reference-edge collection. Salsa-tracked so the AST
/// walk parallelizes via salsa's worker coordination, and cross-file
/// lookups into [`file_to_nodes`] are memoized.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_ref_edges<'db>(db: &'db dyn ProjectDb, file: File) -> FileRefEdges<'db> {
    let self_nodes = file_to_nodes(db, file);
    let mut edges: FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)> = FxHashSet::default();

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
            nested_context: false,
            current_flags: 0,
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
            nested_context: false,
            current_flags: 0,
        };
        walker.visit_stmt(stmt);
    }

    FileRefEdges { edges }
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
    nested_context: bool,
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

    /// Emit an edge from `self.owner` to a synthetic External node
    /// keyed on the module fqname. Top-level segment only — multiple
    /// imports from the same dist collapse to one node via salsa
    /// interning. Simplified vs the existing pipeline: no dist_lookup
    /// canonicalisation, so `[external] X` is the bucket for
    /// everything site-packages-resolved.
    fn emit_external(&mut self, dotted: &str, unresolved: bool) {
        let top_level = dotted.split('.').next().unwrap_or(dotted);
        let fqname = if unresolved {
            format!("[unresolved] {top_level}")
        } else {
            format!("[external] {top_level}")
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
    fn find_local_bindings(&self, name: &ExprName) -> Vec<NodeRef<'db>> {
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
            let bindings = if first {
                let use_id = name.scoped_use_id(db, scope_id.to_scope_id(db, self.file));
                use_def_map.bindings_at_use(use_id)
            } else {
                use_def_map.end_of_scope_symbol_bindings(symbol_id)
            };
            let mut saw_binding = false;
            let mut results: Vec<NodeRef<'db>> = Vec::new();
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
                    results.push(candidate);
                }
                // TODO(file_to_ref_edges follow-up): when the binding
                // is an import in a nested scope (no graph node minted),
                // resolve its upstream via `emit_upstream` and emit
                // parallel reachability edges from `self.owner`.
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
                    results.push(candidate);
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
        for dst in self.find_local_bindings(name) {
            self.emit_edge(dst);
            // If the resolved alias is an import, emit parallel
            // reachability edges through it (Principle 2).
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
            self.emit_external(&loading_target, /*unresolved=*/ true);
            return;
        };
        let Some(start_module) = resolve_module(db, self.file, &start_mn) else {
            self.emit_external(&loading_target, /*unresolved=*/ true);
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
            self.emit_external(&loading_target, /*unresolved=*/ true);
            return;
        };
        // Non-first-party (site-packages / external paths). Mint an
        // External node keyed on the top-level module name. TODO:
        // proper dist_lookup canonicalisation for `[external dist] X`.
        if start_module
            .search_path(db)
            .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
        {
            self.emit_external(&loading_target, /*unresolved=*/ false);
            return;
        }

        // Decl-style alias: emit edge to the upstream module and the
        // decl inside it. Attribute access past a decl is field access
        // on the decl's value, which we don't model.
        if let Some(decl_name) = decl_tail {
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
        // TODO(file_to_ref_edges follow-up): handle string annotations
        // (visit_annotation + walk_string_annotation), dynamic imports
        // (try_emit_dynamic_import).
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
            v.visit_expr(a.value(parsed));
        }
        DefinitionKind::AnnotatedAssignment(a) => {
            v.visit_expr(a.annotation(parsed));
            if let Some(val) = a.value(parsed) {
                v.visit_expr(val);
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
        // Imports, parameters, loop / with / except / match bindings:
        // either no walk-worthy expression or handled at module-level.
        _ => {}
    }
}

fn walk_parameters<'a, 'db>(params: &'a ruff_python_ast::Parameters, v: &mut RefWalker<'a, 'db>) {
    let walk_one = |p: &'a ruff_python_ast::Parameter, v: &mut RefWalker<'a, 'db>| {
        if let Some(ann) = &p.annotation {
            v.visit_expr(ann);
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
