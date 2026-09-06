//! Leftover utilities shared across modules: AST navigation helpers,
//! call-args extraction used by the native plugin query helpers,
//! dead-region detection, noqa scanning, position/file helpers, and the
//! shared `NodeFlags`/`EdgeFlags` constant aliases.

use compact_str::{CompactString, ToCompactString};
use pyo3::prelude::*;
use ruff_db::files::{File, FilePath};
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_db::system::SystemPath;
use ruff_db::PythonFile;
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::visitor::{walk_expr, Visitor};
use ruff_python_ast::{Expr, ExprName, Stmt, StmtClassDef};
use ruff_source_file::LineIndex;
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use ty_module_resolver::Module;
use ty_module_resolver::{
    file_to_module, resolve_module, system_module_search_paths, ImportingFile, ModuleName,
    ResolverEnvironment,
};
use ty_project::metadata::value::RelativePathBuf;
use ty_project::{Db as ProjectDb, ProjectDatabase};
use ty_python_core::definition::{Definition, DefinitionKind};
use ty_python_core::program_file::ProgramFile;
use ty_python_core::{global_scope, place_table, use_def_map};

use crate::file_payload::{
    import_payload_for_pure, ChainStep, ImportPayload, MemberSpec, ModuleValue,
};
use crate::graph::{EdgeFlags, NodeFlags};
use crate::ingest::{
    collapse_attribute_chain, dynamic_call_module, from_module_string, peel_value_chain,
};
use crate::project::BuildOutputs;
use crate::string_fold::{fold_string_expr, StringFoldCtx};

/// Sentinel value stored in a file's local imports map (the value half
/// of ``{local_name: target}`` returned by
/// [`collect_module_imports_local`]) when the local binding is the
/// module object itself (`import flask` / `import flask as f`), not a
/// specific name from inside it. Matchers test against this exact
/// string when classifying attribute-style decorator / call references.
pub(crate) const MODULE_ALIAS_MARKER: &str = "<module>";

/// ``{local name -> imported target}`` map shared by every
/// import-resolving matcher (decorator / construction / call walks).
/// Values are a dotted target (``module.decl``), an absolute module
/// name, or the [`MODULE_ALIAS_MARKER`] sentinel for module-object
/// bindings. `CompactString` keeps the typical short identifier
/// inline; probes with `&str` go through `Borrow<str>`.
pub(crate) type LocalImports = FxHashMap<CompactString, CompactString>;

/// ty keys its semantic queries on a [`ProgramFile`] (file + program).
/// dead-cst analyses every file under the single project program, so
/// the key is derived from the project directly instead of via ty's
/// `Db::program_file`, which re-checks the include globs on every call.
pub(crate) fn program_file<'db>(db: &'db dyn ProjectDb, file: File) -> ProgramFile<'db> {
    db.project().program(db).program_file(db, file)
}

/// Parser key (file + Python version) that `parsed_module` is keyed on.
pub(crate) fn python_file<'db>(db: &'db dyn ProjectDb, file: File) -> PythonFile<'db> {
    program_file(db, file).python_file(db)
}

/// Module-resolution key for `resolve_module` / `from_import_statement`.
/// The `File` variant defers interning the `(file, environment)` pair
/// until the resolver actually needs it.
pub(crate) fn importing_file<'db>(db: &'db dyn ProjectDb, file: File) -> ImportingFile<'db> {
    ImportingFile::File(file, resolver_environment(db))
}

/// The project program's module-resolution environment (search paths +
/// Python version).
pub(crate) fn resolver_environment<'db>(db: &'db dyn ProjectDb) -> ResolverEnvironment<'db> {
    db.project().program(db).resolver_environment(db)
}

pub(crate) fn is_dunder_name(fqname: &str) -> bool {
    let name = fqname.rsplit('.').next().unwrap_or("");
    name.len() > 4 && name.starts_with("__") && name.ends_with("__")
}

/// Split a decorator expression into its *head* -- the callee of the
/// first call in the chain, or the whole expression when nothing is
/// called -- and that first call. Builder-style suffixes after the head
/// call (`@launchable().cpus(16)`, `@app.route("/").tag("x")`) are
/// peeled off, so the head is what a matcher classifies and the head
/// call is where its kwargs come from. The head is returned as written:
/// callers unwrap a subscripted generic (`@route[T]()`) themselves via
/// [`unwrap_subscripted_callee`].
pub(crate) fn decorator_head(expr: &Expr) -> (&Expr, Option<&ruff_python_ast::ExprCall>) {
    let mut first_call: Option<&ruff_python_ast::ExprCall> = None;
    let mut current = expr;
    loop {
        match current {
            Expr::Call(call) => {
                first_call = Some(call);
                current = &call.func;
            }
            Expr::Attribute(attr) => current = &attr.value,
            _ => break,
        }
    }
    match first_call {
        Some(call) => (&call.func, Some(call)),
        None => (expr, None),
    }
}

pub(crate) fn iter_top_level_classes(
    parsed: &ParsedModuleRef,
) -> impl Iterator<Item = &StmtClassDef> {
    parsed.syntax().body.iter().filter_map(|stmt| match stmt {
        Stmt::ClassDef(cls) => Some(cls),
        _ => None,
    })
}

/// Resolve a dotted class fqn to ``(File, name_range)``. Handles both
/// project classes (looked up via ``decl_by_fqname`` + the inverted
/// ``class_by_selection``) and external classes (resolved through
/// [`resolve_member_def`]).
/// Files whose parsed module / semantic index were reloaded on demand
/// *after* the populate-phase eviction: class-base resolution during
/// assembly and class-seed relocation during the plugin pass / Python
/// queries. [`resolve_member_in_file`] — the one chokepoint where
/// those reloads happen — records every file it touches, and the
/// post-plugin-pass sweep re-clears exactly this set instead of
/// probing every project file's salsa slots.
#[derive(Debug, Default)]
pub(crate) struct ReloadLog(parking_lot::Mutex<FxHashSet<File>>);

impl ReloadLog {
    pub(crate) fn record(&self, file: File) {
        self.0.lock().insert(file);
    }

    /// Take the recorded set, leaving the log empty so later on-demand
    /// reloads (post-build Python queries) start a fresh set. Order is
    /// irrelevant to the caller (clears are idempotent and
    /// order-independent).
    pub(crate) fn drain(&self) -> Vec<File> {
        self.0.lock().drain().collect()
    }
}

pub(crate) fn locate_class_seed(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    fqn: &str,
) -> Option<(File, TextRange)> {
    // Project class: cheap path through the existing indices. The
    // inverted `class_by_selection` hands back the name range directly,
    // so no AST re-parse is needed to relocate the def.
    if let Some(idxs) = outputs.decl_by_fqname.get(fqn) {
        for &idx in idxs {
            if outputs.builder.nodes[idx].kind != "class" {
                continue;
            }
            if let Some(&seed) = outputs.class_selection_by_idx.get(&idx) {
                return Some(seed);
            }
        }
    }
    // External class: split the fqn into `(module, member)` and resolve
    // through the same `resolve_member_def` the store/assemble side uses,
    // so a query for `unittest.TestCase` lands on the identical
    // `(File, name_range)` key the assemble pass recorded in
    // `external_base_children` — sibling spellings and re-exports collapse
    // by construction, not via a query-time string scan.
    locate_member_seed(db, outputs, fqn, MemberKinds::CLASS)
}

/// Resolve a dotted `fqn` to the canonical `(File, name_range)` of the
/// definition it names, through the same [`resolve_member_def`] the
/// assemble pass ran on every observed spec — so a query for
/// `pkg.deco` lands on the identical key a `from pkg import deco`
/// spec (or any re-exported / aliased spelling of it) was recorded
/// under. `kinds` selects which definition kinds may terminate the
/// resolution. `None` when nothing resolves (an uninstalled dependency,
/// a member of a non-module such as `pytest.mark.parametrize`, so the
/// dependency must be importable from the analysis environment).
pub(crate) fn locate_member_seed(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    fqn: &str,
    kinds: MemberKinds,
) -> Option<(File, TextRange)> {
    let anchor = *outputs.project_files.first()?;
    let (seed_module, seed_name) = fqn.rsplit_once('.')?;
    // Query-time seed location: the read-set record is irrelevant here
    // (the plugin pass re-runs in full on every materialize), so a
    // scratch recorder is discarded.
    let mut scratch = crate::refspec::Touched::default();
    resolve_member_def(
        db,
        seed_module,
        seed_name,
        anchor,
        0,
        &outputs.reload_log,
        &mut scratch,
        kinds,
    )
}

/// Which definition kinds a member resolution may terminate on. Class
/// bases accept classes only; decorator targets accept classes,
/// functions, and variables (a module-level `deco = Deco(...)` whose
/// value can't be followed any further *is* the decorator's definition).
/// Part of the resolve-memo key, so the two never share an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MemberKinds(u8);

impl MemberKinds {
    pub(crate) const CLASS: MemberKinds = MemberKinds(1);
    pub(crate) const CALLABLE: MemberKinds = MemberKinds(1 | 2 | 4);

    pub(crate) fn accepts_class(self) -> bool {
        self.0 & 1 != 0
    }

    pub(crate) fn accepts_function(self) -> bool {
        self.0 & 2 != 0
    }

    pub(crate) fn accepts_variable(self) -> bool {
        self.0 & 4 != 0
    }
}

/// Maximum hops [`resolve_member_def`] / [`classify_base`] follow through
/// assignment-style re-export chains before giving up. ty's import
/// resolution has its own cycle guard; this only bounds the
/// assignment-following recursion (which ty does *not* perform), so a
/// pathological `A = B; B = A` can't loop. Legitimate re-export chains are
/// only a hop or two deep.
const MEMBER_RESOLVE_DEPTH_CAP: u32 = 16;

/// Raw, *local-scope-only* definitions bound to `name` in `file`'s global
/// scope — bindings and declarations, no cross-file alias resolution.
/// Mirrors ty's internal `find_symbol_in_scope` (which is private to
/// `ide_support`), the building block for the store-side first-hop
/// classification.
fn local_member_defs<'db>(db: &'db dyn ProjectDb, file: File, name: &str) -> Vec<Definition<'db>> {
    let scope = global_scope(db, program_file(db, file));
    let table = place_table(db, scope);
    let Some(symbol_id) = table.symbol_id(name) else {
        return Vec::new();
    };
    let use_def = use_def_map(db, scope);
    let mut defs = Vec::new();
    for binding in use_def.reachable_symbol_bindings(symbol_id) {
        if let Some(def) = binding.binding.definition() {
            defs.push(def);
        }
    }
    for declaration in use_def.reachable_symbol_declarations(symbol_id) {
        if let Some(def) = declaration.declaration.definition() {
            defs.push(def);
        }
    }
    defs
}

/// Describe one class-base expression as a [`MemberSpec`] using only
/// `file`'s own use-def chain — the first symbolic hop, no cross-file
/// resolution (that's [`resolve_member_def`]'s job). Subscripted bases
/// (`Base[T]`, `Generic[T]`) are unwrapped to the bare callee. `depth`
/// bounds same-file alias chains (`Z = Y; class C(Z)`).
pub(crate) fn classify_base(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    expr: &Expr,
    depth: u32,
    kinds: MemberKinds,
) -> Option<MemberSpec> {
    if depth > MEMBER_RESOLVE_DEPTH_CAP {
        return None;
    }
    let expr = match expr {
        Expr::Subscript(subscript) => subscript.value.as_ref(),
        other => other,
    };
    match expr {
        Expr::Name(name) => {
            let symbol = name.id.as_str();
            local_member_defs(db, file, symbol)
                .into_iter()
                .find_map(|def| classify_name_def(db, file, parsed, def, symbol, depth, kinds))
        }
        Expr::Attribute(_) => {
            let (module, name) = attribute_to_module_member(db, file, parsed, expr)?;
            Some(MemberSpec::ModuleMember { module, name })
        }
        _ => None,
    }
}

/// Describe a decorator's head expression (see [`decorator_head`]) as a
/// [`MemberSpec`] using only `file`'s own use-def chain — the decorator
/// twin of [`classify_base`], accepting functions as well as classes:
///
/// * a bare name bound to a same-file `def` / `class`, or to a same-file
///   variable whose value can't be followed (`deco = Deco(...)`) →
///   `Local(range)`;
/// * a bare name bound by `from M import n` / `from M import *` / a
///   same-file alias of one → `ModuleMember { M, n }`;
/// * `mod.deco` / `pkg.mod.deco` / `alias.deco` on an imported module →
///   `ModuleMember` via the file's imports.
///
/// `None` for builtins / unbound names and for decorators whose head is
/// not a module member at all (`@app.route` on an instance — those are
/// the `handler_decorators` family's business).
pub(crate) fn classify_decorator_head(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    expr: &Expr,
) -> Option<MemberSpec> {
    let expr = unwrap_subscripted_callee(expr);
    match expr {
        Expr::Name(name) => {
            let symbol = name.id.as_str();
            local_member_defs(db, file, symbol)
                .into_iter()
                .find_map(|def| {
                    classify_name_def(db, file, parsed, def, symbol, 0, MemberKinds::CALLABLE)
                })
        }
        Expr::Attribute(_) => attribute_to_module_member(db, file, parsed, expr)
            .map(|(module, name)| MemberSpec::ModuleMember { module, name }),
        _ => None,
    }
}

/// Classify one local definition of a bare-name base (`name`, the symbol
/// being referenced) into a [`MemberSpec`]: a same-file class, a
/// `from`-import member, a `from X import *` member, or a same-file
/// assignment alias whose RHS is followed recursively.
fn classify_name_def<'db>(
    db: &'db dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    def: Definition<'db>,
    name: &str,
    depth: u32,
    kinds: MemberKinds,
) -> Option<MemberSpec> {
    match def.kind(db) {
        DefinitionKind::Class(class) if kinds.accepts_class() => Some(MemberSpec::Local(
            range_key(class.node(parsed).name.range()),
        )),
        DefinitionKind::Function(func) if kinds.accepts_function() => {
            Some(MemberSpec::Local(range_key(func.node(parsed).name.range())))
        }
        DefinitionKind::ImportFrom(import_from) => {
            let module = from_module_string(db, file, import_from.import(parsed));
            if module.is_empty() {
                return None;
            }
            Some(MemberSpec::ModuleMember {
                module,
                name: import_from.alias(parsed).name.as_str().to_compact_string(),
            })
        }
        DefinitionKind::StarImport(star_import) => {
            let module = from_module_string(db, file, star_import.import(parsed));
            if module.is_empty() {
                return None;
            }
            Some(MemberSpec::ModuleMember {
                module,
                name: name.to_compact_string(),
            })
        }
        // An assignment is followed through its value (`Alias = Real`);
        // when the value leads nowhere and variables are acceptable, the
        // assignment itself is the definition (`deco = Deco(...)`).
        DefinitionKind::Assignment(assign) => {
            classify_base(db, file, parsed, assign.value(parsed), depth + 1, kinds).or_else(|| {
                kinds
                    .accepts_variable()
                    .then(|| MemberSpec::Local(range_key(assign.target(parsed).range())))
            })
        }
        DefinitionKind::AnnotatedAssignment(assign) => {
            let value = assign.value(parsed)?;
            classify_base(db, file, parsed, value, depth + 1, kinds).or_else(|| {
                kinds
                    .accepts_variable()
                    .then(|| MemberSpec::Local(range_key(assign.target(parsed).range())))
            })
        }
        _ => None,
    }
}

/// The [`ModuleValue`]s a global-scope definition denotes, for
/// [`crate::file_payload::file_to_nodes`]: a variable's assigned value
/// (`m = config`, `m = importlib.import_module('pkg.config')`,
/// `m = pkg.sub`) or, for a function, the value of *calling* it — every
/// `return` expression in its body that denotes a module. Same-file
/// hops (a variable bound to another variable, a call to a same-file
/// function) are folded in here via this file's use-def chain, mirroring
/// [`classify_base`]; nothing cross-file is read. Empty for anything
/// that cannot syntactically denote a module.
pub(crate) fn module_values_for_def(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    kind: &DefinitionKind<'_>,
) -> Vec<ModuleValue> {
    let mut out: Vec<ModuleValue> = Vec::new();
    match kind {
        DefinitionKind::Assignment(assign) => {
            classify_module_value(db, file, parsed, assign.value(parsed), &[], 0, &mut out);
        }
        DefinitionKind::AnnotatedAssignment(assign) => {
            if let Some(value) = assign.value(parsed) {
                classify_module_value(db, file, parsed, value, &[], 0, &mut out);
            }
        }
        DefinitionKind::Function(func) => {
            let mut returns: Vec<&Expr> = Vec::new();
            collect_return_values(&func.node(parsed).body, &mut returns);
            for value in returns {
                classify_module_value(db, file, parsed, value, &[], 0, &mut out);
            }
        }
        _ => {}
    }
    out
}

/// Describe the module(s) `expr` denotes, with `trailing` steps applied
/// after it, into `out`. Peels the access chain to its root, then:
///
/// * a `Name` is classified through each of its reachable definitions
///   ([`classify_module_value_def`]);
/// * a dynamic-import call with a literal target is that module
///   ([`dynamic_call_module`]);
/// * anything else denotes nothing.
fn classify_module_value(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    expr: &Expr,
    trailing: &[ChainStep],
    depth: u32,
    out: &mut Vec<ModuleValue>,
) {
    if depth > MEMBER_RESOLVE_DEPTH_CAP {
        return;
    }
    let (root, mut steps, _calls) = peel_value_chain(expr);
    steps.extend_from_slice(trailing);
    match root {
        Expr::Name(name) => {
            let symbol = name.id.as_str();
            for def in local_member_defs(db, file, symbol) {
                classify_module_value_def(db, file, parsed, def, symbol, &steps, depth, out);
            }
        }
        Expr::Call(call) => {
            if let Some(module) = dynamic_call_module(db, file, call) {
                out.push(ModuleValue {
                    spec: ImportPayload {
                        module,
                        decl: None,
                        star: false,
                    },
                    bound_name: CompactString::default(),
                    steps,
                });
            }
        }
        _ => {}
    }
}

/// One definition of a chain root `name`, with `steps` applied to it:
/// an import binding is the module its payload names; an assignment
/// follows its right-hand side; a function consumes a leading `Call`
/// step and follows its `return` expressions. Anything else (a class,
/// a parameter, …) denotes no module.
#[allow(clippy::too_many_arguments)]
fn classify_module_value_def<'db>(
    db: &'db dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    def: Definition<'db>,
    name: &str,
    steps: &[ChainStep],
    depth: u32,
    out: &mut Vec<ModuleValue>,
) {
    let kind = def.kind(db);
    match kind {
        DefinitionKind::Import(_)
        | DefinitionKind::ImportFrom(_)
        | DefinitionKind::ImportFromSubmodule(_)
        | DefinitionKind::StarImport(_) => {
            let spec = import_payload_for_pure(kind, db, file, parsed);
            if !spec.module.is_empty() {
                out.push(ModuleValue {
                    spec,
                    bound_name: name.to_compact_string(),
                    steps: steps.to_vec(),
                });
            }
        }
        DefinitionKind::Assignment(assign) => {
            classify_module_value(
                db,
                file,
                parsed,
                assign.value(parsed),
                steps,
                depth + 1,
                out,
            );
        }
        DefinitionKind::AnnotatedAssignment(assign) => {
            if let Some(value) = assign.value(parsed) {
                classify_module_value(db, file, parsed, value, steps, depth + 1, out);
            }
        }
        DefinitionKind::Function(func) => {
            let Some((ChainStep::Call, rest)) = steps.split_first() else {
                return;
            };
            let mut returns: Vec<&Expr> = Vec::new();
            collect_return_values(&func.node(parsed).body, &mut returns);
            for value in returns {
                classify_module_value(db, file, parsed, value, rest, depth + 1, out);
            }
        }
        _ => {}
    }
}

/// Every `return <expr>` value in a function body, entering compound
/// statements but not nested `def` / `class` bodies (their returns
/// belong to them).
fn collect_return_values<'ast>(body: &'ast [Stmt], out: &mut Vec<&'ast Expr>) {
    for stmt in body {
        match stmt {
            Stmt::Return(r) => {
                if let Some(value) = &r.value {
                    out.push(value.as_ref());
                }
            }
            Stmt::If(s) => {
                collect_return_values(&s.body, out);
                for clause in &s.elif_else_clauses {
                    collect_return_values(&clause.body, out);
                }
            }
            Stmt::Try(s) => {
                collect_return_values(&s.body, out);
                for handler in &s.handlers {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
                    collect_return_values(&h.body, out);
                }
                collect_return_values(&s.orelse, out);
                collect_return_values(&s.finalbody, out);
            }
            Stmt::For(s) => {
                collect_return_values(&s.body, out);
                collect_return_values(&s.orelse, out);
            }
            Stmt::While(s) => {
                collect_return_values(&s.body, out);
                collect_return_values(&s.orelse, out);
            }
            Stmt::With(s) => collect_return_values(&s.body, out),
            Stmt::Match(s) => {
                for case in &s.cases {
                    collect_return_values(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

/// Decompose an attribute-form base (`mod.Base`, `pkg.mod.Base`,
/// `alias.Base`) into `(absolute_module, member_name)`, resolving the
/// chain root to a module prefix via `file`'s imports and extending it
/// with every segment but the last.
fn attribute_to_module_member(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    expr: &Expr,
) -> Option<(CompactString, CompactString)> {
    let (root, segments) = collapse_attribute_chain(expr)?;
    let (member, middle) = segments.split_last()?;
    let prefix = root_module_prefix(db, file, parsed, root)?;
    let module = if middle.is_empty() {
        CompactString::from(prefix)
    } else {
        compact_str::format_compact!("{prefix}.{}", middle.join("."))
    };
    Some((module, (*member).to_compact_string()))
}

/// Resolve the chain-root name of an attribute-form base to the absolute
/// module it refers to: an `import a.b[ as r]` binding (→ `a.b`), or a
/// `from pkg import mod` binding (→ `pkg.mod`). Returns `None` for
/// anything that isn't an import of a module.
fn root_module_prefix(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
    root: &ExprName,
) -> Option<String> {
    local_member_defs(db, file, root.id.as_str())
        .into_iter()
        .find_map(|def| match def.kind(db) {
            DefinitionKind::Import(import) => {
                let alias = import.alias(parsed);
                Some(if alias.asname.is_some() {
                    alias.name.as_str().to_string()
                } else {
                    root.id.as_str().to_string()
                })
            }
            DefinitionKind::ImportFrom(import_from) => {
                let from_module = from_module_string(db, file, import_from.import(parsed));
                if from_module.is_empty() {
                    return None;
                }
                Some(format!(
                    "{from_module}.{}",
                    import_from.alias(parsed).name.as_str()
                ))
            }
            _ => None,
        })
}

/// Resolve a [`MemberSpec`] to the canonical `(File, name_range)` of
/// the definition it names. `Local` lands on `local_file` (the file
/// the spec was produced/followed in); `ModuleMember` resolves through
/// [`resolve_member_def`] from `anchor`. Both the assemble pass and the
/// assignment-following path funnel through here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_base_spec(
    db: &dyn ProjectDb,
    spec: &MemberSpec,
    local_file: File,
    anchor: File,
    depth: u32,
    reloads: &ReloadLog,
    touched: &mut crate::refspec::Touched,
    kinds: MemberKinds,
) -> Option<(File, TextRange)> {
    match spec {
        MemberSpec::Local((start, end)) => {
            Some((local_file, TextRange::new((*start).into(), (*end).into())))
        }
        MemberSpec::ModuleMember { module, name } => {
            resolve_member_def(db, module, name, anchor, depth, reloads, touched, kinds)
        }
    }
}

/// Resolve an absolute `module` string + member `name` to the canonical
/// `(File, name_range)` of the definition it ultimately names, as seen
/// from `anchor`. Re-exports are followed by [`resolve_member_in_file`],
/// which reuses the store-side decomposition. `depth` bounds the chase.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_member_def(
    db: &dyn ProjectDb,
    module: &str,
    name: &str,
    anchor: File,
    depth: u32,
    reloads: &ReloadLog,
    touched: &mut crate::refspec::Touched,
    kinds: MemberKinds,
) -> Option<(File, TextRange)> {
    if depth > MEMBER_RESOLVE_DEPTH_CAP {
        return None;
    }
    let module_name = ModuleName::new(module)?;
    let module_file = resolve_module(db, importing_file(db, anchor), &module_name)?.file(db)?;
    resolve_member_in_file(
        db,
        module_file,
        name,
        anchor,
        depth,
        reloads,
        touched,
        kinds,
    )
}

/// Resolve `name` in `file`'s global scope to a canonical
/// `(File, name_range)`, reusing [`classify_name_def`] — the same
/// decomposition the store side runs on observed bases. A class binding
/// lands directly; an import re-export (`from x import name`,
/// `from x import *`) recurses cross-file through [`resolve_member_def`];
/// an assignment binding (`Alias = Real`) follows its RHS. `depth` bounds
/// the recursion (cross-file import hops and assignment chains both
/// increment it), standing in for ty's visited-set cycle guard.
#[allow(clippy::too_many_arguments)]
fn resolve_member_in_file(
    db: &dyn ProjectDb,
    file: File,
    name: &str,
    anchor: File,
    depth: u32,
    reloads: &ReloadLog,
    touched: &mut crate::refspec::Touched,
    kinds: MemberKinds,
) -> Option<(File, TextRange)> {
    // This load (and `local_member_defs`'s semantic-index load below)
    // repopulates salsa slots the populate-phase eviction emptied —
    // record the file so the post-plugin-pass sweep can re-clear
    // exactly the touched set. The `touched` record is the resolve
    // cache's eviction twin: every file whose *content* this
    // resolution reads, so a class-base memo entry re-resolves iff one
    // of these files changed.
    reloads.record(file);
    touched.record(file);
    let parsed = parsed_module(db, python_file(db, file)).load(db);
    local_member_defs(db, file, name)
        .into_iter()
        .find_map(|def| {
            let spec = classify_name_def(db, file, &parsed, def, name, depth, kinds)?;
            resolve_base_spec(db, &spec, file, anchor, depth + 1, reloads, touched, kinds)
        })
}

/// Locate the top-level ``if __name__ == "__main__":`` block in a
/// parsed module and return its source range. Both orderings
/// (``__name__ == "__main__"`` and ``"__main__" == __name__``) are
/// recognized; the ``elif`` / ``else`` branches are not matched.
pub(crate) fn find_main_block_range(parsed: &ParsedModuleRef) -> Option<TextRange> {
    for stmt in &parsed.syntax().body {
        let Stmt::If(if_stmt) = stmt else {
            continue;
        };
        if is_name_eq_main(&if_stmt.test) {
            return Some(if_stmt.range);
        }
    }
    None
}

/// Match ``__name__ == "__main__"`` or its reverse.
pub(crate) fn is_name_eq_main(expr: &Expr) -> bool {
    let Expr::Compare(cmp) = expr else {
        return false;
    };
    if cmp.ops.len() != 1 {
        return false;
    }
    if !matches!(cmp.ops[0], ruff_python_ast::CmpOp::Eq) {
        return false;
    }
    let comparators = &cmp.comparators;
    if comparators.len() != 1 {
        return false;
    }
    let left = &*cmp.left;
    let right = &comparators[0];
    (is_name(left, "__name__") && is_string_literal(right, "__main__"))
        || (is_string_literal(left, "__main__") && is_name(right, "__name__"))
}

pub(crate) fn is_name(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Name(n) if n.id.as_str() == value)
}

pub(crate) fn is_string_literal(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::StringLiteral(s) if s.value.to_str() == value)
}

/// Walk every `Stmt::Import` / `Stmt::ImportFrom` in `parsed` and
/// produce `local_name -> upstream_fqname` for every alias.
///
/// Unlike :fn:`collect_module_imports_local` (which filters to a single
/// `module` + `allowed` name set), this captures every top-level
/// import so callers can resolve arbitrary names referenced in
/// arg / kwarg positions.
///
/// Mappings produced:
/// * ``from foo.bar import Baz`` → ``"Baz" -> "foo.bar.Baz"``
/// * ``from foo.bar import Baz as B`` → ``"B" -> "foo.bar.Baz"``
/// * ``from .bases import Baz`` (in package ``pkg``) →
///   ``"Baz" -> "pkg.bases.Baz"`` — relative dots are resolved against
///   ``file_package`` via ``im.level``. ``file_package`` is the
///   importing file's enclosing package (``None`` for a top-level
///   module, in which case a relative import is dropped).
/// * ``import foo`` → ``"foo" -> "foo"``
/// * ``import foo.bar as fb`` → ``"fb" -> "foo.bar"``
/// * ``import foo.bar`` → ``"foo" -> "foo"`` (Python binds the
///   leftmost segment; the runtime module ``foo`` is what the name
///   resolves to).
pub(crate) fn collect_all_imports_local(
    parsed: &ParsedModuleRef,
    file_package: Option<&str>,
) -> LocalImports {
    let mut out: LocalImports = FxHashMap::default();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                // Resolve the source module to an absolute dotted path.
                // `im.module` is only the suffix after the dots; the dot
                // count lives in `im.level`, so a relative re-export like
                // `from .bases import TestCase` resolves against the
                // importing file's package instead of being mis-keyed to
                // a bogus top-level `bases.TestCase`.
                let tail = im.module.as_ref().map(|n| n.as_str()).unwrap_or("");
                let module = if im.level == 0 {
                    (!tail.is_empty()).then(|| tail.to_compact_string())
                } else {
                    resolve_relative_import(im.level, tail, file_package)
                };
                let Some(module) = module else { continue };
                for alias in &im.names {
                    let target = alias.name.as_str();
                    let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                    out.insert(
                        local.to_compact_string(),
                        compact_str::format_compact!("{module}.{target}"),
                    );
                }
            }
            Stmt::Import(im) => {
                for alias in &im.names {
                    let name = alias.name.as_str();
                    if let Some(asname) = alias.asname.as_ref() {
                        // ``import foo.bar as fb`` binds ``fb`` to the
                        // ``foo.bar`` module object.
                        out.insert(
                            asname.as_str().to_compact_string(),
                            name.to_compact_string(),
                        );
                    } else {
                        // ``import foo`` binds ``foo``.
                        // ``import foo.bar`` binds ``foo`` (Python
                        // binds the leftmost segment).
                        let leftmost = name.split('.').next().unwrap_or(name);
                        out.insert(leftmost.to_compact_string(), leftmost.to_compact_string());
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Convenience: build the file-local imports map for names imported
/// from any of ``modules`` (OR semantics — merged into a single map).
/// Later modules win on local-name collisions, which doesn't matter in
/// practice since callers route the matched ``target`` (the upstream
/// constructor name) against an ``allowed`` set; identical targets
/// across modules collapse to the same value.
///
/// ``file_package`` is the importing file's enclosing package (e.g.
/// ``Some("pkg")`` for ``pkg/handlers.py``, ``Some("pkg")`` for
/// ``pkg/__init__.py``, ``None`` for a top-level ``mod.py``). It is
/// used to resolve ``from .foo import N`` / ``from ..bar import N``
/// relative imports against ``modules``.
pub(crate) fn collect_modules_imports_local(
    parsed: &ParsedModuleRef,
    modules: &[String],
    allowed: &FxHashSet<&str>,
    file_package: Option<&str>,
) -> LocalImports {
    let mut out: LocalImports = FxHashMap::default();
    for module in modules {
        let one = collect_module_imports_local(parsed, module, allowed, file_package);
        out.extend(one);
    }
    out
}

/// Compute the absolute module path for a relative ``from`` import.
///
/// ``level`` is the count of leading dots on the ``from`` statement
/// (1 for ``from .x``, 2 for ``from ..x``, ...); ``tail`` is the
/// dotted suffix after the dots (``"x"`` for ``from .x``,
/// ``"x.y"`` for ``from .x.y``, empty for ``from . import name``).
/// ``file_package`` is the importing file's enclosing package.
///
/// Mirrors CPython's relative-import resolution: ``level=1`` anchors
/// at ``file_package``, each additional level strips one trailing
/// segment. Returns ``None`` when the resolution would walk past the
/// top of ``file_package`` (or when ``file_package`` is ``None`` but
/// ``level > 0``).
pub(crate) fn resolve_relative_import(
    level: u32,
    tail: &str,
    file_package: Option<&str>,
) -> Option<CompactString> {
    if level == 0 {
        return None;
    }
    let pkg = file_package?;
    let segments: Vec<&str> = pkg.split('.').filter(|s| !s.is_empty()).collect();
    let levels_up = (level - 1) as usize;
    if levels_up > segments.len() {
        return None;
    }
    let base = &segments[..segments.len() - levels_up];
    let mut parts: Vec<&str> = base.to_vec();
    if !tail.is_empty() {
        parts.push(tail);
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join(".").into())
}

/// Build the file-local imports map ``{local_name: target}`` for
/// names imported from ``module``. ``target`` is the upstream
/// constructor / decl name (e.g. ``"Flask"`` when bound via
/// ``from flask import Flask``) or [`MODULE_ALIAS_MARKER`] when bound
/// via ``import flask`` / ``import flask as f``.
///
/// Only entries whose target is in ``allowed`` survive — keeps the
/// map small and lets call-site matchers do a cheap second check.
pub(crate) fn collect_module_imports_local(
    parsed: &ParsedModuleRef,
    module: &str,
    allowed: &FxHashSet<&str>,
    file_package: Option<&str>,
) -> LocalImports {
    // Submodule binding: ``from <parent> import <last_seg>`` makes
    // ``last_seg`` a local alias for the queried module (e.g.
    // ``from unittest import mock`` for module ``unittest.mock``).
    let parent_last = module.rsplit_once('.');
    let mut out = FxHashMap::default();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                // Compute the import's effective absolute module path,
                // resolving relative dots against the importing file's
                // package. ``mod_name`` is just the dotted suffix; ty
                // exposes the dot count via ``im.level``.
                let tail = im.module.as_ref().map(|n| n.as_str()).unwrap_or("");
                let absolute: Option<CompactString> = if im.level == 0 {
                    if tail.is_empty() {
                        None
                    } else {
                        Some(tail.to_compact_string())
                    }
                } else {
                    resolve_relative_import(im.level, tail, file_package)
                };
                let Some(absolute) = absolute else { continue };
                if absolute == module {
                    for alias in &im.names {
                        let target = alias.name.as_str();
                        if !allowed.contains(target) {
                            continue;
                        }
                        let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                        out.insert(local.to_compact_string(), target.to_compact_string());
                    }
                } else if let Some((parent, last)) = parent_last {
                    if absolute == parent {
                        for alias in &im.names {
                            if alias.name.as_str() != last {
                                continue;
                            }
                            let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(last);
                            out.insert(
                                local.to_compact_string(),
                                CompactString::const_new(MODULE_ALIAS_MARKER),
                            );
                        }
                    }
                }
            }
            Stmt::Import(im) => {
                for alias in &im.names {
                    if alias.name.as_str() != module {
                        continue;
                    }
                    let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(module);
                    out.insert(
                        local.to_compact_string(),
                        CompactString::const_new(MODULE_ALIAS_MARKER),
                    );
                }
            }
            _ => {}
        }
    }
    out
}

/// Unwrap a leading ``Expr::Subscript`` once so subscripted-generic
/// callees (``Worker[int]``, ``app.route[Self]``) classify the same as
/// their bare form. Returns the inner expression for any other shape.
pub(crate) fn unwrap_subscripted_callee(expr: &Expr) -> &Expr {
    match expr {
        Expr::Subscript(sub) => sub.value.as_ref(),
        other => other,
    }
}

/// Multi-module variant of [`matched_call_target`]: returns the matched
/// upstream name as soon as any of ``modules`` produces a hit. Used by
/// the query forms that accept a list of modules to match against.
pub(crate) fn matched_call_target_any(
    call: &ruff_python_ast::ExprCall,
    imports: &LocalImports,
    modules: &[String],
    allowed: &FxHashSet<&str>,
) -> Option<CompactString> {
    for module in modules {
        if let Some(name) = matched_call_target(call, imports, module, allowed) {
            return Some(name);
        }
    }
    None
}

/// Match an expression against a callable-from-module pattern. Three
/// shapes resolve through ``imports`` to the configured ``module``:
///
/// * ``<imported_local>(...)`` — local was bound via ``from <module>
///   import <name>``;
/// * ``<module_alias>.<allowed_name>(...)`` — alias bound via
///   ``import <module> [as <alias>]``;
/// * ``<m>.<n>.…<allowed_name>(...)`` — literal dotted access of a
///   multi-segment ``<module>`` (e.g. ``import unittest.mock;
///   unittest.mock.patch(...)``).
///
/// Returns the matched upstream name (``"Flask"``) on hit, else ``None``.
pub(crate) fn matched_call_target(
    call: &ruff_python_ast::ExprCall,
    imports: &LocalImports,
    module: &str,
    allowed: &FxHashSet<&str>,
) -> Option<CompactString> {
    // Unwrap a leading ``Expr::Subscript`` once so subscripted-generic
    // constructors (``Worker[int](...)``, ``Logger[Self]("foo")``)
    // classify the same as their bare ``Worker(...)`` form. Generics
    // applied to imported classes are syntactically a ``Subscript``
    // whose ``value`` is the bare ``Name`` / ``Attribute`` we already
    // know how to resolve.
    let callee = unwrap_subscripted_callee(call.func.as_ref());
    match callee {
        Expr::Name(name) => {
            let target = imports.get(name.id.as_str())?;
            allowed.contains(target.as_str()).then(|| target.clone())
        }
        Expr::Attribute(attr) => {
            let attr_name = attr.attr.as_str();
            if !allowed.contains(attr_name) {
                return None;
            }
            match attr.value.as_ref() {
                Expr::Name(prefix) => (imports.get(prefix.id.as_str()).map(CompactString::as_str)
                    == Some(MODULE_ALIAS_MARKER))
                .then(|| attr_name.to_compact_string()),
                _ => {
                    let (root, segs) = collapse_attribute_chain(attr.value.as_ref())?;
                    let mut dotted = String::with_capacity(module.len());
                    dotted.push_str(root.id.as_str());
                    for seg in &segs {
                        dotted.push('.');
                        dotted.push_str(seg);
                    }
                    (dotted == module).then(|| attr_name.to_compact_string())
                }
            }
        }
        _ => None,
    }
}

/// Top-level ``X = <expr>`` / ``X: T = <expr>`` where the target is a
/// bare ``Name``. Returns ``(name_range, &expr)`` so the caller can
/// look up ``X``'s node and inspect the RHS.
pub(crate) fn top_level_assign_to_name(stmt: &Stmt) -> Option<(TextRange, &Expr)> {
    match stmt {
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return None;
            }
            let Expr::Name(name) = &assign.targets[0] else {
                return None;
            };
            Some((name.range, assign.value.as_ref()))
        }
        Stmt::AnnAssign(assign) => {
            let value = assign.value.as_deref()?;
            let Expr::Name(name) = assign.target.as_ref() else {
                return None;
            };
            Some((name.range, value))
        }
        _ => None,
    }
}

/// Recursive visitor: walk a function / class body collecting the set
/// of constructor names called anywhere inside it. Matches against any
/// of ``modules`` (OR semantics — for single-module callers pass a
/// one-element slice).
pub(crate) struct FactoryCallFinder<'a> {
    pub(crate) imports: &'a LocalImports,
    pub(crate) modules: &'a [String],
    pub(crate) allowed: &'a FxHashSet<&'a str>,
    pub(crate) kinds: FxHashSet<String>,
}

impl<'ast, 'a> Visitor<'ast> for FactoryCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Some(name) =
                matched_call_target_any(call, self.imports, self.modules, self.allowed)
            {
                self.kinds.insert(name.into_string());
            }
        }
        walk_expr(self, expr);
    }
}

/// A string literal extracted from a call keyword argument, or
/// ``Unknown`` for anything else. The only in-tree consumer is the
/// pytest plugin, which reads the ``name=`` alias on ``@pytest.fixture``;
/// every non-string expression collapses to ``Unknown``. Re-exported from
/// [`crate::native_plugins::plugin_api`] (pyo3-free) so external plugins
/// can read decorator/constructor arguments too.
#[derive(Clone, Debug, PartialEq, Eq, salsa::SalsaValue, get_size2::GetSize)]
pub enum ArgValue {
    Str(CompactString),
    Unknown,
}

/// Captured keyword arguments of a matched call/decorator, keyed by name.
/// The map itself is crate-internal (it carries an `FxHashMap`, which the
/// curated airlock doesn't expose); external plugins read values through
/// the [`CallArgs::get`] / [`CallArgs::str_value`] accessors.
#[derive(Clone, Debug, Default, PartialEq, Eq, salsa::SalsaValue, get_size2::GetSize)]
pub struct CallArgs {
    pub(crate) kwargs: FxHashMap<CompactString, ArgValue>,
}

impl CallArgs {
    /// The captured value of keyword argument `key`, if present.
    pub fn get(&self, key: &str) -> Option<&ArgValue> {
        self.kwargs.get(key)
    }

    /// The string value of keyword argument `key`, if it was captured as a
    /// string literal (the `@pytest.fixture(name="…")` shape). `None` for a
    /// missing key or a non-string value.
    pub fn str_value(&self, key: &str) -> Option<&str> {
        match self.kwargs.get(key) {
            Some(ArgValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Tuple-like result row returned by
/// :meth:`ProjectContext.node_attrs` and the per-query
/// ``.attrs()`` terminals.
///
/// Supports both attribute access (``attr.fqname``) and tuple
/// semantics (``kind, path, fqname, flags = attr``; ``attr[2]``;
/// ``len(attr) == 4``). Not a `typing.NamedTuple` instance, but
/// drop-in compatible for unpacking and subscript access. Frozen —
/// fields are immutable once constructed.
#[pyclass(frozen, get_all)]
pub(crate) struct NodeAttrs {
    pub(crate) kind: String,
    pub(crate) path: String,
    pub(crate) fqname: String,
    pub(crate) flags: u32,
}

#[pymethods]
impl NodeAttrs {
    fn __len__(&self) -> usize {
        4
    }

    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n: isize = 4;
        let i = if idx < 0 { idx + n } else { idx };
        if !(0..n).contains(&i) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "NodeAttrs index {idx} out of range (len=4)"
            )));
        }
        Ok(match i {
            0 => self.kind.clone().into_py(py),
            1 => self.path.clone().into_py(py),
            2 => self.fqname.clone().into_py(py),
            3 => self.flags.into_py(py),
            _ => unreachable!(),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tup = pyo3::types::PyTuple::new_bound(
            py,
            &[
                slf.kind.clone().into_py(py),
                slf.path.clone().into_py(py),
                slf.fqname.clone().into_py(py),
                slf.flags.into_py(py),
            ],
        );
        Ok(tup.call_method0("__iter__")?.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeAttrs(kind={:?}, path={:?}, fqname={:?}, flags={})",
            self.kind, self.path, self.fqname, self.flags
        )
    }
}

/// Convert one AST argument expression to an ``ArgValue``. Anything that
/// folds to a static string under `ctx` (see [`crate::string_fold`]) is
/// captured; every other expression is ``Unknown``.
pub(crate) fn extract_arg_value(expr: &Expr, ctx: &StringFoldCtx<'_>) -> ArgValue {
    match fold_string_expr(expr, ctx) {
        Some(s) => ArgValue::Str(s),
        None => ArgValue::Unknown,
    }
}

/// Extract keyword arguments from a call. Positional args are not
/// captured — no consumer reads them.
pub(crate) fn extract_call_kwargs(
    call: &ruff_python_ast::ExprCall,
    ctx: &StringFoldCtx<'_>,
) -> CallArgs {
    let mut kwargs: FxHashMap<CompactString, ArgValue> = FxHashMap::default();
    for kw in &call.arguments.keywords {
        let Some(name) = kw.arg.as_ref() else {
            continue;
        };
        kwargs.insert(
            name.as_str().to_compact_string(),
            extract_arg_value(&kw.value, ctx),
        );
    }
    CallArgs { kwargs }
}

pub(crate) fn rel_path<P: AsRef<str>>(path: P) -> RelativePathBuf {
    RelativePathBuf::cli(SystemPath::new(path.as_ref()))
}

/// Per-source-type default flags for ``.ipynb``: notebook decls are
/// always alive (cells run top-to-bottom, not imported) and the
/// codemod must skip them (it can't rewrite the cell JSON envelope).
/// The `NOTEBOOK` bit alone is enough — the Python-side
/// `KEEPALIVE_DEFAULT` mask includes `NOTEBOOK`, so reachability seeds
/// from notebook nodes without needing the `ENTRYPOINT` overlay.
pub(crate) const NODE_FLAGS_NOTEBOOK_DEFAULT: u32 = NodeFlags::NOTEBOOK;

/// Bits stamped on every import alias pinned by a noqa directive. The
/// `NOQA` bit alone is enough — the Python-side `KEEPALIVE_DEFAULT`
/// mask includes `NOQA`, so reachability seeds from noqa-pinned
/// aliases and the `kept_alive_by_flags_only(NOQA)` blast-radius query
/// isolates them.
pub(crate) const NODE_FLAGS_NOQA_PIN: u32 = NodeFlags::NOQA;

/// Internal aliases for the pyclass classattrs used by the call sites
/// scattered through this file. Read as bare constants rather than
/// `NodeFlags::ENTRYPOINT` (which would force every reader to chase the
/// `pyclass` macro to decide whether it's a runtime lookup or a const).
pub(crate) const NODE_FLAG_ENTRYPOINT: u32 = NodeFlags::ENTRYPOINT;
pub(crate) const NODE_FLAG_UNRESOLVED: u32 = NodeFlags::UNRESOLVED;
pub(crate) const EDGE_FLAG_DEAD_BRANCH: u8 = EdgeFlags::DEAD_BRANCH;
pub(crate) const EDGE_FLAG_DYNAMIC_IMPORT: u8 = EdgeFlags::DYNAMIC_IMPORT;

/// Collect the source ranges of every statically-dead statement in
/// `parsed` — bodies of `if False` / `else` of `if True` / `while False`
/// / `while True else` branches, statements after an unconditional
/// terminator (`return` / `raise` / `break` / `continue` /
/// `assert <falsy>`), and the suites of compound terminators (an
/// `if/elif/else` whose every branch terminates, a `try` whose body
/// and every handler terminate, a `with` whose body terminates).
///
/// The result is a flat `Vec<TextRange>`; callers use
/// `TextRange::contains_range` to test whether a use site sits
/// inside any of them.
/// Per-scope name → known-truthiness map. Populated by
/// `build_scope_table` from the scope's literal assignments,
/// annotated assignments, and walrus expressions. Function and class
/// bodies inherit and override their enclosing scope's table.
pub(crate) type NameTable = FxHashMap<String, bool>;

pub(crate) fn detect_dead_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut dead = Vec::new();
    let empty: NameTable = FxHashMap::default();
    let module_table = build_scope_table(&parsed.syntax().body, &empty);
    walk_suite_for_dead(&parsed.syntax().body, &module_table, &mut dead);
    dead
}

pub(crate) fn walk_suite_for_dead(stmts: &[Stmt], table: &NameTable, dead: &mut Vec<TextRange>) {
    let mut killed = false;
    for stmt in stmts {
        if killed {
            // Past an in-suite terminator. Mark this stmt's range as
            // dead and don't recurse — descendant uses are covered by
            // the outer range.
            dead.push(stmt.range());
            continue;
        }
        walk_compound_for_dead(stmt, table, dead);
        if stmt_is_terminator(stmt, table) {
            killed = true;
        }
    }
}

pub(crate) fn walk_compound_for_dead(stmt: &Stmt, table: &NameTable, dead: &mut Vec<TextRange>) {
    match stmt {
        Stmt::If(if_stmt) => {
            let mut taken = false;
            match evaluate_truthiness(&if_stmt.test, table) {
                Some(true) => {
                    walk_suite_for_dead(&if_stmt.body, table, dead);
                    taken = true;
                }
                Some(false) => {
                    for s in &if_stmt.body {
                        dead.push(s.range());
                    }
                }
                None => {
                    walk_suite_for_dead(&if_stmt.body, table, dead);
                }
            }
            for clause in &if_stmt.elif_else_clauses {
                if taken {
                    dead.push(clause.range);
                    continue;
                }
                match &clause.test {
                    Some(test) => match evaluate_truthiness(test, table) {
                        Some(true) => {
                            walk_suite_for_dead(&clause.body, table, dead);
                            taken = true;
                        }
                        Some(false) => {
                            for s in &clause.body {
                                dead.push(s.range());
                            }
                        }
                        None => walk_suite_for_dead(&clause.body, table, dead),
                    },
                    None => walk_suite_for_dead(&clause.body, table, dead),
                }
            }
        }
        Stmt::While(w) => match evaluate_truthiness(&w.test, table) {
            Some(false) => {
                for s in &w.body {
                    dead.push(s.range());
                }
                walk_suite_for_dead(&w.orelse, table, dead);
            }
            Some(true) => {
                walk_suite_for_dead(&w.body, table, dead);
                for s in &w.orelse {
                    dead.push(s.range());
                }
            }
            None => {
                walk_suite_for_dead(&w.body, table, dead);
                walk_suite_for_dead(&w.orelse, table, dead);
            }
        },
        Stmt::For(f) => {
            walk_suite_for_dead(&f.body, table, dead);
            walk_suite_for_dead(&f.orelse, table, dead);
        }
        Stmt::With(w) => walk_suite_for_dead(&w.body, table, dead),
        Stmt::Try(t) => {
            walk_suite_for_dead(&t.body, table, dead);
            for handler in &t.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
                walk_suite_for_dead(&h.body, table, dead);
            }
            walk_suite_for_dead(&t.orelse, table, dead);
            walk_suite_for_dead(&t.finalbody, table, dead);
        }
        Stmt::FunctionDef(f) => {
            let nested = build_scope_table(&f.body, table);
            walk_suite_for_dead(&f.body, &nested, dead);
        }
        Stmt::ClassDef(c) => {
            let nested = build_scope_table(&c.body, table);
            walk_suite_for_dead(&c.body, &nested, dead);
        }
        Stmt::Match(m) => {
            for case in &m.cases {
                walk_suite_for_dead(&case.body, table, dead);
            }
        }
        _ => {}
    }
}

/// Source ranges of the statements inside an `if TYPE_CHECKING:` block
/// (the guard spelled bare as `TYPE_CHECKING` or as an attribute like
/// `typing.TYPE_CHECKING`). ty narrows `TYPE_CHECKING` to `True`, so a
/// use whose flow-resolved binding lives in one of these ranges has had
/// its *runtime* binding (in the `else` clause, or after the block)
/// narrowed away. The ref walker uses these ranges to recover the
/// runtime binding. The `else`/`elif` clauses are deliberately not
/// recorded — those are what actually run at runtime.
pub(crate) fn detect_type_checking_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut ranges = Vec::new();
    collect_type_checking_ranges(&parsed.syntax().body, &mut ranges);
    ranges
}

fn collect_type_checking_ranges(stmts: &[Stmt], ranges: &mut Vec<TextRange>) {
    for stmt in stmts {
        match stmt {
            Stmt::If(if_stmt) => {
                if expr_is_type_checking(&if_stmt.test) {
                    for s in &if_stmt.body {
                        ranges.push(s.range());
                    }
                }
                collect_type_checking_ranges(&if_stmt.body, ranges);
                for clause in &if_stmt.elif_else_clauses {
                    collect_type_checking_ranges(&clause.body, ranges);
                }
            }
            Stmt::While(w) => {
                collect_type_checking_ranges(&w.body, ranges);
                collect_type_checking_ranges(&w.orelse, ranges);
            }
            Stmt::For(f) => {
                collect_type_checking_ranges(&f.body, ranges);
                collect_type_checking_ranges(&f.orelse, ranges);
            }
            Stmt::With(w) => collect_type_checking_ranges(&w.body, ranges),
            Stmt::Try(t) => {
                collect_type_checking_ranges(&t.body, ranges);
                for handler in &t.handlers {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
                    collect_type_checking_ranges(&h.body, ranges);
                }
                collect_type_checking_ranges(&t.orelse, ranges);
                collect_type_checking_ranges(&t.finalbody, ranges);
            }
            Stmt::FunctionDef(f) => collect_type_checking_ranges(&f.body, ranges),
            Stmt::ClassDef(c) => collect_type_checking_ranges(&c.body, ranges),
            Stmt::Match(m) => {
                for case in &m.cases {
                    collect_type_checking_ranges(&case.body, ranges);
                }
            }
            _ => {}
        }
    }
}

/// True for the two spellings of the type-checking guard: a bare
/// `TYPE_CHECKING` name or any `<…>.TYPE_CHECKING` attribute access
/// (e.g. `typing.TYPE_CHECKING`).
fn expr_is_type_checking(expr: &Expr) -> bool {
    match expr {
        Expr::Name(n) => n.id.as_str() == "TYPE_CHECKING",
        Expr::Attribute(a) => a.attr.as_str() == "TYPE_CHECKING",
        _ => false,
    }
}

/// Does `stmt` unconditionally hand off control out of the enclosing
/// suite? Statements *after* a terminator in the same suite are dead.
///
/// Bare terminators are `return` / `raise` / `break` / `continue`
/// plus `assert <falsy>`. Compound forms (`if/elif/else`, `try`,
/// `with`, `match`) are terminators when every reachable path inside
/// them ends in one.
pub(crate) fn stmt_is_terminator(stmt: &Stmt, table: &NameTable) -> bool {
    match stmt {
        Stmt::Return(_) | Stmt::Raise(_) | Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::Assert(a) => evaluate_truthiness(&a.test, table) == Some(false),
        Stmt::If(if_stmt) => {
            // A constant-truthy `if` is unconditional: it's a
            // terminator iff its body terminates, with or without
            // any else (`if True: return; …` kills the rest of
            // the enclosing suite).
            if evaluate_truthiness(&if_stmt.test, table) == Some(true) {
                return suite_terminates(&if_stmt.body, table);
            }
            // Otherwise we need an else for completeness — without
            // one, the `if` (or every `elif`) might not fire and the
            // suite continues normally.
            let has_else = if_stmt
                .elif_else_clauses
                .last()
                .is_some_and(|c| c.test.is_none());
            if !has_else {
                return false;
            }
            let test = evaluate_truthiness(&if_stmt.test, table);
            let body_term = suite_terminates(&if_stmt.body, table) || test == Some(false);
            let clauses_term = if_stmt
                .elif_else_clauses
                .iter()
                .all(|c| suite_terminates(&c.body, table));
            body_term && clauses_term
        }
        Stmt::With(w) => suite_terminates(&w.body, table),
        Stmt::Try(t) => {
            if suite_terminates(&t.finalbody, table) {
                return true;
            }
            suite_terminates(&t.body, table)
                && t.handlers.iter().all(|h| {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(eh) = h;
                    suite_terminates(&eh.body, table)
                })
        }
        _ => false,
    }
}

pub(crate) fn suite_terminates(stmts: &[Stmt], table: &NameTable) -> bool {
    stmts.iter().any(|s| stmt_is_terminator(s, table))
}

/// Constant-fold the truthiness of `expr`. Returns `None` when the
/// expression doesn't reduce to a known value — names not in `table`,
/// calls, attribute access, comparisons, etc.
///
/// `BoolOp` (`and` / `or`) short-circuits the way Python does:
/// `True or anything` is `True` even when "anything" is unknown.
/// `not` flips a known truth value through `UnaryOp`.
pub(crate) fn evaluate_truthiness(expr: &Expr, table: &NameTable) -> Option<bool> {
    match expr {
        Expr::BooleanLiteral(b) => Some(b.value),
        Expr::NoneLiteral(_) => Some(false),
        Expr::EllipsisLiteral(_) => Some(true),
        Expr::NumberLiteral(n) => match &n.value {
            ruff_python_ast::Number::Int(i) => i.as_i64().map(|x| x != 0),
            ruff_python_ast::Number::Float(f) => Some(*f != 0.0),
            ruff_python_ast::Number::Complex { real, imag } => Some(*real != 0.0 || *imag != 0.0),
        },
        Expr::StringLiteral(s) => Some(!s.value.is_empty()),
        Expr::BytesLiteral(b) => Some(!b.value.is_empty()),
        Expr::List(l) => Some(!l.elts.is_empty()),
        Expr::Tuple(t) => Some(!t.elts.is_empty()),
        Expr::Set(s) => Some(!s.elts.is_empty()),
        Expr::Dict(d) => Some(!d.items.is_empty()),
        Expr::Name(n) => table.get(n.id.as_str()).copied(),
        Expr::Named(named) => evaluate_truthiness(&named.value, table),
        Expr::BoolOp(b) => match b.op {
            ruff_python_ast::BoolOp::Or => {
                let mut any_unknown = false;
                for v in &b.values {
                    match evaluate_truthiness(v, table) {
                        Some(true) => return Some(true),
                        Some(false) => continue,
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(false)
                }
            }
            ruff_python_ast::BoolOp::And => {
                let mut any_unknown = false;
                for v in &b.values {
                    match evaluate_truthiness(v, table) {
                        Some(false) => return Some(false),
                        Some(true) => continue,
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(true)
                }
            }
        },
        Expr::UnaryOp(u) => {
            if matches!(u.op, ruff_python_ast::UnaryOp::Not) {
                evaluate_truthiness(&u.operand, table).map(|v| !v)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build the scope's name→truthiness table.
///
/// Collects all unconditional assignments at the scope's top level
/// (`X = literal`, `X: T = literal`) plus every walrus binding the
/// scope's expressions contain (PEP 572 leaks walrus bindings to the
/// enclosing function/module). Iterates to fixed point so that
/// chained constants — `foo = False; bar = foo or False; baz = bar`
/// — fold through the chain.
///
/// Conservative on conflicts: a name with multiple disagreeing
/// bindings (`X = True` and `X = False` both at top level), or any
/// binding whose RHS we can't fold, drops out of the table. That
/// matches the libcst pipeline's "only fold when every binding
/// agrees" rule and is what keeps the `does_not_fold_*` tests honest.
///
/// Non-monotonic flips poison the name. A binding like `foo = not foo`
/// in a scope that inherits `foo = False` from the enclosing scope
/// would otherwise oscillate the table between true/false forever
/// (`not false → true → not true → false → ...`). The poison set is
/// the convergence guarantee: once a name produces a different
/// resolved value than the table already holds, we drop it and
/// refuse to re-fold it for the rest of this scope.
pub(crate) fn build_scope_table(stmts: &[Stmt], enclosing: &NameTable) -> NameTable {
    let mut table = enclosing.clone();
    let mut bindings: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
    collect_scope_bindings(stmts, &mut bindings);

    let mut poisoned: FxHashSet<String> = FxHashSet::default();
    let mut changed = true;
    while changed {
        changed = false;
        for (name, exprs) in &bindings {
            if poisoned.contains(name) {
                continue;
            }
            let values: Vec<Option<bool>> = exprs
                .iter()
                .map(|e| evaluate_truthiness(e, &table))
                .collect();
            let resolved = if let Some(first) = values.first().copied() {
                if values.iter().all(|v| *v == first) {
                    first
                } else {
                    None
                }
            } else {
                None
            };
            match (resolved, table.get(name).copied()) {
                (Some(v), None) => {
                    table.insert(name.clone(), v);
                    changed = true;
                }
                (Some(new), Some(prev)) if prev != new => {
                    table.remove(name);
                    poisoned.insert(name.clone());
                    changed = true;
                }
                (None, Some(_)) => {
                    table.remove(name);
                    changed = true;
                }
                _ => {}
            }
        }
    }
    table
}

/// Top-level `name → rhs-expression` bindings for the scope, plus
/// any walrus bindings the scope's *expressions* contain (PEP 572
/// leaks walruses to the enclosing function/module). Does NOT
/// recurse into nested function/class bodies — those are separate
/// scopes with their own tables.
pub(crate) fn collect_scope_bindings<'a>(
    stmts: &'a [Stmt],
    out: &mut FxHashMap<String, Vec<&'a Expr>>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign(a) => {
                if let [Expr::Name(target)] = a.targets.as_slice() {
                    out.entry(target.id.as_str().to_string())
                        .or_default()
                        .push(&a.value);
                }
                collect_walrus_in_expr(&a.value, out);
            }
            Stmt::AnnAssign(a) => {
                if let (Expr::Name(target), Some(value)) = (&*a.target, &a.value) {
                    out.entry(target.id.as_str().to_string())
                        .or_default()
                        .push(value);
                }
                if let Some(value) = &a.value {
                    collect_walrus_in_expr(value, out);
                }
                collect_walrus_in_expr(&a.annotation, out);
            }
            Stmt::Expr(e) => {
                collect_walrus_in_expr(&e.value, out);
            }
            Stmt::If(if_stmt) => {
                collect_walrus_in_expr(&if_stmt.test, out);
            }
            Stmt::While(w) => {
                collect_walrus_in_expr(&w.test, out);
            }
            Stmt::Return(r) => {
                if let Some(value) = &r.value {
                    collect_walrus_in_expr(value, out);
                }
            }
            // Nested function/class bodies are separate scopes —
            // don't drag their bindings into this table.
            Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
            _ => {}
        }
    }
}

/// Scan `expr` for walrus (`:=`) targets that bind names in the
/// enclosing scope.
pub(crate) fn collect_walrus_in_expr<'a>(
    expr: &'a Expr,
    out: &mut FxHashMap<String, Vec<&'a Expr>>,
) {
    match expr {
        Expr::Named(named) => {
            if let Expr::Name(target) = &*named.target {
                out.entry(target.id.as_str().to_string())
                    .or_default()
                    .push(&named.value);
            }
            collect_walrus_in_expr(&named.value, out);
        }
        Expr::BoolOp(b) => {
            for v in &b.values {
                collect_walrus_in_expr(v, out);
            }
        }
        Expr::UnaryOp(u) => collect_walrus_in_expr(&u.operand, out),
        Expr::BinOp(b) => {
            collect_walrus_in_expr(&b.left, out);
            collect_walrus_in_expr(&b.right, out);
        }
        Expr::Compare(c) => {
            collect_walrus_in_expr(&c.left, out);
            for v in &c.comparators {
                collect_walrus_in_expr(v, out);
            }
        }
        _ => {}
    }
}

/// Result of parsing the `noqa[: codes…]` tail of a comment.
///
/// `Bare` means the directive carried no code list and so suppresses
/// every rule (including F401); `F401Present` means F401 is in the
/// comma-separated rule list; `OtherOnly` means a rule list was given
/// but F401 was absent. The first two pin the import alive; the third
/// doesn't.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoqaKind {
    Bare,
    F401Present,
    OtherOnly,
}

impl NoqaKind {
    fn pins_f401(self) -> bool {
        matches!(self, NoqaKind::Bare | NoqaKind::F401Present)
    }
}

/// Parse a `noqa[: codes…]` tail from `content`. Returns `None` if
/// `content` doesn't start with `noqa` (case-insensitive) after
/// optional whitespace.
pub(crate) fn parse_noqa_tail(content: &str) -> Option<NoqaKind> {
    let trimmed = content.trim_start();
    // `str::get` short-circuits when byte 4 isn't a char boundary —
    // e.g. a comment that starts with the 3-byte `─` box-drawing char
    // (common in section banners). Direct slicing would panic.
    let prefix = trimmed.get(..4)?;
    if !prefix.eq_ignore_ascii_case("noqa") {
        return None;
    }
    let after_noqa = &trimmed[4..];
    let after_colon = after_noqa.trim_start();
    let Some(rest) = after_colon.strip_prefix(':') else {
        // Bare `noqa` with no code list.
        return Some(NoqaKind::Bare);
    };
    let rules = rest.trim();
    if rules.is_empty() {
        return Some(NoqaKind::Bare);
    }
    let has_f401 = rules
        .split(',')
        .any(|code| code.trim().eq_ignore_ascii_case("F401"));
    if has_f401 {
        Some(NoqaKind::F401Present)
    } else {
        Some(NoqaKind::OtherOnly)
    }
}

/// Returns `true` when `comment_body` (the text *after* the leading
/// `#`) is a per-line `# noqa` directive that silences F401.
pub(crate) fn is_per_line_pin(comment_body: &str) -> bool {
    parse_noqa_tail(comment_body)
        .map(NoqaKind::pins_f401)
        .unwrap_or(false)
}

/// Returns `true` when `comment_body` is a *file-level*
/// `# ruff: noqa` / `# flake8: noqa` directive that silences F401.
///
/// The `ruff:` / `flake8:` prefix is matched case-sensitively (per
/// ruff's documented behavior); the `noqa` keyword itself is
/// case-insensitive.
pub(crate) fn is_file_pin(comment_body: &str) -> bool {
    let trimmed = comment_body.trim_start();
    let after_prefix = trimmed
        .strip_prefix("ruff:")
        .or_else(|| trimmed.strip_prefix("flake8:"));
    let Some(after_prefix) = after_prefix else {
        return false;
    };
    parse_noqa_tail(after_prefix)
        .map(NoqaKind::pins_f401)
        .unwrap_or(false)
}

/// Scan every Comment token in `parsed` and partition the file's
/// noqa directives into `(file_pinned, per_line_pins)`.
///
/// `file_pinned` is true when *any* comment in the file is a
/// file-level `# ruff: noqa` / `# flake8: noqa` that silences F401 —
/// ruff scans the whole source, not just the header.
/// `per_line_pins` collects the (1-indexed) line numbers carrying a
/// `# noqa[: …F401…]` per-line directive; an import alias on one of
/// those lines is pinned individually.
pub(crate) fn scan_noqa_directives(
    parsed: &ParsedModuleRef,
    source: &str,
    line_index: &LineIndex,
) -> (bool, FxHashSet<usize>) {
    let mut file_pinned = false;
    let mut per_line_pins: FxHashSet<usize> = FxHashSet::default();
    for token in parsed.tokens().iter() {
        if token.kind() != TokenKind::Comment {
            continue;
        }
        let range = token.range();
        let text = &source[range];
        let Some(body) = text.strip_prefix('#') else {
            continue;
        };
        if is_file_pin(body) {
            file_pinned = true;
        }
        if is_per_line_pin(body) {
            let line = line_index.line_column(range.start(), source).line.get();
            per_line_pins.insert(line);
        }
    }
    (file_pinned, per_line_pins)
}

pub(crate) fn position(
    index: &LineIndex,
    source: &str,
    range: TextRange,
) -> (usize, usize, usize, usize) {
    let start = index.line_column(range.start(), source);
    let end = index.line_column(range.end(), source);
    (
        start.line.get(),
        start.column.get() - 1,
        end.line.get(),
        end.column.get() - 1,
    )
}

pub(crate) fn range_key(range: TextRange) -> (u32, u32) {
    (range.start().to_u32(), range.end().to_u32())
}

pub(crate) fn file_path_string(db: &dyn ProjectDb, file: File) -> String {
    match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    }
}

/// Per-source-type default flag mask ORed into every node minted from
/// this file. Only ``.ipynb`` opts in today — cells are inherently
/// entrypoints (run top-to-bottom, not imported) and the codemod must
/// skip the JSON envelope. ``.pyi`` decls get per-decl flagging in
/// ``ingest_decls`` (the flag depends on whether a matching ``.py``
/// runtime decl exists), so the file-level helper returns ``0`` for
/// them.
pub(crate) fn file_default_flags(db: &dyn ProjectDb, file: File) -> u32 {
    if file_path_string(db, file).ends_with(".ipynb") {
        NODE_FLAGS_NOTEBOOK_DEFAULT
    } else {
        0
    }
}

/// Convert a search-path-relative path into a dotted module name.
///
/// `rel` is the file's path relative to a search root (e.g.
/// `"lib_a/__init__.py"` or `"handlers/job.py"`). Strips `.py` / `.pyi`
/// extensions and any trailing `__init__` segment so a package's
/// `__init__.py` collapses to the package's dotted name.
///
/// Returns `None` for paths that aren't valid module containers — most
/// commonly a bare `__init__.py` whose search path *is* the package
/// directory itself (a layout no `.pth` install would produce, but we
/// reject it for safety: there's no name to bind).
fn relative_to_module_name(rel: &str) -> Option<String> {
    let stem = rel
        .strip_suffix(".pyi")
        .or_else(|| rel.strip_suffix(".py"))
        .unwrap_or(rel);
    if stem == "__init__" {
        return None;
    }
    let stem = stem
        .strip_suffix("/__init__")
        .or_else(|| stem.strip_suffix("\\__init__"))
        .unwrap_or(stem);
    if stem.is_empty() {
        return None;
    }
    Some(stem.replace(['/', '\\'], "."))
}

/// Specificity-aware reverse module lookup: which fqname would Python
/// import to reach this file?
///
/// `ty_module_resolver::file_to_module` takes the *first* search path
/// that contains the file. That's correct when search paths reflect
/// Python's resolution order — but when project_root is on the list
/// alongside a `.pth`-derived path inside it, the first match yields a
/// workspace-relative dotted name (`libs.lib_a.src.lib_a`) instead of
/// the import-time name (`lib_a`). The forward resolver still picks
/// the right file via the `.pth`, so cross-file edges built against
/// the workspace-relative fqname never connect.
///
/// This walks every containing search path, builds a candidate
/// module name from each, round-trips it through `resolve_module`, and
/// returns the candidate whose search path is *most* specific (largest
/// component count). The specificity tiebreak picks `lib_a` over
/// `libs.lib_a.src.lib_a` whenever both round-trip — i.e. when both a
/// `.pth` entry and the project root cover the file.
///
/// Returns `None` when no candidate round-trips (the file is shadowed
/// by another module of the same name in a higher-priority search
/// path) or when the file isn't on any search path at all. Callers
/// should fall back to a path-mangle in that case.
pub(crate) fn canonical_module_for_file<'db>(
    db: &'db dyn ProjectDb,
    file: File,
) -> Option<Module<'db>> {
    let name = canonical_module_name_for_file(db, file)?;
    resolve_module(db, importing_file(db, file), &name)
}

/// Same logic as :func:`canonical_module_for_file` but stops at the
/// ``ModuleName``. Saves one ``resolve_module`` call per file in the
/// common single-package case where ``module_fqname_for_file`` only
/// needs the dotted name. Callers that go on to walk the parent module
/// (``parent_module_file``, ``file_package_name``) still go through
/// the ``Module``-returning wrapper above.
pub(crate) fn canonical_module_name_for_file(db: &dyn ProjectDb, file: File) -> Option<ModuleName> {
    let file_path = match file.path(db) {
        FilePath::System(p) => p,
        _ => {
            let resolver_file = program_file(db, file).resolver_file(db);
            return file_to_module(db, resolver_file).map(|m| m.name(db).clone());
        }
    };

    // Collect every search path that physically contains the file.
    // Typical projects have ≤ 5 search paths and a file usually lives
    // under 1–2 of them, so the SmallVec stays inline.
    let mut candidates: smallvec::SmallVec<[(usize, ModuleName); 4]> = smallvec::SmallVec::new();
    for sp_path in system_module_search_paths(db, resolver_environment(db)) {
        let Ok(rel) = file_path.strip_prefix(sp_path) else {
            continue;
        };
        let Some(name_str) = relative_to_module_name(rel.as_str()) else {
            continue;
        };
        let Some(name) = ModuleName::new(&name_str) else {
            continue;
        };
        candidates.push((sp_path.components().count(), name));
    }

    // Fast path: a single containing search path means no ambiguity —
    // skip the ``resolve_module`` round-trip entirely. This is the
    // single-package shape (just ``project_root`` on the search list)
    // and accounts for the bulk of first-party files in typical
    // codebases. Without this short-circuit, the round-trip would cost
    // one ``resolve_module`` call per project file at cold start.
    //
    // Correctness note: ``file_to_module`` round-trips to reject files
    // shadowed by a same-named package (``foo.py`` next to
    // ``foo/__init__.py``). We deliberately don't reject here — the
    // current code path falls back to a path-mangle for those files
    // anyway, returning the same fqname we would. Shadowing precedence
    // for import-edge routing lives in ``file_payload`` / the trie,
    // not here.
    if candidates.len() == 1 {
        // No ambiguity, no round-trip needed.
        return candidates.into_iter().next().map(|(_, name)| name);
    }

    // Multiple containing paths -- specificity tiebreak with round-trip.
    candidates.sort_by_key(|c| std::cmp::Reverse(c.0));
    for (_, name) in candidates {
        let Some(resolved) = resolve_module(db, importing_file(db, file), &name) else {
            continue;
        };
        let Some(resolved_file) = resolved.file(db) else {
            continue;
        };
        if resolved_file == file {
            return Some(name);
        }
    }
    None
}

pub(crate) fn module_fqname_for_file(db: &dyn ProjectDb, file: File) -> CompactString {
    if let Some(name) = canonical_module_name_for_file(db, file) {
        return name.as_str().to_compact_string();
    }
    // Fallback for files ty doesn't classify (e.g. ``gunicorn.conf.py`` at
    // the project root — there's no ``__init__.py`` and the stem contains
    // a dot, so ty's module resolver returns ``None``). Mirror the libcst
    // pipeline's ``strip .py / replace / with .`` derivation so plugins
    // like ``ServerConfigPlugin`` can address the file by its conventional
    // dotted name.
    let path_str = file_path_string(db, file);
    let root_str = db.project().root(db).as_str();
    let rel = path_str
        .strip_prefix(root_str)
        .unwrap_or(&path_str)
        .trim_start_matches('/')
        .trim_start_matches('\\');
    let stem = rel
        .strip_suffix(".pyi")
        .or_else(|| rel.strip_suffix(".py"))
        .unwrap_or(rel);
    stem.replace(['/', '\\'], ".").into()
}

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the AST / string helpers.
    //!
    //! Anything that takes a `Python<'_>` GIL token, a `Py<SymbolNode>`,
    //! a `Bound<'_, PyAny>`, a Salsa `ProjectDatabase`, a `File`, or a
    //! `ParsedModuleRef` (which is also DB-backed) is intentionally NOT
    //! covered here — the FFI surface is exercised end-to-end by the
    //! python suite. We focus on pure functions over primitives + raw
    //! AST nodes built from `parse_module(...)` / `parse_expression(...)`.
    use super::*;
    use ruff_python_parser::{parse_expression, parse_module};

    fn parse_expr(source: &str) -> Box<Expr> {
        parse_expression(source).unwrap().into_syntax().body
    }

    fn parse_stmts(source: &str) -> Vec<Stmt> {
        parse_module(source)
            .unwrap()
            .into_syntax()
            .body
            .into_iter()
            .collect()
    }

    // -- is_dunder_name ----------------------------------------------------

    #[test]
    fn is_dunder_name_true_for_typical_dunders() {
        assert!(is_dunder_name("__init__"));
        assert!(is_dunder_name("__repr__"));
        assert!(is_dunder_name("mod.cls.__init__"));
    }

    #[test]
    fn is_dunder_name_false_for_single_underscore() {
        assert!(!is_dunder_name("_x"));
        assert!(!is_dunder_name("__x"));
        assert!(!is_dunder_name("x__"));
    }

    #[test]
    fn is_dunder_name_false_for_short_dunder() {
        // ``__`` and ``____`` are <= 4 chars after stripping `__` —
        // the helper requires len() > 4 so the dunder body is at least 1 char.
        assert!(!is_dunder_name("__"));
        assert!(!is_dunder_name("____"));
    }

    #[test]
    fn is_dunder_name_empty_string_ok() {
        assert!(!is_dunder_name(""));
        assert!(!is_dunder_name("foo.bar.baz"));
    }

    #[test]
    fn is_dunder_name_checks_only_last_segment() {
        // Only the final dotted segment counts: a dunder module with a
        // plain member is not a dunder, and vice versa.
        assert!(is_dunder_name("pkg.mod.__all__"));
        assert!(!is_dunder_name("pkg.__init__.helper"));
    }

    // -- rel_path ----------------------------------------------------------

    #[test]
    fn rel_path_roundtrips_unix_path() {
        let p = rel_path("foo/bar.py");
        assert_eq!(p.path().as_str(), "foo/bar.py");
    }

    #[test]
    fn rel_path_handles_empty_input() {
        // Should not panic on empty input — the `SystemPath::new` path
        // accepts zero-length strings.
        let p = rel_path("");
        assert_eq!(p.path().as_str(), "");
    }

    // -- is_name / is_string_literal --------------------------------------

    #[test]
    fn is_name_matches_bare_name() {
        let expr = parse_expr("foo");
        assert!(is_name(&expr, "foo"));
        assert!(!is_name(&expr, "bar"));
    }

    #[test]
    fn is_name_rejects_non_name_exprs() {
        let expr = parse_expr("foo.bar");
        assert!(!is_name(&expr, "foo"));
        assert!(!is_name(&expr, "bar"));
        let expr = parse_expr("123");
        assert!(!is_name(&expr, "123"));
    }

    #[test]
    fn is_string_literal_matches_exact_value() {
        let expr = parse_expr("'__main__'");
        assert!(is_string_literal(&expr, "__main__"));
        assert!(!is_string_literal(&expr, "other"));
    }

    #[test]
    fn is_string_literal_rejects_non_string() {
        let expr = parse_expr("123");
        assert!(!is_string_literal(&expr, "123"));
        let expr = parse_expr("foo");
        assert!(!is_string_literal(&expr, "foo"));
    }

    // -- is_name_eq_main --------------------------------------------------

    #[test]
    fn is_name_eq_main_matches_both_orderings() {
        let a = parse_expr("__name__ == '__main__'");
        let b = parse_expr("'__main__' == __name__");
        assert!(is_name_eq_main(&a));
        assert!(is_name_eq_main(&b));
    }

    #[test]
    fn is_name_eq_main_rejects_wrong_constant() {
        let expr = parse_expr("__name__ == 'main'");
        assert!(!is_name_eq_main(&expr));
    }

    #[test]
    fn is_name_eq_main_rejects_other_comparisons() {
        // !=, multiple comparators, non-name LHS, non-string RHS, etc.
        assert!(!is_name_eq_main(&parse_expr("__name__ != '__main__'")));
        assert!(!is_name_eq_main(&parse_expr("x == y == z")));
        assert!(!is_name_eq_main(&parse_expr("foo == '__main__'")));
        assert!(!is_name_eq_main(&parse_expr("123 == 456")));
    }

    // -- top_level_assign_to_name -----------------------------------------

    #[test]
    fn top_level_assign_to_name_matches_simple_assignment() {
        let stmts = parse_stmts("x = 42\n");
        let got = top_level_assign_to_name(&stmts[0]);
        assert!(got.is_some());
        let (_, expr) = got.unwrap();
        assert!(matches!(expr, Expr::NumberLiteral(_)));
    }

    #[test]
    fn top_level_assign_to_name_matches_ann_assignment() {
        let stmts = parse_stmts("x: int = 42\n");
        let got = top_level_assign_to_name(&stmts[0]);
        assert!(got.is_some());
    }

    #[test]
    fn top_level_assign_to_name_rejects_tuple_target() {
        let stmts = parse_stmts("x, y = 1, 2\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    #[test]
    fn top_level_assign_to_name_rejects_attribute_target() {
        let stmts = parse_stmts("obj.attr = 1\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    #[test]
    fn top_level_assign_to_name_rejects_ann_without_value() {
        let stmts = parse_stmts("x: int\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    #[test]
    fn top_level_assign_to_name_rejects_non_assign() {
        let stmts = parse_stmts("def foo(): pass\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    // -- first_call: shared call-parsing helper for the call-target tests --

    fn first_call(source: &str) -> ruff_python_ast::ExprCall {
        let expr = parse_expr(source);
        match *expr {
            Expr::Call(c) => c,
            _ => panic!("expected a call expression"),
        }
    }

    // -- range_key ---------------------------------------------------------

    #[test]
    fn range_key_packs_start_end() {
        use ruff_text_size::TextSize;
        let range = TextRange::new(TextSize::from(5), TextSize::from(12));
        assert_eq!(range_key(range), (5, 12));
    }

    // -- evaluate_truthiness ----------------------------------------------

    #[test]
    fn evaluate_truthiness_literals() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(evaluate_truthiness(&parse_expr("True"), &empty), Some(true));
        assert_eq!(
            evaluate_truthiness(&parse_expr("False"), &empty),
            Some(false)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("None"), &empty),
            Some(false)
        );
        assert_eq!(evaluate_truthiness(&parse_expr("..."), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("0"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("1"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("0.0"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("3.14"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("''"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("'x'"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("[]"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("[1]"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("()"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("(1,)"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("{}"), &empty), Some(false));
        assert_eq!(
            evaluate_truthiness(&parse_expr("{'a': 1}"), &empty),
            Some(true)
        );
    }

    #[test]
    fn evaluate_truthiness_names_via_table() {
        let mut table: NameTable = FxHashMap::default();
        table.insert("DEBUG".into(), true);
        table.insert("RELEASE".into(), false);
        assert_eq!(
            evaluate_truthiness(&parse_expr("DEBUG"), &table),
            Some(true)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("RELEASE"), &table),
            Some(false)
        );
        assert_eq!(evaluate_truthiness(&parse_expr("OTHER"), &table), None);
    }

    #[test]
    fn evaluate_truthiness_unary_not() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(
            evaluate_truthiness(&parse_expr("not True"), &empty),
            Some(false)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("not False"), &empty),
            Some(true)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("not unknown_name"), &empty),
            None
        );
        // Other unary ops aren't folded.
        assert_eq!(evaluate_truthiness(&parse_expr("-1"), &empty), None);
    }

    #[test]
    fn evaluate_truthiness_bool_ops_short_circuit_or() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(
            evaluate_truthiness(&parse_expr("True or unknown"), &empty),
            Some(true)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("False or unknown"), &empty),
            None
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("False or False"), &empty),
            Some(false)
        );
    }

    #[test]
    fn evaluate_truthiness_bool_ops_short_circuit_and() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(
            evaluate_truthiness(&parse_expr("False and unknown"), &empty),
            Some(false)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("True and unknown"), &empty),
            None
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("True and True"), &empty),
            Some(true)
        );
    }

    #[test]
    fn evaluate_truthiness_returns_none_for_unsupported() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(evaluate_truthiness(&parse_expr("f()"), &empty), None);
        assert_eq!(evaluate_truthiness(&parse_expr("a.b"), &empty), None);
        assert_eq!(evaluate_truthiness(&parse_expr("a == b"), &empty), None);
    }

    #[test]
    fn evaluate_truthiness_walrus_unwraps_value() {
        let empty: NameTable = FxHashMap::default();
        // ``(x := True)`` should fold to True (the walrus's value).
        assert_eq!(
            evaluate_truthiness(&parse_expr("(x := True)"), &empty),
            Some(true)
        );
    }

    // -- stmt_is_terminator / suite_terminates ----------------------------

    #[test]
    fn stmt_is_terminator_basic_terminators() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    return 1\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));

        let stmts = parse_stmts("def f():\n    raise X\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_break_continue_inside_loop() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("for _ in []:\n    break\n");
        let body = match &stmts[0] {
            Stmt::For(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));

        let stmts = parse_stmts("for _ in []:\n    continue\n");
        let body = match &stmts[0] {
            Stmt::For(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_assert_falsy() {
        let empty: NameTable = FxHashMap::default();
        // `assert False` is a terminator; `assert True` is not.
        let stmts = parse_stmts("assert False\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("assert True\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("assert 0\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("assert unknown_var\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_constant_truthy_body_terminates() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if True:\n    return\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_without_else_not_terminator() {
        let empty: NameTable = FxHashMap::default();
        // No else clause + a non-constant test means the if might be
        // skipped — not a terminator.
        let stmts = parse_stmts("if cond:\n    return\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_else_both_terminate() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if cond:\n    return\nelse:\n    raise X\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_else_one_falls_through() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if cond:\n    return\nelse:\n    pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_try_finally_terminates() {
        let empty: NameTable = FxHashMap::default();
        let stmts =
            parse_stmts("try:\n    x = 1\nexcept Exception:\n    pass\nfinally:\n    return\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_try_body_and_all_handlers_terminate() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("try:\n    return\nexcept Exception:\n    raise\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_try_handler_falls_through_not_terminator() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("try:\n    return\nexcept Exception:\n    pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_with_body_terminates() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("with cm as c:\n    return\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("with cm as c:\n    pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_pass_is_not() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn suite_terminates_finds_any_terminator() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    x = 1\n    return 2\n    y = 3\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(suite_terminates(body, &empty));

        let stmts = parse_stmts("def f():\n    x = 1\n    y = 2\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(!suite_terminates(body, &empty));
    }

    // -- walk_suite_for_dead / detect_dead_ranges (via module body) -------

    #[test]
    fn walk_suite_for_dead_marks_post_terminator_statements() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    return 1\n    x = 2\n    y = 3\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => f.body.clone(),
            _ => unreachable!(),
        };
        let mut dead = Vec::new();
        walk_suite_for_dead(&body, &empty, &mut dead);
        // x = 2 and y = 3 should both be marked.
        assert_eq!(dead.len(), 2);
    }

    #[test]
    fn walk_suite_for_dead_handles_if_false() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if False:\n    x = 1\n    y = 2\n");
        let mut dead = Vec::new();
        walk_suite_for_dead(&stmts, &empty, &mut dead);
        // Both statements in the False body are marked individually.
        assert_eq!(dead.len(), 2);
    }

    #[test]
    fn walk_suite_for_dead_handles_if_true_drops_else() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if True:\n    x = 1\nelse:\n    y = 2\n");
        let mut dead = Vec::new();
        walk_suite_for_dead(&stmts, &empty, &mut dead);
        // The else clause is dead.
        assert_eq!(dead.len(), 1);
    }

    #[test]
    fn walk_suite_for_dead_recurses_into_function_body() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def outer():\n    return\n    nested = 1\n");
        let mut dead = Vec::new();
        walk_suite_for_dead(&stmts, &empty, &mut dead);
        assert_eq!(dead.len(), 1);
    }

    #[test]
    fn walk_suite_for_dead_empty_input_yields_nothing() {
        let empty: NameTable = FxHashMap::default();
        let mut dead = Vec::new();
        walk_suite_for_dead(&[], &empty, &mut dead);
        assert!(dead.is_empty());
    }

    // -- build_scope_table / collect_scope_bindings -----------------------

    #[test]
    fn build_scope_table_folds_simple_constant() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("DEBUG = True\nX = False\n");
        let table = build_scope_table(&stmts, &empty);
        assert_eq!(table.get("DEBUG"), Some(&true));
        assert_eq!(table.get("X"), Some(&false));
    }

    #[test]
    fn build_scope_table_handles_chained_constants() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("A = False\nB = A or False\nC = B\n");
        let table = build_scope_table(&stmts, &empty);
        assert_eq!(table.get("A"), Some(&false));
        assert_eq!(table.get("B"), Some(&false));
        assert_eq!(table.get("C"), Some(&false));
    }

    #[test]
    fn build_scope_table_drops_conflicting_bindings() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("X = True\nX = False\n");
        let table = build_scope_table(&stmts, &empty);
        assert!(!table.contains_key("X"));
    }

    #[test]
    fn build_scope_table_inherits_enclosing_table() {
        let mut enclosing: NameTable = FxHashMap::default();
        enclosing.insert("OUTER".into(), true);
        let stmts = parse_stmts("INNER = False\n");
        let table = build_scope_table(&stmts, &enclosing);
        assert_eq!(table.get("OUTER"), Some(&true));
        assert_eq!(table.get("INNER"), Some(&false));
    }

    #[test]
    fn build_scope_table_self_negation_does_not_loop() {
        // Inheriting ``foo`` from the enclosing scope and then evaluating
        // ``foo = not foo`` would oscillate the table between true/false
        // forever before the poison set was added.
        let mut enclosing: NameTable = FxHashMap::default();
        enclosing.insert("foo".into(), false);
        let stmts = parse_stmts("foo = not foo\n");
        let table = build_scope_table(&stmts, &enclosing);
        assert!(!table.contains_key("foo"));
    }

    #[test]
    fn build_scope_table_detect_dead_ranges_terminates_for_global_flipper() {
        // End-to-end: module-level ``foo = False`` plus a function that
        // does ``global foo; foo = not foo``. ``detect_dead_ranges``
        // recurses into the function body via ``build_scope_table``,
        // which used to spin forever on the ``not foo`` binding.
        let parsed = ruff_python_parser::parse_module(
            "foo = False\ndef flip():\n    global foo\n    foo = not foo\n",
        )
        .unwrap();
        let mut bindings: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        let body = parsed.syntax().body.clone();
        collect_scope_bindings(&body, &mut bindings);
        // Sanity: module sees ``foo = False`` only, not the in-function rebind.
        assert!(bindings.contains_key("foo"));

        // The inner ``flip`` body inherits ``foo = false``; building its
        // table must terminate and drop ``foo`` rather than fold it.
        let mut enclosing: NameTable = FxHashMap::default();
        enclosing.insert("foo".into(), false);
        if let Stmt::FunctionDef(f) = &body[1] {
            let nested = build_scope_table(&f.body, &enclosing);
            assert!(!nested.contains_key("foo"));
        } else {
            panic!("second stmt should be a FunctionDef");
        }
    }

    #[test]
    fn build_scope_table_ignores_nested_function_bindings() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    X = True\n");
        let table = build_scope_table(&stmts, &empty);
        // ``X`` is defined inside ``f`` — separate scope.
        assert!(!table.contains_key("X"));
    }

    // -- collect_walrus_in_expr -------------------------------------------

    #[test]
    fn collect_walrus_in_expr_picks_up_assignment_target() {
        let expr = parse_expr("(x := 5)");
        let mut out: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        collect_walrus_in_expr(&expr, &mut out);
        assert!(out.contains_key("x"));
    }

    #[test]
    fn collect_walrus_in_expr_descends_compound_exprs() {
        let expr = parse_expr("(a := 1) or (b := 2) and (c := 3)");
        let mut out: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        collect_walrus_in_expr(&expr, &mut out);
        assert!(out.contains_key("a"));
        assert!(out.contains_key("b"));
        assert!(out.contains_key("c"));
    }

    #[test]
    fn collect_walrus_in_expr_no_walrus_yields_empty() {
        let expr = parse_expr("a + b");
        let mut out: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        collect_walrus_in_expr(&expr, &mut out);
        assert!(out.is_empty());
    }

    // -- noqa helpers (extending the existing parse_noqa_tail tests) ------

    #[test]
    fn parse_noqa_tail_case_insensitive_prefix() {
        assert_eq!(parse_noqa_tail("noqa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("NoQa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("nOqA"), Some(NoqaKind::Bare));
    }

    #[test]
    fn parse_noqa_tail_rejects_non_noqa_comment() {
        assert_eq!(parse_noqa_tail("type: ignore"), None);
        assert_eq!(parse_noqa_tail("comment about noqa"), None);
    }

    #[test]
    fn parse_noqa_tail_codes_are_trimmed_and_case_insensitive() {
        assert_eq!(parse_noqa_tail("noqa: f401"), Some(NoqaKind::F401Present));
        assert_eq!(
            parse_noqa_tail("noqa:  F401  ,  E501"),
            Some(NoqaKind::F401Present)
        );
        assert_eq!(
            parse_noqa_tail("noqa: E501,  E502"),
            Some(NoqaKind::OtherOnly)
        );
    }

    #[test]
    fn parse_noqa_tail_handles_leading_whitespace() {
        assert_eq!(
            parse_noqa_tail("    noqa: F401"),
            Some(NoqaKind::F401Present)
        );
    }

    #[test]
    fn is_per_line_pin_basic() {
        assert!(is_per_line_pin(" noqa"));
        assert!(is_per_line_pin(" noqa: F401"));
        assert!(is_per_line_pin(" noqa: E501, F401"));
        assert!(!is_per_line_pin(" noqa: E501"));
        assert!(!is_per_line_pin(" type: ignore"));
        assert!(!is_per_line_pin(""));
    }

    #[test]
    fn is_file_pin_recognizes_ruff_and_flake8_prefixes() {
        assert!(is_file_pin(" ruff: noqa"));
        assert!(is_file_pin(" ruff: noqa: F401"));
        assert!(is_file_pin(" flake8: noqa"));
        assert!(is_file_pin(" flake8: noqa: F401"));
    }

    #[test]
    fn is_file_pin_other_codes_dont_pin_f401() {
        assert!(!is_file_pin(" ruff: noqa: E501"));
        assert!(!is_file_pin(" flake8: noqa: W292, E303"));
    }

    #[test]
    fn is_file_pin_unrelated_comments_rejected() {
        assert!(!is_file_pin(" pyright: ignore"));
        assert!(!is_file_pin(" type: ignore"));
        assert!(!is_file_pin(" noqa"));
        assert!(!is_file_pin(""));
    }

    #[test]
    fn is_file_pin_prefix_is_case_sensitive() {
        // ``ruff:`` / ``flake8:`` are matched case-sensitively per ruff's docs.
        assert!(!is_file_pin(" Ruff: noqa"));
        assert!(!is_file_pin(" FLAKE8: noqa"));
    }

    // -- matched_call_target ----------------------------------------------

    #[test]
    fn matched_call_target_via_local_name() {
        let call = first_call("Flask(__name__)");
        let mut imports: LocalImports = FxHashMap::default();
        imports.insert("Flask".into(), "Flask".into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("Flask");
        assert_eq!(
            matched_call_target(&call, &imports, "flask", &allowed),
            Some("Flask".into())
        );
    }

    #[test]
    fn matched_call_target_via_module_alias_attr() {
        let call = first_call("flask.Flask(__name__)");
        let mut imports: LocalImports = FxHashMap::default();
        imports.insert("flask".into(), MODULE_ALIAS_MARKER.into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("Flask");
        assert_eq!(
            matched_call_target(&call, &imports, "flask", &allowed),
            Some("Flask".into())
        );
    }

    #[test]
    fn matched_call_target_via_multi_seg_module() {
        let call = first_call("unittest.mock.patch('x')");
        // No file imports entry for the leftmost name — fallback to the
        // literal dotted match against ``module``.
        let imports: LocalImports = FxHashMap::default();
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("patch");
        assert_eq!(
            matched_call_target(&call, &imports, "unittest.mock", &allowed),
            Some("patch".into())
        );
    }

    #[test]
    fn matched_call_target_rejects_unknown_name() {
        let call = first_call("unrelated()");
        let imports: LocalImports = FxHashMap::default();
        let allowed: FxHashSet<&str> = FxHashSet::default();
        assert!(matched_call_target(&call, &imports, "x", &allowed).is_none());
    }

    #[test]
    fn matched_call_target_rejects_disallowed_name() {
        let call = first_call("Flask()");
        let mut imports: LocalImports = FxHashMap::default();
        imports.insert("Flask".into(), "Flask".into());
        // Empty allowed set: target is present but not allowed.
        let allowed: FxHashSet<&str> = FxHashSet::default();
        assert!(matched_call_target(&call, &imports, "flask", &allowed).is_none());
    }

    // -- NoqaKind helper --------------------------------------------------

    #[test]
    fn noqakind_pins_f401() {
        assert!(NoqaKind::Bare.pins_f401());
        assert!(NoqaKind::F401Present.pins_f401());
        assert!(!NoqaKind::OtherOnly.pins_f401());
    }
}
