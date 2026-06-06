//! Per-file owned facts, extracted from the AST once during the
//! fan-out. Project-wide plugin queries read these facts (a salsa cache
//! hit) instead of re-walking live ASTs, which lets the parsed modules
//! be freed before the project-wide plugin pass runs — cutting the
//! in-build memory peak rather than just the post-build steady state.
//!
//! Every field is plain owned data — ranges as `(u32, u32)`, names and
//! values as `String` — keyed by name-range so the project-wide query
//! can fan in to a global graph index via `outputs.global_index` /
//! `outputs.decl_by_name_range` without holding anything that points
//! into the AST allocation. That ownership is what makes clearing the
//! parsed module behind this query sound.
//!
//! Sibling of [`crate::file_payload::file_to_nodes`] /
//! [`crate::file_ref_edges::file_to_refspecs`]; unlike `file_to_nodes`
//! (the cross-file lookup primitive) this query is never called
//! cross-file, so it carries no `'db` lifetime and absorbs facts freely.

use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Decorator, Expr, ExprCall, Stmt};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use ty_project::Db as ProjectDb;

use crate::helpers::{
    extract_call_kwargs, find_main_block_range, range_key, resolve_relative_import,
    top_level_assign_to_name, unwrap_subscripted_callee, CallArgs, MODULE_ALIAS_MARKER,
};
use crate::ingest::{collapse_attribute_chain, file_package_name, string_literal_list};

/// `(name_range_key, value)` pair. The range key matches the `(start,
/// end)` `u32` pair the assemble pass stores in `decl_by_name_range` /
/// `class_by_selection`, so a project-wide query fans in with a hashmap
/// probe — no second AST walk.
type ByNameRange<T> = Vec<((u32, u32), T)>;

/// A callee (or decorator) peeled to its `Name`-rooted attribute chain
/// plus any captured keyword args. `root_name` is the bare name at the
/// root (`app` in `@app.route(...)`, `unittest` in `unittest.mock.patch(...)`,
/// `route` in `@route`); `attrs` are the trailing attribute segments in
/// source order (closest-to-root first, so `@owner.via.outer` yields
/// `["via", "outer"]`); `kwargs` holds the call's string-literal keyword
/// args (empty when there is no call form or when the consumer doesn't
/// need them).
///
/// This is the config-independent shape behind the decorator, instance
/// construction, and factory-call queries: the decorator matchers resolve
/// `root_name`/`attrs` through the file's imports (or syntactically), and
/// [`match_callee_descriptor`] reproduces `matched_call_target_any` for
/// the construction/factory queries. We only record callees that bottom
/// out at a bare `Name` (via [`collapse_attribute_chain`]) — every matcher
/// rejects non-`Name` roots, so deeper shapes can't match anyway.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct CalleeDescriptor {
    pub(crate) root_name: String,
    pub(crate) attrs: SmallVec<[String; 2]>,
    pub(crate) kwargs: CallArgs,
}

/// One top-level `import` / `from … import …` statement, resolved to
/// config-independent owned data so the project-wide queries can
/// rebuild a `{local → target}` map ([`imports_local_from_facts`])
/// without re-walking the AST. Relative `from` imports are resolved to
/// their absolute module here (the only file-local input is the file's
/// own package), so the fact carries no `level`/relativity.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) enum ImportFact {
    /// `from <absolute> import <name> [as <local>], …` — `names` is
    /// `(imported_name, local_name)` for each alias.
    From {
        absolute: String,
        names: Vec<(String, String)>,
    },
    /// `import <module> [as <local>], …` — `entries` is
    /// `(module_name, local_name)` for each alias.
    Plain { entries: Vec<(String, String)> },
}

/// The `Name`-rooted callee chain of a call — root name + trailing
/// attribute segments, without captured args. The matchable core of a
/// call site: `find_calls_to_imported` resolves it through the file's
/// imports (via [`match_callee_chain`]) and `find_calls_on_var` matches
/// the `<owner>.<attr>` shape (`attrs == [attr]`, `root_name == owner`).
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct CalleeChain {
    pub(crate) root_name: String,
    pub(crate) attrs: SmallVec<[String; 2]>,
}

/// String content of a positional call argument, recorded so the
/// `nth_positional_string` / `string_or_string_collection` reads can be
/// replayed at a query-time `arg_index` without the AST.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) enum StringArg {
    /// A single `"…"` literal — `nth_positional_string` returns it; a
    /// collection read sees a one-element list.
    Lit(String),
    /// A list/tuple of string literals (non-string elements dropped, as
    /// `string_or_string_collection` does) — `nth_positional_string`
    /// returns `None`; a collection read sees every element.
    Coll(Vec<String>),
}

/// One call site reduced to the config-independent data the three
/// call-site queries replay. Only calls carrying at least one
/// string-bearing positional argument are recorded — every query captures
/// such an argument, so a call without one can never produce a result.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct CallSiteFact {
    /// Last attribute segment of the callee (`load_extension` in
    /// `bot.load_extension(x)`), for *any* receiver shape; `None` when the
    /// callee isn't an attribute access. Powers `find_calls_on_attr`.
    pub(crate) callee_attr: Option<String>,
    /// `Name`-rooted callee chain, when the callee bottoms out at a bare
    /// `Name`. Powers `find_calls_to_imported` / `find_calls_on_var`.
    pub(crate) callee: Option<CalleeChain>,
    /// Positional-argument arity (for `find_calls_on_var`'s
    /// `required_positional` disambiguation).
    pub(crate) positional_len: usize,
    /// String-bearing positional args by index (sparse — non-string args
    /// are omitted). Non-empty by construction.
    pub(crate) string_args: SmallVec<[(usize, StringArg); 1]>,
    /// String-literal keyword args (for `extract_args`).
    pub(crate) kwargs: CallArgs,
}

impl CallSiteFact {
    /// Replay `helpers::nth_positional_string` for `arg_index`: the single
    /// string-literal value, or `None` (missing, non-string, or a
    /// collection).
    pub(crate) fn nth_positional_string(&self, arg_index: usize) -> Option<&str> {
        self.string_args.iter().find_map(|(i, arg)| match arg {
            StringArg::Lit(s) if *i == arg_index => Some(s.as_str()),
            _ => None,
        })
    }

    /// Replay `helpers::string_or_string_collection` for `arg_index`: the
    /// string values (one for a single literal, many for a list/tuple),
    /// or empty.
    pub(crate) fn string_or_collection(&self, arg_index: usize) -> &[String] {
        self.string_args
            .iter()
            .find_map(|(i, arg)| {
                (*i == arg_index).then(|| match arg {
                    StringArg::Lit(s) => std::slice::from_ref(s),
                    StringArg::Coll(v) => v.as_slice(),
                })
            })
            .unwrap_or(&[])
    }
}

/// Owned, AST-free per-file facts that power the project-wide plugin
/// queries. Grows one field per migrated query; see the module docs for
/// the ownership contract that lets the parsed module be cleared behind
/// this query.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileExtraction {
    /// Byte range of the `if __name__ == "__main__":` block, if the
    /// file has one. Powers `find_main_blocks_indices`.
    pub(crate) main_block_range: Option<(u32, u32)>,
    /// Top-level `NAME = [str, ...]` / `NAME: T = [str, ...]`
    /// assignments whose RHS is a list/tuple of string literals, as
    /// `(target_name, entries)`. Powers `find_literal_list_entries` and
    /// the `__all__` export query.
    pub(crate) literal_list_rows: Vec<(String, Vec<String>)>,
    /// Per top-level `def`, its positional + keyword-only parameter
    /// names in source order. Powers `function_parameters`.
    pub(crate) function_params: ByNameRange<Vec<String>>,
    /// Per top-level `class`, the de-duplicated parameter names across
    /// all its methods, excluding `self` / `cls`. Powers
    /// `class_method_parameters`.
    pub(crate) class_method_params: ByNameRange<Vec<String>>,
    /// Per `class` *anywhere* in the file (not just top level), the
    /// names of the methods defined directly in its body. Powers
    /// `find_classes_defining_method`, whose original enumeration went
    /// through the semantic index (so it caught classes nested in
    /// module-level `if`/`try` blocks). We over-collect every class and
    /// let the `class_by_selection` fan-in keep only the global-scope
    /// ones — name-ranges never collide, so nested classes never
    /// produce a false match.
    pub(crate) class_method_defs: ByNameRange<Vec<String>>,
    /// Every top-level `import` / `from … import …` statement, resolved
    /// to absolute modules. Powers the import-resolving decorator /
    /// construction / call queries via [`imports_local_from_facts`].
    pub(crate) import_facts: Vec<ImportFact>,
    /// Per top-level `def` that carries at least one `Name`-rooted
    /// decorator, the decorators on it (source order). Powers
    /// `find_decorated_decls`, `find_handler_decorators`, and
    /// `find_handler_decorators_via`.
    pub(crate) decorator_rows: ByNameRange<Vec<CalleeDescriptor>>,
    /// Per top-level `NAME = <callee>(...)` / `NAME: T = <callee>(...)`
    /// assignment whose callee bottoms out at a bare `Name`, the callee
    /// descriptor keyed by the target's name-range. Powers
    /// `find_instance_constructions`.
    pub(crate) construction_rows: ByNameRange<CalleeDescriptor>,
    /// Per top-level `def` / `class`, every `Name`-rooted call descriptor
    /// found anywhere in its body subtree (recursive). Powers
    /// `find_factory_decls`, which resolves each descriptor through the
    /// file's imports at fan-in and reports the matched constructor kinds.
    /// Kwargs are not captured here (the factory query never reads them).
    pub(crate) factory_rows: ByNameRange<Vec<CalleeDescriptor>>,
    /// Per top-level `def` / `class`, every string-bearing call site found
    /// anywhere in its subtree (recursive, decorators included). Keyed by
    /// the decl's name-range so the project-wide fan-in attributes the call
    /// to that decl. Powers `find_calls_on_attr`, `find_calls_to_imported`,
    /// and `find_calls_on_var`.
    pub(crate) call_sites_by_decl: ByNameRange<Vec<CallSiteFact>>,
    /// Every string-bearing call site reached from a top-level statement
    /// that is *not* a `def` / `class` (module-level assignments,
    /// expressions, `if`/`try`/`with`, …). The fan-in attributes these to
    /// the file's module node, matching `owner_idx_for_stmt_with`.
    pub(crate) module_call_sites: Vec<CallSiteFact>,
}

/// Collects, for every `ClassDef` reachable in the module, the names of
/// the methods defined directly in its body. Recurses through every
/// scope (including function bodies) so module-level conditionally
/// defined classes are captured; the project-wide fan-in filters to
/// global-scope classes via `class_by_selection`.
struct ClassMethodDefsCollector {
    rows: ByNameRange<Vec<String>>,
}

impl<'a> Visitor<'a> for ClassMethodDefsCollector {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if let Stmt::ClassDef(cls) = stmt {
            let methods: Vec<String> = cls
                .body
                .iter()
                .filter_map(|s| match s {
                    Stmt::FunctionDef(f) => Some(f.name.as_str().to_string()),
                    _ => None,
                })
                .collect();
            self.rows.push((range_key(cls.name.range()), methods));
        }
        walk_stmt(self, stmt);
    }
}

/// Collects every `Name`-rooted call descriptor reachable in a decl's
/// body subtree (recursively), for `find_factory_decls`. Mirrors
/// `helpers::FactoryCallFinder` but captures the config-independent
/// descriptor instead of matching against imports — the project-wide
/// query resolves each descriptor through the file's imports at fan-in.
struct FactoryCalleeCollector {
    descriptors: Vec<CalleeDescriptor>,
}

impl<'a> Visitor<'a> for FactoryCalleeCollector {
    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Call(call) = expr {
            if let Some(desc) = callee_descriptor(call, /*capture_kwargs=*/ false) {
                self.descriptors.push(desc);
            }
        }
        walk_expr(self, expr);
    }
}

/// Walk a top-level decl's body, collecting every `Name`-rooted call
/// descriptor, and push a `factory_rows` entry keyed by the decl's
/// name-range when the body contains any. Skips the push for bodies with
/// no resolvable callee so the fan-in query can short-circuit on empty.
fn collect_factory_row(
    rows: &mut ByNameRange<Vec<CalleeDescriptor>>,
    name_range: TextRange,
    body: &[Stmt],
) {
    let mut collector = FactoryCalleeCollector {
        descriptors: Vec::new(),
    };
    for stmt in body {
        collector.visit_stmt(stmt);
    }
    if !collector.descriptors.is_empty() {
        rows.push((range_key(name_range), collector.descriptors));
    }
}

/// Collects every string-bearing call site reachable in a statement's
/// subtree (recursively), for the three call-site queries. Mirrors the
/// live-AST `StringArgCallFinder` / `AttrCallFinder` walk but records the
/// config-independent [`CallSiteFact`] instead of matching against a
/// query's `attr` / imports / owner — the project-wide queries replay the
/// match at fan-in.
struct CallSiteCollector {
    sites: Vec<CallSiteFact>,
}

impl<'a> Visitor<'a> for CallSiteCollector {
    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Call(call) = expr {
            if let Some(fact) = build_call_site_fact(call) {
                self.sites.push(fact);
            }
        }
        walk_expr(self, expr);
    }
}

/// Reduce one call to a [`CallSiteFact`], or `None` when the call carries
/// no string-bearing positional argument — every call-site query captures
/// such an argument, so a call without one can never produce a result and
/// is pruned at extraction time.
fn build_call_site_fact(call: &ExprCall) -> Option<CallSiteFact> {
    let mut string_args: SmallVec<[(usize, StringArg); 1]> = SmallVec::new();
    for (i, arg) in call.arguments.args.iter().enumerate() {
        if let Some(sa) = string_arg(arg) {
            string_args.push((i, sa));
        }
    }
    if string_args.is_empty() {
        return None;
    }
    // `callee_attr` mirrors `AttrCallFinder`: the last attribute segment for
    // *any* receiver shape (incl. `foo().attr(…)`), so it is computed off the
    // unwrapped callee directly rather than the `Name`-rooted chain.
    let callee_attr = match unwrap_subscripted_callee(call.func.as_ref()) {
        Expr::Attribute(attr) => Some(attr.attr.as_str().to_string()),
        _ => None,
    };
    let callee = collapse_callee(call.func.as_ref())
        .map(|(root_name, attrs)| CalleeChain { root_name, attrs });
    Some(CallSiteFact {
        callee_attr,
        callee,
        positional_len: call.arguments.args.len(),
        string_args,
        kwargs: extract_call_kwargs(call),
    })
}

/// Classify a positional argument expression as a string-bearing
/// [`StringArg`], or `None` when it carries no string literal. Mirrors the
/// union of `helpers::nth_positional_string` (single literal) and
/// `helpers::string_or_string_collection` (list/tuple of literals): a
/// list/tuple with no string elements yields `None` (it can never produce
/// a hit), so the call isn't recorded on its account.
fn string_arg(expr: &Expr) -> Option<StringArg> {
    match expr {
        Expr::StringLiteral(s) => Some(StringArg::Lit(s.value.to_str().to_string())),
        Expr::List(list) => {
            let elems = collect_string_elems(&list.elts);
            (!elems.is_empty()).then_some(StringArg::Coll(elems))
        }
        Expr::Tuple(tup) => {
            let elems = collect_string_elems(&tup.elts);
            (!elems.is_empty()).then_some(StringArg::Coll(elems))
        }
        _ => None,
    }
}

/// String-literal elements of a list/tuple, non-string elements dropped —
/// the collection half of `helpers::string_or_string_collection`.
fn collect_string_elems(elts: &[Expr]) -> Vec<String> {
    elts.iter()
        .filter_map(|e| match e {
            Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
            _ => None,
        })
        .collect()
}

/// Walk a top-level statement, collecting every string-bearing call site,
/// and bucket the result the way `owner_idx_for_stmt_with` attributes
/// ownership: a `def` / `class` owns the calls in its subtree (keyed by
/// name-range); every other statement attributes its calls to the module.
fn collect_call_sites(
    stmt: &Stmt,
    by_decl: &mut ByNameRange<Vec<CallSiteFact>>,
    module_sites: &mut Vec<CallSiteFact>,
) {
    let mut collector = CallSiteCollector { sites: Vec::new() };
    collector.visit_stmt(stmt);
    if collector.sites.is_empty() {
        return;
    }
    match stmt {
        Stmt::FunctionDef(f) => by_decl.push((range_key(f.name.range()), collector.sites)),
        Stmt::ClassDef(c) => by_decl.push((range_key(c.name.range()), collector.sites)),
        _ => module_sites.extend(collector.sites),
    }
}

/// Collect a function's positional + keyword-only parameter names in
/// source order (posonly, then args, then kwonly) — matching the order
/// `function_parameters` / `class_method_parameters` produced from the
/// live AST.
fn param_names(params: &ruff_python_ast::Parameters) -> impl Iterator<Item = &str> {
    params
        .posonlyargs
        .iter()
        .chain(params.args.iter())
        .chain(params.kwonlyargs.iter())
        .map(|p| p.parameter.name.id.as_str())
}

/// Peel a callee expression to its `Name` root + trailing attribute
/// segments. Mirrors the live-AST preamble every matcher ran: unwrap a
/// single subscripted-generic callee, then collapse the attribute chain.
/// Returns `None` when the callee doesn't bottom out at a bare `Name`
/// (a call result, a subscript mid chain, …) — those never matched.
fn collapse_callee(expr: &Expr) -> Option<(String, SmallVec<[String; 2]>)> {
    let (root, segs) = collapse_attribute_chain(unwrap_subscripted_callee(expr))?;
    Some((
        root.id.as_str().to_string(),
        segs.iter().map(|s| s.to_string()).collect(),
    ))
}

/// Peel one decorator to a [`CalleeDescriptor`]. Strips the call form
/// (capturing its kwargs) before collapsing the callee; bare decorators
/// (`@route`) carry empty kwargs.
fn decorator_descriptor(dec: &Decorator) -> Option<CalleeDescriptor> {
    let (root_expr, call_form) = match &dec.expression {
        Expr::Call(call) => (&*call.func, Some(call)),
        other => (other, None),
    };
    let (root_name, attrs) = collapse_callee(root_expr)?;
    Some(CalleeDescriptor {
        root_name,
        attrs,
        kwargs: call_form.map(extract_call_kwargs).unwrap_or_default(),
    })
}

/// Peel a call expression's callee to a [`CalleeDescriptor`]. When
/// `capture_kwargs` is false the descriptor's `kwargs` is left empty (the
/// factory walk never reads them, so we skip the work and the alloc).
fn callee_descriptor(call: &ExprCall, capture_kwargs: bool) -> Option<CalleeDescriptor> {
    let (root_name, attrs) = collapse_callee(call.func.as_ref())?;
    Some(CalleeDescriptor {
        root_name,
        attrs,
        kwargs: if capture_kwargs {
            extract_call_kwargs(call)
        } else {
            CallArgs::default()
        },
    })
}

/// Extract every owned fact from a single file's AST in one top-level
/// walk. Warmed during the fan-out (see `project.rs`) so that by the
/// time the project-wide plugin pass runs, this is a pure salsa cache
/// read and the parsed module can already be gone.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_extraction(db: &dyn ProjectDb, file: File) -> FileExtraction {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);

    let main_block_range = if source.contains("__main__") {
        find_main_block_range(&parsed).map(|r| (r.start().to_u32(), r.end().to_u32()))
    } else {
        None
    };

    let mut literal_list_rows: Vec<(String, Vec<String>)> = Vec::new();
    let mut function_params: ByNameRange<Vec<String>> = Vec::new();
    let mut class_method_params: ByNameRange<Vec<String>> = Vec::new();
    let mut class_defs = ClassMethodDefsCollector { rows: Vec::new() };
    let mut import_facts: Vec<ImportFact> = Vec::new();
    let mut decorator_rows: ByNameRange<Vec<CalleeDescriptor>> = Vec::new();
    let mut construction_rows: ByNameRange<CalleeDescriptor> = Vec::new();
    let mut factory_rows: ByNameRange<Vec<CalleeDescriptor>> = Vec::new();
    let mut call_sites_by_decl: ByNameRange<Vec<CallSiteFact>> = Vec::new();
    let mut module_call_sites: Vec<CallSiteFact> = Vec::new();
    // Enclosing package, used to resolve relative `from` imports to
    // absolute modules so the stored facts are relativity-free.
    let file_package = file_package_name(db, file);

    for stmt in &parsed.syntax().body {
        class_defs.visit_stmt(stmt);
        collect_call_sites(stmt, &mut call_sites_by_decl, &mut module_call_sites);
        match stmt {
            Stmt::FunctionDef(func) => {
                let names: Vec<String> =
                    param_names(&func.parameters).map(str::to_string).collect();
                function_params.push((range_key(func.name.range()), names));
                let descriptors: Vec<CalleeDescriptor> = func
                    .decorator_list
                    .iter()
                    .filter_map(decorator_descriptor)
                    .collect();
                if !descriptors.is_empty() {
                    decorator_rows.push((range_key(func.name.range()), descriptors));
                }
                collect_factory_row(&mut factory_rows, func.name.range(), &func.body);
            }
            Stmt::ClassDef(cls) => {
                let mut seen: FxHashSet<&str> = FxHashSet::default();
                let mut names: Vec<String> = Vec::new();
                for body_stmt in &cls.body {
                    let Stmt::FunctionDef(method) = body_stmt else {
                        continue;
                    };
                    for n in param_names(&method.parameters) {
                        if n == "self" || n == "cls" {
                            continue;
                        }
                        if seen.insert(n) {
                            names.push(n.to_string());
                        }
                    }
                }
                class_method_params.push((range_key(cls.name.range()), names));
                collect_factory_row(&mut factory_rows, cls.name.range(), &cls.body);
            }
            Stmt::ImportFrom(im) => {
                let tail = im.module.as_ref().map(|n| n.as_str()).unwrap_or("");
                let absolute = if im.level == 0 {
                    (!tail.is_empty()).then(|| tail.to_string())
                } else {
                    resolve_relative_import(im.level, tail, file_package.as_deref())
                };
                if let Some(absolute) = absolute {
                    let names = im
                        .names
                        .iter()
                        .map(|alias| {
                            let name = alias.name.as_str().to_string();
                            let local = alias
                                .asname
                                .as_ref()
                                .map(|n| n.as_str())
                                .unwrap_or(alias.name.as_str())
                                .to_string();
                            (name, local)
                        })
                        .collect();
                    import_facts.push(ImportFact::From { absolute, names });
                }
            }
            Stmt::Import(im) => {
                let entries = im
                    .names
                    .iter()
                    .map(|alias| {
                        let module_name = alias.name.as_str().to_string();
                        let local = alias
                            .asname
                            .as_ref()
                            .map(|n| n.as_str())
                            .unwrap_or(alias.name.as_str())
                            .to_string();
                        (module_name, local)
                    })
                    .collect();
                import_facts.push(ImportFact::Plain { entries });
            }
            _ => {
                let Some((target_range, value)) = top_level_assign_to_name(stmt) else {
                    continue;
                };
                if let Some(entries) = string_literal_list(value) {
                    let name = source
                        [target_range.start().to_usize()..target_range.end().to_usize()]
                        .to_string();
                    literal_list_rows.push((name, entries.into_iter().map(String::from).collect()));
                } else if let Expr::Call(call) = value {
                    // `NAME = <callee>(...)` — record the callee descriptor so
                    // `find_instance_constructions` can resolve it at fan-in.
                    if let Some(desc) = callee_descriptor(call, /*capture_kwargs=*/ true) {
                        construction_rows.push((range_key(target_range), desc));
                    }
                }
            }
        }
    }

    FileExtraction {
        main_block_range,
        literal_list_rows,
        function_params,
        class_method_params,
        class_method_defs: class_defs.rows,
        import_facts,
        decorator_rows,
        construction_rows,
        factory_rows,
        call_sites_by_decl,
        module_call_sites,
    }
}

/// Rebuild the `{local_name → target}` map a decorator/construction/call
/// query needs from a file's [`ImportFact`]s — the AST-free counterpart
/// of `helpers::collect_modules_imports_local`. `target` is the upstream
/// name (`Flask` for `from flask import Flask`) or [`MODULE_ALIAS_MARKER`]
/// for a module binding (`import flask` / `from unittest import mock`
/// when querying `unittest.mock`). Only entries whose target is in
/// `allowed` survive; later `modules` win on local-name collisions
/// (matching the live-AST helper).
pub(crate) fn imports_local_from_facts(
    facts: &[ImportFact],
    modules: &[String],
    allowed: &FxHashSet<&str>,
) -> FxHashMap<String, String> {
    let mut out: FxHashMap<String, String> = FxHashMap::default();
    for module in modules {
        let parent_last = module.rsplit_once('.');
        for fact in facts {
            match fact {
                ImportFact::From { absolute, names } => {
                    if absolute == module {
                        for (name, local) in names {
                            if allowed.contains(name.as_str()) {
                                out.insert(local.clone(), name.clone());
                            }
                        }
                    } else if let Some((parent, last)) = parent_last {
                        if absolute == parent {
                            for (name, local) in names {
                                if name == last {
                                    out.insert(local.clone(), MODULE_ALIAS_MARKER.to_string());
                                }
                            }
                        }
                    }
                }
                ImportFact::Plain { entries } => {
                    for (module_name, local) in entries {
                        if module_name == module {
                            out.insert(local.clone(), MODULE_ALIAS_MARKER.to_string());
                        }
                    }
                }
            }
        }
    }
    out
}

/// Query-time port of `helpers::matched_call_target_any` operating on a
/// precomputed `Name`-rooted callee chain (`root_name` + `attrs`) instead
/// of a live AST callee. `imports` is the file's `{local → target}` map
/// from [`imports_local_from_facts`]; `modules`/`allowed` are the configured
/// target modules and constructor names. Returns the matched upstream name
/// on hit, else `None`. The three arms mirror the live-AST matcher exactly:
///
/// * `[]` — `<local>(...)` bound via `from <module> import <name>`;
/// * `[attr]` — `<alias>.<name>(...)` where `alias` is a module binding;
/// * `[…, name]` (len ≥ 2) — literal dotted access of a multi-segment
///   module (`unittest.mock.patch(...)`).
///
/// Backs both the descriptor queries (via [`match_callee_descriptor`]) and
/// the call-site query (on [`CallSiteFact::callee`]).
pub(crate) fn match_callee_chain(
    root_name: &str,
    attrs: &[String],
    imports: &FxHashMap<String, String>,
    modules: &[String],
    allowed: &FxHashSet<&str>,
) -> Option<String> {
    match attrs {
        [] => imports
            .get(root_name)
            .filter(|target| allowed.contains(target.as_str()))
            .cloned(),
        [attr] => (imports.get(root_name).map(String::as_str) == Some(MODULE_ALIAS_MARKER)
            && allowed.contains(attr.as_str()))
        .then(|| attr.clone()),
        segs => {
            let last = segs.last().expect("slice has len >= 2");
            if !allowed.contains(last.as_str()) {
                return None;
            }
            // Reconstruct the dotted module prefix: root + all but the
            // final segment (the constructor name).
            let mut dotted = String::with_capacity(root_name.len());
            dotted.push_str(root_name);
            for seg in &segs[..segs.len() - 1] {
                dotted.push('.');
                dotted.push_str(seg);
            }
            modules
                .iter()
                .any(|module| module.as_str() == dotted)
                .then(|| last.clone())
        }
    }
}

/// [`match_callee_chain`] for a precomputed [`CalleeDescriptor`] — the
/// decorator / construction / factory queries call this with the descriptor
/// they stored; the call-site query calls `match_callee_chain` directly on a
/// [`CalleeChain`].
pub(crate) fn match_callee_descriptor(
    desc: &CalleeDescriptor,
    imports: &FxHashMap<String, String>,
    modules: &[String],
    allowed: &FxHashSet<&str>,
) -> Option<String> {
    match_callee_chain(&desc.root_name, &desc.attrs, imports, modules, allowed)
}
