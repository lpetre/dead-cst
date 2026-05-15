//! ty-backed native graph builder for dead-cst.
//!
//! Architecture is governed by the crate's `CLAUDE.md`: ty does every
//! piece of Python semantics, ruff is only used where ty hasn't surfaced
//! the structure we need, and there is **no per-file cache** (ty's
//! Salsa db is the cache).
//!
//! The pipeline is one method (`Project.build()`) returning one
//! project-wide `NativeGraph`:
//!
//! 1. **Phase 1 — decls**. For every project file, iterate every
//!    binding in the file's global scope via
//!    `UseDefMap::all_definitions_with_usage`, minting a node per
//!    binding (including each name brought in by `from foo import *`).
//!    Each node lands in a global `(File, target_range) → node_idx`
//!    index so cross-file edges can find it later.
//! 2. **Phase 2 — chain**. For every module node, emit the submodule
//!    edge to its parent. For every import-kind binding, resolve the
//!    upstream target via `ty_module_resolver::resolve_module` and
//!    emit `alias_node → upstream_node`; lazily mint a module-only
//!    node for any target outside the project (stdlib / site-packages).
//! 3. **Phase 3 — references**. For every Definition that owns an
//!    expression (function body, class body, assignment value,
//!    annotation), walk the contained `Name`s and resolve each to its
//!    reaching def via `visible_ancestor_scopes` +
//!    `end_of_scope_symbol_bindings` (Principle 2 — the local alias
//!    is the target, not the upstream definition). Module-level
//!    non-definition statements attribute to the module.

#![allow(clippy::useless_conversion)]

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::{File, FilePath};
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_db::source::{line_index, source_text};
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Expr, ExprName, Stmt};
use ruff_source_file::LineIndex;
use ruff_text_size::TextRange;
use ty_module_resolver::{file_to_module, resolve_module, ModuleName};
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::{RangedValue, RelativePathBuf};
use ty_project::{Db as ProjectDb, ProjectDatabase, ProjectMetadata};
use ty_python_core::definition::{DefinitionKind, DefinitionState};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::program::UseDefaultStrategy;
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;
use ty_python_semantic::{
    definitions_for_imported_symbol, ImportAliasResolution, ResolvedDefinition, SemanticModel,
};

// ---------------------------------------------------------------------------
// Public Python data classes
// ---------------------------------------------------------------------------

/// Raw record of one cross-file import reference, attached to a
/// `kind="import"` node. Mirrors `dead_cst.graph.Import`.
///
/// `module` is the import's *absolute* dotted target (relative dots
/// are resolved by ty before this field is populated). `decl` is the
/// from-style imported name (`None` for plain `import` and for the
/// per-name nodes minted from `from X import *`). `star` flags the
/// implicit-from-star case.
#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct Import {
    module: String,
    decl: Option<String>,
    star: bool,
}

#[pymethods]
impl Import {
    #[new]
    #[pyo3(signature = (module, decl = None, star = false))]
    fn new(module: String, decl: Option<String>, star: bool) -> Self {
        Self { module, decl, star }
    }

    fn __repr__(&self) -> String {
        format!(
            "Import(module={:?}, decl={:?}, star={})",
            self.module, self.decl, self.star,
        )
    }
}

/// A single node in a `NativeGraph`.
///
/// `imports` is populated for `kind="import"` nodes only (one per
/// alias, plus one per name brought in by `from X import *`). All
/// other kinds carry `None`.
#[pyclass(get_all, frozen)]
struct NativeNode {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    flags: u32,
    imports: Option<Py<Import>>,
}

#[pymethods]
impl NativeNode {
    fn __repr__(&self) -> String {
        format!(
            "NativeNode(fqname={:?}, kind={:?}, path={:?}, start=({}, {}), end=({}, {}), flags={})",
            self.fqname,
            self.kind,
            self.path,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column,
            self.flags,
        )
    }
}

/// One project-wide graph contribution, packed for one FFI hop.
#[pyclass(get_all, frozen)]
struct NativeGraph {
    nodes: Vec<Py<NativeNode>>,
    edges: Vec<(usize, usize, u32)>,
}

#[pymethods]
impl NativeGraph {
    fn __repr__(&self) -> String {
        format!(
            "NativeGraph(nodes={}, edges={})",
            self.nodes.len(),
            self.edges.len(),
        )
    }
}

/// Key into the cross-file decl lookup.
///
/// `target_range` alone is ambiguous for star imports — every name
/// brought in by one `from foo import *` shares the same `*` alias
/// range. Including the bound `place_id` disambiguates star bindings
/// while still distinguishing two `def f` redefinitions by their
/// distinct target ranges.
type DeclKey = (File, ScopedPlaceId, (u32, u32));

type DeclIndex = HashMap<DeclKey, usize>;

// ---------------------------------------------------------------------------
// GraphBuilder + node interning
// ---------------------------------------------------------------------------

/// Positional identity for a node.
///
/// Per `CLAUDE.md` principle 3, `flags` is deliberately *not* part of
/// the key: two nodes for the same `(fqname, kind, path, position)` are
/// the same node regardless of which path computed their flags.
#[derive(Hash, Eq, PartialEq, Clone)]
struct NodeKey {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

struct GraphBuilder {
    nodes: Vec<Py<NativeNode>>,
    node_index: HashMap<NodeKey, usize>,
    edges: Vec<(usize, usize, u32)>,
    edge_set: HashSet<(usize, usize, u32)>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            edges: Vec::new(),
            edge_set: HashSet::new(),
        }
    }

    fn intern_node(&mut self, py: Python<'_>, node: NativeNode) -> PyResult<usize> {
        let key = NodeKey {
            fqname: node.fqname.clone(),
            kind: node.kind,
            path: node.path.clone(),
            start_line: node.start_line,
            start_column: node.start_column,
            end_line: node.end_line,
            end_column: node.end_column,
        };
        if let Some(&idx) = self.node_index.get(&key) {
            return Ok(idx);
        }
        let idx = self.nodes.len();
        self.nodes.push(Py::new(py, node)?);
        self.node_index.insert(key, idx);
        Ok(idx)
    }

    fn add_edge(&mut self, src: usize, dst: usize, flags: u32) {
        let triple = (src, dst, flags);
        if self.edge_set.insert(triple) {
            self.edges.push(triple);
        }
    }
}

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

/// A ty-backed analysis project with explicitly-injected configuration.
#[pyclass(unsendable)]
struct Project {
    db: ProjectDatabase,
}

#[pymethods]
impl Project {
    #[new]
    #[pyo3(signature = (
        root,
        *,
        src_roots = None,
        extra_paths = None,
        python_env = None,
        python_version = None,
        typeshed = None,
    ))]
    fn new(
        root: &str,
        src_roots: Option<Vec<String>>,
        extra_paths: Option<Vec<String>>,
        python_env: Option<&str>,
        python_version: Option<&str>,
        typeshed: Option<&str>,
    ) -> PyResult<Self> {
        let root = SystemPathBuf::from(root);

        let env = EnvironmentOptions {
            root: src_roots.map(|paths| paths.into_iter().map(rel_path).collect()),
            extra_paths: extra_paths.map(|paths| paths.into_iter().map(rel_path).collect()),
            python: python_env.map(rel_path),
            python_version: python_version
                .map(|v| {
                    SupportedPythonVersion::from_str(v).map_err(|e| {
                        PyValueError::new_err(format!("invalid python_version {v:?}: {e}"))
                    })
                })
                .transpose()?
                .map(RangedValue::cli),
            typeshed: typeshed.map(rel_path),
            ..EnvironmentOptions::default()
        };

        let options = Options {
            environment: Some(env),
            ..Options::default()
        };

        let metadata =
            ProjectMetadata::from_options(options, root.clone(), None, &UseDefaultStrategy)
                .map_err(|e| PyValueError::new_err(format!("invalid configuration: {e:?}")))?;

        let cwd = std::env::current_dir()
            .map_err(|e| PyOSError::new_err(format!("cwd unavailable: {e}")))?;
        let cwd = SystemPathBuf::from_path_buf(cwd).map_err(|_| {
            PyValueError::new_err("current working directory is not a valid absolute UTF-8 path")
        })?;
        let system = OsSystem::new(cwd);

        Ok(Self {
            db: ProjectDatabase::use_defaults(metadata, system),
        })
    }

    /// Build the project-wide symbol graph.
    fn build(&self, py: Python<'_>) -> PyResult<NativeGraph> {
        let mut builder = GraphBuilder::new();
        let mut global_index: DeclIndex = HashMap::new();
        let mut module_nodes: HashMap<File, usize> = HashMap::new();

        let project_files: Vec<File> = (&self.db.project().files(&self.db)).into_iter().collect();

        for file in &project_files {
            ingest_decls(
                py,
                &self.db,
                *file,
                &mut builder,
                &mut global_index,
                &mut module_nodes,
            )?;
        }
        for file in &project_files {
            emit_module_hierarchy(&self.db, *file, &module_nodes, &mut builder);
            emit_import_edges(
                py,
                &self.db,
                *file,
                &mut builder,
                &mut global_index,
                &mut module_nodes,
            )?;
        }
        for file in &project_files {
            emit_reference_edges(&self.db, *file, &global_index, &module_nodes, &mut builder);
        }

        Ok(NativeGraph {
            nodes: builder.nodes,
            edges: builder.edges,
        })
    }
}

// ---------------------------------------------------------------------------
// Phase 1: decl enumeration via ty's SemanticIndex
// ---------------------------------------------------------------------------

fn ingest_decls(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = module_fqname_for_file(db, file);

    let (msl, msc, mel, mec) = position(&line_index, &source, parsed.syntax().range);
    let module_idx = builder.intern_node(
        py,
        NativeNode {
            fqname: module_fqname.clone(),
            kind: "module",
            path: path_str.clone(),
            start_line: msl,
            start_column: msc,
            end_line: mel,
            end_column: mec,
            flags: 0,
            imports: None,
        },
    )?;
    module_nodes.insert(file, module_idx);

    // Iterate every binding (including shadowed siblings) — Principle 3.
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);

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
        let local_name = symbol.name().as_str().to_string();

        let target_range = kind.target_range(&parsed);
        let (sl, sc, el, ec) = position(&line_index, &source, target_range);

        let imports = if node_kind == "import" {
            Some(Py::new(py, import_payload_for(kind, db, file, &parsed))?)
        } else {
            None
        };

        let node_idx = builder.intern_node(
            py,
            NativeNode {
                fqname: format!("{module_fqname}.{local_name}"),
                kind: node_kind,
                path: path_str.clone(),
                start_line: sl,
                start_column: sc,
                end_line: el,
                end_column: ec,
                flags: 0,
                imports,
            },
        )?;
        builder.add_edge(node_idx, module_idx, 0);
        global_index.insert((file, place_id, range_key(target_range)), node_idx);
    }

    Ok(())
}

fn decl_kind_str(kind: &DefinitionKind<'_>) -> Option<&'static str> {
    if kind.is_import() {
        return Some("import");
    }
    Some(match kind {
        DefinitionKind::Function(_) => "function",
        DefinitionKind::Class(_) => "class",
        DefinitionKind::TypeAlias(_) => "type_alias",
        DefinitionKind::Assignment(_)
        | DefinitionKind::AnnotatedAssignment(_)
        | DefinitionKind::NamedExpression(_)
        | DefinitionKind::For(_)
        | DefinitionKind::WithItem(_)
        | DefinitionKind::ExceptHandler(_)
        | DefinitionKind::MatchPattern(_) => "variable",
        _ => return None,
    })
}

fn import_payload_for<'db>(
    kind: &DefinitionKind<'db>,
    db: &'db dyn ProjectDb,
    file: File,
    parsed: &ParsedModuleRef,
) -> Import {
    match kind {
        DefinitionKind::Import(k) => {
            let alias = k.alias(parsed);
            Import {
                module: alias.name.id.as_str().to_string(),
                decl: None,
                star: false,
            }
        }
        DefinitionKind::ImportFrom(k) => {
            let alias = k.alias(parsed);
            Import {
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
            Import {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(k.module(parsed).id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::StarImport(k) => Import {
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
fn from_module_string(
    db: &dyn ProjectDb,
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

fn emit_module_hierarchy(
    db: &ProjectDatabase,
    file: File,
    module_nodes: &HashMap<File, usize>,
    builder: &mut GraphBuilder,
) {
    if let Some((self_idx, parent_idx)) = parent_module_edge(db, file, module_nodes) {
        builder.add_edge(self_idx, parent_idx, 0);
    }
}

fn parent_module_edge(
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

fn emit_import_edges(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);
    let model = SemanticModel::new(db, file);

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

        let target = match kind {
            DefinitionKind::Import(k) => resolve_import_target(
                py,
                db,
                k.alias(&parsed).name.id.as_str(),
                file,
                builder,
                module_nodes,
            )?,
            DefinitionKind::ImportFrom(k) => resolve_from_imported(
                py,
                db,
                &model,
                k.import(&parsed),
                k.alias(&parsed).name.id.as_str(),
                builder,
                module_nodes,
                global_index,
            )?,
            DefinitionKind::ImportFromSubmodule(k) => resolve_from_imported(
                py,
                db,
                &model,
                k.import(&parsed),
                k.module(&parsed).id.as_str(),
                builder,
                module_nodes,
                global_index,
            )?,
            DefinitionKind::StarImport(k) => resolve_from_imported(
                py,
                db,
                &model,
                k.import(&parsed),
                place_table.symbol(k.symbol_id()).name().as_str(),
                builder,
                module_nodes,
                global_index,
            )?,
            _ => continue,
        };
        if let Some(target_idx) = target {
            builder.add_edge(alias_idx, target_idx, 0);
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
fn resolve_import_target(
    py: Python<'_>,
    db: &ProjectDatabase,
    dotted: &str,
    importing_file: File,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<Option<usize>> {
    let Some(module_name) = ModuleName::new(dotted) else {
        return Ok(None);
    };
    let Some(module) = resolve_module(db, importing_file, &module_name) else {
        return Ok(None);
    };
    let Some(file) = module.file(db) else {
        return Ok(None);
    };
    Ok(Some(mint_module_node(py, db, file, builder, module_nodes)?))
}

/// Resolve `from <stmt> import <symbol>` to its upstream target node.
///
/// Delegates to ty's `definitions_for_imported_symbol`, which already
/// tries `<module>.<symbol>` as a submodule first, then falls back to a
/// global-scope binding in `<module>`, then to namespace-package
/// submodule resolution. Returns the first `ResolvedDefinition` we can
/// map to a graph node; otherwise mints (or finds) the upstream module
/// node so the alias still has an out-edge.
#[allow(clippy::too_many_arguments)]
fn resolve_from_imported(
    py: Python<'_>,
    db: &ProjectDatabase,
    model: &SemanticModel<'_>,
    stmt: &ruff_python_ast::StmtImportFrom,
    symbol_name: &str,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    global_index: &DeclIndex,
) -> PyResult<Option<usize>> {
    let resolved = definitions_for_imported_symbol(
        model,
        stmt,
        symbol_name,
        ImportAliasResolution::ResolveAliases,
    );
    for r in &resolved {
        match r {
            ResolvedDefinition::Definition(def) => {
                let target_file = def.file(db);
                let parsed = parsed_module(db, target_file).load(db);
                let key = (
                    target_file,
                    def.place(db),
                    range_key(def.kind(db).target_range(&parsed)),
                );
                if let Some(&idx) = global_index.get(&key) {
                    return Ok(Some(idx));
                }
            }
            ResolvedDefinition::Module(target_file) => {
                return Ok(Some(mint_module_node(
                    py,
                    db,
                    *target_file,
                    builder,
                    module_nodes,
                )?));
            }
            ResolvedDefinition::FileWithRange(_) => continue,
        }
    }
    // ty resolved nothing we can graph; fall back to the upstream module
    // so the alias still propagates reachability.
    let Ok(module_name) = ModuleName::from_import_statement(db, model.file(), stmt) else {
        return Ok(None);
    };
    let Some(module) = resolve_module(db, model.file(), &module_name) else {
        return Ok(None);
    };
    let Some(target_file) = module.file(db) else {
        return Ok(None);
    };
    Ok(Some(mint_module_node(
        py,
        db,
        target_file,
        builder,
        module_nodes,
    )?))
}

fn mint_module_node(
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
    let idx = builder.intern_node(
        py,
        NativeNode {
            fqname,
            kind: "module",
            path: path_str,
            start_line: sl,
            start_column: sc,
            end_line: el,
            end_column: ec,
            flags: 0,
            imports: None,
        },
    )?;
    module_nodes.insert(file, idx);
    Ok(idx)
}

// ---------------------------------------------------------------------------
// Phase 3: same-file Name→decl reference edges
// ---------------------------------------------------------------------------

fn emit_reference_edges(
    db: &ProjectDatabase,
    file: File,
    global_index: &DeclIndex,
    module_nodes: &HashMap<File, usize>,
    builder: &mut GraphBuilder,
) {
    let Some(&module_idx) = module_nodes.get(&file) else {
        return;
    };
    let parsed = parsed_module(db, file).load(db);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let use_def_map = index.use_def_map(global);
    let model = SemanticModel::new(db, file);

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

        let mut coll = RefCollector::new(owner_idx, &model, file, &parsed, global_index);
        walk_owned(kind, &parsed, &mut coll);
        coll.flush(builder);
    }

    // (b) Module-level statements that don't carry a Definition (and
    //     so didn't get covered by (a)) attribute to the module node.
    for stmt in &parsed.syntax().body {
        if stmt_creates_top_level_definition(stmt) {
            continue;
        }
        let mut coll = RefCollector::new(module_idx, &model, file, &parsed, global_index);
        coll.visit_stmt(stmt);
        coll.flush(builder);
    }
}

/// True iff this top-level statement is a binding form whose Names
/// have already been attributed by the per-definition walk in (a).
///
/// Compound non-scope statements (``if`` / ``while`` / ``for`` / ...)
/// return ``false`` here: their *bodies* contain definitions that (a)
/// covers, but their *test/iter/etc. expressions* belong to the module
/// and (b) needs to walk them.
fn stmt_creates_top_level_definition(stmt: &Stmt) -> bool {
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
fn walk_owned(kind: &DefinitionKind<'_>, parsed: &ParsedModuleRef, v: &mut RefCollector<'_, '_>) {
    match kind {
        DefinitionKind::Function(func) => {
            for s in &func.node(parsed).body {
                v.visit_stmt(s);
            }
        }
        DefinitionKind::Class(cls) => {
            for s in &cls.node(parsed).body {
                v.visit_stmt(s);
            }
        }
        DefinitionKind::Assignment(a) => v.visit_expr(a.value(parsed)),
        DefinitionKind::AnnotatedAssignment(a) => {
            v.visit_expr(a.annotation(parsed));
            if let Some(val) = a.value(parsed) {
                v.visit_expr(val);
            }
        }
        DefinitionKind::For(for_stmt) => v.visit_expr(for_stmt.iterable(parsed)),
        DefinitionKind::WithItem(item) => v.visit_expr(item.context_expr(parsed)),
        DefinitionKind::NamedExpression(named) => v.visit_expr(named.node(parsed).value.as_ref()),
        DefinitionKind::TypeAlias(alias) => v.visit_expr(alias.node(parsed).value.as_ref()),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Reference collector
// ---------------------------------------------------------------------------

/// Walks an expression / body and records every Name reference,
/// attributing to a single owner decl.
///
/// Resolution is delegated to ty's `definitions_for_name` with
/// `PreserveAliases`: imported-name uses land on the local import
/// node, not on the cross-file target (Principle 2 — edges flow
/// through the local decl). Shadow handling falls out of ty's
/// flow-sensitive use-def chain (Principle 3 — each use lands on its
/// reaching def, never on all same-name siblings).
struct RefCollector<'a, 'db> {
    owner: usize,
    model: &'a SemanticModel<'db>,
    file: File,
    parsed: &'a ParsedModuleRef,
    global_index: &'a DeclIndex,
    edges: HashSet<(usize, usize)>,
}

impl<'a, 'db> RefCollector<'a, 'db> {
    fn new(
        owner: usize,
        model: &'a SemanticModel<'db>,
        file: File,
        parsed: &'a ParsedModuleRef,
        global_index: &'a DeclIndex,
    ) -> Self {
        Self {
            owner,
            model,
            file,
            parsed,
            global_index,
            edges: HashSet::new(),
        }
    }

    fn flush(self, builder: &mut GraphBuilder) {
        for (src, dst) in self.edges {
            builder.add_edge(src, dst, 0);
        }
    }

    fn emit_name(&mut self, name: &ExprName) {
        // Walk the use's scope chain looking for *local* bindings of
        // ``name``. We deliberately do NOT call
        // ``definitions_for_name`` here -- it always chases bare
        // ``from X import y`` to the upstream definition in ``X``, and
        // Principle 2 requires use edges to terminate at the local
        // alias node. The scope walk stops at the first scope that
        // binds the name; bindings inside that scope are emitted as
        // edges (ty's flow tracking already filters reachable ones).
        let db = self.model.db();
        let index = semantic_index(db, self.file);
        let Some(file_scope) = self.model.scope(name.into()) else {
            return;
        };

        for (scope_id, _scope) in index.visible_ancestor_scopes(file_scope) {
            let place_table = index.place_table(scope_id);
            let Some(symbol_id) = place_table.symbol_id(name.id.as_str()) else {
                continue;
            };
            let use_def_map = index.use_def_map(scope_id);
            // ``end_of_scope_symbol_bindings`` is flow-sensitive: it
            // returns only bindings that *reach* the end of the scope
            // (Principle 3 — shadowed-earlier defs that have been
            // overwritten by end-of-scope are excluded). For free
            // references in nested functions/classes this is the
            // right query because the function body runs after the
            // enclosing module finishes loading, by which point the
            // module's bindings are at their end-of-scope state.
            let mut matched = false;
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
                let Some(&dst) = self.global_index.get(&key) else {
                    continue;
                };
                if dst != self.owner {
                    self.edges.insert((self.owner, dst));
                }
                matched = true;
            }
            if matched {
                break;
            }
        }
    }
}

impl<'ast, 'db> Visitor<'ast> for RefCollector<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Name(n) = expr {
            self.emit_name(n);
            return;
        }
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        walk_stmt(self, stmt);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rel_path<P: AsRef<str>>(path: P) -> RelativePathBuf {
    RelativePathBuf::cli(SystemPath::new(path.as_ref()))
}

fn position(index: &LineIndex, source: &str, range: TextRange) -> (usize, usize, usize, usize) {
    let start = index.line_column(range.start(), source);
    let end = index.line_column(range.end(), source);
    (
        start.line.get(),
        start.column.get() - 1,
        end.line.get(),
        end.column.get() - 1,
    )
}

fn range_key(range: TextRange) -> (u32, u32) {
    (range.start().to_u32(), range.end().to_u32())
}

fn file_path_string(db: &dyn ProjectDb, file: File) -> String {
    match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    }
}

fn module_fqname_for_file(db: &dyn ProjectDb, file: File) -> String {
    file_to_module(db, file)
        .map(|m| m.name(db).as_str().to_string())
        .unwrap_or_else(|| file_path_string(db, file))
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

#[pymodule]
fn dead_cst_ty_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Import>()?;
    m.add_class::<NativeNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<Project>()?;
    Ok(())
}
