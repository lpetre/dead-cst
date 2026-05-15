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

/// Cloneable snapshot of an alias's import payload.
///
/// Lives in `alias_imports` (alias_node_idx -> spec) so the reference
/// collector can emit *parallel reachability edges* through an alias
/// without holding a PyO3 reference. Mirrors `Import`'s three fields.
#[derive(Clone, Debug)]
struct ImportSpec {
    module: String,
    decl: Option<String>,
    star: bool,
}

/// (upstream_file, local_decl_name) -> upstream decl node idx.
///
/// Populated during `ingest_decls` for every non-import, non-module
/// node. The last write wins, so the entry tracks the *end-of-scope
/// live* binding for any name that gets rebound during the module's
/// body — matching how the libcst pipeline's trie only keeps the
/// non-`SHADOWED` decl.
type LiveDeclIndex = HashMap<(File, String), usize>;

/// Outcome of resolving a `Name` use to its reaching definition.
///
/// `Alias` is the module-scope path: the use has a local graph node
/// (an import alias or a top-level decl) that takes the in-edge.
/// `NestedImport` is the function-/class-scope path: ty saw an import
/// binding in a non-global scope, so no graph node was minted, and
/// the use's parallel upstream edges flow from the enclosing top-level
/// owner instead.
enum Resolution {
    Alias(usize),
    NestedImport {
        spec: ImportSpec,
        bound_name: String,
    },
}

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
        let mut alias_imports: HashMap<usize, ImportSpec> = HashMap::new();
        let mut live_decls: LiveDeclIndex = HashMap::new();

        let project_files: Vec<File> = (&self.db.project().files(&self.db)).into_iter().collect();

        for file in &project_files {
            ingest_decls(
                py,
                &self.db,
                *file,
                &mut builder,
                &mut global_index,
                &mut module_nodes,
                &mut alias_imports,
                &mut live_decls,
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
            emit_reference_edges(
                &self.db,
                *file,
                &global_index,
                &module_nodes,
                &alias_imports,
                &live_decls,
                &mut builder,
            );
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

#[allow(clippy::too_many_arguments)]
fn ingest_decls(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
    alias_imports: &mut HashMap<usize, ImportSpec>,
    live_decls: &mut LiveDeclIndex,
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

        if let Some(spec) = import_spec {
            alias_imports.insert(node_idx, spec);
        } else if node_kind != "module" {
            // Last-write-wins so the map ends up with the end-of-scope
            // live binding (mirrors libcst's `SHADOWED`-excluding trie).
            live_decls.insert((file, local_name.clone()), node_idx);
        }
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

fn import_payload_for<'db>(
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
fn from_module_string(
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

        let from_stmt = match kind {
            DefinitionKind::ImportFrom(k) => Some(k.import(&parsed)),
            DefinitionKind::ImportFromSubmodule(k) => Some(k.import(&parsed)),
            DefinitionKind::StarImport(k) => Some(k.import(&parsed)),
            _ => None,
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

        // Parallel reachability edge: when `from X import Y` resolved
        // to a *decl* (Y in module X), also link the alias to the
        // upstream module X so reachability can see the file's
        // module-level dependency. Skip when the target is itself a
        // module (the `from X import sub` case where `sub` is a
        // submodule of X) — the submodule's parent-module edge
        // already keeps X alive.
        if let Some(stmt) = from_stmt {
            let target_is_module = target
                .map(|idx| module_nodes.values().any(|&m| m == idx))
                .unwrap_or(false);
            if !target_is_module {
                if let Some(upstream_module_idx) =
                    from_import_module_node(py, db, file, stmt, builder, module_nodes)?
                {
                    if Some(upstream_module_idx) != target && upstream_module_idx != alias_idx {
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

/// Resolve the upstream module of a `from <stmt> import ...` and
/// return (or mint) its module node.
///
/// Returns `Ok(None)` when ty's `from_import_statement` cannot resolve
/// the target (invalid syntax, too many leading dots, missing file).
fn from_import_module_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    importing_file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<Option<usize>> {
    let Ok(module_name) = ModuleName::from_import_statement(db, importing_file, stmt) else {
        return Ok(None);
    };
    let Some(module) = resolve_module(db, importing_file, &module_name) else {
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

#[allow(clippy::too_many_arguments)]
fn emit_reference_edges(
    db: &ProjectDatabase,
    file: File,
    global_index: &DeclIndex,
    module_nodes: &HashMap<File, usize>,
    alias_imports: &HashMap<usize, ImportSpec>,
    live_decls: &LiveDeclIndex,
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

        let mut coll = RefCollector::new(
            owner_idx,
            &model,
            file,
            &parsed,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
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
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
        );
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

fn walk_decorators(decorators: &[ruff_python_ast::Decorator], v: &mut RefCollector<'_, '_>) {
    for d in decorators {
        v.visit_expr(&d.expression);
    }
}

fn walk_parameters(parameters: &ruff_python_ast::Parameters, v: &mut RefCollector<'_, '_>) {
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

fn walk_type_params(type_params: &ruff_python_ast::TypeParams, v: &mut RefCollector<'_, '_>) {
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

/// True iff `target` is the bare `Name("__all__")` LHS of an
/// assignment (`__all__ = ...` or `__all__: list[str] = ...`).
///
/// Subscript / attribute / tuple-unpack targets (e.g.
/// `__all__[0] = "f"`, `mod.__all__ = [...]`) are excluded — those
/// don't redefine the binding in a way the libcst pipeline picks up,
/// and treating them as `__all__` would emit edges from unrelated
/// nodes.
fn target_is_dunder_all(target: &Expr) -> bool {
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
fn emit_dunder_all_edges(v: &mut RefCollector<'_, '_>, value: &Expr) {
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
struct RefCollector<'a, 'db> {
    owner: usize,
    model: &'a SemanticModel<'db>,
    file: File,
    parsed: &'a ParsedModuleRef,
    global_index: &'a DeclIndex,
    module_nodes: &'a HashMap<File, usize>,
    alias_imports: &'a HashMap<usize, ImportSpec>,
    live_decls: &'a LiveDeclIndex,
    edges: HashSet<(usize, usize)>,
    /// `true` while walking a function- or class-body subtree. In
    /// that context, nested `Stmt::Import` / `Stmt::ImportFrom`
    /// statements emit parallel upstream edges from `owner` (no
    /// alias node is minted, since the binding lives in a non-global
    /// scope). At module level the flag stays false, so we don't
    /// double-emit for imports that `emit_import_edges` already
    /// processed via their proper alias nodes.
    nested_context: bool,
}

impl<'a, 'db> RefCollector<'a, 'db> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        owner: usize,
        model: &'a SemanticModel<'db>,
        file: File,
        parsed: &'a ParsedModuleRef,
        global_index: &'a DeclIndex,
        module_nodes: &'a HashMap<File, usize>,
        alias_imports: &'a HashMap<usize, ImportSpec>,
        live_decls: &'a LiveDeclIndex,
    ) -> Self {
        Self {
            owner,
            model,
            file,
            parsed,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            edges: HashSet::new(),
            nested_context: false,
        }
    }

    fn flush(self, builder: &mut GraphBuilder) {
        for (src, dst) in self.edges {
            builder.add_edge(src, dst, 0);
        }
    }

    fn emit_edge(&mut self, dst: usize) {
        if dst != self.owner {
            self.edges.insert((self.owner, dst));
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
        let index = semantic_index(db, self.file);
        let global = FileScopeId::global();
        let place_table = index.place_table(global);
        let symbol_id = place_table.symbol_id(name)?;
        let use_def_map = index.use_def_map(global);
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

    /// Walk the use's scope chain looking for the reaching definition
    /// of `name`. Returns `None` when no binding is found.
    ///
    /// Walks outward through visible scopes until we find one that
    /// *binds* `name` (not merely lists it in its place table — a free
    /// variable can show up there without any local binding, and in
    /// that case we want to keep walking up to the actual definer).
    /// The first binding scope wins; ty's flow-sensitive
    /// `end_of_scope_symbol_bindings` filters within that scope to the
    /// reaching def (Principle 3).
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
    fn find_local_binding(&self, name: &ExprName) -> Option<Resolution> {
        let db = self.model.db();
        let index = semantic_index(db, self.file);
        let file_scope = self.model.scope(name.into())?;
        for (scope_id, _scope) in index.visible_ancestor_scopes(file_scope) {
            let place_table = index.place_table(scope_id);
            let Some(symbol_id) = place_table.symbol_id(name.id.as_str()) else {
                continue;
            };
            let use_def_map = index.use_def_map(scope_id);
            // The first scope that *binds* this name wins, even if the
            // binding has no graph node (function parameter, loop
            // target, except-as, walrus inside a comprehension, …) —
            // outer scopes are shadowed by this one. Free-variable
            // references show up in `place_table` without producing
            // any bindings, and for those we fall through to outer
            // scopes (handled by `place_table.symbol_id` returning
            // `Some` but the bindings iterator yielding nothing).
            let mut saw_binding = false;
            let mut result: Option<Resolution> = None;
            for binding in use_def_map.end_of_scope_symbol_bindings(symbol_id) {
                let Some(def) = binding.binding.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                saw_binding = true;
                if result.is_some() {
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
                    result = Some(Resolution::Alias(idx));
                    continue;
                }
                if kind.is_import() {
                    let PlaceExprRef::Symbol(sym) = place_table.place(place_id) else {
                        continue;
                    };
                    let bound_name = sym.name().as_str().to_string();
                    let spec = import_payload_for(kind, db, self.file, self.parsed);
                    result = Some(Resolution::NestedImport { spec, bound_name });
                }
            }
            if saw_binding {
                return result;
            }
        }
        None
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
    /// edges, using the spec ty handed us.
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
        match self.find_local_binding(name) {
            Some(Resolution::Alias(alias_idx)) => {
                self.emit_edge(alias_idx);
                if let Some(spec) = self.alias_imports.get(&alias_idx).cloned() {
                    self.emit_upstream(&spec, name.id.as_str(), extra_chain);
                }
            }
            Some(Resolution::NestedImport { spec, bound_name }) => {
                self.emit_upstream(&spec, &bound_name, extra_chain);
            }
            None => {}
        }
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

        // Resolve the initial loading target to a project file.
        let Some(start_mn) = ModuleName::new(&loading_target) else {
            return;
        };
        let Some(start_module) = resolve_module(db, self.file, &start_mn) else {
            return;
        };
        let Some(start_file) = start_module.file(db) else {
            return;
        };

        // Decl-style alias: emit edges to the upstream module and the
        // decl inside it, then stop. Attribute access past a decl is
        // field access on the decl's value, which we don't model.
        if let Some(decl_name) = decl_tail {
            if let Some(&idx) = self.module_nodes.get(&start_file) {
                self.emit_edge(idx);
            }
            if let Some(&idx) = self.live_decls.get(&(start_file, decl_name)) {
                self.emit_edge(idx);
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
        let mut terminal_decl_idx: Option<usize> = None;
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
            terminal_decl_idx = self
                .live_decls
                .get(&(current_file, (*seg).to_string()))
                .copied();
            break;
        }

        if let Some(&idx) = self.module_nodes.get(&current_file) {
            self.emit_edge(idx);
        }
        if let Some(idx) = terminal_decl_idx {
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
            .map(|(_, &idx)| idx)
            .collect();
        for idx in targets {
            self.emit_edge(idx);
        }
    }
}

/// Peel an attribute chain back to its `Name` root.
///
/// Returns `Some((root, segments))` when `expr` is `Name`, `Name.s1`,
/// `Name.s1.s2`, ... — i.e. one or more attribute accesses on a bare
/// name. Returns `None` when the chain bottoms out at anything else
/// (a call result, subscript, attribute of attribute of a non-Name, …).
fn collapse_attribute_chain(expr: &Expr) -> Option<(&ExprName, Vec<&str>)> {
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
fn module_name_resolves(dotted: &str, anchor: File, db: &dyn ty_python_semantic::Db) -> bool {
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
        if matches!(expr, Expr::Attribute(_)) {
            if let Some((root, segments)) = collapse_attribute_chain(expr) {
                self.emit_name_use(root, &segments);
                // Don't recurse into the chain — every Name in it has
                // been handled and an attribute access has no other
                // walk-worthy children.
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
        }
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
