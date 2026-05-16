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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::{File, FilePath};
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_db::source::{line_index, source_text};
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Expr, ExprName, Stmt, StmtClassDef};
use ruff_source_file::LineIndex;
use ruff_text_size::{Ranged, TextRange};
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
    definitions_for_imported_symbol, type_hierarchy_subtypes, HasType, ImportAliasResolution,
    ResolvedDefinition, SemanticModel, TypeHierarchyClass,
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

/// Return type of `ProjectContext.find_main_blocks`: one entry per
/// file with a top-level ``if __name__ == "__main__":`` block, paired
/// with the decls that fall inside it.
type MainBlock = (Py<NativeNode>, Vec<Py<NativeNode>>);

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
        let key = node_key_of(&node);
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
        let db = make_db(
            root,
            src_roots,
            extra_paths,
            python_env,
            python_version,
            typeshed,
        )?;
        Ok(Self { db })
    }

    /// Build the project-wide symbol graph.
    fn build(&self, py: Python<'_>) -> PyResult<NativeGraph> {
        let outputs = build_project_graph(py, &self.db)?;
        Ok(NativeGraph {
            nodes: outputs.builder.nodes,
            edges: outputs.builder.edges,
        })
    }
}

/// All state produced by one project-wide build pass.
///
/// Owned by [`ProjectContext`] across the materialize call so plugin
/// queries can re-read indices (e.g. live_decls, global_index) without
/// having to re-derive them from the parsed modules.
struct BuildOutputs {
    builder: GraphBuilder,
    project_files: Vec<File>,
    global_index: DeclIndex,
    /// `file_path_string(file) -> File` so seed lookups don't have to
    /// linear-scan `project_files`. Populated alongside ingest.
    path_to_file: HashMap<String, File>,
    /// `(file, class_target_range_key) -> class node idx`. Lets
    /// `find_subclasses_of` map ty's `TypeHierarchyClass.selection_range`
    /// back to a graph node in O(1).
    class_by_selection: HashMap<(File, (u32, u32)), usize>,
    /// `file -> module node idx`. Lets `find_main_blocks` reach the
    /// file's module node without a linear scan over `builder.nodes`.
    module_nodes_by_file: HashMap<File, usize>,
    /// `(file, target_range_key) -> node idx`. Sister of
    /// ``class_by_selection`` but for every top-level decl ingest minted
    /// (function / class / variable / import). Lets ``find_decorated_decls``
    /// and the dispatch-app queries map an AST node's target range to a
    /// graph node in O(1) instead of scanning the full ``global_index``.
    decl_by_name_range: HashMap<(File, (u32, u32)), usize>,
}

/// Run the three build phases (ingest → hierarchy+imports → references)
/// and return every index the plugin queries need.
fn build_project_graph(py: Python<'_>, db: &ProjectDatabase) -> PyResult<BuildOutputs> {
    let mut builder = GraphBuilder::new();
    let mut global_index: DeclIndex = HashMap::new();
    let mut module_nodes: HashMap<File, usize> = HashMap::new();
    let mut alias_imports: HashMap<usize, ImportSpec> = HashMap::new();
    let mut live_decls: LiveDeclIndex = HashMap::new();
    let mut class_by_selection: HashMap<(File, (u32, u32)), usize> = HashMap::new();
    let mut decl_by_name_range: HashMap<(File, (u32, u32)), usize> = HashMap::new();

    let project_files: Vec<File> = (&db.project().files(db)).into_iter().collect();
    let mut path_to_file: HashMap<String, File> = HashMap::with_capacity(project_files.len());
    for &file in &project_files {
        path_to_file.insert(file_path_string(db, file), file);
    }
    for file in &project_files {
        ingest_decls(
            py,
            db,
            *file,
            &mut builder,
            &mut global_index,
            &mut module_nodes,
            &mut alias_imports,
            &mut live_decls,
            &mut class_by_selection,
            &mut decl_by_name_range,
        )?;
    }
    for file in &project_files {
        emit_module_hierarchy(db, *file, &module_nodes, &mut builder);
        emit_import_edges(
            py,
            db,
            *file,
            &mut builder,
            &mut global_index,
            &mut module_nodes,
        )?;
    }
    for file in &project_files {
        emit_reference_edges(
            db,
            *file,
            &global_index,
            &module_nodes,
            &alias_imports,
            &live_decls,
            &mut builder,
        );
    }
    Ok(BuildOutputs {
        builder,
        project_files,
        global_index,
        path_to_file,
        class_by_selection,
        module_nodes_by_file: module_nodes,
        decl_by_name_range,
    })
}

// ---------------------------------------------------------------------------
// ProjectContext — plugin protocol entry point
// ---------------------------------------------------------------------------

/// Plugin-aware project graph builder.
///
/// Python instantiates a `ProjectContext`, registers Python plugins via
/// `add_plugin`, then calls `materialize()`. `materialize` runs the
/// project-wide build in rust, then for each registered plugin calls
/// `plugin.run(ctx)` back into Python with `ctx` set to the same
/// `ProjectContext` instance — re-entrant calls invoke the rust
/// `add_node` / `add_edge` / `find_*` methods listed below.
///
/// Queries are answered from ty's semantic index: subclass closure goes
/// through `type_hierarchy_subtypes`, method-defines walks each class's
/// `DefinitionKind::Class`, module dunders scan global-scope variable
/// nodes, and comment patterns walk the parser's `Tokens` stream.
#[pyclass(unsendable)]
struct ProjectContext {
    db: ProjectDatabase,
    /// Absolute path of the project root, echoed back to Python via the
    /// :attr:`project_root` getter. Plugins use it to compute paths
    /// relative to the project (e.g. ``ExplicitEntrypointPlugin`` matching
    /// path specs).
    root: String,
    plugins: Vec<PyObject>,
    /// Populated by `materialize` before plugins run. `None` outside a
    /// materialize call — `add_node` / `add_edge` / queries assume it's
    /// `Some` and error if a plugin (incorrectly) caches the ctx and
    /// uses it after materialize returns.
    outputs: RefCell<Option<BuildOutputs>>,
}

#[pymethods]
impl ProjectContext {
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
        let db = make_db(
            root,
            src_roots,
            extra_paths,
            python_env,
            python_version,
            typeshed,
        )?;
        Ok(Self {
            db,
            root: root.to_string(),
            plugins: Vec::new(),
            outputs: RefCell::new(None),
        })
    }

    /// Absolute project root passed at construction.
    #[getter]
    fn project_root(&self) -> &str {
        &self.root
    }

    /// Register a Python plugin. Order of registration is order of
    /// invocation during `materialize`.
    fn add_plugin(&mut self, plugin: PyObject) {
        self.plugins.push(plugin);
    }

    /// Build the project-wide graph, run each plugin's `run(ctx)`,
    /// then snapshot the final state.
    ///
    /// Borrows are released between phases so plugin `run` methods can
    /// re-enter `add_node` / `add_edge` / queries through the same ctx
    /// without aliasing violations.
    fn materialize(slf: Py<Self>, py: Python<'_>) -> PyResult<NativeGraph> {
        {
            let this = slf.borrow(py);
            let outputs = build_project_graph(py, &this.db)?;
            *this.outputs.borrow_mut() = Some(outputs);
        }

        let plugins: Vec<PyObject> = slf
            .borrow(py)
            .plugins
            .iter()
            .map(|p| p.clone_ref(py))
            .collect();
        for plugin in &plugins {
            plugin.bind(py).call_method1("run", (slf.clone_ref(py),))?;
        }

        let outputs =
            slf.borrow(py).outputs.borrow_mut().take().ok_or_else(|| {
                PyRuntimeError::new_err("ProjectContext was already materialized")
            })?;
        Ok(NativeGraph {
            nodes: outputs.builder.nodes,
            edges: outputs.builder.edges,
        })
    }

    // ----- Mutation -------------------------------------------------------

    /// Intern a synthetic node into the graph.
    ///
    /// `path` should be a real source path when the node stands in for a
    /// specific location (so codemod / why-alive output can reach the
    /// file); pass the project root for file-agnostic markers.
    #[pyo3(signature = (
        fqname,
        path,
        *,
        kind = "synthetic",
        start_line = 0,
        start_column = 0,
        end_line = 0,
        end_column = 0,
        flags = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_node(
        &self,
        py: Python<'_>,
        fqname: String,
        path: String,
        kind: &str,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
        flags: u32,
    ) -> PyResult<Py<NativeNode>> {
        let kind = static_kind_str(kind)?;
        let mut outputs = self.outputs.borrow_mut();
        let outputs = outputs
            .as_mut()
            .ok_or_else(|| not_materialized("add_node"))?;
        let idx = outputs.builder.intern_node(
            py,
            NativeNode {
                fqname,
                kind,
                path,
                start_line,
                start_column,
                end_line,
                end_column,
                flags,
                imports: None,
            },
        )?;
        Ok(outputs.builder.nodes[idx].clone_ref(py))
    }

    /// Add an edge between two nodes returned by `add_node` or a query.
    ///
    /// Identity is content-based: each side is looked up in the
    /// builder's intern table via its `(fqname, kind, path, position)`
    /// key, so two `NativeNode` Python references that wrap the same
    /// logical node resolve to the same edge endpoint. Passing a node
    /// that was never interned raises `ValueError`.
    fn add_edge(&self, src: &NativeNode, dst: &NativeNode) -> PyResult<()> {
        let mut outputs = self.outputs.borrow_mut();
        let outputs = outputs
            .as_mut()
            .ok_or_else(|| not_materialized("add_edge"))?;
        let src_idx = lookup_idx(&outputs.builder, src, "src")?;
        let dst_idx = lookup_idx(&outputs.builder, dst, "dst")?;
        outputs.builder.add_edge(src_idx, dst_idx, 0);
        Ok(())
    }

    // ----- Queries (rust-resident, results pass back to Python) -----------

    /// Return every top-level variable node whose name matches `__xxx__`.
    ///
    /// Pure scan over already-interned nodes — no ty re-query needed —
    /// because the visitor's decl pass already minted one node per
    /// global-scope variable binding.
    fn find_module_dunders(&self, py: Python<'_>) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module_dunders"))?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if node.kind != "variable" {
                continue;
            }
            if is_dunder_name(&node.fqname) {
                out.push(node_py.clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Return every import-kind node whose upstream `module` matches.
    ///
    /// Covers both `import <module_name>` and
    /// `from <module_name> import ...` styles — both bind import-kind
    /// nodes whose `Import.module` is the absolute dotted name. Star
    /// reexports synthesized from `from <module_name> import *` are
    /// also included.
    fn find_imports_of(&self, py: Python<'_>, module_name: &str) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_imports_of"))?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if node.kind != "import" {
                continue;
            }
            let Some(import_py) = node.imports.as_ref() else {
                continue;
            };
            if import_py.borrow(py).module == module_name {
                out.push(node_py.clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Return every declaration (function / class / variable / import)
    /// whose fully qualified name matches ``fqname``, walking back
    /// through dotted segments to find the enclosing top-level decl
    /// when the exact name doesn't match.
    ///
    /// ``pkg.lib.Cls.method`` returns ``pkg.lib.Cls`` because methods
    /// aren't represented as their own graph nodes — same rule the
    /// libcst :func:`find_declarations` follows. Modules are never
    /// returned; use :meth:`find_module` for that.
    fn find_declarations(&self, py: Python<'_>, fqname: &str) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_declarations"))?;
        let parts: Vec<&str> = fqname.split('.').collect();
        for split in (1..=parts.len()).rev() {
            let prefix = parts[..split].join(".");
            let mut hits = Vec::new();
            for node_py in &outputs.builder.nodes {
                let node = node_py.borrow(py);
                if !matches!(node.kind, "function" | "class" | "variable" | "import") {
                    continue;
                }
                if node.fqname == prefix {
                    hits.push(node_py.clone_ref(py));
                }
            }
            if !hits.is_empty() {
                return Ok(hits);
            }
        }
        Ok(Vec::new())
    }

    /// Return the module node for the given dotted fqname, if it
    /// exists in the project graph.
    fn find_module(&self, py: Python<'_>, fqname: &str) -> PyResult<Option<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module"))?;
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if node.kind == "module" && node.fqname == fqname {
                return Ok(Some(node_py.clone_ref(py)));
            }
        }
        Ok(None)
    }

    /// Return the module node owning ``path``, if any. O(1) — backed
    /// by the same ``module_nodes_by_file`` index `find_main_blocks`
    /// uses, so plugins don't have to scan ``nodes()`` per call.
    fn module_for(&self, py: Python<'_>, path: &str) -> PyResult<Option<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("module_for"))?;
        let Some(&file) = outputs.path_to_file.get(path) else {
            return Ok(None);
        };
        let Some(&idx) = outputs.module_nodes_by_file.get(&file) else {
            return Ok(None);
        };
        Ok(Some(outputs.builder.nodes[idx].clone_ref(py)))
    }

    /// Return ``(module_node, [decls inside the block])`` for every
    /// file with a top-level ``if __name__ == "__main__":`` block.
    ///
    /// The decls list contains the file's class / function / variable
    /// / import nodes whose source position falls inside the block's
    /// range — same shape ``MainBlockPlugin``'s libcst path computes
    /// from the visitor's payload.
    fn find_main_blocks(&self, py: Python<'_>) -> PyResult<Vec<MainBlock>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_main_blocks"))?;
        let mut out: Vec<MainBlock> = Vec::new();
        for (&file, &module_idx) in &outputs.module_nodes_by_file {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let Some(block_range) = find_main_block_range(&parsed) else {
                continue;
            };
            // Collect decls whose target_range falls within block_range.
            let mut decls: Vec<Py<NativeNode>> = Vec::new();
            for ((entry_file, _place_id, (start, end)), idx) in &outputs.global_index {
                if *entry_file != file {
                    continue;
                }
                let block_start = block_range.start().to_u32();
                let block_end = block_range.end().to_u32();
                if *start >= block_start && *end <= block_end {
                    decls.push(outputs.builder.nodes[*idx].clone_ref(py));
                }
            }
            out.push((outputs.builder.nodes[module_idx].clone_ref(py), decls));
        }
        Ok(out)
    }

    /// Return every class that defines a method with the given name.
    ///
    /// Return every top-level function decorated with ``@<decorator_module>.<name>``
    /// or ``@<name>`` for any ``name`` in ``decorator_names``.
    ///
    /// Both ``@<name>`` (bare) and ``@<name>(...)`` (called) forms
    /// match — the function call is unwrapped before the pattern is
    /// checked. Identity for the attribute prefix is literal
    /// (``@pytest.fixture`` matches; ``@p.fixture`` with
    /// ``import pytest as p`` does not). Bare-name decorators
    /// (``@fixture``) match purely by attribute name regardless of
    /// what the local ``fixture`` refers to — this mirrors the libcst
    /// plugin helpers used by ``PytestPlugin`` etc., which intentionally
    /// keep a loose pattern match rather than trying to chase decorator
    /// imports through ty's resolver.
    fn find_decorated_decls(
        &self,
        py: Python<'_>,
        decorator_module: &str,
        decorator_names: Vec<String>,
    ) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_decorated_decls"))?;
        let names: HashSet<&str> = decorator_names.iter().map(String::as_str).collect();
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            // The file's import map for ``decorator_module`` — if it
            // doesn't import anything from there, no decorator can
            // match. Skips the body walk for the common case.
            let imports = collect_module_imports_local(&parsed, decorator_module, &names);
            if imports.is_empty() {
                continue;
            }
            for stmt in &parsed.syntax().body {
                let Stmt::FunctionDef(func) = stmt else {
                    continue;
                };
                if !decorators_match_imports(&func.decorator_list, &imports, &names) {
                    continue;
                }
                let key = (file, range_key(func.name.range()));
                if let Some(&idx) = outputs.decl_by_name_range.get(&key) {
                    out.push(outputs.builder.nodes[idx].clone_ref(py));
                }
            }
        }
        Ok(out)
    }

    /// Find top-level ``<var> = <Ctor>(...)`` constructions where
    /// ``Ctor`` is imported from ``module`` and is one of ``ctor_names``.
    ///
    /// Recognized shapes (mirroring the libcst plugin helpers):
    /// * ``from <module> import <Ctor>; X = Ctor(...)``
    /// * ``from <module> import <Ctor> as A; X = A(...)``
    /// * ``import <module>; X = <module>.Ctor(...)``
    /// * ``import <module> as m; X = m.Ctor(...)``
    /// * ``X: T = Ctor(...)`` annotated form
    ///
    /// Returns ``[(var_node, ctor_name)]``; ``ctor_name`` is the
    /// upstream constructor's bare name (``"Flask"`` even when imported
    /// as ``F``).
    #[allow(clippy::type_complexity)]
    fn find_instance_constructions(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_instance_constructions"))?;
        let allowed: HashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let imports = collect_module_imports_local(&parsed, module, &allowed);
            if imports.is_empty() {
                continue;
            }
            for stmt in &parsed.syntax().body {
                let (target_range, value) = match top_level_assign_to_name(stmt) {
                    Some(pair) => pair,
                    None => continue,
                };
                if let Some(matched) = matched_call_target(value, &imports, &allowed) {
                    let key = (file, range_key(target_range));
                    if let Some(&idx) = outputs.decl_by_name_range.get(&key) {
                        out.push((outputs.builder.nodes[idx].clone_ref(py), matched));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Find top-level functions decorated with ``@<owner>.<attr>(...)``
    /// where ``attr`` is in ``decorator_attrs``.
    ///
    /// Returns ``[(owner_name, function_node)]``. ``owner_name`` is the
    /// raw textual prefix of the decorator (``"app"`` for ``@app.route``),
    /// not resolved to a graph node — the caller decides which owners
    /// correspond to real framework instances. Multiple decorators on
    /// the same function emit multiple entries.
    #[allow(clippy::type_complexity)]
    fn find_handler_decorators(
        &self,
        py: Python<'_>,
        decorator_attrs: Vec<String>,
    ) -> PyResult<Vec<(String, Py<NativeNode>)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_handler_decorators"))?;
        let attrs: HashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            for stmt in &parsed.syntax().body {
                let Stmt::FunctionDef(func) = stmt else {
                    continue;
                };
                let mut seen_owners: HashSet<String> = HashSet::new();
                for dec in &func.decorator_list {
                    let mut expr = &dec.expression;
                    if let Expr::Call(call) = expr {
                        expr = &call.func;
                    }
                    let Expr::Attribute(attr) = expr else {
                        continue;
                    };
                    if !attrs.contains(attr.attr.as_str()) {
                        continue;
                    }
                    let Expr::Name(owner) = attr.value.as_ref() else {
                        continue;
                    };
                    let owner_name = owner.id.as_str().to_string();
                    if !seen_owners.insert(owner_name.clone()) {
                        continue;
                    }
                    let key = (file, range_key(func.name.range()));
                    if let Some(&idx) = outputs.decl_by_name_range.get(&key) {
                        out.push((owner_name, outputs.builder.nodes[idx].clone_ref(py)));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Find top-level def/class statements whose body constructs one
    /// of ``ctor_names`` imported from ``module``.
    ///
    /// Recursively walks each candidate's body looking for ``<Ctor>(...)``
    /// or ``<module>.<Ctor>(...)`` call expressions. Returns
    /// ``[(decl_node, [kind, ...])]`` where ``kind`` is the matched
    /// constructor's bare name; multiple kinds appear when a single
    /// factory constructs more than one (e.g. a function that returns a
    /// ``Flask`` after mounting several ``Blueprint``s).
    #[allow(clippy::type_complexity)]
    fn find_factory_decls(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
    ) -> PyResult<Vec<(Py<NativeNode>, Vec<String>)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_factory_decls"))?;
        let allowed: HashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let imports = collect_module_imports_local(&parsed, module, &allowed);
            if imports.is_empty() {
                continue;
            }
            for stmt in &parsed.syntax().body {
                let (name_range, body): (TextRange, &[Stmt]) = match stmt {
                    Stmt::FunctionDef(f) => (f.name.range(), &f.body),
                    Stmt::ClassDef(c) => (c.name.range(), &c.body),
                    _ => continue,
                };
                let mut finder = FactoryCallFinder {
                    imports: &imports,
                    allowed: &allowed,
                    kinds: HashSet::new(),
                };
                for inner in body {
                    finder.visit_stmt(inner);
                }
                if finder.kinds.is_empty() {
                    continue;
                }
                let key = (file, range_key(name_range));
                if let Some(&idx) = outputs.decl_by_name_range.get(&key) {
                    let mut kinds_vec: Vec<String> = finder.kinds.into_iter().collect();
                    kinds_vec.sort();
                    out.push((outputs.builder.nodes[idx].clone_ref(py), kinds_vec));
                }
            }
        }
        Ok(out)
    }

    /// Find calls to a callable imported from ``module`` with the name
    /// ``name``. Returns ``(owning_decl, string_literal_arg)`` pairs
    /// where the call resolves through the file's local imports.
    ///
    /// The owning decl is the top-level ``FunctionDef`` / ``ClassDef``
    /// the call lives under (including its decorator subtree) or the
    /// module node for calls at module scope. Mirrors the libcst
    /// ``MockPatchPlugin`` attribution rule.
    #[allow(clippy::type_complexity)]
    fn find_calls_to_imported(
        &self,
        py: Python<'_>,
        module: &str,
        name: &str,
        arg_index: usize,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_to_imported"))?;
        let allowed: HashSet<&str> = [name].into_iter().collect();
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let imports = collect_module_imports_local(&parsed, module, &allowed);
            // No short-circuit on empty imports: the matcher's
            // dotted-attribute branch (``import unittest.mock;
            // unittest.mock.patch(...)``) matches without populating the
            // imports map.
            for stmt in &parsed.syntax().body {
                let Some(owner_idx) = owner_idx_for_stmt(outputs, file, stmt) else {
                    continue;
                };
                let mut finder = ImportedCallFinder {
                    imports: &imports,
                    module,
                    name,
                    arg_index,
                    results: Vec::new(),
                };
                finder.visit_stmt(stmt);
                for arg in finder.results {
                    out.push((outputs.builder.nodes[owner_idx].clone_ref(py), arg));
                }
            }
        }
        Ok(out)
    }

    /// Find ``<owner>.<attr>(...)`` calls where ``owner`` is the textual
    /// prefix (no import resolution — covers pytest fixture conventions
    /// like ``mocker.patch`` / ``monkeypatch.setattr``).
    ///
    /// ``required_positional`` disambiguates fqname-form calls from
    /// object-form calls when the same method name is overloaded:
    /// ``monkeypatch.setattr("X.Y", v)`` has 2 positional args
    /// (fqname + value) while ``monkeypatch.setattr(obj, "name", v)``
    /// has 3. Pass ``None`` to accept any positional-arg count.
    ///
    /// Returns ``(owning_decl, string_literal_arg)`` pairs.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (owner, attr, arg_index, *, required_positional = None))]
    fn find_calls_on_var(
        &self,
        py: Python<'_>,
        owner: &str,
        attr: &str,
        arg_index: usize,
        required_positional: Option<usize>,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_on_var"))?;
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            for stmt in &parsed.syntax().body {
                let Some(owner_idx) = owner_idx_for_stmt(outputs, file, stmt) else {
                    continue;
                };
                let mut finder = VarCallFinder {
                    owner,
                    attr,
                    arg_index,
                    required_positional,
                    results: Vec::new(),
                };
                finder.visit_stmt(stmt);
                for arg in finder.results {
                    out.push((outputs.builder.nodes[owner_idx].clone_ref(py), arg));
                }
            }
        }
        Ok(out)
    }

    /// Walks each class's `DefinitionKind::Class` body for an
    /// `Stmt::FunctionDef` whose name matches. ty's `parsed_module` is
    /// Salsa-cached, so this is just a body scan per class.
    fn find_classes_defining_method(
        &self,
        py: Python<'_>,
        method_name: &str,
    ) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_classes_defining_method"))?;
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let index = semantic_index(&self.db, file);
            let global = FileScopeId::global();
            let use_def_map = index.use_def_map(global);
            for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
                let DefinitionState::Defined(def) = state else {
                    continue;
                };
                if def.file(&self.db) != file || def.file_scope(&self.db) != global {
                    continue;
                }
                let kind = def.kind(&self.db);
                let Some(class_ref) = kind.as_class() else {
                    continue;
                };
                let class_def = class_ref.node(&parsed);
                if !class_body_defines_method(class_def, method_name) {
                    continue;
                }
                let key = (
                    file,
                    def.place(&self.db),
                    range_key(kind.target_range(&parsed)),
                );
                if let Some(&idx) = outputs.global_index.get(&key) {
                    out.push(outputs.builder.nodes[idx].clone_ref(py));
                }
            }
        }
        Ok(out)
    }

    /// Return every transitive subclass of the given class node.
    ///
    /// Direct subtypes come from ty's `type_hierarchy_subtypes`; we BFS
    /// to collect the transitive closure. Results that don't land in
    /// the project (stdlib / external classes) are dropped.
    fn find_subclasses_of(
        &self,
        py: Python<'_>,
        class_node: &NativeNode,
    ) -> PyResult<Vec<Py<NativeNode>>> {
        if class_node.kind != "class" {
            return Ok(Vec::new());
        }
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_subclasses_of"))?;

        // Locate the seed class's File + name range, then ask ty for
        // its inferred Type via the StmtClassDef at that range.
        let Some((seed_file, seed_range)) = locate_class_def(
            &self.db,
            &outputs.path_to_file,
            &class_node.path,
            class_node,
        ) else {
            return Ok(Vec::new());
        };
        let parsed_seed = parsed_module(&self.db, seed_file).load(&self.db);
        let model_seed = SemanticModel::new(&self.db, seed_file);
        let Some(seed_class) = class_def_at(&parsed_seed, seed_range) else {
            return Ok(Vec::new());
        };
        let Some(seed_ty) = seed_class.inferred_type(&model_seed) else {
            return Ok(Vec::new());
        };

        // BFS over Types. Each visited entry is a class identified by
        // `(file, selection_range)` from `TypeHierarchyClass`; we
        // re-derive the next layer's Type by parsing that file and
        // asking the StmtClassDef at the matching range for its type.
        let mut seen: HashSet<(File, (u32, u32))> = HashSet::new();
        let mut frontier: Vec<TypeHierarchyClass> = type_hierarchy_subtypes(&self.db, seed_ty);
        let mut out_idx: Vec<usize> = Vec::new();
        while let Some(thc) = frontier.pop() {
            let file_key = (thc.file, range_key(thc.selection_range));
            if !seen.insert(file_key) {
                continue;
            }
            // `selection_range` is the class name range — same key the
            // class-node was interned under.
            if let Some(&idx) = outputs.class_by_selection.get(&file_key) {
                out_idx.push(idx);
            }
            // Recurse: ask ty for the next layer.
            let parsed = parsed_module(&self.db, thc.file).load(&self.db);
            let Some(class_def) = class_def_at(&parsed, thc.selection_range) else {
                continue;
            };
            let model = SemanticModel::new(&self.db, thc.file);
            let Some(ty) = class_def.inferred_type(&model) else {
                continue;
            };
            frontier.extend(type_hierarchy_subtypes(&self.db, ty));
        }
        let mut out = Vec::with_capacity(out_idx.len());
        for idx in out_idx {
            out.push(outputs.builder.nodes[idx].clone_ref(py));
        }
        Ok(out)
    }

    /// Return `(decl_node, comment_text)` for every comment in the
    /// project that matches `pattern` (a regex), paired with the next
    /// declaration that follows it in the same file.
    ///
    /// Comments are scanned from the parser's `Tokens` stream (no
    /// re-lexing); regex matching is full-text against the comment
    /// content (leading `#` included).
    fn find_comment_patterns(
        &self,
        py: Python<'_>,
        pattern: &str,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let regex = regex::Regex::new(pattern)
            .map_err(|e| PyValueError::new_err(format!("invalid regex {pattern:?}: {e}")))?;
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_comment_patterns"))?;
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let source = source_text(&self.db, file);
            // Lazy — files with no matching comments skip the decl scan.
            let mut file_decls: Option<Vec<(u32, usize)>> = None;
            for token in parsed.tokens() {
                if token.kind() != TokenKind::Comment {
                    continue;
                }
                let range = token.range();
                let text = &source[range];
                if !regex.is_match(text) {
                    continue;
                }
                let decls =
                    file_decls.get_or_insert_with(|| file_decl_sites(file, &outputs.global_index));
                let comment_end = range.end().to_u32();
                let i = decls.partition_point(|(start, _)| *start < comment_end);
                let Some(&(_, decl_idx)) = decls.get(i) else {
                    continue;
                };
                out.push((
                    outputs.builder.nodes[decl_idx].clone_ref(py),
                    text.to_string(),
                ));
            }
        }
        Ok(out)
    }

    // ----- Read-only accessors -------------------------------------------

    /// Live nodes in the in-progress graph. Cheap, no copy.
    fn nodes(&self, py: Python<'_>) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs.as_ref().ok_or_else(|| not_materialized("nodes"))?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.clone_ref(py))
            .collect())
    }

    /// Live edges as `(src_idx, dst_idx, flags)` triples.
    fn edges(&self) -> PyResult<Vec<(usize, usize, u32)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs.as_ref().ok_or_else(|| not_materialized("edges"))?;
        Ok(outputs.builder.edges.clone())
    }
}

// ---------------------------------------------------------------------------
// ProjectContext support helpers
// ---------------------------------------------------------------------------

fn make_db(
    root: &str,
    src_roots: Option<Vec<String>>,
    extra_paths: Option<Vec<String>>,
    python_env: Option<&str>,
    python_version: Option<&str>,
    typeshed: Option<&str>,
) -> PyResult<ProjectDatabase> {
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
    let metadata = ProjectMetadata::from_options(options, root.clone(), None, &UseDefaultStrategy)
        .map_err(|e| PyValueError::new_err(format!("invalid configuration: {e:?}")))?;
    let cwd =
        std::env::current_dir().map_err(|e| PyOSError::new_err(format!("cwd unavailable: {e}")))?;
    let cwd = SystemPathBuf::from_path_buf(cwd).map_err(|_| {
        PyValueError::new_err("current working directory is not a valid absolute UTF-8 path")
    })?;
    let system = OsSystem::new(cwd);
    Ok(ProjectDatabase::use_defaults(metadata, system))
}

fn not_materialized(op: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ProjectContext.{op}() called outside an active materialize() — \
         did you call it from a plugin's run() method?"
    ))
}

/// Project a `NativeNode` onto the `NodeKey` used for intern-table
/// identity. Clones the two `String` fields (`fqname`, `path`); the
/// rest are `Copy`.
fn node_key_of(node: &NativeNode) -> NodeKey {
    NodeKey {
        fqname: node.fqname.clone(),
        kind: node.kind,
        path: node.path.clone(),
        start_line: node.start_line,
        start_column: node.start_column,
        end_line: node.end_line,
        end_column: node.end_column,
    }
}

/// Resolve a `NativeNode` reference to its builder-side index for an
/// edge endpoint. Surfaces a precise `ValueError` (with `side`) when
/// the node was never interned in this context.
fn lookup_idx(builder: &GraphBuilder, node: &NativeNode, side: &str) -> PyResult<usize> {
    builder
        .node_index
        .get(&node_key_of(node))
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "add_edge: {side} node {:?} is not interned in this ProjectContext",
                node.fqname
            ))
        })
}

/// Map a plugin-supplied `kind` string to one of the stable `&'static
/// str` kinds NativeNode carries.
fn static_kind_str(kind: &str) -> PyResult<&'static str> {
    Ok(match kind {
        "synthetic" => "synthetic",
        "module" => "module",
        "import" => "import",
        "function" => "function",
        "class" => "class",
        "variable" => "variable",
        "type_alias" => "type_alias",
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown node kind {other:?} — expected one of synthetic, module, import, \
                 function, class, variable, type_alias"
            )))
        }
    })
}

fn is_dunder_name(fqname: &str) -> bool {
    let name = fqname.rsplit('.').next().unwrap_or("");
    name.len() > 4 && name.starts_with("__") && name.ends_with("__")
}

fn class_body_defines_method(class_def: &StmtClassDef, method_name: &str) -> bool {
    class_def.body.iter().any(|stmt| match stmt {
        Stmt::FunctionDef(f) => f.name.as_str() == method_name,
        _ => false,
    })
}

/// Return ``true`` if any decorator in ``decorators`` matches
/// ``@<module>.<name>`` (literal module name) or ``@<name>`` for
/// any ``name`` in ``names``. Trailing ``(...)`` is unwrapped before
/// the pattern check.
/// Resolves decorator references through the file's local imports map
/// (built by :func:`collect_module_imports_local`) — mirrors the libcst
/// helpers' ``matched_attr_call`` shape. Recognized forms:
///
/// * ``@<name>(...)`` / ``@<name>`` where ``imports[name]`` is in ``names``
///   (covers ``from module import name`` and aliased variants).
/// * ``@<alias>.<attr>(...)`` / ``@<alias>.<attr>`` where
///   ``imports[alias] == "<module>"`` and ``attr`` is in ``names``
///   (covers ``import module`` and ``import module as alias``).
fn decorators_match_imports(
    decorators: &[ruff_python_ast::Decorator],
    imports: &HashMap<String, String>,
    names: &HashSet<&str>,
) -> bool {
    for dec in decorators {
        let mut expr = &dec.expression;
        if let Expr::Call(call) = expr {
            expr = &call.func;
        }
        match expr {
            Expr::Name(n) => {
                if let Some(target) = imports.get(n.id.as_str()) {
                    if names.contains(target.as_str()) {
                        return true;
                    }
                }
            }
            Expr::Attribute(attr) => {
                if let Expr::Name(prefix) = attr.value.as_ref() {
                    if imports.get(prefix.id.as_str()).map(String::as_str) == Some("<module>")
                        && names.contains(attr.attr.as_str())
                    {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn iter_top_level_classes(parsed: &ParsedModuleRef) -> impl Iterator<Item = &StmtClassDef> {
    parsed.syntax().body.iter().filter_map(|stmt| match stmt {
        Stmt::ClassDef(cls) => Some(cls),
        _ => None,
    })
}

/// Locate a class's File + name TextRange from its NativeNode positions.
///
/// We don't store ty `Definition<'db>` references across plugin calls
/// (the `'db` lifetime is tied to the active borrow), so this re-walks
/// Locate a class's File + name TextRange from its NativeNode positions.
///
/// We don't store ty `Definition<'db>` references across plugin calls
/// (the `'db` lifetime is tied to the active borrow), so this re-walks
/// the matching file's top-level classes for one whose name lands on
/// the node's start line. Match-by-line (not line+column) because
/// Function / Class / TypeAlias node columns are snapped to the line's
/// indent — not the bound name's column — to align with libcst, and
/// two top-level classes can't share a source line.
fn locate_class_def(
    db: &ProjectDatabase,
    path_to_file: &HashMap<String, File>,
    path: &str,
    class_node: &NativeNode,
) -> Option<(File, TextRange)> {
    let &file = path_to_file.get(path)?;
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    for cls in iter_top_level_classes(&parsed) {
        let name_range = cls.name.range();
        let (sl, _, _, _) = position(&line_index, &source, name_range);
        if sl == class_node.start_line {
            return Some((file, name_range));
        }
    }
    None
}

/// Find a top-level `StmtClassDef` whose name range equals `selection_range`.
fn class_def_at(parsed: &ParsedModuleRef, selection_range: TextRange) -> Option<&StmtClassDef> {
    iter_top_level_classes(parsed).find(|cls| cls.name.range() == selection_range)
}

/// Locate the top-level ``if __name__ == "__main__":`` block in a
/// parsed module and return its source range. Both orderings
/// (``__name__ == "__main__"`` and ``"__main__" == __name__``) are
/// recognized; the ``elif`` / ``else`` branches are not matched.
fn find_main_block_range(parsed: &ParsedModuleRef) -> Option<TextRange> {
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
fn is_name_eq_main(expr: &Expr) -> bool {
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

fn is_name(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Name(n) if n.id.as_str() == value)
}

fn is_string_literal(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::StringLiteral(s) if s.value.to_str() == value)
}

/// Build the file-local imports map ``{local_name: target}`` for
/// names imported from ``module``. ``target`` is the upstream
/// constructor / decl name (e.g. ``"Flask"`` when bound via
/// ``from flask import Flask``) or the sentinel ``"<module>"``
/// when bound via ``import flask`` / ``import flask as f``.
///
/// Only entries whose target is in ``allowed`` survive — keeps the
/// map small and lets call-site matchers do a cheap second check.
fn collect_module_imports_local(
    parsed: &ParsedModuleRef,
    module: &str,
    allowed: &HashSet<&str>,
) -> HashMap<String, String> {
    // Submodule binding: ``from <parent> import <last_seg>`` makes
    // ``last_seg`` a local alias for the queried module (e.g.
    // ``from unittest import mock`` for module ``unittest.mock``).
    let parent_last = module.rsplit_once('.');
    let mut out = HashMap::new();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                let Some(mod_name) = &im.module else { continue };
                if mod_name.as_str() == module {
                    for alias in &im.names {
                        let target = alias.name.as_str();
                        if !allowed.contains(target) {
                            continue;
                        }
                        let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                        out.insert(local.to_string(), target.to_string());
                    }
                } else if let Some((parent, last)) = parent_last {
                    if mod_name.as_str() == parent {
                        for alias in &im.names {
                            if alias.name.as_str() != last {
                                continue;
                            }
                            let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(last);
                            out.insert(local.to_string(), "<module>".to_string());
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
                    out.insert(local.to_string(), "<module>".to_string());
                }
            }
            _ => {}
        }
    }
    out
}

/// Match an expression against a constructor pattern:
/// * ``<imported_local>(...)`` where the local binds an allowed name; or
/// * ``<module_alias>.<allowed_name>(...)`` where the alias binds the module.
///
/// Returns the matched upstream name (``"Flask"``) on hit, else ``None``.
fn matched_call_target(
    expr: &Expr,
    imports: &HashMap<String, String>,
    allowed: &HashSet<&str>,
) -> Option<String> {
    let Expr::Call(call) = expr else {
        return None;
    };
    match call.func.as_ref() {
        Expr::Name(name) => {
            let target = imports.get(name.id.as_str())?;
            allowed.contains(target.as_str()).then(|| target.clone())
        }
        Expr::Attribute(attr) => {
            let Expr::Name(prefix) = attr.value.as_ref() else {
                return None;
            };
            if imports.get(prefix.id.as_str()).map(String::as_str) != Some("<module>") {
                return None;
            }
            let attr_name = attr.attr.as_str();
            allowed.contains(attr_name).then(|| attr_name.to_string())
        }
        _ => None,
    }
}

/// Top-level ``X = <expr>`` / ``X: T = <expr>`` where the target is a
/// bare ``Name``. Returns ``(name_range, &expr)`` so the caller can
/// look up ``X``'s node and inspect the RHS.
fn top_level_assign_to_name(stmt: &Stmt) -> Option<(TextRange, &Expr)> {
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
/// of constructor names called anywhere inside it.
struct FactoryCallFinder<'a> {
    imports: &'a HashMap<String, String>,
    allowed: &'a HashSet<&'a str>,
    kinds: HashSet<String>,
}

impl<'ast, 'a> Visitor<'ast> for FactoryCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Some(name) = matched_call_target(expr, self.imports, self.allowed) {
            self.kinds.insert(name);
        }
        walk_expr(self, expr);
    }
}

/// Owner-resolution helper shared by the call-finder queries. Matches
/// the libcst MockPatchPlugin attribution rule: top-level
/// ``FunctionDef`` / ``ClassDef`` own calls inside their subtree
/// (including decorators); everything else attributes to the module.
fn owner_idx_for_stmt(outputs: &BuildOutputs, file: File, stmt: &Stmt) -> Option<usize> {
    let module_idx = outputs.module_nodes_by_file.get(&file).copied();
    let name_range = match stmt {
        Stmt::FunctionDef(f) => f.name.range(),
        Stmt::ClassDef(c) => c.name.range(),
        _ => return module_idx,
    };
    outputs
        .decl_by_name_range
        .get(&(file, range_key(name_range)))
        .copied()
        .or(module_idx)
}

/// Extract the string-literal value of a call's ``args[arg_index]``
/// positional argument. ``None`` when out of range or not a string
/// literal. Concatenated / f-string / b-string forms are intentionally
/// not unwrapped — mirrors the libcst ``_first_string_arg`` rule.
fn nth_positional_string(call: &ruff_python_ast::ExprCall, arg_index: usize) -> Option<String> {
    match call.arguments.args.get(arg_index)? {
        Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
        _ => None,
    }
}

fn call_callee_matches_imported(
    call: &ruff_python_ast::ExprCall,
    imports: &HashMap<String, String>,
    module: &str,
    name: &str,
) -> bool {
    match call.func.as_ref() {
        Expr::Name(n) => imports.get(n.id.as_str()).map(String::as_str) == Some(name),
        Expr::Attribute(attr) => {
            if attr.attr.as_str() != name {
                return false;
            }
            match attr.value.as_ref() {
                Expr::Name(prefix) => {
                    imports.get(prefix.id.as_str()).map(String::as_str) == Some("<module>")
                }
                // Nested attribute: literal dotted module access, e.g.
                // ``import unittest.mock; unittest.mock.patch(...)``.
                _ => dotted_attribute_chain(attr.value.as_ref())
                    .as_deref()
                    .is_some_and(|dotted| dotted == module),
            }
        }
        _ => false,
    }
}

/// Walk an Attribute chain rooted in a bare ``Name``, returning its
/// dotted-string form (``unittest.mock`` for
/// ``Attribute(value=Name("unittest"), attr="mock")``). ``None`` when
/// the chain bottoms out at anything but a ``Name``.
fn dotted_attribute_chain(expr: &Expr) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Attribute(attr) => {
                parts.push(attr.attr.as_str());
                current = attr.value.as_ref();
            }
            Expr::Name(n) => {
                parts.push(n.id.as_str());
                parts.reverse();
                return Some(parts.join("."));
            }
            _ => return None,
        }
    }
}

fn call_callee_matches_var(
    call: &ruff_python_ast::ExprCall,
    owner: &str,
    attr: &str,
    required_positional: Option<usize>,
) -> bool {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    if attribute.attr.as_str() != attr {
        return false;
    }
    let Expr::Name(name) = attribute.value.as_ref() else {
        return false;
    };
    if name.id.as_str() != owner {
        return false;
    }
    if let Some(expected) = required_positional {
        if call.arguments.args.len() != expected {
            return false;
        }
    }
    true
}

struct ImportedCallFinder<'a> {
    imports: &'a HashMap<String, String>,
    module: &'a str,
    name: &'a str,
    arg_index: usize,
    results: Vec<String>,
}

impl<'ast, 'a> Visitor<'ast> for ImportedCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if call_callee_matches_imported(call, self.imports, self.module, self.name) {
                if let Some(value) = nth_positional_string(call, self.arg_index) {
                    self.results.push(value);
                }
            }
        }
        walk_expr(self, expr);
    }
}

struct VarCallFinder<'a> {
    owner: &'a str,
    attr: &'a str,
    arg_index: usize,
    required_positional: Option<usize>,
    results: Vec<String>,
}

impl<'ast, 'a> Visitor<'ast> for VarCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if call_callee_matches_var(call, self.owner, self.attr, self.required_positional) {
                if let Some(value) = nth_positional_string(call, self.arg_index) {
                    self.results.push(value);
                }
            }
        }
        walk_expr(self, expr);
    }
}

/// Per-file sorted list of `(target_start_offset, node_idx)` for the
/// file's top-level decls. Sorted ascending so callers can binary-search
/// for the next decl after a comment.
///
/// Reads straight from `global_index` — every key in there already
/// carries the decl's `(start, end)` range tuple, and `ingest_decls`
/// populated it from the same `all_definitions_with_usage()` walk.
fn file_decl_sites(file: File, global_index: &DeclIndex) -> Vec<(u32, usize)> {
    let mut out: Vec<(u32, usize)> = global_index
        .iter()
        .filter(|((f, _, _), _)| *f == file)
        .map(|((_, _, (start, _)), idx)| (*start, *idx))
        .collect();
    out.sort_by_key(|(start, _)| *start);
    out
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
    class_by_selection: &mut HashMap<(File, (u32, u32)), usize>,
    decl_by_name_range: &mut HashMap<(File, (u32, u32)), usize>,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = module_fqname_for_file(db, file);
    let (file_pinned_by_noqa, per_line_noqa_pins) =
        scan_noqa_directives(&parsed, &source, &line_index);

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
        let mut flags: u32 = 0;
        if node_kind == "import" && (file_pinned_by_noqa || per_line_noqa_pins.contains(&sl)) {
            flags |= NODE_FLAGS_NOQA_PIN;
        }

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
                flags,
                imports,
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
    // Stdlib imports (e.g. `import importlib`) don't surface as graph
    // nodes — the libcst pipeline emits no `[stdlib] X` synthetic and
    // no upstream edge. The alias node itself is still minted in
    // `ingest_decls`, so the call site keeps a use edge through it.
    if module
        .search_path(db)
        .is_some_and(|sp| sp.is_standard_library())
    {
        return Ok(None);
    }
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
    let dead_ranges = detect_dead_ranges(&parsed);
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
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            &dead_ranges,
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
    /// Ranges of statically-dead source regions in the current file
    /// (`if False:` bodies, statements after a `return`/`raise`/`break`/
    /// `continue`/`assert <falsy>`, etc.). A use whose source range
    /// is `contains_range`-covered by any of these gets
    /// `EdgeFlags::DEAD_BRANCH` stamped on every edge it emits.
    dead_ranges: &'a [TextRange],
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
        global_index: &'a DeclIndex,
        module_nodes: &'a HashMap<File, usize>,
        alias_imports: &'a HashMap<usize, ImportSpec>,
        live_decls: &'a LiveDeclIndex,
        dead_ranges: &'a [TextRange],
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

    /// Walk the use's scope chain looking for the reaching definitions
    /// of `name`. Returns one entry per reaching def — typically a
    /// single binding, but `if`/`else` or `try`/`except` branches that
    /// both bind the name leave multiple defs live at end-of-scope and
    /// every one gets edges (Principle 3).
    ///
    /// Walks outward through visible scopes until we find one that
    /// *binds* `name` (not merely lists it in its place table — a free
    /// variable can show up there without any local binding, and in
    /// that case we want to keep walking up to the actual definer).
    /// The first binding scope wins; ty's flow-sensitive
    /// `end_of_scope_symbol_bindings` filters within that scope to the
    /// reaching defs.
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
        let index = semantic_index(db, self.file);
        let Some(file_scope) = self.model.scope(name.into()) else {
            return Vec::new();
        };
        for (scope_id, _scope) in index.visible_ancestor_scopes(file_scope) {
            let place_table = index.place_table(scope_id);
            let Some(symbol_id) = place_table.symbol_id(name.id.as_str()) else {
                continue;
            };
            let use_def_map = index.use_def_map(scope_id);
            let mut saw_binding = false;
            let mut results: Vec<Resolution> = Vec::new();
            for binding in use_def_map.end_of_scope_symbol_bindings(symbol_id) {
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

        // Resolve the initial loading target to a project file.
        let Some(start_mn) = ModuleName::new(&loading_target) else {
            return;
        };
        let Some(start_module) = resolve_module(db, self.file, &start_mn) else {
            return;
        };
        // Stdlib targets don't surface as graph nodes; skip the
        // parallel upstream edges so a use of e.g. `importlib` only
        // links through the local alias (whose own upstream edge
        // is also filtered in `resolve_import_target`).
        if start_module
            .search_path(db)
            .is_some_and(|sp| sp.is_standard_library())
        {
            return;
        }
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
            .map(|(_, &idx)| idx)
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
                    Ok(target) => {
                        if let Some(module_name) = ModuleName::new(&target) {
                            self.emit_nested_star(&module_name);
                            // Literal fromlist entries that themselves
                            // resolve as submodules of the target get
                            // fanned out too — they correspond to a
                            // child `from target.entry import *`.
                            for entry in &fromlist {
                                if entry.is_empty() {
                                    continue;
                                }
                                let sub = format!("{target}.{entry}");
                                if let Some(sub_mn) = ModuleName::new(&sub) {
                                    if module_name_resolves(&sub, self.file, db) {
                                        self.emit_nested_star(&sub_mn);
                                    }
                                }
                            }
                        }
                    }
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
}

#[derive(Debug, Clone, Copy)]
enum DynamicKind {
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
fn detect_dynamic_call(func: &Expr) -> Option<DynamicKind> {
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

enum DynamicParseResult<'a> {
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

fn parse_dynamic_args<'a>(
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

fn first_positional(call: &ruff_python_ast::ExprCall) -> Option<&Expr> {
    call.arguments.args.first()
}

fn positional_or_kwarg<'a>(
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

fn string_literal(expr: &Expr) -> Option<&str> {
    if let Expr::StringLiteral(s) = expr {
        Some(s.value.to_str())
    } else {
        None
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    let Expr::NumberLiteral(n) = expr else {
        return None;
    };
    if let ruff_python_ast::Number::Int(i) = &n.value {
        i.as_i64()
    } else {
        None
    }
}

fn string_literal_list(expr: &Expr) -> Option<Vec<&str>> {
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
fn resolve_dynamic_target(
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
fn file_package_name(db: &dyn ty_python_semantic::Db, file: File) -> Option<String> {
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
fn emit_visitor_warning(py: Python<'_>, message: &str) {
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rel_path<P: AsRef<str>>(path: P) -> RelativePathBuf {
    RelativePathBuf::cli(SystemPath::new(path.as_ref()))
}

// ---------------------------------------------------------------------------
// Noqa parsing
// ---------------------------------------------------------------------------

/// Bitwise OR of [`NodeFlags::ENTRYPOINT`] and [`NodeFlags::NOQA`], the two
/// bits stamped on every import alias pinned by a noqa directive.
///
/// Keep these constants in sync with `dead_cst.graph.NodeFlags` —
/// `NodeFlags` is an `enum.IntFlag` with `auto()` assignments, so the
/// values follow declaration order: NONE=0, SHADOWED=1, ENTRYPOINT=2,
/// OVERLOAD=4, TESTCASE=8, NOQA=16, …
const NODE_FLAG_ENTRYPOINT: u32 = 2;
const NODE_FLAG_NOQA: u32 = 16;
const NODE_FLAGS_NOQA_PIN: u32 = NODE_FLAG_ENTRYPOINT | NODE_FLAG_NOQA;

/// `EdgeFlags::DEAD_BRANCH = enum.auto()` in `dead_cst.graph` —
/// first non-`NONE` bit, value `1`. Stamped on every edge produced
/// by a use site that lives inside a statically-dead region.
const EDGE_FLAG_DEAD_BRANCH: u32 = 1;

// ---------------------------------------------------------------------------
// Dead-region detection
// ---------------------------------------------------------------------------

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
type NameTable = HashMap<String, bool>;

fn detect_dead_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut dead = Vec::new();
    let empty: NameTable = HashMap::new();
    let module_table = build_scope_table(&parsed.syntax().body, &empty);
    walk_suite_for_dead(&parsed.syntax().body, &module_table, &mut dead);
    dead
}

fn walk_suite_for_dead(stmts: &[Stmt], table: &NameTable, dead: &mut Vec<TextRange>) {
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

fn walk_compound_for_dead(stmt: &Stmt, table: &NameTable, dead: &mut Vec<TextRange>) {
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

/// Does `stmt` unconditionally hand off control out of the enclosing
/// suite? Statements *after* a terminator in the same suite are dead.
///
/// Bare terminators are `return` / `raise` / `break` / `continue`
/// plus `assert <falsy>`. Compound forms (`if/elif/else`, `try`,
/// `with`, `match`) are terminators when every reachable path inside
/// them ends in one.
fn stmt_is_terminator(stmt: &Stmt, table: &NameTable) -> bool {
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

fn suite_terminates(stmts: &[Stmt], table: &NameTable) -> bool {
    stmts.iter().any(|s| stmt_is_terminator(s, table))
}

/// Constant-fold the truthiness of `expr`. Returns `None` when the
/// expression doesn't reduce to a known value — names not in `table`,
/// calls, attribute access, comparisons, etc.
///
/// `BoolOp` (`and` / `or`) short-circuits the way Python does:
/// `True or anything` is `True` even when "anything" is unknown.
/// `not` flips a known truth value through `UnaryOp`.
fn evaluate_truthiness(expr: &Expr, table: &NameTable) -> Option<bool> {
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
fn build_scope_table(stmts: &[Stmt], enclosing: &NameTable) -> NameTable {
    let mut table = enclosing.clone();
    let mut bindings: HashMap<String, Vec<&Expr>> = HashMap::new();
    collect_scope_bindings(stmts, &mut bindings);

    let mut changed = true;
    while changed {
        changed = false;
        for (name, exprs) in &bindings {
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
                (Some(v), prev) if prev != Some(v) => {
                    table.insert(name.clone(), v);
                    changed = true;
                }
                (None, Some(_)) => {
                    // Previously folded, now conflicting / unknown:
                    // drop it. Don't fall back to the enclosing
                    // scope's value because a same-name binding in
                    // this scope shadows it.
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
fn collect_scope_bindings<'a>(stmts: &'a [Stmt], out: &mut HashMap<String, Vec<&'a Expr>>) {
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
fn collect_walrus_in_expr<'a>(expr: &'a Expr, out: &mut HashMap<String, Vec<&'a Expr>>) {
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
enum NoqaKind {
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
fn parse_noqa_tail(content: &str) -> Option<NoqaKind> {
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
fn is_per_line_pin(comment_body: &str) -> bool {
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
fn is_file_pin(comment_body: &str) -> bool {
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
fn scan_noqa_directives(
    parsed: &ParsedModuleRef,
    source: &str,
    line_index: &LineIndex,
) -> (bool, HashSet<usize>) {
    let mut file_pinned = false;
    let mut per_line_pins: HashSet<usize> = HashSet::new();
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
    if let Some(module) = file_to_module(db, file) {
        return module.name(db).as_str().to_string();
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
    stem.replace(['/', '\\'], ".")
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
    m.add_class::<ProjectContext>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_noqa_tail_multibyte_prefix_does_not_panic() {
        // Regression: byte-slicing `trimmed[..4]` panicked when the
        // comment started with a multi-byte UTF-8 char (e.g. the
        // 3-byte `─` box-drawing char in section banners).
        assert_eq!(parse_noqa_tail("── Top-level model (dataclass) ──"), None,);
        assert_eq!(parse_noqa_tail("─"), None);
        assert_eq!(parse_noqa_tail("héllo"), None);
        assert_eq!(parse_noqa_tail("🙂🙂"), None);
    }

    #[test]
    fn parse_noqa_tail_short_content_returns_none() {
        assert_eq!(parse_noqa_tail(""), None);
        assert_eq!(parse_noqa_tail("no"), None);
        assert_eq!(parse_noqa_tail("noq"), None);
    }

    #[test]
    fn parse_noqa_tail_recognizes_bare_directive() {
        assert_eq!(parse_noqa_tail(" noqa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("NOQA"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("noqa: "), Some(NoqaKind::Bare));
    }

    #[test]
    fn parse_noqa_tail_recognizes_f401() {
        assert_eq!(parse_noqa_tail("noqa: F401"), Some(NoqaKind::F401Present));
        assert_eq!(
            parse_noqa_tail("noqa: E501, F401"),
            Some(NoqaKind::F401Present),
        );
        assert_eq!(parse_noqa_tail("noqa: E501"), Some(NoqaKind::OtherOnly));
    }
}
