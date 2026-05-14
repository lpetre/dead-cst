//! Experimental ty-backed native parser for dead-cst.
//!
//! Exposes a `Project` class that owns a ty `ProjectDatabase`. Unlike
//! `ty check`, this prototype does NOT discover a `pyproject.toml`:
//! the resolver -- whatever produces dead-cst's `Package` tuples
//! today -- is responsible for feeding the project layout in
//! directly. This mirrors dead-cst's `Analysis(project_root,
//! resolver=...)` shape: callers retain control of which paths are
//! first-party, where the venv lives, and which Python version to
//! target.
//!
//! The native API is structured around packages:
//!   * `PackageSpec(name, path, deps)` describes one package and
//!     its dep relationships.
//!   * `Project.package_order()` returns a toposort of package names
//!     (deps before dependents).
//!   * `Project.build_package_graph(name)` walks every `.py` file
//!     in the package's path and returns a single deduplicated
//!     `NativeGraph` envelope. Plugins consume this on the Python
//!     side and add their own nodes / edges through a context.
//!
//! `build_file_graph(path, module_fqname)` is retained for
//! single-file probing during development.

// pyo3 0.22's `#[pymethods]` macro emits `.into()` on already-`PyErr`
// error paths, which clippy 1.95 flags as `useless_conversion` against
// the user-written function signature. Suppress crate-wide; first-party
// useless conversions are still caught by inspection at PR time.
#![allow(clippy::useless_conversion)]

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::{system_path_to_file, File};
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Alias, Expr, ExprName, Stmt, StmtImportFrom};
use ruff_source_file::LineIndex;
use ruff_text_size::{Ranged, TextRange};
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::{RangedValue, RelativePathBuf};
use ty_project::{ProjectDatabase, ProjectMetadata};
use ty_python_core::program::UseDefaultStrategy;
use ty_python_core::scope::FileScopeId;
use ty_python_semantic::SemanticModel;
use ty_python_semantic::{definitions_for_name, ImportAliasResolution};

/// Description of one package the resolver discovered.
///
/// `path` is the package's root directory; it can be absolute or
/// relative to the project root. `deps` lists other package names
/// this one depends on -- used to compute the analysis order.
#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct PackageSpec {
    name: String,
    path: String,
    deps: Vec<String>,
}

#[pymethods]
impl PackageSpec {
    #[new]
    #[pyo3(signature = (name, path, deps = None))]
    fn new(name: String, path: String, deps: Option<Vec<String>>) -> Self {
        Self {
            name,
            path,
            deps: deps.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PackageSpec(name={:?}, path={:?}, deps={:?})",
            self.name, self.path, self.deps,
        )
    }
}

/// Top-level declaration with libcst-compatible position info.
///
/// ``target_range`` is the source location of the bare name being
/// defined -- ``def <here>:`` / ``class <here>:`` / the LHS ``Name``
/// of an assignment. It matches ty's ``Definition::target_range``
/// (a.k.a. ``focus_range``) so that consumers can look up the decl
/// from a definition handle returned by ty without rebuilding any
/// node-to-AST mapping.
#[pyclass(get_all, frozen)]
struct Decl {
    name: String,
    kind: &'static str,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    target_start: u32,
    target_end: u32,
}

#[pymethods]
impl Decl {
    fn __repr__(&self) -> String {
        format!(
            "Decl(name={:?}, kind={:?}, start=({}, {}), end=({}, {}))",
            self.name,
            self.kind,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column,
        )
    }
}

/// Raw, pre-resolution record of one cross-file import reference.
///
/// Mirrors :class:`dead_cst.graph.Import`: ``module`` and ``decl`` are
/// what the source code literally said (``from <module> import <decl>``
/// or ``import <module>``). Cross-file canonicalization -- promoting
/// ``decl`` to a submodule segment of ``module``, classifying the target
/// as first-party / stdlib / external, attaching attribute chains --
/// happens on the Python side after every package has been ingested.
///
/// ``star`` flags a ``from X import *`` reference (currently unused;
/// no per-name node is materialized for star imports yet).
/// ``speculative`` flags an entry the visitor synthesized for a
/// dynamic-import fromlist that may or may not name a submodule
/// (currently unused).
#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct Import {
    module: String,
    decl: Option<String>,
    star: bool,
    speculative: bool,
}

#[pymethods]
impl Import {
    #[new]
    #[pyo3(signature = (module, decl = None, star = false, speculative = false))]
    fn new(module: String, decl: Option<String>, star: bool, speculative: bool) -> Self {
        Self {
            module,
            decl,
            star,
            speculative,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Import(module={:?}, decl={:?}, star={}, speculative={})",
            self.module, self.decl, self.star, self.speculative,
        )
    }
}

/// A single node in a `NativeGraph`.
///
/// ``imports`` is set for ``kind == "import"`` nodes only -- one per
/// module-level alias in an ``import`` / ``from ... import`` statement.
/// All other kinds carry ``None``.
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

/// One graph contribution, packed for one FFI hop.
///
/// `nodes` and `edges` are unique within the envelope (deduplicated
/// in Rust at build time). `package_name` identifies which package
/// produced it; for the per-file `build_file_graph` shape this is
/// empty.
#[pyclass(get_all, frozen)]
struct NativeGraph {
    package_name: String,
    module_fqname: String,
    file_path: String,
    nodes: Vec<Py<NativeNode>>,
    edges: Vec<(usize, usize, u32)>,
}

#[pymethods]
impl NativeGraph {
    fn __repr__(&self) -> String {
        format!(
            "NativeGraph(package_name={:?}, module_fqname={:?}, file_path={:?}, nodes={}, edges={})",
            self.package_name,
            self.module_fqname,
            self.file_path,
            self.nodes.len(),
            self.edges.len(),
        )
    }
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct NodeKey {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    flags: u32,
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
            flags: node.flags,
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

/// A ty-backed analysis project with explicitly-injected configuration.
#[pyclass(unsendable)]
struct Project {
    db: ProjectDatabase,
    root: SystemPathBuf,
    packages: Vec<PackageSpec>,
    packages_by_name: HashMap<String, usize>,
}

#[pymethods]
impl Project {
    #[new]
    #[pyo3(signature = (
        root,
        *,
        packages = None,
        src_roots = None,
        extra_paths = None,
        python_env = None,
        python_version = None,
        typeshed = None,
    ))]
    fn new(
        root: &str,
        packages: Option<Vec<PackageSpec>>,
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

        let db = ProjectDatabase::use_defaults(metadata, system);

        let packages = packages.unwrap_or_default();
        let mut packages_by_name: HashMap<String, usize> = HashMap::new();
        for (i, pkg) in packages.iter().enumerate() {
            if packages_by_name.insert(pkg.name.clone(), i).is_some() {
                return Err(PyValueError::new_err(format!(
                    "duplicate package name: {:?}",
                    pkg.name
                )));
            }
        }
        for pkg in &packages {
            for dep in &pkg.deps {
                if !packages_by_name.contains_key(dep) {
                    return Err(PyValueError::new_err(format!(
                        "package {:?} declares unknown dep {:?}",
                        pkg.name, dep
                    )));
                }
            }
        }

        Ok(Self {
            db,
            root,
            packages,
            packages_by_name,
        })
    }

    /// The packages this project was configured with (read-only view).
    fn packages(&self) -> Vec<PackageSpec> {
        self.packages.clone()
    }

    /// Return package names in toposort order (deps before dependents).
    ///
    /// Errors if a cycle is present. Unknown dep references are caught
    /// at `Project` construction, so this only fails on cycles.
    fn package_order(&self) -> PyResult<Vec<String>> {
        let n = self.packages.len();
        let mut in_degree: Vec<usize> = vec![0; n];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, pkg) in self.packages.iter().enumerate() {
            in_degree[i] = pkg.deps.len();
            for dep in &pkg.deps {
                let dep_idx = self.packages_by_name[dep];
                dependents[dep_idx].push(i);
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut result: Vec<String> = Vec::with_capacity(n);

        while let Some(i) = queue.pop() {
            result.push(self.packages[i].name.clone());
            for &j in &dependents[i] {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    queue.push(j);
                }
            }
        }

        if result.len() != n {
            let unresolved: Vec<&str> = self
                .packages
                .iter()
                .enumerate()
                .filter(|&(i, _)| in_degree[i] > 0)
                .map(|(_, pkg)| pkg.name.as_str())
                .collect();
            return Err(PyValueError::new_err(format!(
                "package dep cycle involving: {unresolved:?}"
            )));
        }
        Ok(result)
    }

    /// Return the top-level declarations of one file (debug helper).
    fn extract_top_level_decls(&self, path: &str) -> PyResult<Vec<Decl>> {
        let path = SystemPath::new(path);
        let file = system_path_to_file(&self.db, path)
            .map_err(|e| PyOSError::new_err(format!("cannot open {path}: {e:?}")))?;
        let parsed = parsed_module(&self.db, file).load(&self.db);
        let source = source_text(&self.db, file);
        let index = LineIndex::from_source_text(&source);
        let mut decls: Vec<Decl> = Vec::new();
        for stmt in &parsed.syntax().body {
            collect_top_level(stmt, &source, &index, &mut decls);
        }
        Ok(decls)
    }

    /// Build a graph for one file (development / debug shape).
    ///
    /// For real per-package processing, prefer `build_package_graph`.
    fn build_file_graph(
        &self,
        py: Python<'_>,
        path: &str,
        module_fqname: &str,
    ) -> PyResult<NativeGraph> {
        let mut builder = GraphBuilder::new();
        ingest_file(py, &self.db, path, module_fqname, &mut builder)?;
        Ok(NativeGraph {
            package_name: String::new(),
            module_fqname: module_fqname.to_string(),
            file_path: path.to_string(),
            nodes: builder.nodes,
            edges: builder.edges,
        })
    }

    /// Build the graph contribution for one package.
    ///
    /// Walks every `.py` file under the package's path, ingesting
    /// each into a shared `GraphBuilder`. The resulting `NativeGraph`
    /// is deduplicated across all files in the package -- the
    /// in-package module + decl nodes appear at most once.
    ///
    /// `file_path` on the returned envelope is empty (the package
    /// spans many files; each node carries its own `path` field).
    fn build_package_graph(&self, py: Python<'_>, name: &str) -> PyResult<NativeGraph> {
        let idx = self
            .packages_by_name
            .get(name)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("unknown package: {name:?}")))?;
        let package = &self.packages[idx];
        let pkg_abs = absolute_package_path(&self.root, &package.path);

        let mut builder = GraphBuilder::new();
        let mut files = Vec::new();
        collect_py_files(&pkg_abs, &mut files)
            .map_err(|e| PyOSError::new_err(format!("walking {pkg_abs:?}: {e}")))?;
        files.sort();

        for file_abs in files {
            let module_fqname = derive_module_fqname(&file_abs, &pkg_abs, &package.name);
            let file_str = file_abs.to_string_lossy().into_owned();
            ingest_file(py, &self.db, &file_str, &module_fqname, &mut builder)?;
        }

        Ok(NativeGraph {
            package_name: package.name.clone(),
            module_fqname: String::new(),
            file_path: String::new(),
            nodes: builder.nodes,
            edges: builder.edges,
        })
    }
}

fn absolute_package_path(root: &SystemPath, package_path: &str) -> PathBuf {
    let p = PathBuf::from(package_path);
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(root.as_str()).join(p)
    }
}

fn collect_py_files(dir: &PathBuf, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str == "__pycache__" || name_str.starts_with('.') {
                continue;
            }
            collect_py_files(&path, out)?;
        } else if file_type.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "py" {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}

fn derive_module_fqname(file: &Path, package_dir: &Path, package_name: &str) -> String {
    let rel = file.strip_prefix(package_dir).unwrap_or(file);
    let s = rel.to_string_lossy().replace('\\', "/");
    let stripped = s
        .strip_suffix(".py")
        .unwrap_or(&s)
        .trim_end_matches("/__init__")
        .trim_end_matches("__init__");
    if stripped.is_empty() {
        package_name.to_string()
    } else {
        format!("{package_name}.{}", stripped.replace('/', "."))
    }
}

fn ingest_file(
    py: Python<'_>,
    db: &ProjectDatabase,
    path: &str,
    module_fqname: &str,
    builder: &mut GraphBuilder,
) -> PyResult<()> {
    let sys_path = SystemPath::new(path);
    let file = system_path_to_file(db, sys_path)
        .map_err(|e| PyOSError::new_err(format!("cannot open {path}: {e:?}")))?;
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let index = LineIndex::from_source_text(&source);

    let module_range = parsed.syntax().range;
    let (msl, msc, mel, mec) = position(&index, &source, module_range);

    let module_idx = builder.intern_node(
        py,
        NativeNode {
            fqname: module_fqname.to_string(),
            kind: "module",
            path: path.to_string(),
            start_line: msl,
            start_column: msc,
            end_line: mel,
            end_column: mec,
            flags: 0,
            imports: None,
        },
    )?;

    let mut decls: Vec<Decl> = Vec::new();
    for stmt in &parsed.syntax().body {
        collect_top_level(stmt, &source, &index, &mut decls);
    }

    // ``target_range -> decl index`` is the primary lookup, keyed on
    // the same range ty's ``Definition::target_range`` returns. When a
    // name resolves through ty we can map the binding straight to the
    // matching decl (which is critical for redeclaration cases: two
    // ``def f`` at different lines get distinct nodes, and the
    // reference must land on the live one).
    //
    // ``by_name`` is kept as a fallback for the rare case where ty
    // returns a definition whose target_range we didn't index (e.g.
    // walrus targets, for-loop targets that we haven't promoted to
    // top-level decls yet).
    let mut decl_by_range: HashMap<(u32, u32), usize> = HashMap::new();
    let mut decl_by_name: HashMap<String, Vec<usize>> = HashMap::new();
    for decl in &decls {
        let decl_idx = builder.intern_node(
            py,
            NativeNode {
                fqname: format!("{module_fqname}.{}", decl.name),
                kind: decl.kind,
                path: path.to_string(),
                start_line: decl.start_line,
                start_column: decl.start_column,
                end_line: decl.end_line,
                end_column: decl.end_column,
                flags: 0,
                imports: None,
            },
        )?;
        builder.add_edge(decl_idx, module_idx, 0);
        decl_by_range.insert((decl.target_start, decl.target_end), decl_idx);
        decl_by_name
            .entry(decl.name.clone())
            .or_default()
            .push(decl_idx);
    }

    // Imports get the same shape as other top-level decls (one node per
    // bound name, edge to the module). Their ``target_range`` follows
    // ty's ``DefinitionKind::{Import, ImportFrom}::target_range``, which
    // is the full alias range -- so ``RefCollector`` can resolve a
    // ``Name`` use straight to the import node via ``by_range`` without
    // any special-casing for aliased / dotted / from-style imports.
    let is_init = std::path::Path::new(path)
        .file_name()
        .map(|n| n == "__init__.py")
        .unwrap_or(false);

    let mut import_sites: Vec<ImportSite> = Vec::new();
    for stmt in &parsed.syntax().body {
        collect_top_level_imports(stmt, module_fqname, is_init, &mut import_sites);
    }

    for site in &import_sites {
        let (sl, sc, el, ec) = position(&index, &source, site.alias_range);
        let import_obj = Py::new(
            py,
            Import {
                module: site.module.clone(),
                decl: site.decl.clone(),
                star: false,
                speculative: false,
            },
        )?;
        let decl_idx = builder.intern_node(
            py,
            NativeNode {
                fqname: format!("{module_fqname}.{}", site.local_name),
                kind: "import",
                path: path.to_string(),
                start_line: sl,
                start_column: sc,
                end_line: el,
                end_column: ec,
                flags: 0,
                imports: Some(import_obj),
            },
        )?;
        builder.add_edge(decl_idx, module_idx, 0);
        decl_by_range.insert(
            (
                site.target_range.start().to_u32(),
                site.target_range.end().to_u32(),
            ),
            decl_idx,
        );
        decl_by_name
            .entry(site.local_name.clone())
            .or_default()
            .push(decl_idx);
    }

    let model = SemanticModel::new(db, file);
    let module_ref = parsed_module(db, file).load(db);
    let index = DeclIndex {
        by_range: decl_by_range,
        by_name: decl_by_name,
    };
    collect_reference_edges(
        &parsed.syntax().body,
        module_idx,
        &index,
        &model,
        file,
        &module_ref,
        builder,
    );

    Ok(())
}

/// Per-file lookup tables used to resolve a name reference back to the
/// matching graph node. ``by_range`` is keyed on ty's
/// ``Definition::target_range`` for exact same-file binding lookups;
/// ``by_name`` is the fallback when the resolved definition's target
/// range falls outside the indexed set (e.g. an ``import`` alias whose
/// definition target_range we haven't materialized into a node yet).
struct DeclIndex {
    by_range: HashMap<(u32, u32), usize>,
    by_name: HashMap<String, Vec<usize>>,
}

/// Walk top-level statements emitting same-file Name-reference edges.
///
/// Top-level FunctionDef / ClassDef bodies attribute every Name they
/// contain to the enclosing top-level decl (the "top-level only"
/// model: nested defs do not get their own graph node). Assign /
/// AnnAssign RHS / annotation expressions attribute to their LHS
/// target decls. Every other module-level statement attributes its
/// Names to the synthetic module node. Compound statements that don't
/// introduce a new scope (``if`` / ``while`` / ``for`` / ``with`` /
/// ``try`` / ``match``) are descended into so that assignments inside
/// them still get edge attribution.
fn collect_reference_edges<'db>(
    body: &[Stmt],
    module_idx: usize,
    decl_index: &DeclIndex,
    model: &SemanticModel<'db>,
    file: File,
    module_ref: &ruff_db::parsed::ParsedModuleRef,
    builder: &mut GraphBuilder,
) {
    for stmt in body {
        emit_top_level_refs(
            stmt, module_idx, decl_index, model, file, module_ref, builder,
        );
    }
}

fn emit_top_level_refs<'db>(
    stmt: &Stmt,
    module_idx: usize,
    decl_index: &DeclIndex,
    model: &SemanticModel<'db>,
    file: File,
    module_ref: &ruff_db::parsed::ParsedModuleRef,
    builder: &mut GraphBuilder,
) {
    match stmt {
        Stmt::FunctionDef(f) => {
            // Resolve the specific decl for this FunctionDef via its
            // name's source range -- the by-name list also contains
            // shadowed siblings (``def f`` redefined), which must not
            // share the body's reference edges.
            let key = (
                f.name.range().start().to_u32(),
                f.name.range().end().to_u32(),
            );
            if let Some(&decl_idx) = decl_index.by_range.get(&key) {
                let mut coll =
                    RefCollector::new(vec![decl_idx], decl_index, model, file, module_ref);
                coll.visit_stmt(stmt);
                coll.flush(builder);
            }
        }
        Stmt::ClassDef(c) => {
            let key = (
                c.name.range().start().to_u32(),
                c.name.range().end().to_u32(),
            );
            if let Some(&decl_idx) = decl_index.by_range.get(&key) {
                let mut coll =
                    RefCollector::new(vec![decl_idx], decl_index, model, file, module_ref);
                coll.visit_stmt(stmt);
                coll.flush(builder);
            }
        }
        Stmt::Assign(a) => {
            let mut pairs: Vec<(&ExprName, &Expr)> = Vec::new();
            for t in &a.targets {
                pair_targets(t, &a.value, &mut pairs);
            }
            // Group by the RHS expression (chained assignment
            // ``b = c = f`` lets ``b`` and ``c`` share the same RHS
            // identity) so we walk each RHS once with every LHS decl
            // it should attribute to.
            let mut by_rhs: HashMap<*const Expr, Vec<usize>> = HashMap::new();
            for (lhs, rhs) in &pairs {
                let key = (lhs.range().start().to_u32(), lhs.range().end().to_u32());
                if let Some(&idx) = decl_index.by_range.get(&key) {
                    by_rhs.entry(*rhs as *const Expr).or_default().push(idx);
                }
            }
            for (rhs_ptr, decls) in by_rhs {
                // SAFETY: the pointer was derived from a borrow of
                // `a.value` (or a descendant), still alive in this arm.
                let rhs = unsafe { &*rhs_ptr };
                let mut coll = RefCollector::new(decls, decl_index, model, file, module_ref);
                coll.visit_expr(rhs);
                coll.flush(builder);
            }
        }
        Stmt::AnnAssign(a) => {
            if let Expr::Name(n) = a.target.as_ref() {
                let key = (n.range().start().to_u32(), n.range().end().to_u32());
                if let Some(&decl_idx) = decl_index.by_range.get(&key) {
                    let mut coll =
                        RefCollector::new(vec![decl_idx], decl_index, model, file, module_ref);
                    coll.visit_expr(&a.annotation);
                    if let Some(v) = &a.value {
                        coll.visit_expr(v);
                    }
                    coll.flush(builder);
                }
            }
        }
        Stmt::AugAssign(a) => {
            // ``a += b`` is treated like a module-level expression:
            // only the RHS read is emitted, and it attributes to the
            // module (not to the LHS decl). Mirrors libcst's visitor.
            let mut coll = RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
            coll.visit_expr(&a.value);
            coll.flush(builder);
        }
        // No-new-scope compound statements: ``if`` / ``while`` /
        // ``for`` / ``with`` / ``try`` / ``match``. Their test / iter /
        // context-manager / pattern expressions attribute to the
        // module node; their bodies recurse so nested assignments
        // still get attributed to their LHS decls.
        Stmt::If(i) => {
            {
                let mut coll =
                    RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
                coll.visit_expr(&i.test);
                coll.flush(builder);
            }
            for s in &i.body {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
            for clause in &i.elif_else_clauses {
                if let Some(test) = &clause.test {
                    let mut coll =
                        RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
                    coll.visit_expr(test);
                    coll.flush(builder);
                }
                for s in &clause.body {
                    emit_top_level_refs(
                        s, module_idx, decl_index, model, file, module_ref, builder,
                    );
                }
            }
        }
        Stmt::While(w) => {
            {
                let mut coll =
                    RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
                coll.visit_expr(&w.test);
                coll.flush(builder);
            }
            for s in &w.body {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
            for s in &w.orelse {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
        }
        Stmt::For(f) => {
            // ``for x in iter:`` binds ``x`` (its own decl) and the
            // iterable is a read for that decl. Pair them up so
            // ``for x in y:`` produces ``mod.x -> mod.y``.
            let mut targets = Vec::new();
            collect_assign_targets(&f.target, &mut targets);
            let lhs_decls: Vec<usize> = targets
                .iter()
                .filter_map(|(_, range)| {
                    let k = (range.start().to_u32(), range.end().to_u32());
                    decl_index.by_range.get(&k).copied()
                })
                .collect();
            if !lhs_decls.is_empty() {
                let mut coll = RefCollector::new(lhs_decls, decl_index, model, file, module_ref);
                coll.visit_expr(&f.iter);
                coll.flush(builder);
            } else {
                let mut coll =
                    RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
                coll.visit_expr(&f.iter);
                coll.flush(builder);
            }
            for s in &f.body {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
            for s in &f.orelse {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
        }
        Stmt::With(w) => {
            for item in &w.items {
                // The context expression attributes to the LHS decl
                // when ``with X as y:`` binds ``y`` at module level.
                let target_decls: Vec<usize> = if let Some(target) = &item.optional_vars {
                    let mut targets = Vec::new();
                    collect_assign_targets(target, &mut targets);
                    targets
                        .iter()
                        .filter_map(|(_, range)| {
                            let k = (range.start().to_u32(), range.end().to_u32());
                            decl_index.by_range.get(&k).copied()
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let owner = if target_decls.is_empty() {
                    vec![module_idx]
                } else {
                    target_decls
                };
                let mut coll = RefCollector::new(owner, decl_index, model, file, module_ref);
                coll.visit_expr(&item.context_expr);
                coll.flush(builder);
            }
            for s in &w.body {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
        }
        Stmt::Try(t) => {
            for s in &t.body {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
            for handler in &t.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(eh) = handler;
                if let Some(ty) = &eh.type_ {
                    let mut coll =
                        RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
                    coll.visit_expr(ty);
                    coll.flush(builder);
                }
                for s in &eh.body {
                    emit_top_level_refs(
                        s, module_idx, decl_index, model, file, module_ref, builder,
                    );
                }
            }
            for s in &t.orelse {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
            for s in &t.finalbody {
                emit_top_level_refs(s, module_idx, decl_index, model, file, module_ref, builder);
            }
        }
        Stmt::Match(m) => {
            {
                let mut coll =
                    RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
                coll.visit_expr(&m.subject);
                coll.flush(builder);
            }
            for case in &m.cases {
                for s in &case.body {
                    emit_top_level_refs(
                        s, module_idx, decl_index, model, file, module_ref, builder,
                    );
                }
            }
        }
        _ => {
            let mut coll = RefCollector::new(vec![module_idx], decl_index, model, file, module_ref);
            coll.visit_stmt(stmt);
            coll.flush(builder);
        }
    }
}

/// Visitor that records every ``Name`` reference encountered against
/// the active ``current_decls`` stack.
///
/// "Top-level only" attribution: ``current_decls`` is set once per
/// top-level statement by :func:`collect_reference_edges` and is *not*
/// pushed when descending into nested ``def`` / ``class`` -- inner
/// references fold into the outer decl. Edges collapse to a unique
/// ``(src, dst)`` set so a repeated reference within the same decl
/// emits one edge.
///
/// Scope resolution is delegated to ty's :type:`SemanticModel` (the
/// same use-def map ty uses for type inference): a ``Name`` is emitted
/// as a same-file edge only when ``definitions_for_name`` resolves the
/// reference to a definition in the file's global scope. Closures,
/// parameter shadowing, walrus targets, ``import`` aliases, and the
/// rest of Python's LEGB rules fall out of ty's index for free, so
/// this visitor stays a thin shell over name -> top-level-decl
/// emission.
struct RefCollector<'a, 'db> {
    current_decls: Vec<usize>,
    decl_index: &'a DeclIndex,
    edges: HashSet<(usize, usize)>,
    model: &'a SemanticModel<'db>,
    file: File,
    module_ref: &'a ruff_db::parsed::ParsedModuleRef,
}

impl<'a, 'db> RefCollector<'a, 'db> {
    fn new(
        current_decls: Vec<usize>,
        decl_index: &'a DeclIndex,
        model: &'a SemanticModel<'db>,
        file: File,
        module_ref: &'a ruff_db::parsed::ParsedModuleRef,
    ) -> Self {
        Self {
            current_decls,
            decl_index,
            edges: HashSet::new(),
            model,
            file,
            module_ref,
        }
    }

    fn flush(self, builder: &mut GraphBuilder) {
        for (src, dst) in self.edges {
            builder.add_edge(src, dst, 0);
        }
    }

    fn emit_name(&mut self, name: &ExprName) {
        let defs = definitions_for_name(
            self.model,
            name.id.as_str(),
            name.into(),
            ImportAliasResolution::PreserveAliases,
        );
        // Collect every same-file global-scope binding ty found for
        // this name. We map each to a specific decl by target_range
        // (so two ``def f`` at different lines resolve to different
        // nodes); the per-name fallback handles the very rare case
        // where a target_range fell outside the indexed set.
        let mut targets: HashSet<usize> = HashSet::new();
        let db = self.model.db();
        for resolved in defs {
            let Some(def) = resolved.definition() else {
                continue;
            };
            if def.file(db) != self.file {
                continue;
            }
            if def.file_scope(db) != FileScopeId::global() {
                continue;
            }
            let tr = def.kind(db).target_range(self.module_ref);
            let key = (tr.start().to_u32(), tr.end().to_u32());
            if let Some(&idx) = self.decl_index.by_range.get(&key) {
                targets.insert(idx);
            } else if let Some(by_name) = self.decl_index.by_name.get(name.id.as_str()) {
                // Fallback: ty pinned the binding to the global scope
                // but we never minted a decl at that exact target
                // range (e.g. an ``import`` alias we don't model yet).
                // Land on every same-name decl so reachability is at
                // least conservative.
                for &idx in by_name {
                    targets.insert(idx);
                }
            }
        }
        for dst in targets {
            for &src in &self.current_decls {
                if src == dst {
                    continue;
                }
                self.edges.insert((src, dst));
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

fn push_decl(
    name: String,
    kind: &'static str,
    range: TextRange,
    target_range: TextRange,
    source: &str,
    index: &LineIndex,
    out: &mut Vec<Decl>,
) {
    let (sl, sc, el, ec) = position(index, source, range);
    out.push(Decl {
        name,
        kind,
        start_line: sl,
        start_column: sc,
        end_line: el,
        end_column: ec,
        target_start: target_range.start().to_u32(),
        target_end: target_range.end().to_u32(),
    });
}

/// Yield ``(name, rhs)`` pairs for an assignment target pattern.
///
/// ``a, b = x, y`` -> ``[(a, x), (b, y)]``. ``a, b = call()`` (RHS
/// arity mismatches the target tuple) -> the whole ``call()`` is
/// broadcast to both ``a`` and ``b``. Nested tuple / list patterns
/// recurse; non-Name leaves (``Attribute``, ``Subscript``) are
/// skipped. Mirrors :func:`dead_cst._visitor._pair_targets`. Each
/// pair carries the LHS ``ExprName`` so the caller can recover the
/// per-target ``target_range`` and disambiguate redeclarations.
fn pair_targets<'ast>(
    target: &'ast Expr,
    rhs: &'ast Expr,
    out: &mut Vec<(&'ast ExprName, &'ast Expr)>,
) {
    match target {
        Expr::Name(n) => {
            out.push((n, rhs));
        }
        Expr::Tuple(t) => {
            let rhs_elts = match rhs {
                Expr::Tuple(rt) if rt.elts.len() == t.elts.len() => Some(&rt.elts),
                Expr::List(rl) if rl.elts.len() == t.elts.len() => Some(&rl.elts),
                _ => None,
            };
            if let Some(rhs_elts) = rhs_elts {
                for (te, re) in t.elts.iter().zip(rhs_elts.iter()) {
                    pair_targets(te, re, out);
                }
            } else {
                for te in &t.elts {
                    pair_targets(te, rhs, out);
                }
            }
        }
        Expr::List(t) => {
            let rhs_elts = match rhs {
                Expr::Tuple(rt) if rt.elts.len() == t.elts.len() => Some(&rt.elts),
                Expr::List(rl) if rl.elts.len() == t.elts.len() => Some(&rl.elts),
                _ => None,
            };
            if let Some(rhs_elts) = rhs_elts {
                for (te, re) in t.elts.iter().zip(rhs_elts.iter()) {
                    pair_targets(te, re, out);
                }
            } else {
                for te in &t.elts {
                    pair_targets(te, rhs, out);
                }
            }
        }
        Expr::Starred(s) => pair_targets(&s.value, rhs, out),
        _ => {}
    }
}

fn collect_assign_targets(target: &Expr, out: &mut Vec<(String, TextRange)>) {
    match target {
        Expr::Name(n) => out.push((n.id.to_string(), n.range)),
        Expr::Tuple(t) => {
            for e in &t.elts {
                collect_assign_targets(e, out);
            }
        }
        Expr::List(l) => {
            for e in &l.elts {
                collect_assign_targets(e, out);
            }
        }
        Expr::Starred(s) => collect_assign_targets(&s.value, out),
        _ => {}
    }
}

/// Per-alias record emitted by :func:`collect_top_level_imports`.
///
/// ``target_range`` matches ty's ``DefinitionKind::{Import, ImportFrom}::target_range``
/// (the full alias range -- including ``as asname`` if present), so a
/// :class:`RefCollector` lookup against ``decl_by_range`` lands on this
/// site whenever a ``Name`` use resolves through the alias. ``module``
/// is what the source code literally said for ``import <module>`` /
/// ``from <module> import ...``, with relative dots resolved against the
/// file's enclosing package; ``decl`` is the ``from``-style imported
/// name (``None`` for plain ``import``).
struct ImportSite {
    local_name: String,
    module: String,
    decl: Option<String>,
    target_range: TextRange,
    alias_range: TextRange,
}

fn collect_top_level_imports(
    stmt: &Stmt,
    module_fqname: &str,
    is_init: bool,
    out: &mut Vec<ImportSite>,
) {
    match stmt {
        Stmt::Import(s) => {
            for alias in &s.names {
                if let Some(site) = build_import_site(alias, "", false) {
                    out.push(site);
                }
            }
        }
        Stmt::ImportFrom(s) => {
            let from_module = resolve_from_module(s, module_fqname, is_init);
            for alias in &s.names {
                if let Some(site) = build_import_site(alias, &from_module, true) {
                    out.push(site);
                }
            }
        }
        // Compound non-scope statements (mirrors ``collect_top_level``):
        // imports nested under ``if TYPE_CHECKING:`` and friends still
        // bind at module scope.
        Stmt::If(i) => {
            for s in &i.body {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
            for clause in &i.elif_else_clauses {
                for s in &clause.body {
                    collect_top_level_imports(s, module_fqname, is_init, out);
                }
            }
        }
        Stmt::While(w) => {
            for s in &w.body {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
            for s in &w.orelse {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
        }
        Stmt::For(f) => {
            for s in &f.body {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
            for s in &f.orelse {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
        }
        Stmt::With(w) => {
            for s in &w.body {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
        }
        Stmt::Try(t) => {
            for s in &t.body {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
            for handler in &t.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(eh) = handler;
                for s in &eh.body {
                    collect_top_level_imports(s, module_fqname, is_init, out);
                }
            }
            for s in &t.orelse {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
            for s in &t.finalbody {
                collect_top_level_imports(s, module_fqname, is_init, out);
            }
        }
        Stmt::Match(m) => {
            for case in &m.cases {
                for s in &case.body {
                    collect_top_level_imports(s, module_fqname, is_init, out);
                }
            }
        }
        _ => {}
    }
}

fn build_import_site(alias: &Alias, from_module: &str, is_from: bool) -> Option<ImportSite> {
    let name_id = alias.name.id.as_str();
    if name_id == "*" {
        // ``from X import *`` doesn't bind any individual name and so
        // doesn't get a per-name node. It will land in a future raw
        // ``imports`` field on ``NativeGraph`` so the Python edge
        // stitcher can fan it out across the star target's surface.
        return None;
    }

    let local_name = if let Some(asname) = &alias.asname {
        asname.id.as_str().to_string()
    } else if is_from {
        name_id.to_string()
    } else {
        // ``import a.b.c`` binds the first segment ``a`` in the local
        // scope; everything after the first dot is attribute access on
        // the bound module object.
        let dot = name_id.find('.').unwrap_or(name_id.len());
        name_id[..dot].to_string()
    };

    let (module, decl) = if is_from {
        (from_module.to_string(), Some(name_id.to_string()))
    } else {
        (name_id.to_string(), None)
    };

    Some(ImportSite {
        local_name,
        module,
        decl,
        target_range: alias.range,
        alias_range: alias.range,
    })
}

/// Resolve a ``from .x import y`` clause's module to its absolute form.
///
/// ``level == 0`` is an absolute import -- return the module string
/// verbatim. ``level >= 1`` resolves relative to the file's enclosing
/// package: ``__init__.py`` resolves dots starting from the file's own
/// FQN, every other file drops its module segment first. Each additional
/// dot beyond the first pops one more parent off the segment stack.
fn resolve_from_module(s: &StmtImportFrom, module_fqname: &str, is_init: bool) -> String {
    let base = s.module.as_ref().map(|i| i.id.as_str()).unwrap_or("");
    if s.level == 0 {
        return base.to_string();
    }
    let mut segments: Vec<&str> = if module_fqname.is_empty() {
        Vec::new()
    } else {
        module_fqname.split('.').collect()
    };
    if !is_init && !segments.is_empty() {
        segments.pop();
    }
    let pop_count = (s.level as usize).saturating_sub(1);
    for _ in 0..pop_count {
        if segments.pop().is_none() {
            break;
        }
    }
    if base.is_empty() {
        segments.join(".")
    } else if segments.is_empty() {
        base.to_string()
    } else {
        format!("{}.{}", segments.join("."), base)
    }
}

fn collect_top_level(stmt: &Stmt, source: &str, index: &LineIndex, out: &mut Vec<Decl>) {
    match stmt {
        Stmt::FunctionDef(f) => {
            push_decl(
                f.name.to_string(),
                "function",
                f.range,
                f.name.range(),
                source,
                index,
                out,
            );
        }
        Stmt::ClassDef(c) => {
            push_decl(
                c.name.to_string(),
                "class",
                c.range,
                c.name.range(),
                source,
                index,
                out,
            );
        }
        Stmt::Assign(a) => {
            let mut targets = Vec::new();
            for t in &a.targets {
                collect_assign_targets(t, &mut targets);
            }
            for (name, range) in targets {
                push_decl(name, "variable", range, range, source, index, out);
            }
        }
        Stmt::AnnAssign(a) => {
            if let Expr::Name(n) = a.target.as_ref() {
                push_decl(
                    n.id.to_string(),
                    "variable",
                    n.range,
                    n.range,
                    source,
                    index,
                    out,
                );
            }
        }
        // Compound statements that don't introduce a new scope: their
        // assignments still create top-level decls. Mirrors libcst's
        // ScopeProvider rules.
        Stmt::If(i) => {
            for s in &i.body {
                collect_top_level(s, source, index, out);
            }
            for clause in &i.elif_else_clauses {
                for s in &clause.body {
                    collect_top_level(s, source, index, out);
                }
            }
        }
        Stmt::While(w) => {
            for s in &w.body {
                collect_top_level(s, source, index, out);
            }
            for s in &w.orelse {
                collect_top_level(s, source, index, out);
            }
        }
        Stmt::For(f) => {
            // ``for x in ...:`` binds ``x`` at module scope.
            let mut targets = Vec::new();
            collect_assign_targets(&f.target, &mut targets);
            for (name, range) in targets {
                push_decl(name, "variable", range, range, source, index, out);
            }
            for s in &f.body {
                collect_top_level(s, source, index, out);
            }
            for s in &f.orelse {
                collect_top_level(s, source, index, out);
            }
        }
        Stmt::With(w) => {
            for item in &w.items {
                if let Some(target) = &item.optional_vars {
                    let mut targets = Vec::new();
                    collect_assign_targets(target, &mut targets);
                    for (name, range) in targets {
                        push_decl(name, "variable", range, range, source, index, out);
                    }
                }
            }
            for s in &w.body {
                collect_top_level(s, source, index, out);
            }
        }
        Stmt::Try(t) => {
            for s in &t.body {
                collect_top_level(s, source, index, out);
            }
            for handler in &t.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(eh) = handler;
                for s in &eh.body {
                    collect_top_level(s, source, index, out);
                }
            }
            for s in &t.orelse {
                collect_top_level(s, source, index, out);
            }
            for s in &t.finalbody {
                collect_top_level(s, source, index, out);
            }
        }
        Stmt::Match(m) => {
            for case in &m.cases {
                for s in &case.body {
                    collect_top_level(s, source, index, out);
                }
            }
        }
        _ => {}
    }
}

#[pymodule]
fn dead_cst_ty_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Decl>()?;
    m.add_class::<Import>()?;
    m.add_class::<NativeNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<PackageSpec>()?;
    m.add_class::<Project>()?;
    Ok(())
}
