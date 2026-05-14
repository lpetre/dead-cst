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

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::system_path_to_file;
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_python_ast::{Expr, Stmt};
use ruff_source_file::LineIndex;
use ruff_text_size::TextRange;
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::{RangedValue, RelativePathBuf};
use ty_project::{ProjectDatabase, ProjectMetadata};
use ty_python_core::program::UseDefaultStrategy;

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
#[pyclass(get_all, frozen)]
struct Decl {
    name: String,
    kind: &'static str,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
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

/// A single node in a `NativeGraph`.
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
        let cwd = SystemPathBuf::from_path_buf(cwd.try_into().map_err(|_| {
            PyValueError::new_err("current working directory is not valid UTF-8")
        })?)
        .map_err(|_| PyValueError::new_err("current working directory is not absolute"))?;
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

fn derive_module_fqname(file: &PathBuf, package_dir: &PathBuf, package_name: &str) -> String {
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
        },
    )?;

    let mut decls: Vec<Decl> = Vec::new();
    for stmt in &parsed.syntax().body {
        collect_top_level(stmt, &source, &index, &mut decls);
    }
    for decl in decls {
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
            },
        )?;
        builder.add_edge(decl_idx, module_idx, 0);
    }
    Ok(())
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
    });
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
        _ => {}
    }
}

fn collect_top_level(stmt: &Stmt, source: &str, index: &LineIndex, out: &mut Vec<Decl>) {
    match stmt {
        Stmt::FunctionDef(f) => {
            push_decl(f.name.to_string(), "function", f.range, source, index, out);
        }
        Stmt::ClassDef(c) => {
            push_decl(c.name.to_string(), "class", c.range, source, index, out);
        }
        Stmt::Assign(a) => {
            let mut targets = Vec::new();
            for t in &a.targets {
                collect_assign_targets(t, &mut targets);
            }
            for (name, range) in targets {
                push_decl(name, "variable", range, source, index, out);
            }
        }
        Stmt::AnnAssign(a) => {
            if let Expr::Name(n) = a.target.as_ref() {
                push_decl(n.id.to_string(), "variable", n.range, source, index, out);
            }
        }
        _ => {}
    }
}

#[pymodule]
fn dead_cst_ty_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Decl>()?;
    m.add_class::<NativeNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<PackageSpec>()?;
    m.add_class::<Project>()?;
    Ok(())
}
