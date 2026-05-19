//! Binary persistence for project graphs.
//!
//! The on-disk format is a 12-byte header followed by a bincode body:
//!
//! ```text
//! magic       : "DEADCSTG"   (8 bytes)
//! version     : u32 LE       (4 bytes)
//! body        : bincode(GraphFile)
//! ```
//!
//! The body holds the project-wide node + edge lists and a small
//! [`GraphMetadata`] block (creation timestamp, counts, user-supplied
//! `--meta key=value` pairs). The format intentionally captures only
//! the graph — plugins must rebuild on load — so the file stays small
//! and the rust ↔ Python round-trip is one allocation per node.

use std::fs::File as StdFile;
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph::{intern_kind, Import, NativeGraph, SymbolNode};

/// Magic bytes prefixing every persisted graph file. Lets the loader
/// reject unrelated files before attempting any deserialization.
pub(crate) const MAGIC: &[u8; 8] = b"DEADCSTG";

/// On-disk format version. Bumped on any breaking layout change to
/// [`GraphFileBody`]; the loader rejects mismatched values outright
/// (graphs are cheap to rebuild — no migration logic).
pub(crate) const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct GraphFileBody {
    metadata: GraphMetadataRecord,
    nodes: Vec<NodeRecord>,
    edges: Vec<(u32, u32, u32)>,
}

#[derive(Serialize, Deserialize, Clone)]
struct GraphMetadataRecord {
    /// Unix epoch seconds at which the file was written.
    created_at: u64,
    node_count: u64,
    edge_count: u64,
    file_count: u64,
    line_count: u64,
    user_meta: Vec<(String, String)>,
}

#[derive(Serialize, Deserialize)]
struct NodeRecord {
    fqname: String,
    kind: String,
    path: String,
    start_line: u32,
    start_column: u32,
    end_line: u32,
    end_column: u32,
    flags: u32,
    imports: Option<ImportRecord>,
}

#[derive(Serialize, Deserialize)]
struct ImportRecord {
    module: String,
    decl: Option<String>,
    star: bool,
}

/// Metadata block returned alongside the graph by [`read_graph`].
///
/// `created_at` is unix-epoch seconds; `user_meta` carries the
/// `--meta key=value` pairs passed at write time. The counts are the
/// values stamped into the file at write time — they double as a
/// sanity check against the deserialized graph.
#[pyclass(get_all, frozen, name = "GraphMetadata")]
#[derive(Clone)]
pub(crate) struct GraphMetadata {
    pub(crate) format_version: u32,
    pub(crate) created_at: u64,
    pub(crate) node_count: u64,
    pub(crate) edge_count: u64,
    pub(crate) file_count: u64,
    pub(crate) line_count: u64,
    pub(crate) user_meta: Vec<(String, String)>,
}

#[pymethods]
impl GraphMetadata {
    fn __repr__(&self) -> String {
        format!(
            "GraphMetadata(format_version={}, created_at={}, nodes={}, edges={}, files={}, lines={}, user_meta={} entries)",
            self.format_version,
            self.created_at,
            self.node_count,
            self.edge_count,
            self.file_count,
            self.line_count,
            self.user_meta.len(),
        )
    }
}

fn distinct_files_and_lines<I, F>(nodes: I, get: F) -> (u64, u64)
where
    I: IntoIterator,
    F: Fn(&I::Item) -> (Option<String>, u32),
{
    use std::collections::HashMap;
    let mut max_line: HashMap<String, u32> = HashMap::new();
    for node in nodes {
        let (path, end_line) = get(&node);
        let Some(path) = path else { continue };
        if path.is_empty() {
            continue;
        }
        let slot = max_line.entry(path).or_insert(0);
        if end_line > *slot {
            *slot = end_line;
        }
    }
    let files = max_line.len() as u64;
    let lines: u64 = max_line.values().map(|v| *v as u64).sum();
    (files, lines)
}

/// Write a project graph to `path`.
///
/// `nodes` / `edges` are the same payload `NativeGraph` exposes —
/// passed as plain Python objects so callers don't need to keep the
/// `NativeGraph` instance around. `meta` is the user-supplied list of
/// `(key, value)` pairs from `--meta`.
#[pyfunction]
#[pyo3(signature = (path, nodes, edges, meta))]
pub(crate) fn write_graph(
    path: String,
    nodes: Vec<Py<SymbolNode>>,
    edges: Vec<(u32, u32, u32)>,
    meta: Vec<(String, String)>,
    py: Python<'_>,
) -> PyResult<()> {
    let mut node_records: Vec<NodeRecord> = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let n = node.borrow(py);
        let imports = if let Some(imp) = &n.imports {
            let imp = imp.borrow(py);
            Some(ImportRecord {
                module: imp.module.clone(),
                decl: imp.decl.clone(),
                star: imp.star,
            })
        } else {
            None
        };
        node_records.push(NodeRecord {
            fqname: n.fqname.clone(),
            kind: n.kind.to_string(),
            path: n.path.clone(),
            start_line: n.start_line as u32,
            start_column: n.start_column as u32,
            end_line: n.end_line as u32,
            end_column: n.end_column as u32,
            flags: n.flags,
            imports,
        });
    }

    let (file_count, line_count) = distinct_files_and_lines(&node_records, |n: &&NodeRecord| {
        (Some(n.path.clone()), n.end_line)
    });

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let body = GraphFileBody {
        metadata: GraphMetadataRecord {
            created_at,
            node_count: node_records.len() as u64,
            edge_count: edges.len() as u64,
            file_count,
            line_count,
            user_meta: meta,
        },
        nodes: node_records,
        edges,
    };

    let file = StdFile::create(&path)
        .map_err(|e| PyIOError::new_err(format!("could not open {path:?} for write: {e}")))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(MAGIC)
        .map_err(|e| PyIOError::new_err(format!("write magic to {path:?}: {e}")))?;
    writer
        .write_all(&FORMAT_VERSION.to_le_bytes())
        .map_err(|e| PyIOError::new_err(format!("write version to {path:?}: {e}")))?;
    bincode::serialize_into(&mut writer, &body)
        .map_err(|e| PyIOError::new_err(format!("bincode encode {path:?}: {e}")))?;
    writer
        .flush()
        .map_err(|e| PyIOError::new_err(format!("flush {path:?}: {e}")))?;
    Ok(())
}

/// Read a project graph from `path`.
///
/// Returns a `(NativeGraph, GraphMetadata)` pair. The graph object can
/// be fed straight into the Python bridge to build a `SymbolGraph`.
/// Format-version mismatch raises `ValueError`; missing magic raises
/// `ValueError` too — both are hard errors with no migration path.
#[pyfunction]
pub(crate) fn read_graph(
    path: String,
    py: Python<'_>,
) -> PyResult<(Py<NativeGraph>, Py<GraphMetadata>)> {
    let file = StdFile::open(&path)
        .map_err(|e| PyIOError::new_err(format!("could not open {path:?}: {e}")))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| PyIOError::new_err(format!("read magic from {path:?}: {e}")))?;
    if &magic != MAGIC {
        return Err(PyValueError::new_err(format!(
            "{path:?} is not a dead-cst graph file (bad magic)"
        )));
    }

    let mut version_bytes = [0u8; 4];
    reader
        .read_exact(&mut version_bytes)
        .map_err(|e| PyIOError::new_err(format!("read version from {path:?}: {e}")))?;
    let version = u32::from_le_bytes(version_bytes);
    if version != FORMAT_VERSION {
        return Err(PyValueError::new_err(format!(
            "{path:?} has format version {version}; this build expects {FORMAT_VERSION}. \
             Rebuild the graph with this dead-cst version."
        )));
    }

    let body: GraphFileBody = bincode::deserialize_from(&mut reader)
        .map_err(|e| PyIOError::new_err(format!("bincode decode {path:?}: {e}")))?;

    let GraphFileBody {
        metadata,
        nodes,
        edges,
    } = body;

    let mut node_objs: Vec<Py<SymbolNode>> = Vec::with_capacity(nodes.len());
    for rec in nodes {
        let kind = intern_kind(&rec.kind)?;
        let imports = match rec.imports {
            Some(imp) => {
                let import = Import {
                    module: imp.module,
                    decl: imp.decl,
                    star: imp.star,
                };
                Some(Py::new(py, import)?)
            }
            None => None,
        };
        let node = SymbolNode {
            fqname: rec.fqname,
            kind,
            path: rec.path,
            start_line: rec.start_line as usize,
            start_column: rec.start_column as usize,
            end_line: rec.end_line as usize,
            end_column: rec.end_column as usize,
            flags: rec.flags,
            imports,
            cached_hash: std::sync::OnceLock::new(),
        };
        node_objs.push(Py::new(py, node)?);
    }

    let graph = NativeGraph {
        nodes: node_objs,
        edges: edges
            .into_iter()
            .map(|(s, d, f)| (s as usize, d as usize, f))
            .collect(),
    };
    let meta = GraphMetadata {
        format_version: version,
        created_at: metadata.created_at,
        node_count: metadata.node_count,
        edge_count: metadata.edge_count,
        file_count: metadata.file_count,
        line_count: metadata.line_count,
        user_meta: metadata.user_meta,
    };
    Ok((Py::new(py, graph)?, Py::new(py, meta)?))
}
