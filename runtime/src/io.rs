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
//! The body holds the project-wide node + edge lists, a small
//! [`GraphMetadata`] block (creation timestamp, counts, user-supplied
//! `--meta key=value` pairs), and the node + edge flag registries (so a
//! reader can decode a flag bit to its `owner/name`). The format
//! intentionally captures only the graph — plugins must rebuild on load
//! — so the file stays small and the rust ↔ Python round-trip is one
//! allocation per node.

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
/// (graphs are cheap to rebuild — no migration logic). Bumped 1 → 2
/// when the flag registries were added to the body and edge flags
/// narrowed to `u8`; 2 → 3 when node paths moved behind the
/// [`GraphFileBody::paths`] table indirection.
pub(crate) const FORMAT_VERSION: u32 = 3;

#[derive(Serialize, Deserialize)]
struct GraphFileBody {
    metadata: GraphMetadataRecord,
    /// Path table: every distinct node path, in first-encounter order
    /// (assembled graphs write nodes file-contiguously, so this is the
    /// build's file order). [`NodeRecord::path_idx`] indexes into it.
    /// Besides deduplicating what was one owned string per node, the
    /// table gives the file an explicit file-identity axis — the
    /// foundation graph surgery keys per-file node blocks on.
    paths: Vec<String>,
    nodes: Vec<NodeRecord>,
    edges: Vec<(u32, u32, u8)>,
    /// Node-flag registry (engine built-ins + plugin declarations) so a
    /// reader can decode a node's flag bits to `owner/name`. Top-level
    /// (structural), not user metadata.
    node_flag_registry: Vec<FlagRecord>,
    /// Edge-flag registry — the edge-space twin of [`Self::node_flag_registry`].
    edge_flag_registry: Vec<FlagRecord>,
}

/// One serialized flag-registry entry: a `owner/name` flag and the bit it
/// occupies, plus its reachability semantics. `bit` is stored as `u64` for
/// headroom against a future width widen; the runtime narrows it to `u32`
/// (node) / `u8` (edge) on read.
#[derive(Serialize, Deserialize, Clone)]
struct FlagRecord {
    name: String,
    bit: u64,
    seed: bool,
    default_on: bool,
    description: String,
}

/// Python-facing flag-registry entry tuple: `(name, bit, seed, default_on,
/// description)`. The shape `GraphMetadata` exposes and `write_graph` accepts,
/// so a loaded graph can be re-written without reshaping.
type FlagTuple = (String, u64, bool, bool, String);

impl FlagRecord {
    fn from_tuple((name, bit, seed, default_on, description): FlagTuple) -> Self {
        Self {
            name,
            bit,
            seed,
            default_on,
            description,
        }
    }

    fn into_tuple(self) -> FlagTuple {
        (
            self.name,
            self.bit,
            self.seed,
            self.default_on,
            self.description,
        )
    }
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
    /// Index into [`GraphFileBody::paths`].
    path_idx: u32,
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
    /// Node-flag registry entries `(name, bit, seed, default_on, description)`,
    /// bit-sorted. Lets a Python reader decode a node's flags by name and a
    /// re-write preserve the registry.
    pub(crate) node_flag_registry: Vec<FlagTuple>,
    /// Edge-flag registry entries — the edge-space twin of
    /// [`Self::node_flag_registry`].
    pub(crate) edge_flag_registry: Vec<FlagTuple>,
}

#[pymethods]
impl GraphMetadata {
    fn __repr__(&self) -> String {
        format!(
            "GraphMetadata(format_version={}, created_at={}, nodes={}, edges={}, files={}, lines={}, user_meta={} entries, node_flags={}, edge_flags={})",
            self.format_version,
            self.created_at,
            self.node_count,
            self.edge_count,
            self.file_count,
            self.line_count,
            self.user_meta.len(),
            self.node_flag_registry.len(),
            self.edge_flag_registry.len(),
        )
    }
}

/// Write a project graph to `path`.
///
/// `nodes` / `edges` are the same payload `NativeGraph` exposes —
/// passed as plain Python objects so callers don't need to keep the
/// `NativeGraph` instance around. `meta` is the user-supplied list of
/// `(key, value)` pairs from `--meta`. `node_flag_registry` /
/// `edge_flag_registry` are the `(name, bit, seed, default_on,
/// description)` tuples from `ProjectContext.{node,edge}_flag_registry`,
/// persisted so a reader can decode flag bits by name.
#[pyfunction]
#[pyo3(signature = (path, nodes, edges, meta, node_flag_registry, edge_flag_registry))]
pub(crate) fn write_graph(
    path: String,
    nodes: Vec<Py<SymbolNode>>,
    edges: Vec<(u32, u32, u8)>,
    meta: Vec<(String, String)>,
    node_flag_registry: Vec<FlagTuple>,
    edge_flag_registry: Vec<FlagTuple>,
    py: Python<'_>,
) -> PyResult<()> {
    let mut node_records: Vec<NodeRecord> = Vec::with_capacity(nodes.len());
    let mut paths: Vec<String> = Vec::new();
    let mut path_to_idx: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
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
        let path_idx = *path_to_idx.entry(n.path.clone()).or_insert_with(|| {
            paths.push(n.path.clone());
            (paths.len() - 1) as u32
        });
        node_records.push(NodeRecord {
            fqname: n.fqname.clone(),
            kind: n.kind.to_string(),
            path_idx,
            start_line: n.start_line as u32,
            start_column: n.start_column as u32,
            end_line: n.end_line as u32,
            end_column: n.end_column as u32,
            flags: n.flags,
            imports,
        });
    }

    // The path table IS the distinct-file set; per-file line counts
    // fold over `path_idx` instead of re-hashing path strings.
    let file_count = paths.iter().filter(|p| !p.is_empty()).count() as u64;
    let mut max_line: Vec<u32> = vec![0; paths.len()];
    for n in &node_records {
        let slot = &mut max_line[n.path_idx as usize];
        *slot = (*slot).max(n.end_line);
    }
    let line_count: u64 = paths
        .iter()
        .zip(&max_line)
        .filter(|(p, _)| !p.is_empty())
        .map(|(_, &l)| l as u64)
        .sum();

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
        paths,
        nodes: node_records,
        edges,
        node_flag_registry: node_flag_registry
            .into_iter()
            .map(FlagRecord::from_tuple)
            .collect(),
        edge_flag_registry: edge_flag_registry
            .into_iter()
            .map(FlagRecord::from_tuple)
            .collect(),
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
        paths,
        nodes,
        edges,
        node_flag_registry,
        edge_flag_registry,
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
        let path = paths
            .get(rec.path_idx as usize)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{path:?}: node path_idx {} out of range ({} paths)",
                    rec.path_idx,
                    paths.len()
                ))
            })?
            .clone();
        let node = SymbolNode {
            fqname: rec.fqname,
            kind,
            path,
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
        node_flag_registry: node_flag_registry
            .into_iter()
            .map(FlagRecord::into_tuple)
            .collect(),
        edge_flag_registry: edge_flag_registry
            .into_iter()
            .map(FlagRecord::into_tuple)
            .collect(),
    };
    Ok((Py::new(py, graph)?, Py::new(py, meta)?))
}
