//! Public Python data classes (`Import`, `SymbolNode`, `NativeGraph`),
//! the type aliases that describe graph indices, and the `NodeFlags`/
//! `EdgeFlags` constant catalogs exposed to plugins.
//!
//! See `src/CLAUDE.md` for the architectural rules these types follow.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use ruff_db::files::File;
use ty_python_core::place::ScopedPlaceId;

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
pub(crate) struct Import {
    pub(crate) module: String,
    pub(crate) decl: Option<String>,
    pub(crate) star: bool,
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

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.module.hash(&mut hasher);
        self.decl.hash(&mut hasher);
        self.star.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<Import>>() {
            Ok(other) => {
                self.module == other.module && self.decl == other.decl && self.star == other.star
            }
            Err(_) => false,
        }
    }
}

/// A single node in a `NativeGraph`.
///
/// `imports` is populated for `kind="import"` nodes only (one per
/// alias, plus one per name brought in by `from X import *`). All
/// other kinds carry `None`.
///
/// `cached_hash` memoizes `__hash__` so repeated hashing (graph
/// builds, set membership, codemod walks) pays the full SipHash cost
/// once per node instead of on every call. `OnceLock<u64>` is the
/// minimal `Send + Sync` cell that's compatible with `#[pyclass(frozen)]`.
#[pyclass(frozen)]
pub(crate) struct SymbolNode {
    #[pyo3(get)]
    pub(crate) fqname: String,
    #[pyo3(get)]
    pub(crate) kind: &'static str,
    #[pyo3(get)]
    pub(crate) path: String,
    #[pyo3(get)]
    pub(crate) start_line: usize,
    #[pyo3(get)]
    pub(crate) start_column: usize,
    #[pyo3(get)]
    pub(crate) end_line: usize,
    #[pyo3(get)]
    pub(crate) end_column: usize,
    #[pyo3(get)]
    pub(crate) flags: u32,
    #[pyo3(get)]
    pub(crate) imports: Option<Py<Import>>,
    pub(crate) cached_hash: OnceLock<u64>,
}

const VALID_KINDS: &[&str] = &[
    "function",
    "class",
    "variable",
    "import",
    "type_alias",
    "module",
    "synthetic",
];

pub(crate) fn intern_kind(kind: &str) -> PyResult<&'static str> {
    VALID_KINDS
        .iter()
        .find(|&&v| v == kind)
        .copied()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid SymbolNode.kind: {kind:?} — expected one of {VALID_KINDS:?}"
            ))
        })
}

#[pymethods]
impl SymbolNode {
    #[new]
    #[pyo3(signature = (
        fqname,
        kind,
        path,
        *,
        start_line = 0,
        start_column = 0,
        end_line = 0,
        end_column = 0,
        flags = 0,
        imports = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fqname: String,
        kind: &str,
        path: String,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
        flags: u32,
        imports: Option<Py<Import>>,
    ) -> PyResult<Self> {
        Ok(Self {
            fqname,
            kind: intern_kind(kind)?,
            path,
            start_line,
            start_column,
            end_line,
            end_column,
            flags,
            imports,
            cached_hash: OnceLock::new(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SymbolNode(fqname={:?}, kind={:?}, path={:?}, start=({}, {}), end=({}, {}), flags={})",
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

    fn __hash__(&self, py: Python<'_>) -> u64 {
        *self.cached_hash.get_or_init(|| {
            let mut hasher = DefaultHasher::new();
            self.fqname.hash(&mut hasher);
            self.kind.hash(&mut hasher);
            self.path.hash(&mut hasher);
            self.start_line.hash(&mut hasher);
            self.start_column.hash(&mut hasher);
            self.end_line.hash(&mut hasher);
            self.end_column.hash(&mut hasher);
            self.flags.hash(&mut hasher);
            if let Some(imp) = &self.imports {
                let imp_ref = imp.borrow(py);
                imp_ref.module.hash(&mut hasher);
                imp_ref.decl.hash(&mut hasher);
                imp_ref.star.hash(&mut hasher);
            } else {
                // Discriminate "no imports" from "imports = ()".
                0u8.hash(&mut hasher);
            }
            hasher.finish()
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.extract::<PyRef<SymbolNode>>() else {
            return false;
        };
        if self.fqname != other.fqname
            || self.kind != other.kind
            || self.path != other.path
            || self.start_line != other.start_line
            || self.start_column != other.start_column
            || self.end_line != other.end_line
            || self.end_column != other.end_column
            || self.flags != other.flags
        {
            return false;
        }
        let py = other.py();
        match (&self.imports, &other.imports) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                let a = a.borrow(py);
                let b = b.borrow(py);
                a.module == b.module && a.decl == b.decl && a.star == b.star
            }
            _ => false,
        }
    }
}

/// One project-wide graph contribution, packed for one FFI hop.
#[pyclass(get_all, frozen)]
pub(crate) struct NativeGraph {
    pub(crate) nodes: Vec<Py<SymbolNode>>,
    pub(crate) edges: Vec<(usize, usize, u32)>,
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
pub(crate) type DeclKey = (File, ScopedPlaceId, (u32, u32));

pub(crate) type DeclIndex = HashMap<DeclKey, usize>;

/// Return type of `ProjectContext.find_main_blocks`: one entry per
/// file with a top-level ``if __name__ == "__main__":`` block, paired
/// with the decls that fall inside it.
pub(crate) type MainBlock = (Py<SymbolNode>, Vec<Py<SymbolNode>>);

/// Bit values stamped into [`SymbolNode::flags`]. Mirrors
/// `dead_cst.graph.NodeFlags` exactly so plugin code can mix
/// rust-emitted and libcst-emitted nodes.
///
/// Exposed as Python class attributes — the classattr values are plain
/// `int`s so `NodeFlags.ENTRYPOINT | NodeFlags.NOQA` works the same as
/// the libcst `IntFlag` (just without the `.name` / `.value` surface).
#[pyclass(frozen)]
pub(crate) struct NodeFlags;

#[pymethods]
impl NodeFlags {
    #[classattr]
    pub(crate) const NONE: u32 = 0;
    /// Decl rebound by a later assignment in the same file. Kept in the
    /// graph (with its parent-module edge) but excluded from the
    /// cross-module lookup so consumers of an exported name route to
    /// the live binding.
    #[classattr]
    pub(crate) const SHADOWED: u32 = 1;
    /// Explicit entrypoint — plugin-emitted seeds, the CLI's `-e`
    /// flag, `[project.scripts]` targets, factory-app synthetics, etc.
    /// One of the keepalive bits ORed into `KEEPALIVE_DEFAULT` on the
    /// Python side, so reachability seeds from `ENTRYPOINT`-flagged
    /// nodes by default.
    #[classattr]
    pub(crate) const ENTRYPOINT: u32 = 2;
    /// `typing.overload` stub (or any same-name decl whose lifetime is
    /// anchored to a matching impl). Excluded from the lookup trie like
    /// `SHADOWED`; kept alive by an explicit `impl -> overload` edge.
    #[classattr]
    pub(crate) const OVERLOAD: u32 = 4;
    /// Pytest / unittest test discoveries. One of the keepalive bits in
    /// `KEEPALIVE_DEFAULT`, so tests are alive by default. The
    /// `kept_alive_by_flags_only(TESTCASE)` blast-radius query isolates
    /// "what's only alive because of tests" by computing the diff
    /// against `reachable(seed_flags=KEEPALIVE_DEFAULT & ~TESTCASE)`.
    #[classattr]
    pub(crate) const TESTCASE: u32 = 8;
    /// Import alias preserved by a user noqa directive (bare `# noqa`,
    /// `# noqa: F401`, multi-rule `# noqa: E501, F401`, or the
    /// file-level `# ruff: noqa` / `# flake8: noqa`). One of the
    /// keepalive bits in `KEEPALIVE_DEFAULT`.
    #[classattr]
    pub(crate) const NOQA: u32 = 16;
    /// Every node sourced from a Jupyter `.ipynb` file. Cells run
    /// top-to-bottom rather than being imported, so the bit alone keeps
    /// the node alive (no `ENTRYPOINT` overlay needed — `NOTEBOOK` is in
    /// `KEEPALIVE_DEFAULT`). The codemod also reads the bit to skip
    /// notebook nodes (it can't rewrite the cell JSON envelope).
    #[classattr]
    pub(crate) const NOTEBOOK: u32 = 32;
    /// Every node sourced from a file under the package's `exported`
    /// glob. Used by the cross-package merge to filter to entries the
    /// owning package opts into exposing.
    #[classattr]
    pub(crate) const EXPORTED: u32 = 64;
    /// Import decl synthesized from `from X import *` — one per name
    /// the star statement brought in. Set so the cross-module trie can
    /// distinguish "real" import aliases from per-name star fan-out.
    #[classattr]
    pub(crate) const STAR_REEXPORT: u32 = 128;
}

/// Bit values stamped into the third tuple slot of each `NativeGraph`
/// edge. Mirrors `dead_cst.graph.EdgeFlags`.
#[pyclass(frozen)]
pub(crate) struct EdgeFlags;

#[pymethods]
impl EdgeFlags {
    #[classattr]
    pub(crate) const NONE: u32 = 0;
    /// Reference originated inside a statically-dead region (the body of
    /// `if False:`, the else of `if True:`, after an unconditional
    /// `return` / `raise` / `break` / `continue`, …). Metadata only —
    /// the edge still participates in default reachability; pass to
    /// `descendants(..., skip_flags=EdgeFlags.DEAD_BRANCH)` to compute
    /// the kept-alive-only-by-dead-branches set.
    #[classattr]
    pub(crate) const DEAD_BRANCH: u32 = 1;
    /// Edge emitted from a runtime-import call (`__import__('X')` /
    /// `importlib.import_module('X')`). Lets plugins read which edges
    /// the visitor produced from dynamic-import shapes and choose to
    /// fan out / specialize.
    #[classattr]
    pub(crate) const DYNAMIC_IMPORT: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_kind_accepts_every_known_kind() {
        for kind in [
            "function",
            "class",
            "variable",
            "import",
            "type_alias",
            "module",
            "synthetic",
        ] {
            let interned = intern_kind(kind).expect("known kinds must intern");
            // The returned string is the static slice — pointer-equal to
            // the entry in VALID_KINDS — so equality on bytes is enough.
            assert_eq!(interned, kind);
        }
    }

    #[test]
    fn intern_kind_rejects_unknown_kinds() {
        // pyo3 PyResult requires GIL to inspect the err; we can still
        // assert that the call returns Err without a Python interpreter
        // attached — the error construction itself doesn't touch the GIL
        // (only message formatting does).
        assert!(intern_kind("").is_err());
        assert!(intern_kind("Function").is_err()); // case-sensitive
        assert!(intern_kind("functio").is_err());
        assert!(intern_kind("functions").is_err());
        assert!(intern_kind("decorator").is_err());
    }

    #[test]
    fn node_flags_constants_are_distinct_bits() {
        // Each flag should be a single distinct bit (or zero for NONE).
        let flags = [
            NodeFlags::SHADOWED,
            NodeFlags::ENTRYPOINT,
            NodeFlags::OVERLOAD,
            NodeFlags::TESTCASE,
            NodeFlags::NOQA,
            NodeFlags::NOTEBOOK,
            NodeFlags::EXPORTED,
            NodeFlags::STAR_REEXPORT,
        ];
        assert_eq!(NodeFlags::NONE, 0);
        for &f in &flags {
            // Single-bit invariant.
            assert!(f.is_power_of_two(), "flag {f} is not a single bit");
        }
        // All distinct.
        let mut seen = std::collections::HashSet::new();
        for &f in &flags {
            assert!(seen.insert(f), "duplicate flag value {f}");
        }
    }

    #[test]
    fn node_flags_combine_via_bitwise_or() {
        let combo = NodeFlags::ENTRYPOINT | NodeFlags::NOQA;
        assert!(combo & NodeFlags::ENTRYPOINT != 0);
        assert!(combo & NodeFlags::NOQA != 0);
        assert!(combo & NodeFlags::TESTCASE == 0);
    }

    #[test]
    fn edge_flags_are_distinct_bits() {
        assert_eq!(EdgeFlags::NONE, 0);
        assert!(EdgeFlags::DEAD_BRANCH.is_power_of_two());
        assert!(EdgeFlags::DYNAMIC_IMPORT.is_power_of_two());
        assert_ne!(EdgeFlags::DEAD_BRANCH, EdgeFlags::DYNAMIC_IMPORT);
    }
}
