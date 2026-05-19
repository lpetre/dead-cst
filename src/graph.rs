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
    for valid in VALID_KINDS {
        if *valid == kind {
            return Ok(*valid);
        }
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "invalid SymbolNode.kind: {kind:?}"
    )))
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

/// Cloneable snapshot of an alias's import payload.
///
/// Lives in `alias_imports` (alias_node_idx -> spec) so the reference
/// collector can emit *parallel reachability edges* through an alias
/// without holding a PyO3 reference. Mirrors `Import`'s three fields.
#[derive(Clone, Debug)]
pub(crate) struct ImportSpec {
    pub(crate) module: String,
    pub(crate) decl: Option<String>,
    pub(crate) star: bool,
}

/// (upstream_file, local_decl_name) -> upstream decl node idx.
///
/// Populated during `ingest_decls` for every non-import, non-module
/// node. The value is a `Vec` so branch-bound names (try/except,
/// if/else where both branches assign) keep every live binding;
/// sequentially-rebound names collapse to the latest via the
/// post-pass that populates this from ty's
/// `end_of_scope_symbol_bindings`. Mirrors how the libcst pipeline's
/// trie excludes `SHADOWED` decls but keeps multi-branch ones.
pub(crate) type LiveDeclIndex = HashMap<(File, String), Vec<usize>>;

/// (file, name) -> idx of *any* live module-scope binding, decl or
/// import alias. Last-write-wins like `LiveDeclIndex`.
///
/// Used by `resolve_from_imported` so it can shortcut ty's full
/// `definitions_for_imported_symbol` walk (which recursively chases
/// alias chains across files) into a single hashmap probe.
///
/// The value is a `Vec` so that branch-bound names (try/except, if/else
/// where both branches assign) keep every live binding instead of
/// the last-write only. A cross-module `from lib import f` then
/// resolves to every reaching def of `f` in `lib`, matching the
/// libcst pipeline's `SHADOWED`-excluding trie merge over multiple
/// bindings.
pub(crate) type GlobalsByName = HashMap<(File, String), Vec<usize>>;

/// (file, name) -> upstream module name when `name` in `file` is bound
/// by `from <upstream> import *`. Populated alongside `GlobalsByName`
/// during Phase 1.
///
/// Lets `resolve_from_imported` walk a star-reexport chain
/// `A → B → C` and emit `consumer → C.g` directly when `from A import g`
/// resolves through the chain, matching the libcst pipeline's
/// fixed-point trie merge. Cycle-safe via a `seen` set in the caller.
pub(crate) type StarReexports = HashMap<(File, String), String>;

/// Return type of `ProjectContext.find_main_blocks`: one entry per
/// file with a top-level ``if __name__ == "__main__":`` block, paired
/// with the decls that fall inside it.
pub(crate) type MainBlock = (Py<SymbolNode>, Vec<Py<SymbolNode>>);

/// Outcome of resolving a `Name` use to its reaching definition.
///
/// `Alias` is the module-scope path: the use has a local graph node
/// (an import alias or a top-level decl) that takes the in-edge.
/// `NestedImport` is the function-/class-scope path: ty saw an import
/// binding in a non-global scope, so no graph node was minted, and
/// the use's parallel upstream edges flow from the enclosing top-level
/// owner instead.
pub(crate) enum Resolution {
    Alias(usize),
    NestedImport {
        spec: ImportSpec,
        bound_name: String,
    },
}

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
