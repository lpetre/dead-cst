//! Native (rust-side) plugins — in-tree only.
//!
//! A native plugin is a rust implementation of the plugin contract
//! that skips the Python `.run(ctx)` call entirely: the harness
//! detects a :class:`NativePlugin` pyclass via downcast in
//! ``collect_prepared_plugin_ops``, invokes the inner
//! [`NativePluginImpl::run`] directly, and the impl pushes its
//! ops into the shared [`PreparedOp`] sink — no Python ``GraphOp``
//! instances are ever constructed, no per-op ``extract`` /
//! ``prepare_graph_op`` extraction step.
//!
//! **Scope.** Native plugins are an in-tree fast-path for bundled
//! plugins whose logic is fixed and hot. They're not a public
//! extension mechanism: the trait and its dependent types
//! ([`PreparedOp`], [`ProjectContext`]) are `pub(crate)` by design,
//! the `dead-cst-native` crate is not published to crates.io, and
//! the rust API has no stability commitment. Out-of-tree plugin
//! authors continue to use the Python :class:`Plugin` protocol —
//! they can still write hot code in rust, but they ship it as a
//! pyo3 extension that their `run(ctx)` body calls into, and they
//! emit ops via the public Python ``AddNodeByIdx`` / ``AddEdgeByIdx``
//! / ``AddEntrypointByIdx`` graph ops.
//!
//! Trade-off vs the Python-side plugin path (for in-tree authors):
//!
//! * **Lighter per-op cost** — every Python plugin op pays one
//!   ``Py::new(AddNodeByIdx { ... })`` allocation on the yield side,
//!   then one ``extract::<PyRef<...>>`` + ``.clone()`` of the inner
//!   fields on the harness side. Native plugins skip both ends.
//! * **No GIL release between op yields** — the impl runs entirely
//!   in rust under one GIL hold; rust queries inside the impl can
//!   still drop the GIL via ``py.allow_threads`` if they want to.
//! * **Costs flexibility** — a native plugin's logic is compiled
//!   into the wheel; it can't be authored / overridden / dataclass-
//!   configured externally. Python plugins remain the right home
//!   for anything user-configurable.
//!
//! The frozen-graph contract is identical on both paths: the impl
//! observes the base graph mid-pass, emits ops, ops apply in a
//! single batch after every plugin completes. Native plugins are
//! drop-in interchangeable with Python ones in
//! ``Analysis(plugins=[...])`` and inside the harness's
//! :class:`ThreadPoolExecutor` fan-out.

use pyo3::prelude::*;

use crate::builder::PreparedOp;
use crate::graph::{intern_kind, NodeFlags};
use crate::project::ProjectContext;

/// Rust-side plugin contract. Each impl describes how to derive ops
/// from a frozen :class:`ProjectContext` and push them into the
/// shared [`PreparedOp`] sink the harness drains at the end of the
/// plugin pass.
///
/// Implementations don't construct Python ``GraphOp`` instances —
/// they push pure-rust [`PreparedOp`] variants directly, skipping
/// the prepare-from-Python round-trip. No ``Python<'_>`` parameter
/// either: every ctx accessor a native plugin needs (``node_attrs``,
/// ``find_main_blocks_indices``, the query DSL helpers) is GIL-free,
/// reading ``Sync`` ``#[pyclass(frozen)]`` data via ``Py::get`` rather
/// than ``Py::borrow``.
pub(crate) trait NativePluginImpl: Send + Sync {
    /// Human-readable name surfaced by ``NativePlugin.name`` and used
    /// for progress reporting (``plugin_start`` / ``plugin_end``
    /// events). Should match the conventional name of the equivalent
    /// Python plugin so existing harness logs read the same.
    fn name(&self) -> &'static str;

    /// Walk the frozen ``ctx`` and append the plugin's ops to
    /// ``sink``. Same frozen-graph contract as the Python path: the
    /// impl observes the base graph only; its emissions are folded in
    /// by the apply pass after every plugin returns.
    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()>;
}

/// Python-visible wrapper for a [`NativePluginImpl`]. Constructed via
/// static factories (one per concrete impl, e.g.
/// :meth:`NativePlugin.main_block`); ``__init__`` is intentionally
/// unsupported so plugin authors discover the existing factories
/// (and we keep the configuration surface inside the wrapper rust-
/// side).
///
/// Sendable because the inner trait bound (``Send + Sync``) carries
/// across: the harness drives plugins from a
/// :class:`ThreadPoolExecutor`, so the ``Py<NativePlugin>`` handle
/// shuttles between threads alongside Python plugins. The impl's
/// :meth:`run` is invoked under the GIL on whichever worker picks
/// up the task.
#[pyclass(name = "NativePlugin")]
pub(crate) struct NativePlugin {
    pub(crate) inner: Box<dyn NativePluginImpl>,
}

#[pymethods]
impl NativePlugin {
    /// Plugin name. Matches the conventional name of the equivalent
    /// Python plugin (e.g. ``"MainBlockPlugin"``), so harness logs
    /// and ``progress_callback`` events look the same whether a
    /// native or Python instance is registered.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// ``Plugin`` protocol's ``prepare(project_root)`` no-op hook.
    /// Native plugins don't have a pre-graph prepare phase today —
    /// every impl reads from the frozen ctx in :meth:`run`.
    fn prepare(&self, _project_root: PyObject) {}

    /// Construct the native [``MainBlockPlugin``](
    /// crate::native_plugins::MainBlockPluginImpl) — same observable
    /// behaviour as ``dead_cst.plugins.MainBlockPlugin``, no Python
    /// loop or ``AddNodeByIdx`` allocation in the hot path.
    #[staticmethod]
    fn main_block() -> Self {
        Self {
            inner: Box::new(MainBlockPluginImpl),
        }
    }
}

// ---------------------------------------------------------------------------
// MainBlockPlugin — first native impl. Equivalent to
// ``dead_cst.plugins.main_block.MainBlockPlugin``: emit one synthetic
// ``<__main__>:<module_fqname>`` entrypoint per file with a top-level
// ``if __name__ == "__main__":`` block, with edges to the containing
// module and to every top-level decl inside the block.
// ---------------------------------------------------------------------------

const MAIN_BLOCK_PREFIX: &str = "<__main__>:";

pub(crate) struct MainBlockPluginImpl;

impl NativePluginImpl for MainBlockPluginImpl {
    fn name(&self) -> &'static str {
        "MainBlockPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        let pairs = ctx.find_main_blocks_indices()?;
        if pairs.is_empty() {
            return Ok(());
        }
        // One batched ``node_attrs`` for every matched module — same
        // shape the Python ``MainBlockPlugin`` uses, but without the
        // ``Py<NodeAttrs>`` round-trip: we read straight off the
        // ``Vec<NodeAttrs>`` the rust helper returns. Note: no ``py``
        // argument — ``node_attrs`` reads via ``Py::get`` (frozen-Sync
        // pyclass fast path).
        let module_idxs: Vec<usize> = pairs.iter().map(|(m, _)| *m).collect();
        let attrs = ctx.node_attrs(module_idxs)?;
        let synthetic_kind = intern_kind("synthetic")?;
        for ((module_idx, decl_idxs), attr) in pairs.iter().zip(attrs.iter()) {
            let mut edges_to_idx: Vec<usize> = Vec::with_capacity(decl_idxs.len() + 1);
            edges_to_idx.push(*module_idx);
            edges_to_idx.extend(decl_idxs);
            sink.push(PreparedOp::NodeByIdx {
                fqname: format!("{MAIN_BLOCK_PREFIX}{}", attr.fqname),
                kind: synthetic_kind,
                path: attr.path.clone(),
                flags: NodeFlags::ENTRYPOINT,
                edges_from_idx: Vec::new(),
                edges_to_idx,
            });
        }
        Ok(())
    }
}
