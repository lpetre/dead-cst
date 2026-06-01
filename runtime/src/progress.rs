//! GIL-free progress counters shared between the rust build pipeline
//! and a Python-side polling thread.
//!
//! The rust side increments [`AtomicUsize`] / [`AtomicU64`] / [`AtomicBool`]
//! fields with [`Ordering::Relaxed`] — no GIL, no contention. A Python
//! polling thread reads a snapshot at ~100 ms cadence via
//! [`crate::project::ProjectContext::read_progress_snapshot`] and fires
//! structured callback events.
//!
//! Phase discriminants live in [`PHASE_*`] constants and are compared
//! to ``AtomicUsize::load`` on the Python side. Adding a new phase:
//! append a new ``PHASE_*`` constant, increment the matching counter
//! fields, and extend the Python-side dispatch in
//! ``Analysis.materialize_all``.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// "No phase active yet" — what the counters report before the build
/// pass calls [`ProgressCounters::start_phase`].
pub(crate) const PHASE_NONE: usize = 0;
/// File enumeration (`db.project().files(db)`).
pub(crate) const PHASE_ENUM: usize = 1;
/// Parallel rayon ingest (`file_to_nodes` / `file_to_edges` /
/// `file_to_ref_edges`).
pub(crate) const PHASE_POPULATE: usize = 2;
/// Serial fan-in (`assemble_graph`).
pub(crate) const PHASE_ASSEMBLE: usize = 3;
/// Post-assemble fqname-index build (`build_fqname_indices`).
pub(crate) const PHASE_FQNAME: usize = 4;
/// Per-plugin `run(ctx)` invocations driven from Python.
pub(crate) const PHASE_PLUGINS: usize = 5;

/// Shared atomic counters describing build progress.
///
/// Every field is GIL-free; rust workers (rayon, serial passes, plugin
/// driver) bump them with [`Ordering::Relaxed`]. The Python polling
/// thread reads them via [`Self::snapshot`].
///
/// `*_total` of `0` always means "unknown / not yet computed". Phase
/// totals are written exactly once at the start of each phase, before
/// the matching `*_done` counter starts moving.
pub(crate) struct ProgressCounters {
    /// Current phase discriminant (one of the `PHASE_*` constants).
    pub(crate) phase: AtomicUsize,
    /// `true` once the rust build finishes (success or error). Signals
    /// the Python polling thread to flush any pending `phase_end`
    /// events and exit.
    pub(crate) finished: AtomicBool,

    /// Files enumerated so far (cheap; one increment after the project
    /// scan completes).
    pub(crate) enum_files: AtomicUsize,
    /// Total files enumerated (0 = unknown until enum completes).
    pub(crate) enum_total: AtomicUsize,
    /// `Instant`-relative microseconds at which the enum phase started.
    pub(crate) enum_started_us: AtomicU64,
    /// Microseconds elapsed in the enum phase (set at phase end; 0
    /// while running).
    pub(crate) enum_elapsed_us: AtomicU64,

    /// Files whose populate-side salsa queries have completed.
    pub(crate) populate_done: AtomicUsize,
    /// Total files the populate phase will process (set at phase start).
    pub(crate) populate_total: AtomicUsize,
    pub(crate) populate_started_us: AtomicU64,
    pub(crate) populate_elapsed_us: AtomicU64,

    /// Files whose serial-assemble fold has completed.
    pub(crate) assemble_done: AtomicUsize,
    /// Total files the assemble phase will process.
    pub(crate) assemble_total: AtomicUsize,
    pub(crate) assemble_started_us: AtomicU64,
    pub(crate) assemble_elapsed_us: AtomicU64,

    /// Nodes the fqname-indices builder has folded so far.
    pub(crate) fqname_done: AtomicUsize,
    /// Total nodes the fqname-indices builder will fold.
    pub(crate) fqname_total: AtomicUsize,
    pub(crate) fqname_started_us: AtomicU64,
    pub(crate) fqname_elapsed_us: AtomicU64,

    /// Plugins whose `run(ctx)` returned.
    pub(crate) plugins_done: AtomicUsize,
    /// Total plugins registered.
    pub(crate) plugins_total: AtomicUsize,
    pub(crate) plugins_started_us: AtomicU64,
    pub(crate) plugins_elapsed_us: AtomicU64,

    /// Per-plugin slabs initialised once at the start of the plugin
    /// pass via [`Self::init_plugin_slots`]. Each plugin worker writes
    /// only to its own slot, so the parallel ``ThreadPoolExecutor``
    /// path can attribute completions to the actual plugin (not the
    /// "next-unseen plugin in registration order" approximation the
    /// global counter forces).
    pub(crate) plugin_slots: OnceLock<PluginSlots>,

    /// Process-wide monotonic clock anchor used to compute the
    /// `*_started_us` markers. Set at construction.
    start: Instant,
}

/// Pre-sized per-plugin counter slabs. The per-plugin rayon worker in
/// [`crate::project::ProjectContext::materialize`] stamps its own index
/// as it starts / finishes; the started / finished timestamps land in
/// the matching slot. All three vectors are the same length and indexed
/// by the plugin's registration order.
pub(crate) struct PluginSlots {
    /// Display names — ``type(plugin).__qualname__`` (or a fallback).
    /// Read-only after [`ProgressCounters::init_plugin_slots`] returns.
    pub(crate) names: Vec<String>,
    /// Microseconds (relative to [`ProgressCounters::start`]) at which
    /// the indexed plugin's ``run`` entered. `0` until stamped.
    pub(crate) started_us: Box<[AtomicU64]>,
    /// Microseconds at which the indexed plugin's ``run`` returned.
    /// `0` while still running.
    pub(crate) finished_us: Box<[AtomicU64]>,
}

impl ProgressCounters {
    pub(crate) fn new() -> Self {
        Self {
            phase: AtomicUsize::new(PHASE_NONE),
            finished: AtomicBool::new(false),
            enum_files: AtomicUsize::new(0),
            enum_total: AtomicUsize::new(0),
            enum_started_us: AtomicU64::new(0),
            enum_elapsed_us: AtomicU64::new(0),
            populate_done: AtomicUsize::new(0),
            populate_total: AtomicUsize::new(0),
            populate_started_us: AtomicU64::new(0),
            populate_elapsed_us: AtomicU64::new(0),
            assemble_done: AtomicUsize::new(0),
            assemble_total: AtomicUsize::new(0),
            assemble_started_us: AtomicU64::new(0),
            assemble_elapsed_us: AtomicU64::new(0),
            fqname_done: AtomicUsize::new(0),
            fqname_total: AtomicUsize::new(0),
            fqname_started_us: AtomicU64::new(0),
            fqname_elapsed_us: AtomicU64::new(0),
            plugins_done: AtomicUsize::new(0),
            plugins_total: AtomicUsize::new(0),
            plugins_started_us: AtomicU64::new(0),
            plugins_elapsed_us: AtomicU64::new(0),
            plugin_slots: OnceLock::new(),
            start: Instant::now(),
        }
    }

    /// Allocate the per-plugin counter slabs once. ``names`` is the
    /// plugin list in registration order; indices passed to
    /// [`Self::plugin_started`] / [`Self::plugin_finished`] match.
    /// Calls after the first are silently ignored (the
    /// [`OnceLock`] preserves the first set) — a [`ProjectContext`]
    /// drives at most one materialize pass, so this matches the
    /// expected lifecycle.
    pub(crate) fn init_plugin_slots(&self, names: Vec<String>) {
        let n = names.len();
        let _ = self.plugin_slots.set(PluginSlots {
            names,
            started_us: (0..n).map(|_| AtomicU64::new(0)).collect(),
            finished_us: (0..n).map(|_| AtomicU64::new(0)).collect(),
        });
    }

    /// Stamp the per-plugin start time. Called from the per-plugin
    /// rayon worker in [`crate::project::ProjectContext::materialize`]
    /// on entry. No-op if the slot wasn't pre-allocated (e.g. a caller
    /// that skipped [`Self::init_plugin_slots`]).
    pub(crate) fn plugin_started(&self, idx: usize) {
        if let Some(slots) = self.plugin_slots.get() {
            if let Some(cell) = slots.started_us.get(idx) {
                cell.store(self.now_us(), Ordering::Relaxed);
            }
        }
    }

    /// Stamp the per-plugin finish time. Called from the per-plugin
    /// rayon worker in [`crate::project::ProjectContext::materialize`]
    /// once the plugin's ``run`` returns (whether it succeeded or
    /// produced an error to be folded in later).
    pub(crate) fn plugin_finished(&self, idx: usize) {
        if let Some(slots) = self.plugin_slots.get() {
            if let Some(cell) = slots.finished_us.get(idx) {
                cell.store(self.now_us(), Ordering::Relaxed);
            }
        }
    }

    /// Microseconds since [`Self::new`].
    fn now_us(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    /// Stamp the current phase discriminant and (optionally) its total.
    /// `total` is `None` when the phase doesn't pre-compute a total.
    pub(crate) fn start_phase(&self, phase: usize, total: Option<usize>) {
        let now = self.now_us();
        match phase {
            PHASE_ENUM => self.enum_started_us.store(now, Ordering::Relaxed),
            PHASE_POPULATE => {
                if let Some(t) = total {
                    self.populate_total.store(t, Ordering::Relaxed);
                }
                self.populate_started_us.store(now, Ordering::Relaxed);
            }
            PHASE_ASSEMBLE => {
                if let Some(t) = total {
                    self.assemble_total.store(t, Ordering::Relaxed);
                }
                self.assemble_started_us.store(now, Ordering::Relaxed);
            }
            PHASE_FQNAME => {
                if let Some(t) = total {
                    self.fqname_total.store(t, Ordering::Relaxed);
                }
                self.fqname_started_us.store(now, Ordering::Relaxed);
            }
            PHASE_PLUGINS => {
                if let Some(t) = total {
                    self.plugins_total.store(t, Ordering::Relaxed);
                }
                self.plugins_started_us.store(now, Ordering::Relaxed);
            }
            _ => {}
        }
        self.phase.store(phase, Ordering::Relaxed);
    }

    /// Record the phase's elapsed time (microseconds). Called once at
    /// phase end. The Python polling thread reads this when it sees
    /// the phase discriminant advance.
    pub(crate) fn finish_phase(&self, phase: usize) {
        let now = self.now_us();
        let elapsed = |started: &AtomicU64| now.saturating_sub(started.load(Ordering::Relaxed));
        match phase {
            PHASE_ENUM => self
                .enum_elapsed_us
                .store(elapsed(&self.enum_started_us), Ordering::Relaxed),
            PHASE_POPULATE => self
                .populate_elapsed_us
                .store(elapsed(&self.populate_started_us), Ordering::Relaxed),
            PHASE_ASSEMBLE => self
                .assemble_elapsed_us
                .store(elapsed(&self.assemble_started_us), Ordering::Relaxed),
            PHASE_FQNAME => self
                .fqname_elapsed_us
                .store(elapsed(&self.fqname_started_us), Ordering::Relaxed),
            PHASE_PLUGINS => self
                .plugins_elapsed_us
                .store(elapsed(&self.plugins_started_us), Ordering::Relaxed),
            _ => {}
        }
    }

    /// Mark the whole pipeline finished — the polling thread will
    /// flush pending end events and exit on its next tick.
    pub(crate) fn mark_finished(&self) {
        self.finished.store(true, Ordering::Relaxed);
    }

    /// Atomic-load every counter into a plain-int tuple. Called from
    /// the Python polling thread under the GIL.
    ///
    /// The tuple order is the public contract between this rust
    /// helper and `Analysis._poll_progress` — do not reorder without
    /// updating both sides.
    pub(crate) fn snapshot(&self) -> ProgressSnapshot {
        let plugin_states = self
            .plugin_slots
            .get()
            .map(|slots| {
                slots
                    .names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| PluginState {
                        name: name.clone(),
                        started_us: slots.started_us[i].load(Ordering::Relaxed),
                        finished_us: slots.finished_us[i].load(Ordering::Relaxed),
                    })
                    .collect()
            })
            .unwrap_or_default();
        ProgressSnapshot {
            phase: self.phase.load(Ordering::Relaxed),
            finished: self.finished.load(Ordering::Relaxed),
            enum_done: self.enum_files.load(Ordering::Relaxed),
            enum_total: self.enum_total.load(Ordering::Relaxed),
            enum_elapsed_us: self.enum_elapsed_us.load(Ordering::Relaxed),
            populate_done: self.populate_done.load(Ordering::Relaxed),
            populate_total: self.populate_total.load(Ordering::Relaxed),
            populate_elapsed_us: self.populate_elapsed_us.load(Ordering::Relaxed),
            assemble_done: self.assemble_done.load(Ordering::Relaxed),
            assemble_total: self.assemble_total.load(Ordering::Relaxed),
            assemble_elapsed_us: self.assemble_elapsed_us.load(Ordering::Relaxed),
            fqname_done: self.fqname_done.load(Ordering::Relaxed),
            fqname_total: self.fqname_total.load(Ordering::Relaxed),
            fqname_elapsed_us: self.fqname_elapsed_us.load(Ordering::Relaxed),
            plugins_done: self.plugins_done.load(Ordering::Relaxed),
            plugins_total: self.plugins_total.load(Ordering::Relaxed),
            plugins_elapsed_us: self.plugins_elapsed_us.load(Ordering::Relaxed),
            plugin_states,
        }
    }

    /// Stash the file-enumeration total (and bump `enum_files` to
    /// match, since enumeration is atomic from the polling-thread's
    /// perspective).
    pub(crate) fn set_enum_total(&self, total: usize) {
        self.enum_total.store(total, Ordering::Relaxed);
        self.enum_files.store(total, Ordering::Relaxed);
    }

    /// Bump the populate-files-done counter. Called from rayon workers
    /// after each file's salsa queries return. Relaxed ordering is
    /// fine — the Python side only reads the running count.
    pub(crate) fn populate_inc(&self) {
        self.populate_done.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the assemble-files-done counter.
    pub(crate) fn assemble_inc(&self) {
        self.assemble_done.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the fqname-nodes-done counter.
    pub(crate) fn fqname_inc(&self) {
        self.fqname_done.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the plugins-done counter.
    pub(crate) fn plugins_inc(&self) {
        self.plugins_done.fetch_add(1, Ordering::Relaxed);
    }
}

/// Plain-data snapshot of [`ProgressCounters`] — what
/// [`crate::project::ProjectContext::read_progress_snapshot`] returns
/// to Python. All fields are `usize` / `u64` / `bool` so the pyo3
/// conversion is a cheap memcpy.
#[derive(Clone, Debug)]
pub(crate) struct ProgressSnapshot {
    pub(crate) phase: usize,
    pub(crate) finished: bool,
    pub(crate) enum_done: usize,
    pub(crate) enum_total: usize,
    pub(crate) enum_elapsed_us: u64,
    pub(crate) populate_done: usize,
    pub(crate) populate_total: usize,
    pub(crate) populate_elapsed_us: u64,
    pub(crate) assemble_done: usize,
    pub(crate) assemble_total: usize,
    pub(crate) assemble_elapsed_us: u64,
    pub(crate) fqname_done: usize,
    pub(crate) fqname_total: usize,
    pub(crate) fqname_elapsed_us: u64,
    pub(crate) plugins_done: usize,
    pub(crate) plugins_total: usize,
    pub(crate) plugins_elapsed_us: u64,
    /// Per-plugin slot snapshot in registration order. Empty when
    /// [`ProgressCounters::init_plugin_slots`] hasn't been called
    /// (e.g. mid-build during a phase that precedes plugins).
    pub(crate) plugin_states: Vec<PluginState>,
}

/// One plugin's slot at snapshot time. ``started_us == 0`` ⇔ not yet
/// running; ``finished_us == 0`` ⇔ still running (or never ran).
#[derive(Clone, Debug)]
pub(crate) struct PluginState {
    pub(crate) name: String,
    pub(crate) started_us: u64,
    pub(crate) finished_us: u64,
}

impl ProgressSnapshot {
    /// Convert the snapshot to a Python dict. The key set is the
    /// public contract that ``Analysis._ProgressPoller`` reads.
    pub(crate) fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("phase", self.phase)?;
        dict.set_item("finished", self.finished)?;
        dict.set_item("enum_done", self.enum_done)?;
        dict.set_item("enum_total", self.enum_total)?;
        dict.set_item("enum_elapsed_us", self.enum_elapsed_us)?;
        dict.set_item("populate_done", self.populate_done)?;
        dict.set_item("populate_total", self.populate_total)?;
        dict.set_item("populate_elapsed_us", self.populate_elapsed_us)?;
        dict.set_item("assemble_done", self.assemble_done)?;
        dict.set_item("assemble_total", self.assemble_total)?;
        dict.set_item("assemble_elapsed_us", self.assemble_elapsed_us)?;
        dict.set_item("fqname_done", self.fqname_done)?;
        dict.set_item("fqname_total", self.fqname_total)?;
        dict.set_item("fqname_elapsed_us", self.fqname_elapsed_us)?;
        dict.set_item("plugins_done", self.plugins_done)?;
        dict.set_item("plugins_total", self.plugins_total)?;
        dict.set_item("plugins_elapsed_us", self.plugins_elapsed_us)?;
        // Per-plugin slot snapshot. ``[(name, started_us, finished_us), ...]``
        // in registration order. The polling thread uses these to fire
        // ``plugin_start`` / ``plugin_end`` with accurate attribution
        // when plugins run concurrently.
        let plugin_states: Vec<(String, u64, u64)> = self
            .plugin_states
            .iter()
            .map(|s| (s.name.clone(), s.started_us, s.finished_us))
            .collect();
        dict.set_item("plugin_states", plugin_states)?;
        Ok(dict)
    }
}

/// Python-visible handle over the [`Arc<ProgressCounters>`] live in
/// the rust pipeline. The polling thread reads this handle's
/// :meth:`snapshot` without touching the parent
/// :class:`crate::project::ProjectContext`'s pyo3 borrow flag, so it
/// can poll concurrently with a long-running ``materialize`` call
/// that holds the context's ``borrow_mut`` token.
///
/// Cloning the handle is cheap — it's just an ``Arc::clone``. The
/// counters live as long as either the originating ``ProjectContext``
/// or any outstanding handle.
#[pyclass(name = "ProgressHandle", module = "dead_cst._native")]
pub(crate) struct ProgressHandle {
    pub(crate) counters: Arc<ProgressCounters>,
}

#[pymethods]
impl ProgressHandle {
    /// Atomic snapshot of every counter as a plain Python dict.
    /// Each value is a non-negative integer (`bool` for ``finished``).
    pub(crate) fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.counters.snapshot().to_pydict(py)
    }
}
