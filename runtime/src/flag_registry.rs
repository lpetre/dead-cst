//! Node- and edge-flag registries: one [`FlagSpec`] mechanism with two
//! registration paths.
//!
//! Every flag — engine built-ins included — is described by a [`FlagSpec`].
//! The engine registers first via [`FlagRegistry::register_engine`] with an
//! **explicit mask** (the `graph.rs` const, so hot-path `flags & CONST` reads
//! still constant-fold). Plugins register via
//! [`FlagRegistry::register_plugin`] **without** a mask; the host allocates
//! the next free bit **above the last engine mask**, in plugin registration
//! order — a pure function of that order, hence deterministic.
//!
//! [`FlagSpec`] is the only author-facing type (re-exported from
//! `plugin_api`); it is pyo3-free. [`FlagRegistry`] is host-internal and
//! never crosses the plugin airlock — plugins read a resolved bit through
//! `PluginCtx::node_flag` / `edge_flag`, which hand back a plain `Option`.

use rustc_hash::FxHashMap;

use crate::graph::{BUILTIN_EDGE_FLAGS, BUILTIN_NODE_FLAGS};
use crate::native_plugins::plugin_api::PluginError;

/// Declarative description of one flag. Carries no bit value: engine flags
/// pass their mask separately to [`FlagRegistry::register_engine`], and
/// plugin flags are assigned a bit by the host.
///
/// `name` is `owner/name` namespaced; the `engine/` owner is reserved for
/// built-ins. `seed` marks a flag that seeds reachability; `default_on`
/// (only meaningful with `seed`) puts it in the default keepalive mask.
#[derive(Debug, Clone)]
pub struct FlagSpec {
    pub name: String,
    pub seed: bool,
    pub default_on: bool,
    pub description: String,
}

/// A name↔bit registry over a fixed bit width (32 for node flags, 8 for edge
/// flags). Seeded with the engine built-ins, then frozen after the plugin
/// declaration pass and lent read-only into the parallel plugin run.
pub(crate) struct FlagRegistry {
    width_bits: u32,
    by_name: FxHashMap<String, u64>,
    /// `(bit, spec)` in registration order (engine ascending, then plugins
    /// ascending — already bit-sorted, but [`entries`](Self::entries) sorts
    /// defensively).
    specs: Vec<(u64, FlagSpec)>,
    /// Lowest bit a plugin flag may take: one above the highest engine mask,
    /// advanced as plugin flags are allocated.
    next_plugin_bit: u64,
}

impl FlagRegistry {
    fn new(width_bits: u32) -> Self {
        Self {
            width_bits,
            by_name: FxHashMap::default(),
            specs: Vec::new(),
            next_plugin_bit: 0,
        }
    }

    /// A node-flag registry (`u32`) seeded with the engine node built-ins.
    pub(crate) fn with_node_builtins() -> Self {
        Self::seeded(32, BUILTIN_NODE_FLAGS)
    }

    /// An edge-flag registry (`u8`) seeded with the engine edge built-ins.
    pub(crate) fn with_edge_builtins() -> Self {
        Self::seeded(8, BUILTIN_EDGE_FLAGS)
    }

    fn seeded(width_bits: u32, builtins: &[(u64, &str, bool, bool, &str)]) -> Self {
        let mut reg = Self::new(width_bits);
        for &(mask, name, seed, default_on, description) in builtins {
            reg.register_engine(
                mask,
                FlagSpec {
                    name: name.to_string(),
                    seed,
                    default_on,
                    description: description.to_string(),
                },
            )
            .expect("built-in flag table must be self-consistent");
        }
        reg
    }

    /// Register an engine flag with its explicit (constant-folded) mask.
    /// Crate-internal; the engine owns the `engine/` namespace.
    pub(crate) fn register_engine(&mut self, mask: u64, spec: FlagSpec) -> Result<(), PluginError> {
        debug_assert!(
            mask.is_power_of_two() && mask < (1u64 << self.width_bits),
            "engine flag {:?} mask {mask:#x} must be a single in-range bit",
            spec.name,
        );
        if self.by_name.contains_key(&spec.name) {
            return Err(PluginError::value(format!(
                "duplicate engine flag {:?}",
                spec.name
            )));
        }
        self.by_name.insert(spec.name.clone(), mask);
        self.specs.push((mask, spec));
        let next = mask << 1;
        if next > self.next_plugin_bit {
            self.next_plugin_bit = next;
        }
        Ok(())
    }

    /// Register a plugin flag, allocating the next free bit above the last
    /// engine mask. Idempotent on an identical spec (so a flag two plugins
    /// share — e.g. `test/testcase` — collapses to one bit); a conflicting
    /// re-registration or an `engine/`-prefixed name fails loudly, as does
    /// exhausting the bit width.
    pub(crate) fn register_plugin(&mut self, spec: FlagSpec) -> Result<u64, PluginError> {
        if spec.name.starts_with("engine/") {
            return Err(PluginError::value(format!(
                "plugin flag {:?}: the 'engine/' namespace is reserved for built-in flags",
                spec.name
            )));
        }
        if let Some(&bit) = self.by_name.get(&spec.name) {
            let existing = &self
                .specs
                .iter()
                .find(|(b, _)| *b == bit)
                .expect("by_name bit must have a spec")
                .1;
            if existing.seed != spec.seed
                || existing.default_on != spec.default_on
                || existing.description != spec.description
            {
                return Err(PluginError::value(format!(
                    "flag {:?} re-registered with a conflicting spec",
                    spec.name
                )));
            }
            return Ok(bit);
        }
        let bit = self.next_plugin_bit;
        if bit == 0 || bit >= (1u64 << self.width_bits) {
            return Err(PluginError::value(format!(
                "flag registry exhausted: no free bit for {:?} within {} bits",
                spec.name, self.width_bits
            )));
        }
        self.by_name.insert(spec.name.clone(), bit);
        self.specs.push((bit, spec));
        self.next_plugin_bit <<= 1;
        Ok(bit)
    }

    /// The bit for `name`, if registered.
    pub(crate) fn get(&self, name: &str) -> Option<u64> {
        self.by_name.get(name).copied()
    }

    /// All `(bit, spec)` entries, sorted by bit — for serialization and the
    /// `ProjectContext` getters.
    pub(crate) fn entries(&self) -> Vec<(u64, FlagSpec)> {
        let mut out = self.specs.clone();
        out.sort_by_key(|(bit, _)| *bit);
        out
    }

    /// OR of every bit whose flag both seeds reachability and is on by
    /// default — the default keepalive mask the Python layer consumes.
    pub(crate) fn default_seed_mask(&self) -> u64 {
        self.specs
            .iter()
            .filter(|(_, s)| s.seed && s.default_on)
            .map(|(bit, _)| *bit)
            .fold(0, |acc, bit| acc | bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, seed: bool, default_on: bool) -> FlagSpec {
        FlagSpec {
            name: name.to_string(),
            seed,
            default_on,
            description: String::new(),
        }
    }

    #[test]
    fn node_builtins_seed_engine_bits() {
        let reg = FlagRegistry::with_node_builtins();
        assert_eq!(reg.get("engine/entrypoint"), Some(2));
        assert_eq!(reg.get("engine/star_reexport"), Some(128));
        // entries are bit-sorted and all engine-namespaced.
        let entries = reg.entries();
        assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(entries.iter().all(|(_, s)| s.name.starts_with("engine/")));
    }

    #[test]
    fn plugins_allocate_above_last_engine_mask() {
        let mut reg = FlagRegistry::with_node_builtins();
        // Highest engine node mask is 128, so the first plugin bit is 256.
        assert_eq!(
            reg.register_plugin(spec("acme/a", false, false)).unwrap(),
            256
        );
        assert_eq!(
            reg.register_plugin(spec("acme/b", false, false)).unwrap(),
            512
        );
    }

    #[test]
    fn shared_flag_registration_is_idempotent() {
        let mut reg = FlagRegistry::with_node_builtins();
        let first = reg
            .register_plugin(spec("test/testcase", true, true))
            .unwrap();
        let again = reg
            .register_plugin(spec("test/testcase", true, true))
            .unwrap();
        assert_eq!(first, again);
    }

    #[test]
    fn conflicting_reregistration_fails() {
        let mut reg = FlagRegistry::with_node_builtins();
        reg.register_plugin(spec("test/testcase", true, true))
            .unwrap();
        assert!(reg
            .register_plugin(spec("test/testcase", false, false))
            .is_err());
    }

    #[test]
    fn engine_prefix_is_reserved_for_plugins() {
        let mut reg = FlagRegistry::with_node_builtins();
        assert!(reg
            .register_plugin(spec("engine/sneaky", false, false))
            .is_err());
    }

    #[test]
    fn edge_width_exhaustion_is_loud() {
        // Edge registry is u8: engine masks 1 and 2, plugins from bit 4.
        // Free plugin bits are 4,8,16,32,64,128 (six), then exhaustion.
        let mut reg = FlagRegistry::with_edge_builtins();
        for i in 0..6 {
            reg.register_plugin(spec(&format!("acme/e{i}"), false, false))
                .expect("six plugin edge bits should fit in u8");
        }
        assert!(reg
            .register_plugin(spec("acme/overflow", false, false))
            .is_err());
    }

    #[test]
    fn default_seed_mask_ors_seed_and_default_on() {
        let mut reg = FlagRegistry::with_node_builtins();
        // engine/entrypoint|noqa|notebook are seed+default_on (2|16|32 = 50);
        // star_reexport/dead_branch are not.
        assert_eq!(reg.default_seed_mask(), 2 | 16 | 32);
        reg.register_plugin(spec("test/testcase", true, true))
            .unwrap();
        assert_eq!(reg.default_seed_mask(), 2 | 16 | 32 | 256);
    }
}
