//! Topic registry: the declare side of the per-file → project-wide *fact*
//! channel.
//!
//! A plugin declares topics via
//! [`ExternalPlugin::declare_topics`](crate::native_plugins::plugin_api::ExternalPlugin::declare_topics);
//! the host assigns each a small integer handle in registration order. A
//! per-file plugin then publishes facts under a topic **name** (a salsa-stable
//! string it hard-codes — a registry-allocated handle can't ride the per-file
//! salsa cache without coupling it to the plugin set), and the project-wide
//! side resolves a handle back to its name to read the collected facts via
//! [`PluginCtx::facts_for_topic`](crate::native_plugins::plugin_api::PluginCtx::facts_for_topic).
//!
//! [`TopicSpec`] is the only author-facing type (re-exported from
//! `plugin_api`); it is pyo3-free. [`TopicRegistry`] is host-internal and
//! never crosses the plugin airlock — plugins resolve a name to a handle
//! through `PluginCtx::topic`, which hands back a plain `Option<u32>`. The
//! shape mirrors [`FlagRegistry`](crate::flag_registry::FlagRegistry); topics
//! carry no bit-width cap (the handle is a sequential index, not a packed
//! bit), so there is no engine-registration or exhaustion path.

use rustc_hash::FxHashMap;

use crate::native_plugins::plugin_api::PluginError;

/// Declarative description of one topic. `name` is `owner/name` namespaced;
/// the `engine/` owner is reserved (there are no engine topics today, but the
/// namespace is held for symmetry with [`FlagSpec`](crate::flag_registry::FlagSpec)).
#[derive(Debug, Clone)]
pub struct TopicSpec {
    pub name: String,
    pub description: String,
}

/// A name↔handle registry over topics. Seeded empty (topics are
/// plugin-declared — no engine built-ins), then frozen after the plugin
/// declaration pass and lent read-only into the parallel plugin run.
pub(crate) struct TopicRegistry {
    by_name: FxHashMap<String, u32>,
    /// Specs in registration order; the `Vec` index is the topic's handle.
    specs: Vec<TopicSpec>,
}

impl TopicRegistry {
    pub(crate) fn new() -> Self {
        Self {
            by_name: FxHashMap::default(),
            specs: Vec::new(),
        }
    }

    /// Register a plugin topic, allocating the next sequential handle.
    /// Idempotent on an identical spec (so a topic two plugins share collapses
    /// to one handle); a conflicting re-registration (same name, different
    /// description) or an `engine/`-prefixed name fails loudly.
    pub(crate) fn register_plugin(&mut self, spec: TopicSpec) -> Result<u32, PluginError> {
        if spec.name.starts_with("engine/") {
            return Err(PluginError::value(format!(
                "plugin topic {:?}: the 'engine/' namespace is reserved",
                spec.name
            )));
        }
        if let Some(&handle) = self.by_name.get(&spec.name) {
            let existing = &self.specs[handle as usize];
            if existing.description != spec.description {
                return Err(PluginError::value(format!(
                    "topic {:?} re-registered with a conflicting description",
                    spec.name
                )));
            }
            return Ok(handle);
        }
        let handle = self.specs.len() as u32;
        self.by_name.insert(spec.name.clone(), handle);
        self.specs.push(spec);
        Ok(handle)
    }

    /// The handle for `name`, if registered.
    pub(crate) fn get(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    /// The name for `handle`, if in range — the inverse of [`Self::get`], used
    /// to look up a topic's collected facts from a handle.
    pub(crate) fn name_of(&self, handle: u32) -> Option<&str> {
        self.specs.get(handle as usize).map(|s| s.name.as_str())
    }

    /// All `(handle, spec)` entries in handle order — for the
    /// `ProjectContext` getter.
    pub(crate) fn entries(&self) -> Vec<(u32, TopicSpec)> {
        self.specs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, spec)| (i as u32, spec))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, description: &str) -> TopicSpec {
        TopicSpec {
            name: name.to_string(),
            description: description.to_string(),
        }
    }

    #[test]
    fn handles_allocate_sequentially_in_registration_order() {
        let mut reg = TopicRegistry::new();
        assert_eq!(reg.register_plugin(spec("acme/a", "")).unwrap(), 0);
        assert_eq!(reg.register_plugin(spec("acme/b", "")).unwrap(), 1);
        assert_eq!(reg.get("acme/a"), Some(0));
        assert_eq!(reg.get("acme/b"), Some(1));
        assert_eq!(reg.name_of(1), Some("acme/b"));
        assert_eq!(reg.name_of(2), None);
    }

    #[test]
    fn shared_topic_registration_is_idempotent() {
        let mut reg = TopicRegistry::new();
        let first = reg.register_plugin(spec("test/cases", "decl ids")).unwrap();
        let again = reg.register_plugin(spec("test/cases", "decl ids")).unwrap();
        assert_eq!(first, again);
        assert_eq!(reg.entries().len(), 1);
    }

    #[test]
    fn conflicting_reregistration_fails() {
        let mut reg = TopicRegistry::new();
        reg.register_plugin(spec("test/cases", "decl ids")).unwrap();
        assert!(reg
            .register_plugin(spec("test/cases", "something else"))
            .is_err());
    }

    #[test]
    fn engine_prefix_is_reserved() {
        let mut reg = TopicRegistry::new();
        assert!(reg.register_plugin(spec("engine/sneaky", "")).is_err());
    }

    #[test]
    fn entries_are_handle_ordered() {
        let mut reg = TopicRegistry::new();
        reg.register_plugin(spec("acme/first", "1")).unwrap();
        reg.register_plugin(spec("acme/second", "2")).unwrap();
        let entries = reg.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, 0);
        assert_eq!(entries[0].1.name, "acme/first");
        assert_eq!(entries[1].0, 1);
        assert_eq!(entries[1].1.name, "acme/second");
    }
}
