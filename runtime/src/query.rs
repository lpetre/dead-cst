//! Shared per-file scan helpers for the native query surface.
//!
//! What remains here after the chainable `query()` DSL was retired:
//! the identifier prefilter (`_contains_identifier` and friends) that
//! lets a per-file walk skip the parse on files that don't even
//! mention the target name, the `_compile_path_regex` /
//! `_path_re_matches` path-scoping pair, and `par_scan_files` — the
//! generic GIL-free parallel per-file walk that every `find_*` query
//! on [`crate::project::ProjectContext`] drives. These are pure-rust,
//! Salsa-snapshot-based, and shared by `project.rs` and `helpers.rs`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ruff_db::files::File;
use ty_project::Db as ProjectDb;

use crate::helpers::file_path_string;

/// Compile an optional path regex once for a query's file-iteration
/// loop. Centralized so every ``find_*`` method that takes a
/// ``path_regex`` parameter shares the same error reporting.
pub(crate) fn _compile_path_regex(re_str: Option<&str>) -> PyResult<Option<regex::Regex>> {
    match re_str {
        None => Ok(None),
        Some(s) => regex::Regex::new(s)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("invalid path regex {s:?}: {e}"))),
    }
}

/// Per-file predicate-fusion check. ``true`` when the file should be
/// processed (no regex, or regex matches its absolute path).
pub(crate) fn _path_re_matches(re: &Option<regex::Regex>, db: &dyn ProjectDb, file: File) -> bool {
    match re {
        None => true,
        Some(re) => re.is_match(&file_path_string(db, file)),
    }
}

/// Cheap text prefilter for identifier references before AST/semantic
/// validation. Mirrors `ty_ide::references::contains_identifier`
/// (vendor/ruff/crates/ty_ide/src/references.rs:198) so per-file
/// queries can skip the parse + walk when the file source doesn't
/// even mention the target identifier.
///
/// Matches an ASCII approximation of `\b<name>\b`: every occurrence
/// of `needle` in `source` whose surrounding bytes aren't identifier
/// continuations. Used by every decorator / construction / call /
/// method query that walks ``project_files``; saves the parse on the
/// (typically large) majority of files that don't reference the
/// query's target name at all.
pub(crate) fn _contains_identifier(source: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = source.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut start = 0;
    while let Some(rel) = source[start..].find(needle) {
        let pos = start + rel;
        let after = pos + needle_bytes.len();
        let boundary_before = pos == 0 || !_is_ident_continue(bytes[pos - 1]);
        let boundary_after = bytes
            .get(after)
            .is_none_or(|byte| !_is_ident_continue(*byte));
        if boundary_before && boundary_after {
            return true;
        }
        start = pos + 1;
    }
    false
}

pub(crate) fn _is_ident_continue(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// Multi-needle variant of [`_contains_identifier`]: returns ``true``
/// as soon as any one of ``needles`` is found. Used by queries that
/// take a list of names (decorator name set, ctor name set, …) so the
/// per-file prefilter passes when the file mentions any of them.
pub(crate) fn _contains_any_identifier(source: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| _contains_identifier(source, n))
}

/// Generic parallel per-file walk. ``per_file`` runs on a Salsa
/// snapshot of ``db`` (one ``Db::dyn_clone`` per worker, mirroring
/// the ty_ide find_references pattern at
/// ``vendor/ruff/crates/ty_ide/src/references.rs:107-130``) and
/// returns a ``Vec<T>`` of opaque per-file results.
///
/// Caller is responsible for releasing the GIL with
/// :meth:`pyo3::Python::allow_threads` — the closure passed in must
/// be ``Send + Sync`` and ``T`` must be ``Send``. Materializing
/// ``Py<SymbolNode>`` values (which are GIL-bound) belongs in the
/// caller AFTER ``allow_threads`` returns.
pub(crate) fn par_scan_files<T, F>(
    db: Box<dyn ProjectDb>,
    files: &[File],
    path_re: &Option<regex::Regex>,
    per_file: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(&dyn ProjectDb, File) -> Vec<T> + Send + Sync,
{
    let result = std::sync::Mutex::new(Vec::<T>::new());
    let per_file_ref = &per_file;
    let result_ref = &result;
    // `move` captures `db: Box<dyn ProjectDb>` by value — `dyn Db`
    // has a `Send` supertrait via `salsa::Database`, so the box is
    // Send, but `&dyn Db` is NOT Send (the trait isn't Sync), which
    // is why the box can't be borrowed across the rayon scope.
    rayon::scope(move |s| {
        for &file in files {
            if !_path_re_matches(path_re, &*db, file) {
                continue;
            }
            let db_t: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&*db);
            s.spawn(move |_| {
                let local = per_file_ref(&*db_t, file);
                if !local.is_empty() {
                    result_ref.lock().unwrap().extend(local);
                }
            });
        }
    });
    result.into_inner().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the identifier-prefilter helpers used by
    //! every per-file query loop. The per-file query methods on
    //! `ProjectContext` depend on `Python<'_>` / `Py<ProjectContext>`
    //! and are covered end-to-end by the python suite.
    use super::*;

    // -- _is_ident_continue -----------------------------------------------

    #[test]
    fn is_ident_continue_recognizes_word_bytes() {
        for byte in b'a'..=b'z' {
            assert!(_is_ident_continue(byte), "{byte} should be ident continue");
        }
        for byte in b'A'..=b'Z' {
            assert!(_is_ident_continue(byte));
        }
        for byte in b'0'..=b'9' {
            assert!(_is_ident_continue(byte));
        }
        assert!(_is_ident_continue(b'_'));
    }

    #[test]
    fn is_ident_continue_rejects_punctuation_and_whitespace() {
        for byte in [b'.', b',', b' ', b'\t', b'\n', b'(', b')', b'-', b'#', b':'] {
            assert!(!_is_ident_continue(byte));
        }
    }

    // -- _contains_identifier ---------------------------------------------

    #[test]
    fn contains_identifier_finds_isolated_match() {
        assert!(_contains_identifier("foo", "foo"));
        assert!(_contains_identifier("call(foo)", "foo"));
        assert!(_contains_identifier(" foo ", "foo"));
        assert!(_contains_identifier("foo.bar", "foo"));
        assert!(_contains_identifier("a.foo", "foo"));
    }

    #[test]
    fn contains_identifier_rejects_substring_inside_other_word() {
        assert!(!_contains_identifier("foobar", "foo"));
        assert!(!_contains_identifier("xfoo", "foo"));
        assert!(!_contains_identifier("foo_bar", "foo"));
        assert!(!_contains_identifier("foo1", "foo"));
        // Underscore prefix is also an identifier continuation.
        assert!(!_contains_identifier("_foo", "foo"));
        // Digit prefix
        assert!(!_contains_identifier("1foo", "foo"));
    }

    #[test]
    fn contains_identifier_handles_match_at_boundaries() {
        assert!(_contains_identifier("foo()", "foo"));
        assert!(_contains_identifier("(foo)", "foo"));
        assert!(_contains_identifier("foo+bar", "foo"));
    }

    #[test]
    fn contains_identifier_empty_needle_returns_false() {
        // Empty needle would otherwise match everywhere — explicitly handled.
        assert!(!_contains_identifier("anything", ""));
        assert!(!_contains_identifier("", ""));
    }

    #[test]
    fn contains_identifier_empty_source_returns_false() {
        assert!(!_contains_identifier("", "foo"));
    }

    #[test]
    fn contains_identifier_multiple_occurrences_one_valid() {
        // The first ("xfoo") is invalid; the second (" foo") matches.
        assert!(_contains_identifier("xfoo foo", "foo"));
        // Both fail.
        assert!(!_contains_identifier("xfooy zfoow", "foo"));
    }

    #[test]
    fn contains_identifier_handles_overlapping_advance() {
        // Even when the first start position fails the boundary check,
        // the scan must advance and try subsequent occurrences.
        assert!(_contains_identifier("aaa a", "a"));
    }

    #[test]
    fn contains_identifier_unicode_source_does_not_panic() {
        // The function uses byte indexing for the boundary check. ASCII
        // identifier continuations are all single-byte, and the
        // multi-byte UTF-8 leading/continuation bytes all have the high
        // bit set (>= 0x80), so they fail `_is_ident_continue` and the
        // boundary holds — which makes `foo` adjacent to emoji a hit,
        // even though Python would treat the unicode chars as
        // identifier-continuation. The prefilter is approximate by
        // design (`ty_ide::references::contains_identifier` has the
        // same shape) — we exercise the byte path here to lock in the
        // documented behavior.
        assert!(_contains_identifier("héllo foo", "foo"));
        assert!(_contains_identifier("foo🙂bar", "foo"));
        // No needle present.
        assert!(!_contains_identifier("héllo", "foo"));
        // ASCII boundary still wins over surrounding unicode.
        assert!(_contains_identifier(" foo ", "foo"));
    }

    #[test]
    fn contains_identifier_multi_char_needle() {
        assert!(_contains_identifier("def my_func(): pass", "my_func"));
        assert!(!_contains_identifier("def my_function(): pass", "my_func"));
        assert!(!_contains_identifier("my_funcx", "my_func"));
    }

    // -- _contains_any_identifier -----------------------------------------

    #[test]
    fn contains_any_identifier_returns_true_on_first_hit() {
        assert!(_contains_any_identifier("call(foo)", &["bar", "foo"]));
        assert!(_contains_any_identifier("foo()", &["foo"]));
    }

    #[test]
    fn contains_any_identifier_empty_list_returns_false() {
        assert!(!_contains_any_identifier("anything", &[]));
        assert!(!_contains_any_identifier("", &[]));
    }

    #[test]
    fn contains_any_identifier_returns_false_when_no_match() {
        assert!(!_contains_any_identifier("this has none", &["foo", "bar"]));
    }

    #[test]
    fn contains_any_identifier_respects_boundaries() {
        // Same substring-inside-word rule as _contains_identifier.
        assert!(!_contains_any_identifier("foobar", &["foo", "bar"]));
        // But the standalone "foo" still hits if present.
        assert!(_contains_any_identifier("foobar foo", &["foo"]));
    }
}
