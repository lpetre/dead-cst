"""SQLite-backed cache of per-file :class:`VisitorPayload` blobs.

The visitor pass is the dominant cost in :func:`build_symbol_graph` --
LibCST's :class:`ScopeProvider` and :class:`FullyQualifiedNameProvider`
are both O(file) and not cheap. The :class:`VisitorPayload` shape that
landed in PR #53 captures everything one file contributes to the
symbol graph in four serializable fields, so we can pickle the
visitor's output keyed by the file's content hash and skip the visit
entirely when the file hasn't changed.

Two tables make up the database:

``meta`` holds a single ``fingerprint`` row covering everything that
could change a payload's *interpretation* without touching file
contents -- Python version, search paths, and the visitor / resolver
/ plugin / detector chain (each contributing its own
``Cacheable`` ``(name, version)`` pair). A fingerprint mismatch on
open wipes ``file_cache`` and writes the new fingerprint, so the
very next analysis fully rebuilds the graph.

``file_cache`` is one row per analyzed file, keyed by absolute path,
with the file's SHA-256 content hash and the pickled payload. A hash
mismatch invalidates that single row; the visitor reruns and the new
payload replaces the old. Per-base :func:`resolve_edges` runs
unconditionally every analysis, so mutating one file's exports
re-stitches every importer's edges in the same base for free -- the
cache only short-circuits the per-file visitor work, never the
graph-stitching work.

Plugins are intentionally **not** part of the fingerprint: they
contribute graph ops *after* per-file payloads are folded in, so
swapping plugins between runs reuses the cached visitor output and
only re-executes the plugin pass.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

from ._visitor import SymbolVisitor
from .branches import (
    DefaultUnreachableRegionDetector,
    UnreachableRegionDetector,
)
from .graph import VisitorPayload
from .plugins._core import EdgePlugin
from .resolvers import PathMap, PathResolver

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".dead-cst-cache"
CACHE_DB_NAME = "cache.db"

# Bump when ``VisitorPayload``'s on-disk shape changes in a way the
# unpickler can't handle. The fingerprint already covers ``__version__``,
# but a major shape break wants an explicit signal so older databases
# from sibling installs are wiped on open even at the same package
# version (e.g. an editable checkout running ahead of the published
# version).
SCHEMA_VERSION = 1


def default_cache_path(project_root: Path) -> Path:
    """Default ``.dead-cst-cache/cache.db`` location under ``project_root``."""
    return project_root / CACHE_DIR_NAME / CACHE_DB_NAME


def compute_fingerprint(
    *,
    paths: PathMap,
    resolvers: Sequence[PathResolver],
    plugins: Sequence[EdgePlugin] = (),
    unreachable_detector: UnreachableRegionDetector | None = None,
) -> str:
    """SHA-256 of every input that affects payload semantics for one analysis run.

    Includes Python version (pickle protocol stability), the
    ``PathMap`` (the search-path layout governs ``Import.path``
    resolution), and the visitor / resolver / plugin / detector
    chain. Each component satisfies
    :class:`~dead_cst._cacheable.Cacheable` and is fingerprinted by
    its ``(name, version)`` pair: bumping ``version`` invalidates
    the file_cache so new output replaces the old. ``version`` is a
    Unix epoch int by convention, so concurrent bumps on different
    branches merge with ``max()``-wins semantics rather than
    colliding on a re-used integer label.

    The dead-cst package ``__version__`` is *not* in the fingerprint:
    every component whose output could shift between releases carries
    its own ``Cacheable`` knob, and folding ``__version__`` in on top
    would let lazily-unbumped components ride for free on a release
    bump. Bumping the relevant component's ``version`` is the
    discipline; ``__version__`` is not a substitute.

    Each value is normalized to a stable string before hashing so
    equivalent inputs produce equal keys.
    """
    h = hashlib.sha256()
    h.update(f"schema={SCHEMA_VERSION}\n".encode())
    h.update(f"python={sys.version_info.major}.{sys.version_info.minor}\n".encode())
    h.update(f"visitor={SymbolVisitor.name}@{SymbolVisitor.version}\n".encode())

    h.update(b"paths=\n")
    for base in sorted(paths, key=lambda p: str(p)):
        deps = sorted(str(d) for d in paths[base])
        h.update(f"  {base}:{','.join(deps)}\n".encode())

    h.update(b"resolvers=\n")
    for r_name, r_version in sorted((r.name, r.version) for r in resolvers):
        h.update(f"  {r_name}@{r_version}\n".encode())

    h.update(b"plugins=\n")
    for p_name, p_version in sorted((p.name, p.version) for p in plugins):
        h.update(f"  {p_name}@{p_version}\n".encode())

    detector = (
        unreachable_detector
        if unreachable_detector is not None
        else DefaultUnreachableRegionDetector()
    )
    h.update(f"unreachable_detector={detector.name}@{detector.version}\n".encode())

    return h.hexdigest()


def file_hash(path: Path) -> str | None:
    """SHA-256 of ``path``'s bytes; ``None`` if the file is unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


class GraphCache:
    """SQLite-backed lookup of pickled :class:`VisitorPayload` per file.

    Open with a fingerprint string covering the full per-run inputs
    (see :func:`compute_fingerprint`). On open, if the stored
    fingerprint or schema version differs, ``file_cache`` is wiped and
    the new fingerprint is recorded -- the next run is effectively a
    full rebuild, and only the bumped fingerprint is preserved.

    :meth:`get` returns the cached payload for a path when the file's
    SHA-256 matches the stored hash, otherwise ``None``. :meth:`put`
    writes (or replaces) one row. :meth:`close` flushes and closes the
    connection; the class is also a context manager.

    Concurrency note: SQLite handles a single writer fine, but the
    cache is intentionally process-local for the duration of one
    analysis. Don't share an instance across threads.
    """

    def __init__(self, db_path: Path, fingerprint: str) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_gitignore(db_path.parent)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()
        self._reconcile_fingerprint(fingerprint)

    @staticmethod
    def _write_gitignore(cache_dir: Path) -> None:
        """Drop a ``.gitignore`` so the cache dir is auto-ignored.

        Idempotent. Failures (read-only filesystem, permissions) are
        logged at debug level -- the cache itself still works, the dir
        just won't be hidden from version control.
        """
        gi = cache_dir / ".gitignore"
        if gi.exists():
            return
        try:
            gi.write_text("*\n")
        except OSError as exc:
            logger.debug("Could not write %s: %s", gi, exc)

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS file_cache ("
                "path TEXT PRIMARY KEY, "
                "content_hash TEXT NOT NULL, "
                "payload BLOB NOT NULL)"
            )

    def _reconcile_fingerprint(self, fingerprint: str) -> None:
        cur = self._conn.execute("SELECT value FROM meta WHERE key='fingerprint'")
        row = cur.fetchone()
        stored = row[0] if row else None
        if stored == fingerprint:
            return
        logger.debug(
            "Cache fingerprint mismatch (stored=%s, current=%s); wiping file_cache",
            stored,
            fingerprint,
        )
        with self._conn:
            self._conn.execute("DELETE FROM file_cache")
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('fingerprint', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (fingerprint,),
            )

    def get(self, path: Path) -> VisitorPayload | None:
        """Return the cached payload for ``path`` if its content hash matches.

        Returns ``None`` on first sight of the file, on hash mismatch
        (the row is left in place; :meth:`put` will overwrite it), or
        if the stored blob fails to unpickle (the row is dropped to
        avoid trapping the cache).
        """
        h = file_hash(path)
        if h is None:
            return None
        cur = self._conn.execute(
            "SELECT content_hash, payload FROM file_cache WHERE path=?",
            (str(path),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        stored_hash, blob = row
        if stored_hash != h:
            return None
        try:
            payload = pickle.loads(blob)
        except Exception:
            logger.warning("Corrupt cache entry for %s; dropping", path)
            with self._conn:
                self._conn.execute("DELETE FROM file_cache WHERE path=?", (str(path),))
            return None
        if not isinstance(payload, VisitorPayload):
            logger.warning("Cache entry for %s is not a VisitorPayload; dropping", path)
            with self._conn:
                self._conn.execute("DELETE FROM file_cache WHERE path=?", (str(path),))
            return None
        return payload

    def put(self, path: Path, payload: VisitorPayload) -> None:
        """Pickle ``payload`` and record it under the file's current hash."""
        h = file_hash(path)
        if h is None:
            return
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with self._conn:
            self._conn.execute(
                "INSERT INTO file_cache(path, content_hash, payload) VALUES(?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "content_hash=excluded.content_hash, payload=excluded.payload",
                (str(path), h, blob),
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> GraphCache:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def clear_cache(db_path: Path) -> bool:
    """Delete the cache database file (and its auto-created ``.gitignore``).

    Returns ``True`` when the file was removed, ``False`` when it
    didn't exist. The parent ``.dead-cst-cache`` directory is removed
    only when empty after cleanup, matching ``rm -r`` semantics for a
    pristine project tree.
    """
    if not db_path.exists():
        return False
    db_path.unlink()
    # WAL mode leaves -wal / -shm sidecars; clean those too.
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            side.unlink()
    parent = db_path.parent
    gi = parent / ".gitignore"
    if gi.exists() and parent.name == CACHE_DIR_NAME:
        gi.unlink()
    if parent.name == CACHE_DIR_NAME and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
    return True


__all__ = [
    "GraphCache",
    "SCHEMA_VERSION",
    "clear_cache",
    "compute_fingerprint",
    "default_cache_path",
]
