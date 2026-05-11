"""SQLite-backed cache of per-file :class:`VisitorPayload` blobs.

The visitor pass is the dominant cost in :func:`build_symbol_graph` --
LibCST's :class:`ScopeProvider` and :class:`FullyQualifiedNameProvider`
are both O(file) and not cheap. The :class:`VisitorPayload` shape that
landed in PR #53 captures everything one file contributes to the
symbol graph in four serializable fields, so we can pickle the
visitor's output keyed by the file's content hash and skip the visit
entirely when the file hasn't changed.

Two tables make up the database:

``meta`` holds the schema version (an integer) so reads from a database
written by an incompatible package version are detected on open: a
schema-version mismatch drops ``file_cache`` and re-writes the current
version.

``file_cache`` is one row per analyzed file, keyed by absolute path,
with the file's SHA-256 content hash, the analysis fingerprint
(:func:`compute_fingerprint`) it was written under, and the pickled
payload. A hash *or* fingerprint mismatch on read is treated as a
miss; the row is left in place and overwritten by the next
:meth:`GraphCache.put`. :func:`resolve_edges` runs unconditionally
every analysis, so mutating one file's exports re-stitches every
importer's edges for free -- the cache only short-circuits the
per-file visitor work, never the graph-stitching work.

The fingerprint covers the visitor / plugin / detector chain, the
schema version, and the Python version: it does *not* depend on the
package the file lives under, so the same payload is reusable across
analyses with different package layouts. Resolvers and search paths
deliberately do not enter the fingerprint either -- cross-file import
resolution moved out of the visitor and runs unconditionally on every
analysis, so swapping a resolver or rebinding ``sys.path`` re-stitches
edges without invalidating any cached payloads.

Plugins are intentionally part of the fingerprint (their ``observe``
output is folded into the cached payload) but the ``finalize`` pass
runs unconditionally on every analysis, so swapping ``finalize``-only
plugins between runs does not require a plugin ``version`` bump.
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

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".dead-cst-cache"
CACHE_DB_NAME = "cache.db"

# Bump when the per-row shape of ``file_cache`` or the ``meta`` schema
# changes in a way the unpickler / reader can't handle. Schema-version
# mismatch on open drops ``file_cache`` so older databases from sibling
# installs (or pre-per-base-fingerprint releases) are wiped on first
# use even at the same package version.
SCHEMA_VERSION = 3


def default_cache_path(project_root: Path) -> Path:
    """Default ``.dead-cst-cache/cache.db`` location under ``project_root``."""
    return project_root / CACHE_DIR_NAME / CACHE_DB_NAME


def compute_fingerprint(
    *,
    plugins: Sequence[EdgePlugin] = (),
    unreachable_detector: UnreachableRegionDetector | None = None,
) -> str:
    """SHA-256 of every input that affects :class:`VisitorPayload` semantics.

    Covers exactly the inputs the visitor + observe pass depend on:
    the visitor / plugin / detector ``(name, version)`` chain, the
    schema version, and the Python version. The package a file lives
    under is *not* part of the key -- the visitor's output is purely
    a function of the file's source plus the plugin/detector chain,
    so a payload computed under one package can be reused if the
    same file appears under a differently-named package later.

    ``search_paths`` and the resolver are *not* in the fingerprint:
    cross-file import resolution moved out of the visitor and into
    :func:`dead_cst._edges.resolve_edges` (which runs unconditionally
    on every analysis), so swapping a resolver or rebinding
    ``sys.path`` re-stitches edges without invalidating the cached
    :class:`VisitorPayload` blobs. :class:`~dead_cst.resolvers.PathResolver`
    deliberately does *not* satisfy :class:`Cacheable` for the same
    reason -- there is no fingerprint to invalidate, and the (uncached)
    edge-stitching pass picks up a swapped resolver on the next run.

    Each component (visitor, plugins, detector) satisfies
    :class:`~dead_cst._cacheable.Cacheable` and is fingerprinted by
    its ``(name, version)`` pair: bumping ``version`` invalidates
    every cached entry that referenced the old version. ``version``
    is a Unix epoch int by convention so concurrent bumps on
    different branches merge with ``max()``-wins semantics rather
    than colliding on a re-used label.

    The dead-cst package ``__version__`` is *not* in the fingerprint:
    every component whose output could shift between releases carries
    its own ``Cacheable`` knob, and folding ``__version__`` in on top
    would let lazily-unbumped components ride for free on a release
    bump.

    Each value is normalized to a stable string before hashing so
    equivalent inputs produce equal keys.
    """
    h = hashlib.sha256()
    h.update(f"schema={SCHEMA_VERSION}\n".encode())
    h.update(f"python={sys.version_info.major}.{sys.version_info.minor}\n".encode())
    h.update(f"visitor={SymbolVisitor.name}@{SymbolVisitor.version}\n".encode())

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

    Open with just the database path -- there is no fingerprint at
    the database level. Each :meth:`get` and :meth:`put` takes the
    analysis fingerprint the caller computed via
    :func:`compute_fingerprint`; the row stores its own fingerprint
    and a mismatch on read returns ``None`` (treated as a cache miss).
    On open, a schema-version mismatch wipes ``file_cache`` and
    re-writes the current schema -- this handles upgrading from older
    databases without intervention.

    :meth:`get` returns the cached payload for a path when both the
    file's SHA-256 and the stored fingerprint match the caller's;
    otherwise ``None``. :meth:`put` writes (or replaces) one row.
    :meth:`close` flushes and closes the connection; the class is
    also a context manager.

    Concurrency note: SQLite handles a single writer fine, but the
    cache is intentionally process-local for the duration of one
    analysis. Don't share an instance across threads.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_gitignore(db_path.parent)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

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
        """Create or migrate ``meta`` + ``file_cache`` to ``SCHEMA_VERSION``.

        Schema-version mismatch (or first-run, or upgrade from an
        older release that stored a single project-wide fingerprint
        in ``meta.fingerprint``) drops ``file_cache`` and clears
        ``meta`` so the next run starts clean. The next round of
        :meth:`put` calls re-populates with rows that carry the
        current analysis fingerprint.
        """
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
        cur = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        stored = int(row[0]) if row else None
        if stored == SCHEMA_VERSION:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS file_cache ("
                    "path TEXT PRIMARY KEY, "
                    "content_hash TEXT NOT NULL, "
                    "fingerprint TEXT NOT NULL, "
                    "payload BLOB NOT NULL)"
                )
            return
        logger.debug(
            "Cache schema mismatch (stored=%s, current=%s); wiping file_cache",
            stored,
            SCHEMA_VERSION,
        )
        with self._conn:
            self._conn.execute("DROP TABLE IF EXISTS file_cache")
            self._conn.execute("DELETE FROM meta")
            self._conn.execute(
                "CREATE TABLE file_cache ("
                "path TEXT PRIMARY KEY, "
                "content_hash TEXT NOT NULL, "
                "fingerprint TEXT NOT NULL, "
                "payload BLOB NOT NULL)"
            )
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def get(self, path: Path, fingerprint: str) -> VisitorPayload | None:
        """Return the cached payload for ``path`` if its content hash and
        fingerprint match.

        Returns ``None`` on first sight of the file, on hash or
        fingerprint mismatch (the row is left in place; :meth:`put`
        will overwrite it), or if the stored blob fails to unpickle
        (the row is dropped to avoid trapping the cache).
        """
        h = file_hash(path)
        if h is None:
            return None
        cur = self._conn.execute(
            "SELECT content_hash, fingerprint, payload FROM file_cache WHERE path=?",
            (str(path),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        stored_hash, stored_fp, blob = row
        if stored_hash != h or stored_fp != fingerprint:
            return None
        try:
            payload = pickle.loads(blob)
        except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, ValueError):
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

    def put(self, path: Path, payload: VisitorPayload, fingerprint: str) -> None:
        """Pickle ``payload`` and record it under the file's current hash
        and the caller's analysis ``fingerprint``."""
        h = file_hash(path)
        if h is None:
            return
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with self._conn:
            self._conn.execute(
                "INSERT INTO file_cache(path, content_hash, fingerprint, payload) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "content_hash=excluded.content_hash, "
                "fingerprint=excluded.fingerprint, "
                "payload=excluded.payload",
                (str(path), h, fingerprint, blob),
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
