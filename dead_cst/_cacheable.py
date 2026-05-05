"""Shared :class:`Cacheable` protocol for fingerprintable analyzer inputs.

Plugins and the unreachable-region detector both fold their per-file
output into the cached :class:`~dead_cst.graph.VisitorPayload`, so
swapping or reconfiguring either must invalidate stale entries. Both
contribute a stable ``(name, version)`` pair to the cache fingerprint;
:class:`Cacheable` is the structural type that captures that contract
in one place. :func:`~dead_cst.cache.compute_fingerprint` accepts any
``Cacheable`` directly so it can read ``.name`` / ``.version`` without
``getattr`` defaults.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Cacheable(Protocol):
    """Stable ``(name, version)`` identity used by the cache fingerprint.

    ``version`` is a Unix epoch int by convention. Bump it (to the
    current epoch) on any change that shouldn't be served from older
    caches. Epoch ints merge with ``max()`` semantics, so two
    branches bumping the same component never collide on a re-used
    label.
    """

    name: str
    version: int
