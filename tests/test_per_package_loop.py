"""Tests for the per-package edge pass in :class:`Analysis`.

The pipeline iterates packages in BFS dep order, swapping
``Program::search_paths`` between iterations so each package's
phase 1-3 runs under env_roots scoped to *exactly* what its
lockfile declares: its own exports + its deps' exports.
Non-dep cross-package imports therefore resolve as
``[unresolved]`` / ``[external]`` rather than silently routing
into a non-dep, faithfully reflecting the lockfile's dep graph
at analysis time.

Layout convention these tests follow (mirrors uv flat-layout
workspaces): each owned member dir contains a subdirectory
matching the wheel's published package name -- e.g.
``app/app/main.py``, ``libx/libx/__init__.py``. The member dir
is what goes on the search path; the inner subdir is what
consumers actually import.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst import Analysis
from dead_cst.resolvers import ManualResolver


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def _edges_by_node(graph) -> dict[int, list[tuple[int, int]]]:
    """``src_idx -> [(dst_idx, flags)]`` adjacency built from the rust
    ``(src_idx, dst_idx, flags)`` triples."""
    out: dict[int, list[tuple[int, int]]] = {}
    for s, d, f in graph.edges():
        out.setdefault(s, []).append((d, f))
    return out


def test_per_package_loop_resolves_declared_dep_import(tmp_path, has_edge):
    """Baseline: when ``app`` declares ``libx`` as a dep,
    ``from libx import value`` in app's code must resolve
    first-party to libx's decl.

    Under the new loop, app's iteration runs with
    env_roots = [app.exports, app.path, libx.exports], so the
    resolver finds libx as first-party. Pins that env composition
    doesn't accidentally drop deps' exports.
    """
    _write(
        tmp_path,
        {
            "app/app/main.py": "from libx import value\nuse = value",
            "libx/libx/__init__.py": "value = 42",
        },
    )
    analysis = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["app:libx", "libx"]),
    )
    graph = analysis.materialize_all()

    value_node = next(n for n in graph.nodes() if n.fqname == "libx.value" and n.kind == "variable")
    app_import = next(
        n
        for n in graph.nodes()
        if n.kind == "import" and n.path.endswith("app/app/main.py") and "value" in n.fqname
    )
    assert has_edge(graph, app_import, value_node)


def test_per_package_loop_isolates_non_dep_cross_package_import(tmp_path):
    """Behavior the per-package loop ships: when ``app`` imports
    from ``other`` but does NOT declare ``other`` as a dep, the
    import does NOT resolve into ``other``'s first-party decl.

    Today's flat-env behavior would route the import to ``other``'s
    decl anyway (both are in src_roots) -- silently OK at analysis
    time even though it would fail at runtime when the wheel-built
    app is installed without ``other`` in its deps. The per-package
    loop's app iteration excludes other.exports, so the resolver
    classifies the import as non-first-party.
    """
    _write(
        tmp_path,
        {
            "app/app/main.py": "from other import y\nuse = y",
            "other/other/__init__.py": "y = 1",
        },
    )
    # No dep declared from app -> other.
    analysis = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["app", "other"]),
    )
    graph = analysis.materialize_all()

    nodes = graph.nodes()
    app_import = next(n for n in nodes if n.kind == "import" and n.path.endswith("app/app/main.py"))
    adj = _edges_by_node(graph)
    src_idx = nodes.index(app_import)
    outgoing = [nodes[d] for (d, _f) in adj.get(src_idx, ())]
    # No edge into a first-party decl whose source path is under the
    # non-dep ``other/`` member dir.
    first_party_other = [n for n in outgoing if n.kind == "variable" and "/other/other/" in n.path]
    assert not first_party_other, (
        "Per-package loop must NOT route app.main's non-dep import "
        f"into other's decl, but found edges to: {first_party_other}"
    )


def test_per_package_loop_dep_chain_resolves_transitively_via_imports(tmp_path, has_edge):
    """A diamond: ``app -> mid -> base``. Imports in ``app`` that
    go through ``mid``'s public API must resolve into ``mid``'s
    decls, and ``mid``'s import of ``base`` must resolve into
    ``base``'s decls. Each package's iteration sees only its own
    declared deps -- transitive resolution is by composition, not
    by giving every package the union env.
    """
    _write(
        tmp_path,
        {
            "app/app/main.py": "from mid import bridge\nuse = bridge",
            "mid/mid/__init__.py": "from base import core\nbridge = core",
            "base/base/__init__.py": "core = 'hello'",
        },
    )
    analysis = Analysis(
        tmp_path,
        resolver=ManualResolver(specs=["app:mid", "mid:base", "base"]),
    )
    graph = analysis.materialize_all()

    core_node = next(n for n in graph.nodes() if n.fqname == "base.core" and n.kind == "variable")
    bridge_node = next(
        n for n in graph.nodes() if n.fqname == "mid.bridge" and n.kind == "variable"
    )
    # mid's import of base.core resolved -- the mid pass had base in env.
    mid_imports_core = next(
        n
        for n in graph.nodes()
        if n.kind == "import" and "mid/mid/__init__" in n.path and "core" in n.fqname
    )
    assert has_edge(graph, mid_imports_core, core_node)

    # app's import of mid.bridge resolved -- the app pass had mid in env.
    app_imports_bridge = next(
        n
        for n in graph.nodes()
        if n.kind == "import" and "app/app/main" in n.path and "bridge" in n.fqname
    )
    assert has_edge(graph, app_imports_bridge, bridge_node)
