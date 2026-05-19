"""Tests for ``@overload`` anchoring and ``.pyi`` stub ingestion
(compiled-extension layout only; peer ``.pyi`` is dropped at enumeration)."""

from __future__ import annotations

import textwrap


from dead_cst import NodeFlags
from dead_cst.graph import KEEPALIVE_DEFAULT
from dead_cst.codemod import remove_code
from dead_cst.plugins import ExplicitEntrypointPlugin


def _normalise(s: str) -> str:
    s = textwrap.dedent(s)
    return s[1:] if s.startswith("\n") else s


# ---------------------------------------------------------------------------
# In-file ``@overload``
# ---------------------------------------------------------------------------


def test_overloads_are_excluded_from_cross_module_lookup(tmp_path, make_analysis, successors_of):
    """``from mod import f`` must reach the impl, never an overload."""
    (tmp_path / "mod.py").write_text(
        _normalise(
            """
            from typing import overload

            @overload
            def f(x: int) -> int: ...
            def f(x):
                return x
            """
        )
    )
    (tmp_path / "main.py").write_text("from mod import f\nf(1)\n")
    graph = make_analysis().materialize_all()

    main_f_import = next(n for n in graph.nodes() if n.fqname == "main.f")
    targets = [
        s
        for s in successors_of(graph, main_f_import)
        if s.fqname == "mod.f" and s.kind == "function"
    ]
    assert targets, "main.f should reach mod.f"
    assert all(not (t.flags & NodeFlags.OVERLOAD) for t in targets), (
        "Cross-module imports should resolve to the impl, not the overload"
    )


def test_dead_overloads_are_removed_with_impl(tmp_path, make_analysis):
    """When the impl is dead, the codemod removes the overloads alongside it."""
    (tmp_path / "mod.py").write_text(
        _normalise(
            """
            from typing import overload

            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x):
                return x

            def keep():
                return 1
            """
        )
    )
    graph = make_analysis().materialize_all()
    keep = next(n for n in graph.nodes() if n.fqname == "mod.keep")
    reachable = set(graph.descendants(keep)) | {keep}
    unreachable = [n for n in graph.nodes() if n not in reachable]
    remove_code(unreachable, tmp_path)

    rewritten = (tmp_path / "mod.py").read_text()
    assert "def f" not in rewritten
    assert "@overload" not in rewritten
    assert "def keep" in rewritten


def test_live_overloads_survive_codemod(tmp_path, make_analysis):
    """When the impl is alive, the overloads are kept too."""
    (tmp_path / "mod.py").write_text(
        _normalise(
            """
            from typing import overload

            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x):
                return x
            """
        )
    )
    a = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["mod.f"])])
    graph = a.materialize_all()
    reachable = set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))
    unreachable = [n for n in graph.nodes() if n not in reachable]
    remove_code(unreachable, tmp_path)

    rewritten = (tmp_path / "mod.py").read_text()
    assert rewritten.count("def f") == 3
    assert rewritten.count("@overload") == 2


# ---------------------------------------------------------------------------
# ``.pyi`` ingestion -- compiled-extension orphan stubs
# ---------------------------------------------------------------------------


def test_orphan_pyi_stub_uses_runtime_fqname(tmp_path, make_analysis, successors_of):
    """Compiled-extension shape: ``from mypkg._native import compute``
    must resolve to the stub decl when no ``.py`` twin exists."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg._native import compute\n")
    # No mypkg/_native.py -- emulating a binary that ships only its stub.
    (pkg / "_native.pyi").write_text("def compute(x: int) -> int: ...\n")
    (tmp_path / "main.py").write_text("from mypkg import compute\ncompute(1)\n")

    a = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["main"])])
    graph = a.materialize_all()
    reachable = set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))

    stub_compute = next(
        n for n in graph.nodes() if n.fqname == "mypkg._native.compute" and n.kind == "function"
    )
    assert stub_compute.path.endswith("/_native.pyi")
    assert stub_compute in reachable, "orphan stub decl should be alive when imported"

    pkg_compute = next(n for n in graph.nodes() if n.fqname == "mypkg.compute")
    assert stub_compute in successors_of(graph, pkg_compute)
