"""Tests for ``@overload`` anchoring and ``.pyi`` stub ingestion
(compiled-extension layout only; peer ``.pyi`` is dropped at enumeration)."""

from __future__ import annotations

import textwrap

import pytest

from dead_cst import NodeFlags
from dead_cst.analyze import _entrypoint_seeds, _find_reachable as find_reachable
from dead_cst.codemod import remove_code
from dead_cst.plugins import ExplicitEntrypointPlugin


def _normalise(s: str) -> str:
    s = textwrap.dedent(s)
    return s[1:] if s.startswith("\n") else s


# ---------------------------------------------------------------------------
# In-file ``@overload``
# ---------------------------------------------------------------------------


def test_overload_decls_are_flagged_and_anchored_to_impl(tmp_path, make_analysis):
    src = """
    from typing import overload

    @overload
    def f(x: int) -> int: ...
    @overload
    def f(x: str) -> str: ...
    def f(x):
        return x

    f(1)
    """
    (tmp_path / "mod.py").write_text(_normalise(src))
    graph = make_analysis().materialize_all()

    f_decls = sorted(
        (n for n in graph.nodes if n.type == "function" and n.fqname == "mod.f"),
        key=lambda n: n.position.start.line,
    )
    assert len(f_decls) == 3, [d.position.start.line for d in f_decls]

    overloads = [d for d in f_decls if d.flags & NodeFlags.OVERLOAD]
    impls = [d for d in f_decls if not (d.flags & NodeFlags.OVERLOAD)]
    assert len(overloads) == 2
    assert len(impls) == 1
    impl = impls[0]
    for ov in overloads:
        assert impl.position.start.line > ov.position.start.line
    successors = list(graph.successors(impl))
    assert all(o in successors for o in overloads)


def test_overloads_are_excluded_from_cross_module_lookup(tmp_path, make_analysis):
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

    main_f_import = next(n for n in graph.nodes if n.fqname == "main.f")
    targets = [
        s for s in graph.successors(main_f_import) if s.fqname == "mod.f" and s.type == "function"
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
    seeds = [n for n in graph.nodes if n.fqname == "mod.keep"]
    reachable = find_reachable(graph, seeds)
    unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()
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
    reachable = find_reachable(graph, _entrypoint_seeds(graph))
    unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()
    remove_code(unreachable, tmp_path)

    rewritten = (tmp_path / "mod.py").read_text()
    assert rewritten.count("def f") == 3
    assert rewritten.count("@overload") == 2


@pytest.mark.parametrize("decorator", ["overload", "typing.overload"])
def test_overload_recognized_under_alternate_decorator_forms(tmp_path, make_analysis, decorator):
    src = (
        "import typing\n" if "." in decorator else "from typing import overload\n"
    ) + textwrap.dedent(
        f"""
            @{decorator}
            def f(x: int) -> int: ...
            def f(x):
                return x
            """
    ).lstrip()
    (tmp_path / "mod.py").write_text(src)
    graph = make_analysis().materialize_all()
    overloads = [
        n
        for n in graph.nodes
        if n.fqname == "mod.f" and n.type == "function" and n.flags & NodeFlags.OVERLOAD
    ]
    assert len(overloads) == 1


# ---------------------------------------------------------------------------
# ``.pyi`` ingestion -- compiled-extension orphan stubs
# ---------------------------------------------------------------------------


def test_peer_pyi_is_skipped_when_py_twin_exists(tmp_path, make_analysis):
    (tmp_path / "mod.py").write_text("def f(x):\n    return x\n")
    (tmp_path / "mod.pyi").write_text("def stub_only(x: int) -> int: ...\n")

    graph = make_analysis().materialize_all()
    paths = {n.path.name for n in graph.nodes if n.type == "module"}
    assert paths == {"mod.py"}
    function_names = {n.fqname for n in graph.nodes if n.type == "function"}
    assert function_names == {"mod.f"}


def test_orphan_pyi_stub_uses_runtime_fqname(tmp_path, make_analysis):
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
    reachable = find_reachable(graph, _entrypoint_seeds(graph))

    stub_compute = next(
        n for n in graph.nodes if n.fqname == "mypkg._native.compute" and n.type == "function"
    )
    assert stub_compute.path.name == "_native.pyi"
    assert stub_compute in reachable, "orphan stub decl should be alive when imported"

    pkg_compute = next(n for n in graph.nodes if n.fqname == "mypkg.compute")
    assert stub_compute in graph.successors(pkg_compute)


def test_orphan_pyi_stub_deleted_when_unused(tmp_path, make_analysis):
    """An orphan stub that no entrypoint reaches is still removed by the codemod."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_native.pyi").write_text(
        "def compute(x: int) -> int: ...\ndef other(x: int) -> int: ...\n"
    )
    (pkg / "live.py").write_text("def keep():\n    return 1\n")

    a = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["mypkg.live.keep"])])
    pkg_view = a.package(tmp_path)
    pkg_view.remove_dead_code()

    assert not (pkg / "_native.pyi").exists()
