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


def test_overload_stub_flag_set_on_stubs_only(build_decl_graph):
    """`NodeFlags.OVERLOAD` is set on `@typing.overload`-decorated stubs and
    not on the impl. Recognise both `@overload` (from-import / aliased) and
    the `@typing.overload` attribute form (plain `import typing`)."""
    graph = build_decl_graph(
        {
            "mod.py": """
            from typing import overload as ovl
            import typing

            @ovl
            def f(x: int) -> int: ...
            @typing.overload
            def f(x: str) -> str: ...
            def f(x):
                return x
            """
        }
    )
    # After build_decl_graph's dedent+strip the lines are:
    #   1: from typing import overload as ovl
    #   2: import typing
    #   3: (blank)
    #   4: @ovl
    #   5: def f(x: int) -> int: ...      <- stub
    #   6: @typing.overload
    #   7: def f(x: str) -> str: ...      <- stub
    #   8: def f(x):                      <- impl
    f_nodes = [n for n in graph.nodes() if n.fqname == "mod.f" and n.kind == "function"]
    by_line = {n.start_line: n for n in f_nodes}
    assert set(by_line) == {5, 7, 8}, f"unexpected lines: {sorted(by_line)}"
    assert by_line[5].flags & NodeFlags.OVERLOAD
    assert by_line[7].flags & NodeFlags.OVERLOAD
    assert not (by_line[8].flags & NodeFlags.OVERLOAD)


def test_impl_to_overload_anchor_edges(build_decl_graph):
    """Each in-file `@overload` stub gets an explicit `impl -> stub` edge,
    visible in `descendants(impl)`."""
    graph = build_decl_graph(
        {
            "mod.py": """
            from typing import overload

            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x):
                return x
            """
        }
    )
    # After dedent+strip lines are:
    #   1: from typing import overload
    #   2: (blank)
    #   3: @overload
    #   4: def f(x: int) -> int: ...
    #   5: @overload
    #   6: def f(x: str) -> str: ...
    #   7: def f(x):
    impl = next(n for n in graph.nodes() if n.fqname == "mod.f" and n.start_line == 7)
    stubs = [
        n
        for n in graph.nodes()
        if n.fqname == "mod.f" and n.kind == "function" and n.start_line in (4, 6)
    ]
    assert len(stubs) == 2
    descendants = set(graph.descendants(impl))
    assert all(s in descendants for s in stubs), (
        "Each overload stub should be a descendant of the impl via the anchor edge"
    )


def test_cross_module_import_reaches_impl_not_stubs(build_decl_graph):
    """`from mod import f` should produce exactly one decl edge per consumer
    use — to the impl, not the stubs (which are deliberately excluded from
    the cross-module trie via `NodeFlags.OVERLOAD`)."""
    graph = build_decl_graph(
        {
            "mod.py": """
            from typing import overload

            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x):
                return x
            """,
            "main.py": "from mod import f\nf(1)\n",
        }
    )
    main_alias = next(n for n in graph.nodes() if n.fqname == "main.f" and n.kind == "import")
    nodes = graph.nodes()
    # Direct successors of the import alias: filter to `mod.f` function
    # decls and assert none of them carry the OVERLOAD flag.
    targets = [
        nodes[v]
        for u, v, _ in graph.edges()
        if nodes[u] == main_alias and nodes[v].fqname == "mod.f" and nodes[v].kind == "function"
    ]
    assert targets, "import alias should reach at least one mod.f decl"
    assert all(not (t.flags & NodeFlags.OVERLOAD) for t in targets)


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
