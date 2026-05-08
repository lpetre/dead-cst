"""Tests for ``@overload`` flagging and ``.pyi`` stub linking."""

from __future__ import annotations

import textwrap

import pytest

from dead_cst import NodeFlags
from dead_cst.analyze import _find_reachable as find_reachable
from dead_cst.codemod import remove_code
from dead_cst.plugins import ExplicitEntrypointPlugin, PyiStubPlugin


def _normalise(s: str) -> str:
    s = textwrap.dedent(s)
    return s[1:] if s.startswith("\n") else s


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
    # Last def in source is the live impl, earlier are overloads.
    for ov in overloads:
        assert impl.position.start.line > ov.position.start.line
    # impl -> each overload edge so they share lifetime.
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
    for node in graph.nodes:
        if node.fqname == "mod.keep":
            graph.nodes[node]["entrypoint"] = True
    reachable = find_reachable(graph)
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
    reachable = find_reachable(graph)
    unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()
    remove_code(unreachable, tmp_path)

    rewritten = (tmp_path / "mod.py").read_text()
    assert rewritten.count("def f") == 3  # two overloads + impl
    assert rewritten.count("@overload") == 2


def test_pyi_module_gets_distinct_fqn(tmp_path, make_analysis):
    """Same-named ``.py`` and ``.pyi`` coexist under disjoint FQNs."""
    (tmp_path / "mod.py").write_text("def f(x):\n    return x\n")
    (tmp_path / "mod.pyi").write_text("def f(x: int) -> int: ...\n")

    graph = make_analysis().materialize_all()
    fqnames = {n.fqname for n in graph.nodes if n.type == "module"}
    assert "mod" in fqnames
    assert "mod.__pyi__" in fqnames


def test_pyi_decls_track_runtime_lifetime(tmp_path, make_analysis):
    """A ``.pyi`` decl is alive iff its ``.py`` twin is alive."""
    (tmp_path / "mod.py").write_text(
        _normalise(
            """
            def alive(x):
                return x

            def dead(x):
                return x
            """
        )
    )
    (tmp_path / "mod.pyi").write_text(
        _normalise(
            """
            def alive(x: int) -> int: ...
            def dead(x: int) -> int: ...
            """
        )
    )
    a = make_analysis(
        plugins=[
            PyiStubPlugin(),
            ExplicitEntrypointPlugin(specs=["mod.alive"]),
        ],
    )
    graph = a.materialize_all()
    reachable = find_reachable(graph)

    alive_stub = next(
        n for n in graph.nodes if n.fqname == "mod.__pyi__.alive" and n.type == "function"
    )
    dead_stub = next(
        n for n in graph.nodes if n.fqname == "mod.__pyi__.dead" and n.type == "function"
    )
    assert alive_stub in reachable, "stub for live runtime decl should be alive"
    assert dead_stub not in reachable, "stub for dead runtime decl should be dead"


def test_pyi_overloads_removed_with_dead_runtime_impl(tmp_path, make_analysis):
    """Dead runtime impl drags its ``.pyi`` overloads into deletion."""
    (tmp_path / "mod.py").write_text(
        _normalise(
            """
            def keep():
                return 1
            def kill():
                return 2
            keep()
            """
        )
    )
    (tmp_path / "mod.pyi").write_text(
        _normalise(
            """
            from typing import overload

            @overload
            def kill(x: int) -> int: ...
            @overload
            def kill(x: str) -> str: ...
            def kill(x): ...

            def keep() -> int: ...
            """
        )
    )
    a = make_analysis(
        plugins=[
            PyiStubPlugin(),
            ExplicitEntrypointPlugin(specs=["mod.keep"]),
        ],
    )
    pkg = a.package(tmp_path)
    pkg.remove_dead_code()

    rewritten = (tmp_path / "mod.pyi").read_text()
    assert "def keep" in rewritten
    assert "def kill" not in rewritten
    assert "@overload" not in rewritten


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
