"""Tests for the ``_codemod`` module.

Each transformer case is a single-module Python source paired with the
dead FQN set to feed ``RemoveDeadSymbols`` and the exact rewritten
output expected. ``remove_code`` cases describe a small in-memory
project, the entrypoints to seed reachability, and the rewritten
contents of the file under inspection.

Both ``src`` and ``expected`` are dedented and have a single leading
newline stripped, so the literal indented-triple-quoted form lines up
visually inside ``pytest.param``. Trailing blank lines are preserved
because the codemod's whitespace decisions are part of what we're
pinning -- compare ``test_declarations`` for the matching declaration
side.
"""

import textwrap

import pytest
from libcst.metadata import FullRepoManager, MetadataWrapper

from dead_cst import Analysis
from dead_cst.analyze import _find_reachable as find_reachable
from dead_cst.codemod import RemoveDeadSymbols, remove_code
from dead_cst._fqn import FixedFullyQualifiedNameProvider
from dead_cst.resolvers import ManualResolver


def _normalise(s: str) -> str:
    """Dedent and strip up to one leading newline.

    The single-newline rule lets the literal triple-quoted form put
    its opening quote on its own line (which is the readable
    convention) without losing genuine leading blank lines that the
    codemod leaves behind.
    """
    s = textwrap.dedent(s)
    return s[1:] if s.startswith("\n") else s


@pytest.fixture
def apply_transformer(tmp_path):
    """Write ``src`` to ``tmp_path/mod.py`` and run ``RemoveDeadSymbols``.

    Resolves each requested FQN to its ``(fqname, position)`` pair via the
    symbol graph so the codemod's position-aware matching has the data it
    needs.
    """

    def _apply(src: str, dead_fqnames: set[str]) -> str:
        path = tmp_path / "mod.py"
        path.write_text(_normalise(src))
        graph = Analysis(tmp_path, resolvers=[ManualResolver(specs=["."])]).materialize_all()
        dead_decls = {(n.fqname, n.position) for n in graph.nodes if n.fqname in dead_fqnames}
        mgr = FullRepoManager(str(tmp_path), [str(path)], {FixedFullyQualifiedNameProvider})
        wrapper: MetadataWrapper = mgr.get_metadata_wrapper_for_path(str(path))
        return wrapper.visit(RemoveDeadSymbols(dead_decls)).code

    return _apply


@pytest.fixture
def apply_transformer_at_lines(tmp_path):
    """Run ``RemoveDeadSymbols`` keyed on ``(fqname, start_line)`` pairs.

    Used by shadowing cases where the same FQN binds at multiple
    positions; the FQN-only fixture above cannot disambiguate them.
    """

    def _apply(src: str, dead: set[tuple[str, int]]) -> str:
        path = tmp_path / "mod.py"
        path.write_text(_normalise(src))
        graph = Analysis(tmp_path, resolvers=[ManualResolver(specs=["."])]).materialize_all()
        dead_decls = {
            (n.fqname, n.position) for n in graph.nodes if (n.fqname, n.position.start.line) in dead
        }
        mgr = FullRepoManager(str(tmp_path), [str(path)], {FixedFullyQualifiedNameProvider})
        wrapper: MetadataWrapper = mgr.get_metadata_wrapper_for_path(str(path))
        return wrapper.visit(RemoveDeadSymbols(dead_decls)).code

    return _apply


@pytest.fixture
def run_remove_code(tmp_path):
    """Materialise ``files`` under ``tmp_path``, run ``remove_code``, return paths.

    Returns the ``tmp_path`` so the test can inspect rewritten contents
    and file existence directly. ``entrypoints`` is the set of FQNs to
    mark as graph entrypoints before computing reachability.
    """

    def _run(files: dict[str, str], entrypoints: set[str]) -> None:
        for name, src in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_normalise(src))
        graph = Analysis(tmp_path, resolvers=[ManualResolver(specs=["."])]).materialize_all()
        for node in graph.nodes:
            if node.fqname in entrypoints:
                graph.nodes[node]["entrypoint"] = True
        reachable = find_reachable(graph)
        unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()
        remove_code(unreachable, tmp_path)

    return _run


# ---------------------------------------------------------------------------
# RemoveDeadSymbols -- per-CST-shape removal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, dead_fqnames, expected",
    [
        # ------------------------------------------------------------------
        # Functions and decorators (FunctionDef, sync + async)
        # ------------------------------------------------------------------
        pytest.param(
            """
            def dead(): return 1
            def keep(): pass
            """,
            {"mod.dead"},
            """
            def keep(): pass
            """,
            id="function-removed",
        ),
        pytest.param(
            """
            async def dead(): return 1
            def keep(): pass
            """,
            {"mod.dead"},
            """
            def keep(): pass
            """,
            id="async-function-removed",
        ),
        pytest.param(
            """
            import functools

            @functools.cache
            def dead():
                return 1

            def keep(): pass
            """,
            {"mod.dead"},
            """
            import functools

            def keep(): pass
            """,
            id="decorated-function-takes-decorators",
        ),
        pytest.param(
            """
            def deco_a(f): return f
            def deco_b(f): return f

            @deco_a
            @deco_b
            def dead():
                return 1

            def keep(): pass
            """,
            {"mod.dead"},
            """
            def deco_a(f): return f
            def deco_b(f): return f

            def keep(): pass
            """,
            id="stacked-decorators-all-removed",
        ),
        pytest.param(
            """
            def deco(f): return f

            @deco
            async def dead():
                return 1

            def keep(): pass
            """,
            {"mod.dead"},
            """
            def deco(f): return f

            def keep(): pass
            """,
            id="decorated-async-function-removed",
        ),
        # ------------------------------------------------------------------
        # Classes (ClassDef) -- bases, metaclass, decorators
        # ------------------------------------------------------------------
        pytest.param(
            """
            class Dead:
                x = 1
                def m(self): pass
            def keep(): pass
            """,
            {"mod.Dead"},
            """
            def keep(): pass
            """,
            id="class-with-body-removed",
        ),
        pytest.param(
            """
            def reg(cls): return cls

            @reg
            class Dead:
                pass

            class Keep:
                pass
            """,
            {"mod.Dead"},
            """
            def reg(cls): return cls

            class Keep:
                pass
            """,
            id="decorated-class-takes-decorator",
        ),
        pytest.param(
            """
            class Base: pass
            class Dead(Base):
                pass
            def keep(): pass
            """,
            {"mod.Dead"},
            """
            class Base: pass
            def keep(): pass
            """,
            id="class-with-single-base",
        ),
        pytest.param(
            """
            class A: pass
            class B: pass
            class Meta(type): pass
            class Dead(A, B, metaclass=Meta):
                pass
            def keep(): pass
            """,
            {"mod.Dead"},
            """
            class A: pass
            class B: pass
            class Meta(type): pass
            def keep(): pass
            """,
            id="class-with-multiple-bases-and-metaclass",
        ),
        # ------------------------------------------------------------------
        # Assign -- chained, single, destructuring
        # ------------------------------------------------------------------
        pytest.param(
            """
            x = 1
            def keep(): pass
            """,
            {"mod.x"},
            """
            def keep(): pass
            """,
            id="single-target-assign-removed-entirely",
        ),
        pytest.param(
            """
            a = b = 1
            def keep(): return b
            """,
            {"mod.a"},
            """
            b = 1
            def keep(): return b
            """,
            id="chained-assign-drops-first-target",
        ),
        pytest.param(
            """
            a = b = 1
            def keep(): return a
            """,
            {"mod.b"},
            """
            a = 1
            def keep(): return a
            """,
            id="chained-assign-drops-last-target",
        ),
        pytest.param(
            """
            a = b = c = 1
            def keep(): return a + c
            """,
            {"mod.b"},
            """
            a = c = 1
            def keep(): return a + c
            """,
            id="chained-assign-drops-middle-target",
        ),
        pytest.param(
            """
            a = b = 1
            def keep(): pass
            """,
            {"mod.a", "mod.b"},
            """
            def keep(): pass
            """,
            id="chained-assign-all-targets-dead-removes-statement",
        ),
        pytest.param(
            """
            a, b, c = 1, 2, 3
            def keep(): return a + b + c
            """,
            {"mod.b"},
            """
            a, b, c = 1, 2, 3
            def keep(): return a + b + c
            """,
            id="tuple-unpacking-not-pruned-per-name",
        ),
        pytest.param(
            """
            [a, b] = [1, 2]
            def keep(): return a + b
            """,
            {"mod.b"},
            """
            [a, b] = [1, 2]
            def keep(): return a + b
            """,
            id="list-target-unpacking-not-pruned-per-name",
        ),
        pytest.param(
            """
            ((a, b), c) = ((1, 2), 3)
            def keep(): return a + b + c
            """,
            {"mod.b"},
            """
            ((a, b), c) = ((1, 2), 3)
            def keep(): return a + b + c
            """,
            id="nested-tuple-unpacking-not-pruned-per-name",
        ),
        # ------------------------------------------------------------------
        # AnnAssign -- with and without value
        # ------------------------------------------------------------------
        pytest.param(
            """
            dead_var: int = 1
            keep_var: str = "x"
            """,
            {"mod.dead_var"},
            """
            keep_var: str = "x"
            """,
            id="ann-assign-with-value-removed",
        ),
        pytest.param(
            """
            dead_var: int
            keep_var: str = "x"
            """,
            {"mod.dead_var"},
            """
            keep_var: str = "x"
            """,
            id="ann-assign-without-value-removed",
        ),
        # ------------------------------------------------------------------
        # PEP 695 ``type`` statements
        # ------------------------------------------------------------------
        pytest.param(
            """
            type Dead = int
            type Keep = str
            """,
            {"mod.Dead"},
            """
            type Keep = str
            """,
            id="type-alias-removed",
        ),
        pytest.param(
            """
            type Dead[T] = list[T]
            type Keep = str
            """,
            {"mod.Dead"},
            """
            type Keep = str
            """,
            id="generic-type-alias-removed",
        ),
        pytest.param(
            """
            type Keep = int
            type Dead = Keep
            def f(x: Keep) -> Keep: return x
            """,
            {"mod.Dead"},
            """
            type Keep = int
            def f(x: Keep) -> Keep: return x
            """,
            id="type-alias-removed-keeps-live-sibling-and-users",
        ),
        # ------------------------------------------------------------------
        # Whitespace handling around removed statements
        # ------------------------------------------------------------------
        pytest.param(
            """
            def keep_first():
                pass


            def dead_middle():
                pass


            def keep_last():
                pass
            """,
            {"mod.dead_middle"},
            """
            def keep_first():
                pass


            def keep_last():
                pass
            """,
            id="middle-function-removal-collapses-blank-lines",
        ),
        pytest.param(
            """
            def keep(): pass

            # explainer for dead_fn
            def dead_fn():
                pass
            """,
            {"mod.dead_fn"},
            """
            def keep(): pass
            """,
            id="leading-comment-attached-to-removed-def-goes-with-it",
        ),
    ],
)
def test_remove_dead_symbols(apply_transformer, src, dead_fqnames, expected):
    assert apply_transformer(src, dead_fqnames) == _normalise(expected)


# ---------------------------------------------------------------------------
# RemoveDeadSymbols -- shadowing (same FQN, multiple positions)
#
# The codemod keys on ``(fqname, position)``, so a dead shadowed binding
# does not drag its live sibling out, and a live binding can be removed
# without touching a shadowed predecessor that happens to share its
# name. ``dead`` is a set of ``(fqname, start_line)`` pairs; lines are
# 1-based and counted from the first non-blank line of the dedented
# source (matching ``_normalise``).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, dead, expected",
    [
        pytest.param(
            """
            def f(): return 1
            def f(): return 2
            """,
            {("mod.f", 1)},
            """
            def f(): return 2
            """,
            id="shadowed-function-removed-keeps-live",
        ),
        pytest.param(
            """
            def f(): return 1
            def f(): return 2
            """,
            {("mod.f", 2)},
            """
            def f(): return 1
            """,
            id="live-function-removed-keeps-shadowed",
        ),
        pytest.param(
            """
            def f(): return 1
            def f(): return 2
            def f(): return 3
            """,
            {("mod.f", 2)},
            """
            def f(): return 1
            def f(): return 3
            """,
            id="middle-of-three-shadowed-removed",
        ),
        pytest.param(
            """
            class C:
                a = 1
            class C:
                b = 2
            """,
            {("mod.C", 1)},
            """
            class C:
                b = 2
            """,
            id="shadowed-class-removed",
        ),
        pytest.param(
            """
            x = 1
            x = 2
            """,
            {("mod.x", 1)},
            """
            x = 2
            """,
            id="shadowed-variable-removed",
        ),
        pytest.param(
            """
            x: int = 1
            x: str = "y"
            """,
            {("mod.x", 1)},
            """
            x: str = "y"
            """,
            id="shadowed-ann-assign-removed",
        ),
        pytest.param(
            """
            if cond:
                def f(): return 1
            else:
                def f(): return 2
            """,
            {("mod.f", 2)},
            """
            if cond:
                pass
            else:
                def f(): return 2
            """,
            id="conditional-branch-def-removed-leaves-pass",
        ),
        pytest.param(
            """
            def f(): return 1
            def f(): return 2
            def keep(): pass
            """,
            {("mod.f", 1), ("mod.f", 2)},
            """
            def keep(): pass
            """,
            id="all-same-name-decls-removed",
        ),
    ],
)
def test_remove_dead_symbols_disambiguates_by_position(
    apply_transformer_at_lines, src, dead, expected
):
    assert apply_transformer_at_lines(src, dead) == _normalise(expected)


# ---------------------------------------------------------------------------
# remove_code end-to-end -- import pruning via RemoveImportsVisitor
# ---------------------------------------------------------------------------


_LIB = "def used(): pass\ndef unused(): pass\n"


@pytest.mark.parametrize(
    "files, entrypoints, expected_mod",
    [
        pytest.param(
            {
                "lib.py": _LIB,
                "mod.py": """
                from lib import used, unused
                def main(): used()
                """,
            },
            {"mod", "mod.main"},
            """
            from lib import used
            def main(): used()
            """,
            id="partial-from-import-drops-named-alias",
        ),
        pytest.param(
            {
                "lib.py": "def a(): pass\ndef b(): pass\ndef c(): pass\n",
                "mod.py": ("from lib import (\n    a,\n    b,\n    c,\n)\ndef main(): a(); c()\n"),
            },
            {"mod", "mod.main"},
            ("from lib import (\n    a,\n    c,\n)\ndef main(): a(); c()\n"),
            id="multiline-from-import-loses-dead-alias",
        ),
        pytest.param(
            {
                "lib.py": _LIB,
                "mod.py": """
                from lib import used as kept
                def main(): pass
                """,
            },
            {"mod", "mod.main"},
            """
            def main(): pass
            """,
            id="aliased-import-drops-using-asname",
        ),
        pytest.param(
            {
                "lib.py": _LIB,
                "mod.py": """
                import lib
                def main(): pass
                """,
            },
            {"mod", "mod.main"},
            """
            def main(): pass
            """,
            id="bare-import-dropped",
        ),
        pytest.param(
            {
                "lib.py": _LIB,
                "mod.py": """
                from lib import used
                def helper(): used()
                def main(): pass
                """,
            },
            {"mod", "mod.main"},
            """
            def main(): pass
            """,
            id="import-only-used-by-removed-def-also-pruned",
        ),
        pytest.param(
            {
                "lib.py": _LIB,
                "mod.py": """
                from lib import used
                def main(): used()
                """,
            },
            {"mod", "mod.main"},
            """
            from lib import used
            def main(): used()
            """,
            id="live-import-preserved",
        ),
        pytest.param(
            {
                "mod.py": """
                type Dead = int
                type Keep = str
                def main() -> Keep: return "x"
                """,
            },
            {"mod", "mod.main"},
            """
            type Keep = str
            def main() -> Keep: return "x"
            """,
            id="dead-type-alias-removed-end-to-end",
        ),
    ],
)
def test_remove_code_rewrites_imports(run_remove_code, tmp_path, files, entrypoints, expected_mod):
    run_remove_code(files, entrypoints)
    assert (tmp_path / "mod.py").read_text() == _normalise(expected_mod)


# ---------------------------------------------------------------------------
# remove_code end-to-end -- module-level removal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files, entrypoints, expected_files",
    [
        pytest.param(
            {
                "kept.py": "def main(): pass\n",
                "dropped.py": "def helper(): pass\n",
            },
            {"kept"},
            {"kept.py": True, "dropped.py": False},
            id="unreachable-module-file-unlinked",
        ),
    ],
)
def test_remove_code_unlinks_dead_module_files(
    run_remove_code, tmp_path, files, entrypoints, expected_files
):
    run_remove_code(files, entrypoints)
    for relpath, should_exist in expected_files.items():
        assert (tmp_path / relpath).exists() is should_exist
