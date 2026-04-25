"""Tests for the ``_codemod`` module.

``RemoveDeadSymbols`` rewrites a single module, dropping declarations
whose FQN is in the supplied dead set. ``remove_code`` is the public
entry point: it routes a graph of dead nodes per file, runs
``RemoveDeadSymbols`` for defs / classes / variables, then hands the
``"import"``-typed nodes to libcst's ``RemoveImportsVisitor`` for a
second pass.

The transformer-level tests drive ``RemoveDeadSymbols`` directly with
hand-picked FQN sets so trailing-comma, decorator, and blank-line
behaviour can be exercised in isolation. The import-pruning tests go
through ``remove_code`` end-to-end because the wiring -- deriving
``(module, obj, asname)`` from each ``"import"`` node and sequencing
the two passes -- is what we own; the alias-rewriting itself is
libcst's responsibility.
"""

import textwrap

import pytest
from libcst.metadata import FullRepoManager, MetadataWrapper

from dead_cst._codemod import RemoveDeadSymbols, remove_code
from dead_cst._fqn import FixedFullyQualifiedNameProvider


@pytest.fixture
def apply_transformer(tmp_path):
    """Run ``RemoveDeadSymbols`` against ``src`` with the given dead FQNs.

    Writes ``src`` to ``tmp_path / filename`` (default ``mod.py``) so
    libcst's metadata pipeline can compute fully-qualified names the
    same way it does in production.
    """

    def _apply(src: str, dead_fqnames: set[str], filename: str = "mod.py") -> str:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src).lstrip("\n"))

        mgr = FullRepoManager(str(tmp_path), [str(path)], {FixedFullyQualifiedNameProvider})
        wrapper: MetadataWrapper = mgr.get_metadata_wrapper_for_path(str(path))
        return wrapper.visit(RemoveDeadSymbols(dead_fqnames)).code

    return _apply


@pytest.fixture
def run_remove_code(tmp_path):
    """Run ``remove_code`` end-to-end against an in-memory project.

    ``files`` maps relative paths to source. ``entrypoints`` lists FQNs
    to mark as graph entrypoints before computing reachability. Returns
    the rewritten contents of the file at ``primary``.
    """
    from dead_cst import build_symbol_graph, find_reachable

    def _run(
        files: dict[str, str],
        entrypoints: set[str],
        primary: str,
    ) -> str:
        for name, src in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(src).lstrip("\n"))
        graph = build_symbol_graph({tmp_path: []})
        for node in graph.nodes:
            if node.fqname in entrypoints:
                graph.nodes[node]["entrypoint"] = True
        reachable = find_reachable(graph)
        unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()
        remove_code(unreachable, tmp_path)
        return (tmp_path / primary).read_text()

    return _run


# ---------------------------------------------------------------------------
# Chained assignment / trailing-comma handling on ``Assign.targets``
# ---------------------------------------------------------------------------


class TestChainedAssign:
    """``leave_Assign`` rebuilds ``targets`` so dead names drop out cleanly."""

    def test_drops_first_target_in_chain(self, apply_transformer):
        result = apply_transformer(
            """
            a = b = 1
            def keep(): return b
            """,
            {"mod.a"},
        )
        assert result == "b = 1\ndef keep(): return b\n"

    def test_drops_last_target_in_chain(self, apply_transformer):
        result = apply_transformer(
            """
            a = b = 1
            def keep(): return a
            """,
            {"mod.b"},
        )
        assert result == "a = 1\ndef keep(): return a\n"

    def test_drops_middle_target_in_three_way_chain(self, apply_transformer):
        result = apply_transformer(
            """
            a = b = c = 1
            def keep(): return a + c
            """,
            {"mod.b"},
        )
        assert result == "a = c = 1\ndef keep(): return a + c\n"

    def test_dropping_all_targets_removes_whole_statement(self, apply_transformer):
        result = apply_transformer(
            """
            a = b = 1
            def keep(): pass
            """,
            {"mod.a", "mod.b"},
        )
        assert result == "def keep(): pass\n"

    def test_single_target_assign_removed_entirely(self, apply_transformer):
        result = apply_transformer(
            """
            x = 1
            def keep(): pass
            """,
            {"mod.x"},
        )
        assert result == "def keep(): pass\n"

    def test_tuple_unpacking_target_is_left_alone(self, apply_transformer):
        # ``a, b, c = 1, 2, 3`` is a single ``Assign`` with one tuple
        # target, so per-name pruning does not apply and the statement
        # is preserved even when ``mod.b`` is in the dead set.
        src = """
            a, b, c = 1, 2, 3
            def keep(): return a + b + c
            """
        result = apply_transformer(src, {"mod.b"})
        assert result == "a, b, c = 1, 2, 3\ndef keep(): return a + b + c\n"

    def test_list_target_unpacking_is_left_alone(self, apply_transformer):
        # ``[a, b] = ...`` is the list-target variant of tuple unpacking
        # -- the visitor produces decls for ``a`` and ``b`` but the
        # codemod still sees a single Assign target, so per-name
        # pruning is a no-op (parity with tuple unpacking).
        src = """
            [a, b] = [1, 2]
            def keep(): return a + b
            """
        result = apply_transformer(src, {"mod.b"})
        assert result == "[a, b] = [1, 2]\ndef keep(): return a + b\n"

    def test_nested_tuple_target_is_left_alone(self, apply_transformer):
        # Same idea for nested patterns like ``((a, b), c) = ...``.
        src = """
            ((a, b), c) = ((1, 2), 3)
            def keep(): return a + b + c
            """
        result = apply_transformer(src, {"mod.b"})
        assert result == "((a, b), c) = ((1, 2), 3)\ndef keep(): return a + b + c\n"


# ---------------------------------------------------------------------------
# Decorator stripping when removing a decorated def / class
# ---------------------------------------------------------------------------


class TestDecoratorStripping:
    """Removing a ``FunctionDef`` / ``ClassDef`` takes its decorators with it."""

    def test_function_decorators_removed_with_def(self, apply_transformer):
        result = apply_transformer(
            """
            import functools

            @functools.cache
            def dead():
                return 1

            def keep(): pass
            """,
            {"mod.dead"},
        )
        assert "functools.cache" not in result
        assert "dead" not in result
        assert "def keep(): pass" in result

    def test_stacked_function_decorators_all_removed(self, apply_transformer):
        result = apply_transformer(
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
        )
        assert "@deco_a" not in result
        assert "@deco_b" not in result
        assert "def keep(): pass" in result

    def test_class_decorators_removed_with_class(self, apply_transformer):
        result = apply_transformer(
            """
            def reg(cls): return cls

            @reg
            class Dead:
                pass

            class Keep:
                pass
            """,
            {"mod.Dead"},
        )
        assert "@reg" not in result
        assert "class Dead" not in result
        assert "class Keep:" in result

    def test_async_function_removed(self, apply_transformer):
        # ``async def`` is a ``FunctionDef`` with ``asynchronous`` set,
        # so ``leave_FunctionDef`` covers it without a separate handler.
        result = apply_transformer(
            """
            async def dead():
                return 1

            def keep(): pass
            """,
            {"mod.dead"},
        )
        assert "async def dead" not in result
        assert "def keep(): pass" in result

    def test_decorated_async_function_removed(self, apply_transformer):
        result = apply_transformer(
            """
            def deco(f): return f

            @deco
            async def dead():
                return 1

            def keep(): pass
            """,
            {"mod.dead"},
        )
        assert "@deco" not in result
        assert "async def dead" not in result
        assert "def keep(): pass" in result


# ---------------------------------------------------------------------------
# Whitespace handling around removed statements
# ---------------------------------------------------------------------------


class TestBlankLineCleanup:
    """libcst collapses leading blank lines on the removed node."""

    def test_blank_line_separated_middle_function(self, apply_transformer):
        result = apply_transformer(
            """
            def keep_first():
                pass


            def dead_middle():
                pass


            def keep_last():
                pass
            """,
            {"mod.dead_middle"},
        )
        # The two surrounding defs remain separated by a single blank
        # line each, with no orphan run of blank lines left behind.
        assert result == ("def keep_first():\n    pass\n\n\ndef keep_last():\n    pass\n")

    def test_leading_comment_attached_to_removed_def_goes_with_it(self, apply_transformer):
        # Leading comments and blank lines are owned by the following
        # statement node, so removing the def deletes them too.
        result = apply_transformer(
            """
            def keep(): pass

            # explainer for dead_fn
            def dead_fn():
                pass
            """,
            {"mod.dead_fn"},
        )
        assert "explainer for dead_fn" not in result
        assert "dead_fn" not in result
        assert result.startswith("def keep(): pass")


# ---------------------------------------------------------------------------
# AnnAssign and ClassDef sanity checks
# ---------------------------------------------------------------------------


class TestAnnAssignAndClass:
    def test_ann_assign_dropped_when_target_is_dead(self, apply_transformer):
        result = apply_transformer(
            """
            dead_var: int = 1
            keep_var: str = "x"
            """,
            {"mod.dead_var"},
        )
        assert result == 'keep_var: str = "x"\n'

    def test_ann_assign_without_value_also_dropped(self, apply_transformer):
        result = apply_transformer(
            """
            dead_var: int
            keep_var: str = "x"
            """,
            {"mod.dead_var"},
        )
        assert result == 'keep_var: str = "x"\n'

    def test_class_with_body_removed_whole(self, apply_transformer):
        result = apply_transformer(
            """
            class Dead:
                x = 1
                def m(self): pass

            def keep(): pass
            """,
            {"mod.Dead"},
        )
        assert "class Dead" not in result
        assert "def m" not in result
        assert "def keep(): pass" in result

    def test_class_with_single_base_removed(self, apply_transformer):
        result = apply_transformer(
            """
            class Base: pass

            class Dead(Base):
                pass

            def keep(): pass
            """,
            {"mod.Dead"},
        )
        assert "class Dead" not in result
        assert "class Base: pass" in result
        assert "def keep(): pass" in result

    def test_class_with_multiple_bases_and_metaclass_removed(self, apply_transformer):
        # Whole-node removal handles inheritance and the ``metaclass=``
        # keyword without any special-case logic.
        result = apply_transformer(
            """
            class A: pass
            class B: pass
            class Meta(type): pass

            class Dead(A, B, metaclass=Meta):
                pass

            def keep(): pass
            """,
            {"mod.Dead"},
        )
        assert "class Dead" not in result
        assert "metaclass=Meta" not in result
        assert "class A: pass" in result
        assert "class B: pass" in result
        assert "class Meta(type): pass" in result
        assert "def keep(): pass" in result


# ---------------------------------------------------------------------------
# Import pruning -- end-to-end via ``remove_code`` + ``RemoveImportsVisitor``
# ---------------------------------------------------------------------------


class TestImports:
    """``remove_code`` derives ``(module, obj, asname)`` per dead import.

    The actual alias rewriting is delegated to libcst's
    ``RemoveImportsVisitor``; what we own is the derivation from each
    ``"import"``-typed ``SymbolNode``: ``module`` from
    ``node.imports.module``, ``obj`` from ``node.imports.decl``, and
    ``asname`` from the bound name (last segment of the FQN) when it
    differs from the natural binding. These tests exercise that wiring
    by building a real graph and checking the rewritten output.
    """

    LIB = {"lib.py": "def used(): pass\ndef unused(): pass\n"}

    def test_partial_from_import_drops_named_alias(self, run_remove_code):
        result = run_remove_code(
            files={
                **self.LIB,
                "mod.py": """
                    from lib import used, unused
                    def main(): used()
                """,
            },
            entrypoints={"mod", "mod.main"},
            primary="mod.py",
        )
        assert result == "from lib import used\ndef main(): used()\n"

    def test_multiline_from_import_loses_dead_alias(self, run_remove_code):
        # Multi-line, parenthesised. The middle alias dies; libcst's
        # remover keeps the parens with the surviving aliases.
        files = {
            "lib.py": "def a(): pass\ndef b(): pass\ndef c(): pass\n",
            "mod.py": ("from lib import (\n    a,\n    b,\n    c,\n)\ndef main(): a(); c()\n"),
        }
        result = run_remove_code(
            files=files,
            entrypoints={"mod", "mod.main"},
            primary="mod.py",
        )
        assert "    b,\n" not in result
        assert "    a,\n" in result and "    c,\n" in result
        assert "def main(): a(); c()\n" in result

    def test_aliased_import_drops_using_asname(self, run_remove_code):
        # ``from lib import used as kept`` binds ``kept``. When ``kept``
        # is unused, ``remove_code`` should pass ``asname="kept"`` to
        # ``RemoveImportsVisitor`` so it matches and removes the line.
        result = run_remove_code(
            files={
                **self.LIB,
                "mod.py": """
                    from lib import used as kept
                    def main(): pass
                """,
            },
            entrypoints={"mod", "mod.main"},
            primary="mod.py",
        )
        assert result == "def main(): pass\n"

    def test_bare_import_dropped(self, run_remove_code):
        result = run_remove_code(
            files={
                **self.LIB,
                "mod.py": """
                    import lib
                    def main(): pass
                """,
            },
            entrypoints={"mod", "mod.main"},
            primary="mod.py",
        )
        assert result == "def main(): pass\n"

    def test_dead_import_used_only_by_removed_def_is_pruned(self, run_remove_code):
        # ``helper`` imports ``used`` but ``helper`` itself is dead.
        # After pass 1 strips ``helper``, pass 2 should drop the import.
        result = run_remove_code(
            files={
                **self.LIB,
                "mod.py": """
                    from lib import used
                    def helper(): used()
                    def main(): pass
                """,
            },
            entrypoints={"mod", "mod.main"},
            primary="mod.py",
        )
        assert result == "def main(): pass\n"

    def test_live_import_is_preserved(self, run_remove_code):
        # Defensive: when the import is reachable, ``remove_code`` does
        # not touch it (the dead set never contained it).
        result = run_remove_code(
            files={
                **self.LIB,
                "mod.py": """
                    from lib import used
                    def main(): used()
                """,
            },
            entrypoints={"mod", "mod.main"},
            primary="mod.py",
        )
        assert result == "from lib import used\ndef main(): used()\n"


# ---------------------------------------------------------------------------
# Module-level removal via ``remove_code``
# ---------------------------------------------------------------------------


def test_remove_code_unlinks_dead_module_files(tmp_path):
    """Module nodes in the unreachable graph trigger ``Path.unlink``."""
    from dead_cst import build_symbol_graph, find_reachable

    (tmp_path / "kept.py").write_text("def main(): pass\n")
    (tmp_path / "dropped.py").write_text("def helper(): pass\n")

    graph = build_symbol_graph({tmp_path: []})
    for node in graph.nodes:
        if node.fqname == "kept":
            graph.nodes[node]["entrypoint"] = True
    reachable = find_reachable(graph)
    unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()

    remove_code(unreachable, tmp_path)

    assert (tmp_path / "kept.py").exists()
    assert not (tmp_path / "dropped.py").exists()
