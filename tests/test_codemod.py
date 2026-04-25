"""Tests for the ``RemoveDeadSymbols`` libcst transformer.

The transformer takes a set of dead fully-qualified names and rewrites a
single module, dropping or trimming declarations whose FQN is in the
set. These tests drive the transformer directly with a hand-picked
``dead_fqnames`` set so each behaviour is exercised in isolation,
without involving graph construction or reachability analysis.

The cases below pin the formatting decisions libcst makes when nodes
are removed: trailing commas in chained assignments, decorators
attached to a removed def, and the blank-line whitespace between
top-level statements. The import-related cases document the *current*
gap -- ``RemoveDeadSymbols`` has no ``leave_Import`` /
``leave_ImportFrom`` handlers, and ``remove_code`` skips nodes of type
``"import"`` outright -- and follow the same "pin the limitation"
pattern as ``test_limitations``.
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


# ---------------------------------------------------------------------------
# Import handling -- documenting the current gap
# ---------------------------------------------------------------------------


class TestImports:
    """``RemoveDeadSymbols`` does not currently rewrite import lines.

    The transformer has no ``leave_Import`` / ``leave_ImportFrom``
    handlers, and ``remove_code`` only forwards nodes of type
    ``function`` / ``class`` / ``variable`` / ``module`` to it. Even
    when an import's local name is in ``dead_fqnames``, the source is
    untouched. These tests pin that behaviour so a future patch that
    adds import pruning will surface here and force the maintainer to
    update the expectations (compare ``test_limitations``).
    """

    def test_partial_from_import_is_not_pruned(self, apply_transformer):
        src = textwrap.dedent(
            """
            from foo import a, b
            def keep(): return b
            """
        ).lstrip("\n")
        result = apply_transformer(src, {"mod.a"})
        assert result == src

    def test_multiline_from_import_is_not_pruned(self, apply_transformer):
        src = textwrap.dedent(
            """
            from foo import (
                a,
                b,
                c,
            )
            def keep(): return b
            """
        ).lstrip("\n")
        result = apply_transformer(src, {"mod.a", "mod.c"})
        assert result == src

    def test_aliased_import_is_not_pruned(self, apply_transformer):
        src = textwrap.dedent(
            """
            from foo import a as renamed
            def keep(): pass
            """
        ).lstrip("\n")
        result = apply_transformer(src, {"mod.renamed"})
        assert result == src

    def test_remove_code_silently_skips_import_typed_nodes(
        self, apply_transformer, tmp_path, monkeypatch
    ):
        # End-to-end check: ``remove_code`` does not raise when handed a
        # graph that only contains import nodes -- it just leaves the
        # file alone. This guards against accidentally adding a ``case``
        # for ``"import"`` without writing the corresponding transformer
        # logic.
        from dead_cst import build_symbol_graph, find_reachable

        (tmp_path / "lib.py").write_text("def used(): pass\ndef unused(): pass\n")
        (tmp_path / "mod.py").write_text("from lib import used, unused\ndef main(): used()\n")
        graph = build_symbol_graph({tmp_path: []})
        for node in graph.nodes:
            if node.fqname == "mod" or node.fqname == "mod.main":
                graph.nodes[node]["entrypoint"] = True
        reachable = find_reachable(graph)
        unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable]).copy()

        # ``mod.unused`` is the only dead node in ``mod.py`` and it is
        # an import, so the file should be unchanged after ``remove_code``.
        before = (tmp_path / "mod.py").read_text()
        remove_code(unreachable, tmp_path)
        after = (tmp_path / "mod.py").read_text()
        assert before == after


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
