import textwrap

from typer.testing import CliRunner

from dead_cst._symbols import SymbolNode
from dead_cst.cli import _is_module_dunder, app


def test_is_module_dunder_matches_double_underscore_variables(tmp_path):
    path = tmp_path / "m.py"
    assert _is_module_dunder(SymbolNode("pkg.m.__all__", "variable", path))
    assert _is_module_dunder(SymbolNode("pkg.m.__version__", "variable", path))
    assert _is_module_dunder(SymbolNode("pkg.m.__author__", "variable", path))


def test_is_module_dunder_rejects_non_dunder_and_non_variables(tmp_path):
    path = tmp_path / "m.py"
    assert not _is_module_dunder(SymbolNode("pkg.m.foo", "variable", path))
    assert not _is_module_dunder(SymbolNode("pkg.m._private", "variable", path))
    assert not _is_module_dunder(SymbolNode("pkg.m.__mangled", "variable", path))
    assert not _is_module_dunder(SymbolNode("pkg.m.____", "variable", path))
    # Same-name function/class/module shouldn't be matched - only variables.
    assert not _is_module_dunder(SymbolNode("pkg.m.__all__", "function", path))
    assert not _is_module_dunder(SymbolNode("pkg.m.__all__", "module", path))


def _write_pkg(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "main.py").write_text(
        textwrap.dedent(
            """
            __version__ = "1.0.0"
            __author__ = "someone"
            __all__ = ["used"]

            used = 1
            unused = 2

            def main():
                return used
            """
        ).strip()
    )
    return tmp_path


def test_analyze_preserves_module_dunders_by_default(tmp_path):
    root = _write_pkg(tmp_path)
    result = CliRunner().invoke(app, ["analyze", str(root), "-e", "pkg.main.main"])
    # `unused` is dead, but the dunder vars should not be reported as dead.
    assert "pkg.main.unused" in result.stdout
    assert "pkg.main.__version__" not in result.stdout
    assert "pkg.main.__author__" not in result.stdout
    assert "pkg.main.__all__" not in result.stdout


def test_analyze_no_preserve_dunders_marks_them_dead(tmp_path):
    root = _write_pkg(tmp_path)
    result = CliRunner().invoke(
        app, ["analyze", str(root), "-e", "pkg.main.main", "--no-preserve-dunders"]
    )
    assert "pkg.main.__version__" in result.stdout
    assert "pkg.main.__author__" in result.stdout
    assert "pkg.main.__all__" in result.stdout
