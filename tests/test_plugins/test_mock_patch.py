"""Tests for :class:`MockPatchPlugin`."""

from __future__ import annotations

from dead_cst import Analysis
from dead_cst.plugins import MockPatchPlugin, PytestPlugin, UnittestPlugin


def test_decorator_form_keeps_target_alive(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": """
            def helper():
                return 1
            """,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("pkg.lib.helper")
            def test_helper(mock_helper):
                pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_with_block_keeps_target_alive(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            def test_helper():
                with patch("pkg.lib.helper") as m:
                    m.return_value = 2
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_aliased_patch_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch as p

            @p("pkg.lib.helper")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_module_prefixed_patch(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest import mock

            @mock.patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_dotted_unittest_mock_patch(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import unittest.mock

            @unittest.mock.patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_third_party_mock_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from mock import patch

            @patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_aliased_mock_module_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import unittest.mock as um

            @um.patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_mocker_fixture_patch(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            def test_helper(mocker):
                mocker.patch("pkg.lib.helper")
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_class_method_target_resolves_to_class(tmp_path, write_files, reachable_fqnames):
    """``patch("pkg.lib.Cls.method")`` walks back to the class node.

    Methods are not represented as their own graph nodes, so resolution
    must climb dotted segments until it finds the enclosing top-level
    decl (the class).
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": """
            class Cls:
                def method(self): return 1
            """,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("pkg.lib.Cls.method")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.Cls" in reachable_fqnames(graph)


def test_unrelated_patch_method_not_recognized(tmp_path, write_files, reachable_fqnames):
    """``<flask_app>.patch("/url")`` is not a mock call and must not
    promote the string to an fqname reference."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "app.py": """
            class FakeApp:
                def patch(self, *_a, **_kw):
                    def deco(f): return f
                    return deco

            app = FakeApp()

            @app.patch("pkg.lib.helper")
            def handler(): return 1
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    # ``handler`` itself isn't reachable (no entrypoint), so neither is
    # ``helper`` -- the plugin must not have hijacked the unrelated
    # ``app.patch`` call into an fqname reference.
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_no_imports_no_effect(tmp_path, write_files, reachable_fqnames):
    """Files that don't import a recognized mock binding contribute
    nothing -- a bare ``patch("X")`` call could be anything."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from somewhere_else import patch

            @patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    # ``patch`` came from a non-mock module, so the plugin does not
    # treat the string as an fqname reference.
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_unittest_testcase_patch(tmp_path, write_files, reachable_fqnames):
    """``@patch("X")`` on a ``unittest.TestCase`` test method works."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import unittest
            from unittest.mock import patch

            class MyTests(unittest.TestCase):
                @patch("pkg.lib.helper")
                def test_helper(self, _): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_unresolved_target_is_harmless(tmp_path, write_files, reachable_fqnames):
    """Patches against third-party / non-existent fqnames don't
    promote anything but also don't raise."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("third_party.lib.something")
            def test_one(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "tests.test_lib.test_one" in reached
    # Unrelated symbol still alive because tests are entrypoints; the
    # patch target just has no first-party decl to keep alive.
    assert "pkg.lib.helper" not in reached


def test_module_target_keeps_module_alive(tmp_path, write_files, reachable_fqnames):
    """``patch("pkg.lib")`` keeps the module itself alive."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "VALUE = 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("pkg.lib")
            def test_helper(_): pass
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib" in reachable_fqnames(graph)


def test_only_marks_target_when_test_alive(tmp_path, write_files, reachable_fqnames):
    """Patches in a non-pytest, non-test file don't keep targets alive
    unless the enclosing decl is itself reachable.

    A file with no entrypoint plugin and no ``-e`` reference has no
    reachable nodes, so the plugin's edges from the dead test function
    to the patch synthetic never fire.
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "isolated.py": """
            from unittest.mock import patch

            def isolated_helper():
                with patch("pkg.lib.helper") as m:
                    return m
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    # No entrypoint reaches isolated.py's body, so the patch reference
    # doesn't promote anything.
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_monkeypatch_setattr_keeps_target_alive(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            def test_helper(monkeypatch):
                monkeypatch.setattr("pkg.lib.helper", lambda: 2)
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_monkeypatch_delattr_keeps_target_alive(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            def test_helper(monkeypatch):
                monkeypatch.delattr("pkg.lib.helper")
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_monkeypatch_setattr_object_form_not_treated_as_fqname(
    tmp_path, write_files, reachable_fqnames
):
    """``monkeypatch.setattr(obj, "attr", value)`` has 3 positional args
    and is the object form -- the ``"attr"`` string must not be treated
    as a fqname reference."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import pkg.lib

            def test_helper(monkeypatch):
                # 3 positional args: object form, "helper" is just an
                # attribute name on the pkg.lib module object, not an
                # fqname string. The reference to ``pkg.lib`` is what
                # keeps the module alive (and ``helper`` with it).
                monkeypatch.setattr(pkg.lib, "helper", lambda: 2)
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    # Sanity: "pkg.lib" is alive (imported) but the plugin specifically
    # didn't synthesize an extra ``<patch-target>:helper`` edge.
    nodes = list(graph.nodes)
    synthetics = [n.fqname for n in nodes if n.type == "synthetic"]
    assert "<patch-target>:helper" not in synthetics


def test_monkeypatch_setattr_with_raising_kwarg(tmp_path, write_files, reachable_fqnames):
    """``raising=False`` is a kwarg; the fqname form still has 2
    positional args."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            def test_helper(monkeypatch):
                monkeypatch.setattr("pkg.lib.helper", lambda: 2, raising=False)
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_monkeypatch_setitem_not_recognized(tmp_path, write_files, reachable_fqnames):
    """``monkeypatch.setitem(d, "key", value)`` patches a dict, not a
    symbol -- the string is a key, not a fqname."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            def test_helper(monkeypatch):
                d = {}
                monkeypatch.setitem(d, "pkg.lib.helper", 1)
            """,
        }
    )
    graph = Analysis(
        {tmp_path: []},
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    # ``setitem`` is not in our recognized methods; the string is a
    # dict key, not a symbol fqname.
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_mock_patch_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("mock_patch")
    assert isinstance(plugin, MockPatchPlugin)
