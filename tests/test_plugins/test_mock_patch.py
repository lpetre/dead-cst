"""Tests for :class:`MockPatchPlugin`."""

from __future__ import annotations

import pytest

from dead_cst import Analysis
from dead_cst.contrib.mock_patch import PATCH_TARGET_PREFIX
from dead_cst.plugins import MockPatchPlugin, PytestPlugin, UnittestPlugin
from conftest import build_trees

# Each entry is a ``tests/test_lib.py`` body that should keep
# ``pkg.lib.helper`` alive via a string-fqname patch reference. The id
# names the recognized form so failures point at the specific shape.
_RECOGNIZED_FORMS: list[tuple[str, str]] = [
    (
        "decorator-from-unittest-mock",
        """
        from unittest.mock import patch

        @patch("pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "with-block-from-unittest-mock",
        """
        from unittest.mock import patch

        def test_helper():
            with patch("pkg.lib.helper") as m:
                m.return_value = 2
        """,
    ),
    (
        "aliased-patch-import",
        """
        from unittest.mock import patch as p

        @p("pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "from-unittest-import-mock",
        """
        from unittest import mock

        @mock.patch("pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "import-unittest-mock-dotted",
        """
        import unittest.mock

        @unittest.mock.patch("pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "third-party-mock-package",
        """
        from mock import patch

        @patch("pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "import-unittest-mock-as-um",
        """
        import unittest.mock as um

        @um.patch("pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "mocker-fixture",
        """
        def test_helper(mocker):
            mocker.patch("pkg.lib.helper")
        """,
    ),
    (
        "monkeypatch-setattr",
        """
        def test_helper(monkeypatch):
            monkeypatch.setattr("pkg.lib.helper", lambda: 2)
        """,
    ),
    (
        "monkeypatch-setattr-with-raising-kwarg",
        """
        def test_helper(monkeypatch):
            monkeypatch.setattr("pkg.lib.helper", lambda: 2, raising=False)
        """,
    ),
    (
        "monkeypatch-delattr",
        """
        def test_helper(monkeypatch):
            monkeypatch.delattr("pkg.lib.helper")
        """,
    ),
]


@pytest.mark.parametrize(
    "test_source",
    [src for _, src in _RECOGNIZED_FORMS],
    ids=[name for name, _ in _RECOGNIZED_FORMS],
)
def test_recognized_form_keeps_target_alive(tmp_path, write_files, reachable_fqnames, test_source):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": test_source,
        }
    )
    graph = Analysis(
        build_trees({tmp_path: []}),
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
        build_trees({tmp_path: []}),
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
        build_trees({tmp_path: []}),
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
        build_trees({tmp_path: []}),
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
        build_trees({tmp_path: []}),
        plugins=[MockPatchPlugin(), UnittestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_unresolved_target_is_harmless(tmp_path, write_files, reachable_fqnames):
    """Patches against third-party / non-existent fqnames have no
    first-party decl to keep alive; the test still runs to completion."""
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
        build_trees({tmp_path: []}),
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


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
        build_trees({tmp_path: []}),
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib" in reachable_fqnames(graph)


def test_only_marks_target_when_test_alive(tmp_path, write_files, reachable_fqnames):
    """A patch reference inside a dead decl doesn't promote anything.

    No entrypoint reaches ``isolated_helper``, so the synthesized
    ``isolated_helper -> <patch-target>:pkg.lib.helper`` edge never
    fires.
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
        build_trees({tmp_path: []}),
        plugins=[MockPatchPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


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
        build_trees({tmp_path: []}),
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    synthetics = {n.fqname for n in graph.nodes if n.type == "synthetic"}
    assert f"{PATCH_TARGET_PREFIX}helper" not in synthetics


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
        build_trees({tmp_path: []}),
        plugins=[MockPatchPlugin(), PytestPlugin()],
        project_root=tmp_path,
    ).materialize_all()
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_mock_patch_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("mock_patch")
    assert isinstance(plugin, MockPatchPlugin)
