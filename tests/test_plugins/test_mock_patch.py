"""Tests for the native ``mock_patch`` plugin (``NativePlugin.mock_patch``)."""

from __future__ import annotations

import pytest

from dead_cst import _native as native

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
    # -- statically foldable string targets (see runtime/src/string_fold.rs)
    (
        "f-string-without-placeholders",
        """
        from unittest.mock import patch

        @patch(f"pkg.lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "implicit-concatenation",
        """
        from unittest.mock import patch

        @patch("pkg." "lib.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "string-addition",
        """
        from unittest.mock import patch

        @patch("pkg.lib" + ".helper")
        def test_helper(_): pass
        """,
    ),
    (
        "f-string-over-module-constant",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "f-string-over-chained-constants",
        """
        from unittest.mock import patch

        PKG = "pkg"
        MODULE = f"{PKG}.lib"

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "constant-declared-after-use",
        """
        from unittest.mock import patch

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass

        MODULE = "pkg.lib"
        """,
    ),
    (
        "constant-plus-literal-in-with-block",
        """
        from unittest.mock import patch

        MODULE: str = "pkg.lib"

        def test_helper():
            with patch(MODULE + ".helper") as m:
                m.return_value = 2
        """,
    ),
    (
        "mocker-f-string-over-constant",
        """
        MODULE = "pkg.lib"

        def test_helper(mocker):
            mocker.patch(f"{MODULE}.helper")
        """,
    ),
]


@pytest.mark.parametrize(
    "test_source",
    [src for _, src in _RECOGNIZED_FORMS],
    ids=[name for name, _ in _RECOGNIZED_FORMS],
)
def test_recognized_form_keeps_target_alive(build_plugin_graph, reachable_fqnames, test_source):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": test_source,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_class_method_target_resolves_to_class(build_plugin_graph, reachable_fqnames):
    """``patch("pkg.lib.Cls.method")`` walks back to the class node.

    Methods are not represented as their own graph nodes, so resolution
    must climb dotted segments until it finds the enclosing top-level
    decl (the class).
    """
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    assert "pkg.lib.Cls" in reachable_fqnames(graph)


def test_unrelated_patch_method_not_recognized(build_plugin_graph, reachable_fqnames):
    """``<flask_app>.patch("/url")`` is not a mock call and must not
    promote the string to an fqname reference."""
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.mock_patch()],
    )
    # ``handler`` itself isn't reachable (no entrypoint), so neither is
    # ``helper`` -- the plugin must not have hijacked the unrelated
    # ``app.patch`` call into an fqname reference.
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_no_imports_no_effect(build_plugin_graph, reachable_fqnames):
    """Files that don't import a recognized mock binding contribute
    nothing -- a bare ``patch("X")`` call could be anything."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from somewhere_else import patch

            @patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    # ``patch`` came from a non-mock module, so the plugin does not
    # treat the string as an fqname reference.
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_unittest_testcase_patch(build_plugin_graph, reachable_fqnames):
    """``@patch("X")`` on a ``unittest.TestCase`` test method works."""
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.unittest()],
    )
    assert "pkg.lib.helper" in reachable_fqnames(graph)


def test_unresolved_target_is_harmless(build_plugin_graph, reachable_fqnames):
    """Patches against third-party / non-existent fqnames have no
    first-party decl to keep alive; the test still runs to completion."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("third_party.lib.something")
            def test_one(_): pass
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_module_target_keeps_module_alive(build_plugin_graph, reachable_fqnames):
    """``patch("pkg.lib")`` keeps the module itself alive."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "VALUE = 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("pkg.lib")
            def test_helper(_): pass
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    assert "pkg.lib" in reachable_fqnames(graph)


def test_only_marks_target_when_test_alive(build_plugin_graph, reachable_fqnames):
    """A patch reference inside a dead decl doesn't promote anything.

    No entrypoint reaches ``isolated_helper``, so the synthesized
    ``isolated_helper -> pkg.lib.helper`` keep-alive edge never fires.
    """
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "isolated.py": """
            from unittest.mock import patch

            def isolated_helper():
                with patch("pkg.lib.helper") as m:
                    return m
            """,
        },
        [native.NativePlugin.mock_patch()],
    )
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_monkeypatch_setattr_object_form_not_treated_as_fqname(build_plugin_graph):
    """``monkeypatch.setattr(obj, "attr", value)`` has 3 positional args
    and is the object form -- the ``"attr"`` string must not be treated
    as a fqname reference, so no ``test -> helper`` patch edge is wired."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import pkg.lib

            def test_helper(monkeypatch):
                # 3 positional args: object form, "helper" is just an
                # attribute name on the pkg.lib module object, not an
                # fqname string.
                monkeypatch.setattr(pkg.lib, "helper", lambda: 2)
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    test_idx = by_fqname["tests.test_lib.test_helper"]
    helper_idx = by_fqname["pkg.lib.helper"]
    assert (test_idx, helper_idx, 0) not in [(s, d, f) for (s, d, f) in graph.edges()]


def test_mock_patch_emits_direct_owner_to_target_edge(build_plugin_graph):
    """The keep-alive is a direct ``owner -> target`` edge, not a relay
    node. Verify the edge is in the graph and no patch-target marker node
    remains."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            @patch("pkg.lib.helper")
            def test_helper(_): pass
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    nodes = graph.nodes()
    by_fqname = {n.fqname: i for i, n in enumerate(nodes)}
    owner_idx = by_fqname["tests.test_lib.test_helper"]
    target_idx = by_fqname["pkg.lib.helper"]
    assert (owner_idx, target_idx, 0) in [(s, d, f) for (s, d, f) in graph.edges()]
    assert not any("patch-target" in n.fqname for n in nodes)


def test_monkeypatch_setitem_not_recognized(build_plugin_graph, reachable_fqnames):
    """``monkeypatch.setitem(d, "key", value)`` patches a dict, not a
    symbol -- the string is a key, not a fqname."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            def test_helper(monkeypatch):
                d = {}
                monkeypatch.setitem(d, "pkg.lib.helper", 1)
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    assert "pkg.lib.helper" not in reachable_fqnames(graph)


def test_mock_patch_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("mock_patch")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "MockPatchPlugin"


def test_f_string_over_dunder_name_folds_to_the_test_module(build_plugin_graph, reachable_fqnames):
    """``patch(f"{__name__}.helper")`` folds ``__name__`` to the file's own
    module name, so the target resolves to ``tests.test_lib.helper``."""
    graph = build_plugin_graph(
        {
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from unittest.mock import patch

            def helper(): return 1

            def unpatched(): return 2

            @patch(f"{__name__}.helper")
            def test_helper(_): pass
            """,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    reached = reachable_fqnames(graph)
    assert "tests.test_lib.helper" in reached
    assert "tests.test_lib.unpatched" not in reached


# Each entry is a ``tests/test_lib.py`` body whose patch target must *not*
# fold: ``pkg.lib.helper`` stays dead because the string is genuinely
# unknown to static analysis and a wrong fold would fabricate a reference.
_REFUSED_FOLDS: list[tuple[str, str]] = [
    (
        "constant-shadowed-by-local-assignment",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        def test_helper():
            MODULE = "elsewhere"
            with patch(f"{MODULE}.helper"):
                pass
        """,
    ),
    (
        "constant-shadowed-by-parameter",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        def test_helper(MODULE):
            with patch(f"{MODULE}.helper"):
                pass
        """,
    ),
    (
        "constant-shadowed-by-loop-target",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        def test_helper():
            for MODULE in ["elsewhere"]:
                with patch(f"{MODULE}.helper"):
                    pass
        """,
    ),
    (
        "constant-rebound-at-module-level",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"
        MODULE = "elsewhere"

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "constant-declared-global-in-a-function",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        def setup():
            global MODULE
            MODULE = "elsewhere"

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "constant-bound-by-a-call",
        """
        from unittest.mock import patch

        MODULE = str("pkg.lib")

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "repr-conversion",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        @patch(f"{MODULE!r}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "format-spec",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        @patch(f"{MODULE:>7}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "debug-text",
        """
        from unittest.mock import patch

        MODULE = "pkg.lib"

        @patch(f"{MODULE=}.helper")
        def test_helper(_): pass
        """,
    ),
    (
        "imported-constant-is-not-followed",
        """
        from unittest.mock import patch
        from tests.consts import MODULE

        @patch(f"{MODULE}.helper")
        def test_helper(_): pass
        """,
    ),
]


@pytest.mark.parametrize(
    "test_source",
    [src for _, src in _REFUSED_FOLDS],
    ids=[name for name, _ in _REFUSED_FOLDS],
)
def test_refused_fold_leaves_target_dead(build_plugin_graph, reachable_fqnames, test_source):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def helper(): return 1",
            "tests/__init__.py": "",
            "tests/consts.py": 'MODULE = "pkg.lib"',
            "tests/test_lib.py": test_source,
        },
        [native.NativePlugin.mock_patch(), native.NativePlugin.pytest()],
    )
    assert "pkg.lib.helper" not in reachable_fqnames(graph)
