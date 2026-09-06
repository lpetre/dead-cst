"""End-to-end tests for :meth:`Analysis.dead_tests`.

A test is *dead* when nothing it references is reachable from a
non-test seed: it only exercises code that is itself dead in production.
The query returns the blast radius of dropping those tests — the tests
plus every decl only they keep alive — mirroring the shape of the
``test-only`` query, but scoped to tests that back nothing live.
"""

from __future__ import annotations

from dead_cst import _native as native

# ``pkg.app`` is the production entrypoint: it calls ``used`` and nothing
# else, so ``dead_helper`` / ``CONST`` / ``pkg.util.shared`` are alive
# only through the tests.
_PROJECT = {
    "pkg/__init__.py": "",
    "pkg/app.py": """
    from pkg.lib import used

    used()
    """,
    "pkg/lib.py": """
    from pkg import util

    def used():
        return 1

    def dead_helper():
        return util.shared()

    def mixed_helper():
        return used()

    CONST = 1
    """,
    "pkg/util.py": """
    def shared():
        return 1
    """,
}


def _pytest_analysis(make_analysis, extra_plugins=()):
    return make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.app"], []),
            native.NativePlugin.pytest(),
            *extra_plugins,
        ]
    )


def _fqnames(ctx, indices):
    return {n.fqname for n in ctx.nodes_at(sorted(indices))}


def test_test_exercising_only_dead_code_is_dead(make_analysis, write_files):
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import dead_helper, mixed_helper

            def test_dead():
                assert dead_helper() == 1

            def test_mixed():
                assert mixed_helper() == 1
            """,
        }
    )
    analysis = _pytest_analysis(make_analysis)
    ctx = analysis.materialize_all()
    dead = _fqnames(ctx, analysis.dead_tests())
    # The dead test plus everything only it kept alive: the helper, the
    # module it reaches through, and the now-unused import aliases.
    assert "tests.test_lib.test_dead" in dead
    assert "pkg.lib.dead_helper" in dead
    assert "pkg.util.shared" in dead
    assert "tests.test_lib.dead_helper" in dead
    # ``mixed_helper`` calls ``used`` (production-reachable), so the test
    # exercising it is live and nothing it reaches is reported.
    assert "tests.test_lib.test_mixed" not in dead
    assert "pkg.lib.mixed_helper" not in dead
    assert "pkg.lib.used" not in dead


def test_module_attribute_use_of_live_code_keeps_test_alive(make_analysis, write_files):
    """``pkg.lib.used()`` resolves to a direct decl edge, so a test that
    reaches live code only through a module import is still live."""
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import pkg.lib
            from pkg import util

            def test_attr():
                assert pkg.lib.used() == 1

            def test_submodule_attr():
                assert util.shared() == 1
            """,
        }
    )
    analysis = _pytest_analysis(make_analysis)
    ctx = analysis.materialize_all()
    dead = _fqnames(ctx, analysis.dead_tests())
    assert "tests.test_lib.test_attr" not in dead
    assert dead >= {"tests.test_lib.test_submodule_attr", "pkg.util.shared"}


def test_module_level_call_in_production_module_does_not_leak(make_analysis, write_files):
    """Importing from a module whose body calls a live function is not
    the test exercising that function: the per-test walk stops at
    module nodes."""
    write_files(
        {
            **_PROJECT,
            "pkg/lib.py": """
            def used():
                return 1

            def dead_helper():
                return 2

            used()
            """,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import dead_helper

            def test_dead():
                assert dead_helper() == 2
            """,
        }
    )
    analysis = _pytest_analysis(make_analysis)
    ctx = analysis.materialize_all()
    dead = _fqnames(ctx, analysis.dead_tests())
    assert dead >= {"tests.test_lib.test_dead", "pkg.lib.dead_helper"}
    assert "pkg.lib.used" not in dead


def test_test_touching_no_project_code_is_dead(make_analysis, write_files):
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_misc.py": """
            import os

            def test_nothing():
                assert os.getcwd()
            """,
        }
    )
    analysis = _pytest_analysis(make_analysis)
    ctx = analysis.materialize_all()
    assert "tests.test_misc.test_nothing" in _fqnames(ctx, analysis.dead_tests())


def test_decl_shared_with_live_test_survives(make_analysis, write_files):
    """A dead helper a *live* test also reaches is kept by that test, so
    it stays out of the blast radius even though a dead test uses it."""
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import dead_helper, used

            def test_dead():
                assert dead_helper() == 1

            def test_both():
                assert used() == 1
                assert dead_helper() == 1
            """,
        }
    )
    analysis = _pytest_analysis(make_analysis)
    ctx = analysis.materialize_all()
    dead = _fqnames(ctx, analysis.dead_tests())
    assert "tests.test_lib.test_dead" in dead
    assert "tests.test_lib.test_both" not in dead
    assert "pkg.lib.dead_helper" not in dead
    assert "pkg.util.shared" not in dead


def test_fixture_chain_to_live_code_keeps_test_alive(make_analysis, write_files):
    """A test reaching live code only through a fixture parameter is
    live; a fixture only dead tests use is a seed of its own
    (``test/fixture``) and so is not part of the blast radius."""
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import pytest
            from pkg.lib import used, dead_helper

            @pytest.fixture
            def live_fixture():
                return used()

            @pytest.fixture
            def dead_fixture():
                return dead_helper()

            def test_via_live_fixture(live_fixture):
                assert live_fixture == 1

            def test_via_dead_fixture(dead_fixture):
                assert dead_fixture == 1
            """,
        }
    )
    analysis = _pytest_analysis(make_analysis)
    ctx = analysis.materialize_all()
    dead = _fqnames(ctx, analysis.dead_tests())
    assert "tests.test_lib.test_via_live_fixture" not in dead
    assert "tests.test_lib.test_via_dead_fixture" in dead
    assert "tests.test_lib.dead_fixture" not in dead
    # The fixture seed still keeps ``dead_helper`` alive on its own.
    assert "pkg.lib.dead_helper" not in dead


def test_unittest_testcase_is_a_test(make_analysis, write_files):
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            import unittest
            from pkg.lib import dead_helper, used

            class DeadCase(unittest.TestCase):
                def test_it(self):
                    self.assertEqual(dead_helper(), 1)

            class LiveCase(unittest.TestCase):
                def test_it(self):
                    self.assertEqual(used(), 1)
            """,
        }
    )
    analysis = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.app"], []),
            native.NativePlugin.unittest(),
        ]
    )
    ctx = analysis.materialize_all()
    dead = _fqnames(ctx, analysis.dead_tests())
    assert dead >= {"tests.test_lib.DeadCase", "pkg.lib.dead_helper", "pkg.util.shared"}
    assert "tests.test_lib.LiveCase" not in dead


def test_no_test_plugin_means_no_dead_tests(make_analysis, write_files):
    write_files(
        {
            **_PROJECT,
            "tests/__init__.py": "",
            "tests/test_lib.py": """
            from pkg.lib import dead_helper

            def test_dead():
                assert dead_helper() == 1
            """,
        }
    )
    analysis = make_analysis(plugins=[native.NativePlugin.explicit([], ["pkg.app"], [])])
    analysis.materialize_all()
    assert analysis.dead_tests() == set()
