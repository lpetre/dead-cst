"""Tests for :class:`ServerConfigPlugin`."""

from __future__ import annotations

from dead_cst import _native as native
from dead_cst.graph import NodeFlags
from dead_cst.contrib import ServerConfigPlugin


def test_gunicorn_conf_module_stays_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "gunicorn.conf.py": """
            bind = "0.0.0.0:8000"
            workers = 4

            def on_starting(server):
                pass

            def post_fork(server, worker):
                pass
            """,
        },
        [ServerConfigPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "gunicorn.conf" in reached
    assert "gunicorn.conf.bind" in reached
    assert "gunicorn.conf.workers" in reached
    assert "gunicorn.conf.on_starting" in reached
    assert "gunicorn.conf.post_fork" in reached


def test_hypercorn_conf_module_stays_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "hypercorn.conf.py": """
            bind = ["0.0.0.0:8000"]

            async def shutdown_trigger():
                pass
            """,
        },
        [ServerConfigPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "hypercorn.conf" in reached
    assert "hypercorn.conf.bind" in reached
    assert "hypercorn.conf.shutdown_trigger" in reached


def test_underscore_naming_variants_match(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "gunicorn_conf.py": "workers = 4",
            "hypercorn_conf.py": "bind = ['0.0.0.0:8000']",
        },
        [ServerConfigPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "gunicorn_conf.workers" in reached
    assert "hypercorn_conf.bind" in reached


def test_imports_used_only_in_config_stay_alive(build_plugin_graph, reachable_fqnames):
    # ``helpers.read_env`` is referenced only from ``gunicorn.conf.py``;
    # without the plugin marking the config module's imports as alive,
    # the cross-file edge would dead-end at an unreachable import node.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": """
            def read_env(name, default):
                return default
            """,
            "gunicorn.conf.py": """
            from pkg.helpers import read_env

            workers = read_env("WORKERS", 4)
            """,
        },
        [ServerConfigPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.helpers.read_env" in reached


def test_unrelated_modules_are_not_affected(build_plugin_graph, reachable_fqnames):
    # A file whose stem happens to be ``gunicorn`` but isn't the config
    # module shouldn't get any free entrypoints.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/gunicorn.py": """
            def helper(): pass
            """,
        },
        [ServerConfigPlugin()],
    )
    assert "pkg.gunicorn.helper" not in reachable_fqnames(graph)


def test_filenames_override(make_analysis, write_files, reachable_fqnames):
    # Custom deployment naming: the user's config lives at
    # ``deploy/prod_gunicorn.py``. Default ``filenames`` won't match;
    # explicit override does.
    write_files(
        {
            "pkg/__init__.py": "",
            "deploy/__init__.py": "",
            "deploy/prod_gunicorn.py": "workers = 8",
        }
    )
    reached_default = reachable_fqnames(
        make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    )
    assert "deploy.prod_gunicorn.workers" not in reached_default

    reached_override = reachable_fqnames(
        make_analysis(
            plugins=[ServerConfigPlugin(filenames=("prod_gunicorn.py",))]
        ).materialize_all()
    )
    assert "deploy.prod_gunicorn.workers" in reached_override


def test_classes_in_config_stay_alive(build_plugin_graph, reachable_fqnames):
    # A common gunicorn pattern: define a custom logging class inline
    # and reference it via ``logger_class``.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "gunicorn.conf.py": """
            class CustomLogger:
                pass

            logger_class = CustomLogger
            """,
        },
        [ServerConfigPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "gunicorn.conf.CustomLogger" in reached
    assert "gunicorn.conf.logger_class" in reached


def test_server_config_plugin_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("server_config")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "ServerConfigPlugin"


def test_native_server_config_matches_python_plugin(build_plugin_graph, reachable_fqnames):
    """``NativePlugin.server_config()`` produces the same reachable set
    as the default-filename ``ServerConfigPlugin()``."""
    files = {
        "pkg/__init__.py": "",
        "gunicorn.conf.py": """
        import os

        bind = "0.0.0.0:8000"
        workers = 4

        class CustomLogger: pass

        def on_starting(server):
            pass
        """,
        "hypercorn_conf.py": "loglevel = 'info'",
        "regular.py": "def untouched(): pass",
    }
    py_ctx = build_plugin_graph(files, [ServerConfigPlugin()])
    rs_ctx = build_plugin_graph(files, [native.NativePlugin.server_config()])
    assert reachable_fqnames(py_ctx) == reachable_fqnames(rs_ctx)


def test_seeds_are_not_tagged_testcase(build_plugin_graph):
    # Server-config entrypoints are production code, not tests -- they
    # should seed reachability but not get filtered by
    # ``kept_alive_by_flags_only(NodeFlags.TESTCASE)``.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "gunicorn.conf.py": "workers = 4",
        },
        [ServerConfigPlugin()],
    )
    seeds = [n for n in graph.nodes() if n.flags & NodeFlags.ENTRYPOINT]
    server_seeds = [s for s in seeds if s.fqname.startswith("<server-config>:")]
    assert server_seeds
    for seed in server_seeds:
        assert not (seed.flags & NodeFlags.TESTCASE), seed.fqname
