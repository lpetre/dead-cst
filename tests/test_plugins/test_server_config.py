"""Tests for :class:`ServerConfigPlugin`."""

from __future__ import annotations

from dead_cst.plugins import ServerConfigPlugin


def test_gunicorn_conf_module_stays_alive(make_analysis, write_files, reachable_fqnames):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "gunicorn.conf" in reached
    assert "gunicorn.conf.bind" in reached
    assert "gunicorn.conf.workers" in reached
    assert "gunicorn.conf.on_starting" in reached
    assert "gunicorn.conf.post_fork" in reached


def test_hypercorn_conf_module_stays_alive(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "hypercorn.conf.py": """
            bind = ["0.0.0.0:8000"]

            async def shutdown_trigger():
                pass
            """,
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "hypercorn.conf" in reached
    assert "hypercorn.conf.bind" in reached
    assert "hypercorn.conf.shutdown_trigger" in reached


def test_underscore_naming_variants_match(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "gunicorn_conf.py": "workers = 4",
            "hypercorn_conf.py": "bind = ['0.0.0.0:8000']",
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "gunicorn_conf.workers" in reached
    assert "hypercorn_conf.bind" in reached


def test_imports_used_only_in_config_stay_alive(make_analysis, write_files, reachable_fqnames):
    # ``helpers.read_env`` is referenced only from ``gunicorn.conf.py``;
    # without the plugin marking the config module's imports as alive,
    # the cross-file edge would dead-end at an unreachable import node.
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.helpers.read_env" in reached


def test_unrelated_modules_are_not_affected(make_analysis, write_files, reachable_fqnames):
    # A file whose stem happens to be ``gunicorn`` but isn't the config
    # module shouldn't get any free entrypoints.
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/gunicorn.py": """
            def helper(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
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


def test_classes_in_config_stay_alive(make_analysis, write_files, reachable_fqnames):
    # A common gunicorn pattern: define a custom logging class inline
    # and reference it via ``logger_class``.
    write_files(
        {
            "pkg/__init__.py": "",
            "gunicorn.conf.py": """
            class CustomLogger:
                pass

            logger_class = CustomLogger
            """,
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "gunicorn.conf.CustomLogger" in reached
    assert "gunicorn.conf.logger_class" in reached


def test_server_config_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("server_config")
    assert isinstance(plugin, ServerConfigPlugin)


def test_seeds_are_not_tagged_testcase(make_analysis, write_files):
    # Server-config entrypoints are production code, not tests -- they
    # should seed reachability but not get filtered by
    # ``kept_alive_by_flags_only(NodeFlags.TESTCASE)``.
    write_files(
        {
            "pkg/__init__.py": "",
            "gunicorn.conf.py": "workers = 4",
        }
    )
    graph = make_analysis(plugins=[ServerConfigPlugin()]).materialize_all()
    seeds = [n for n, attrs in graph.nodes(data=True) if attrs.get("entrypoint")]
    server_seeds = [s for s in seeds if s.fqname.startswith("<server-config>:")]
    assert server_seeds
    for seed in server_seeds:
        assert not graph.nodes[seed].get("testcase"), seed.fqname
