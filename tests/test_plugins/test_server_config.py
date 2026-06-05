"""Tests for the native ``ServerConfigPlugin`` (``NativePlugin.server_config()``).

It is a *configured* per-file plugin: the matched filename set is carried as
config and the per-file pass is salsa-cached, keyed on the config (identical
filename sets share a cache entry). The behavioural tests assert the reachable
surface; the cache tests use the rust-side run-counter
(``_server_config_run_count`` / reset) to assert the salsa cache fires.
"""

from __future__ import annotations

import textwrap

from dead_cst import Analysis
from dead_cst import _native as native
from dead_cst.graph import NodeFlags


def _write(path, src: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src).strip() + "\n")


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
        [native.NativePlugin.server_config()],
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
        [native.NativePlugin.server_config()],
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
        [native.NativePlugin.server_config()],
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
        [native.NativePlugin.server_config()],
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
        [native.NativePlugin.server_config()],
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
        make_analysis(plugins=[native.NativePlugin.server_config()]).materialize_all()
    )
    assert "deploy.prod_gunicorn.workers" not in reached_default

    reached_override = reachable_fqnames(
        make_analysis(
            plugins=[native.NativePlugin.server_config(filenames=["prod_gunicorn.py"])]
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
        [native.NativePlugin.server_config()],
    )
    reached = reachable_fqnames(graph)
    assert "gunicorn.conf.CustomLogger" in reached
    assert "gunicorn.conf.logger_class" in reached


def test_server_config_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("server_config")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "ServerConfigPlugin"


def test_seeds_are_not_tagged_testcase(build_plugin_graph):
    # Server-config entrypoints are production code, not tests -- they
    # should seed reachability but never carry the ``test/testcase`` flag
    # (so ``kept_alive_by_flags_only(test/testcase)`` doesn't filter them).
    # Register pytest alongside so the ``test/testcase`` flag is registered
    # and there's a bit to check against.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "gunicorn.conf.py": "workers = 4",
        },
        [native.NativePlugin.server_config(), native.NativePlugin.pytest()],
    )
    testcase = graph.node_flag("test/testcase")
    assert testcase is not None, "pytest plugin should register the test/testcase flag"
    seeds = [n for n in graph.nodes() if n.flags & NodeFlags.ENTRYPOINT]
    server_seeds = [s for s in seeds if s.fqname.startswith("gunicorn.conf")]
    assert server_seeds
    for seed in server_seeds:
        assert not (seed.flags & testcase), seed.fqname


# ---------------------------------------------------------------------------
# Per-file salsa caching + config interning. server_config is a *configured*
# per-file plugin: the per-file pass is keyed on (file, Configured(id)), where
# id is hash-interned on the filename set. Unchanged files reuse cached ops
# across re_materialize, and identical configs collapse to one cache key.
# ---------------------------------------------------------------------------


def test_per_file_server_config_caches_unchanged_files(tmp_path):
    """Editing one file should re-run the per-file server_config plugin for
    *only* that file on ``re_materialize`` — every other file's result is
    served from the salsa cache (even though only the matched config file
    emits ops, the impl is invoked once per file)."""
    _write(tmp_path / "pkg/__init__.py", "")
    _write(tmp_path / "gunicorn.conf.py", "workers = 4\n")
    _write(tmp_path / "pkg/a.py", "def f(): pass\n")
    _write(tmp_path / "pkg/b.py", "def g(): pass\n")

    analysis = Analysis(tmp_path, plugins=[native.NativePlugin.server_config()])
    native._reset_server_config_run_count()
    analysis.materialize_all()
    # Cold build: the impl runs once per project file (gunicorn.conf, a, b, __init__).
    assert native._server_config_run_count() >= 4

    # Edit only b.py (never a server-config match). On re_materialize, every
    # other file hits the salsa cache; only b.py re-runs the impl.
    native._reset_server_config_run_count()
    _write(tmp_path / "pkg/b.py", "def g(): pass\ndef extra(): pass\n")
    analysis.re_materialize(analysis.materialize_all().detect_changes())
    assert native._server_config_run_count() == 1, (
        f"expected exactly 1 per-file re-run (the edited b.py), got "
        f"{native._server_config_run_count()} — salsa cache for unchanged files "
        "should have served their ops"
    )


def test_per_file_server_config_cache_invalidates_on_edit(tmp_path):
    """Editing the matched config file re-runs its per-file plugin and
    reflects the new top-level surface."""
    _write(tmp_path / "pkg/__init__.py", "")
    _write(tmp_path / "gunicorn.conf.py", "workers = 4\n")

    analysis = Analysis(tmp_path, plugins=[native.NativePlugin.server_config()])
    ctx = analysis.materialize_all()
    mod = next(n for n in ctx.nodes() if n.fqname == "gunicorn.conf")
    assert mod.flags & NodeFlags.ENTRYPOINT

    native._reset_server_config_run_count()
    _write(tmp_path / "gunicorn.conf.py", "workers = 4\nbind = '0.0.0.0:8000'\n")
    ctx2 = analysis.re_materialize(analysis.materialize_all().detect_changes())
    assert native._server_config_run_count() >= 1  # gunicorn.conf re-ran
    reached = {n.fqname for n in ctx2.reachable()}
    assert "gunicorn.conf.bind" in reached


def test_identical_filenames_intern_to_one_cache_key(tmp_path):
    """Two server_config plugins with the *same* filenames intern to one id,
    so they share a salsa cache entry — the per-file impl runs once per file,
    not once per plugin. Distinct filenames key separately (twice per file)."""
    _write(tmp_path / "pkg/__init__.py", "")
    _write(tmp_path / "gunicorn.conf.py", "workers = 4\n")
    _write(tmp_path / "pkg/a.py", "def f(): pass\n")

    # Same config twice -> one interned id -> one run per file.
    native._reset_server_config_run_count()
    Analysis(
        tmp_path,
        plugins=[
            native.NativePlugin.server_config(filenames=["gunicorn.conf.py"]),
            native.NativePlugin.server_config(filenames=["gunicorn.conf.py"]),
        ],
    ).materialize_all()
    count_same = native._server_config_run_count()

    # Distinct configs -> two ids -> two runs per file.
    native._reset_server_config_run_count()
    Analysis(
        tmp_path,
        plugins=[
            native.NativePlugin.server_config(filenames=["gunicorn.conf.py"]),
            native.NativePlugin.server_config(filenames=["other.conf.py"]),
        ],
    ).materialize_all()
    count_diff = native._server_config_run_count()

    assert count_same >= 1
    assert count_diff == 2 * count_same, (
        f"identical filename configs should share one cache key (got "
        f"{count_same} runs) and distinct configs should key separately (got "
        f"{count_diff}); expected the latter to be exactly double"
    )


def test_filename_order_and_dupes_intern_equal(tmp_path):
    """The config hash canonicalises filenames (sort + dedup), so configs that
    differ only in order/duplication share a cache key."""
    _write(tmp_path / "pkg/__init__.py", "")
    _write(tmp_path / "gunicorn.conf.py", "workers = 4\n")

    native._reset_server_config_run_count()
    Analysis(
        tmp_path,
        plugins=[
            native.NativePlugin.server_config(filenames=["a.py", "gunicorn.conf.py"]),
            native.NativePlugin.server_config(filenames=["gunicorn.conf.py", "a.py", "a.py"]),
        ],
    ).materialize_all()
    # Both plugins canonicalise to the same {a.py, gunicorn.conf.py} set ->
    # one interned id -> one run per file (2: gunicorn.conf, __init__).
    assert native._server_config_run_count() == 2
