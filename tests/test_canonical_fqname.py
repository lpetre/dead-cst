"""Tests for canonical (specificity-aware) file → fqname resolution.

dead-cst's reverse module lookup must agree with the forward import
resolver: if ``from lib_a import greet`` resolves to file F, then F's
fqname-in-graph must be ``lib_a`` so the cross-file edge connects. ty's
own ``file_to_module`` returns the *first* containing search path's
relative name, which goes wrong when ``project_root`` is on the list
alongside a ``.pth``-derived path it physically contains.

These tests pin the two endpoints of the fix from #222:

1. **Single-package, no flat ``.pth`` covering first-party code** (e.g.
   setuptools' default PEP 660 ``MetaPathFinder`` editable install):
   ``project_root`` must still rescue first-party resolution. Before
   the fix, ``Analysis(..., venv=...)`` suppressed ``project_root``
   from ty's search paths entirely; every cross-file import resolved
   to ``[unresolved]`` and the whole graph fell apart.

2. **Monorepo with flat ``.pth`` per workspace member**: the deeper
   ``.pth`` path must still win the fqname for files it covers — the
   workspace-rooted path mustn't shadow it. ``module`` nodes for
   ``packages/lib_a/src/lib_a/__init__.py`` should still come out as
   ``lib_a``, not ``packages.lib_a.src.lib_a``.
"""

from __future__ import annotations


from dead_cst import Analysis


def _module_fqnames(graph) -> dict[str, str]:
    """``{file_path: fqname}`` over every ``kind == "module"`` node."""
    return {n.path: n.fqname for n in graph.nodes() if n.kind == "module"}


def test_pep660_finder_install_resolves_via_project_root(
    tmp_path, write_files, make_workspace_venv
):
    """Reproduces #222: a venv whose ``.pth`` is a PEP 660 finder stub
    (no flat path entries) must not break first-party resolution. We
    simulate by handing ``Analysis`` a venv that has *no* editable
    ``.pth`` covering the project — ``project_root`` on ty's search
    paths is now the only thing keeping ``myapp.routes.job`` reachable.
    """
    write_files(
        {
            "myapp/__init__.py": "",
            "myapp/main.py": (
                "from myapp.routes.job import router as job_router\napp = job_router\n"
            ),
            "myapp/routes/__init__.py": "",
            "myapp/routes/job.py": "router = object()\n",
        }
    )
    # Empty venv -- mimics a setuptools PEP 660 finder install whose
    # ``.pth`` we can't see into. No flat path covers ``myapp/``, so the
    # only first-party search root is ``project_root`` itself.
    venv = make_workspace_venv({})

    analysis = Analysis(tmp_path, venv=venv)
    graph = analysis.materialize_all()
    by_path = _module_fqnames(graph)

    assert by_path[str(tmp_path / "myapp" / "__init__.py")] == "myapp"
    assert by_path[str(tmp_path / "myapp" / "main.py")] == "myapp.main"
    assert by_path[str(tmp_path / "myapp" / "routes" / "job.py")] == "myapp.routes.job"

    # The import alias in main.py must resolve to the real router var,
    # not to ``[unresolved] myapp``.
    job_router_alias = next(n for n in graph.nodes() if n.fqname == "myapp.main.job_router")
    router_var = next(n for n in graph.nodes() if n.fqname == "myapp.routes.job.router")
    # The alias's descendant set should reach the upstream var.
    assert router_var in graph.descendants(job_router_alias)


def test_monorepo_pth_beats_project_root_in_fqname(tmp_path, write_files, make_workspace_venv):
    """Specificity tiebreak: when both a deep ``.pth`` and ``project_root``
    contain the same file, the ``.pth``-derived fqname must win.

    Layout mirrors a uv workspace: ``packages/lib_a/src/lib_a/`` is the
    real source, ``packages/lib_a/src`` is the editable ``.pth`` entry,
    and ``project_root`` (=tmp_path) is also on ty's search paths now
    that suppression is gone.
    """
    write_files(
        {
            "packages/lib_a/src/lib_a/__init__.py": "def greet(name): return name\n",
            "packages/lib_b/src/lib_b/__init__.py": (
                "from lib_a import greet\ndef shout(n): return greet(n).upper()\n"
            ),
            "main.py": "from lib_a import greet\nprint(greet('x'))\n",
        }
    )
    venv = make_workspace_venv(
        {
            "lib_a": "packages/lib_a/src",
            "lib_b": "packages/lib_b/src",
        }
    )

    analysis = Analysis(tmp_path, venv=venv)
    graph = analysis.materialize_all()
    by_path = _module_fqnames(graph)

    # Deepest ``.pth`` wins: not ``packages.lib_a.src.lib_a``.
    assert (
        by_path[str(tmp_path / "packages" / "lib_a" / "src" / "lib_a" / "__init__.py")] == "lib_a"
    )
    assert (
        by_path[str(tmp_path / "packages" / "lib_b" / "src" / "lib_b" / "__init__.py")] == "lib_b"
    )
    # Files at the workspace root with no ``.pth`` coverage fall through
    # to ``project_root``, which is fine -- their short name is already
    # the canonical one.
    assert by_path[str(tmp_path / "main.py")] == "main"


def test_pth_with_top_level_module_keeps_short_fqname(tmp_path, write_files, make_workspace_venv):
    """An app published as a flat ``.pth`` (no ``src/`` layer) must
    surface its top-level module as the bare name (``handlers.job``),
    not as a workspace-rooted dotted name (``apps.app_two.handlers.job``).
    """
    write_files(
        {
            "apps/app_two/main.py": "from handlers.job import work\nwork()\n",
            "apps/app_two/handlers/__init__.py": "",
            "apps/app_two/handlers/job.py": "def work(): return 0\n",
        }
    )
    venv = make_workspace_venv({"app_two": "apps/app_two"})

    analysis = Analysis(tmp_path, venv=venv)
    graph = analysis.materialize_all()
    by_path = _module_fqnames(graph)

    assert by_path[str(tmp_path / "apps" / "app_two" / "handlers" / "job.py")] == "handlers.job"
    assert by_path[str(tmp_path / "apps" / "app_two" / "main.py")] == "main"
