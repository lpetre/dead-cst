"""Negative tests that document known gaps in the analysis.

Each case asserts the *current* edge set / graph shape and includes a
comment about the ideal behaviour. When the analyser is improved
these tests will start producing the commented-out result and will
begin to fail -- that is the signal to promote them into a real
behaviour test in ``test_declarations``, ``test_imports``,
``test_notebooks``, ``test_overloads_and_pyi``, etc.
"""

import pytest

from dead_cst.graph import NodeFlags
from dead_cst.plugins import ExplicitEntrypointPlugin


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        pytest.param(
            {
                "mod.py": """
                nums = [1, 2, 3]
                result = [last := n for n in nums]
                def use(): return last
                """,
            },
            # Per PEP 572, a walrus inside a comprehension binds its
            # target in the *containing* scope -- ``mod.last`` should
            # surface as a top-level decl and ``use``'s reference to
            # ``last`` should route to it.
            #
            # ty has a ``// TODO walrus in comprehensions is implicitly
            # nonlocal`` at
            # ``vendor/ruff/crates/ty_python_core/src/builder.rs:3605``,
            # so the walrus's ``DefinitionKind::NamedExpression``
            # currently lives in the comprehension scope rather than
            # the enclosing module scope. Our ``ingest_decls`` loop
            # iterates the module's global scope and so doesn't see
            # the leaked binding -- ``mod.last`` is never minted, the
            # ``use``-site reference goes unresolved, and reachability
            # treats ``last`` as if it were never written.
            {
                "mod.nums -> mod",
                "mod.result -> mod",
                "mod.result -> mod.nums",
                "mod.use -> mod",
            },
            id="comprehension-walrus-doesnt-leak-to-enclosing-scope",
        ),
    ],
)
def test_limitation(build_decl_graph, assert_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)


# ---------------------------------------------------------------------------
# Graph-shape limitations that don't fit the edge-set parametrize above.
# ---------------------------------------------------------------------------


def test_overload_decls_not_flagged_with_overload_bit(tmp_path, make_analysis):
    """Ideal: ``@overload`` stubs are flagged ``NodeFlags.OVERLOAD`` and
    excluded from cross-module lookup so ``from mod import f`` reaches
    only the impl, never an overload stub.

    Current rust behaviour: every ``def f`` (overload stubs + impl) is
    emitted as an unflagged function node. Three nodes for the FQN
    ``mod.f`` exist (good — they're disambiguated by position), but no
    bit marks which one is the impl. The codemod and cross-module
    lookup still work today because every same-FQN node is treated
    uniformly, but the ``OVERLOAD`` blast-radius query
    (``kept_alive_by_flags_only(NodeFlags.OVERLOAD)``) is meaningless.
    """
    (tmp_path / "mod.py").write_text(
        "from typing import overload\n"
        "\n"
        "@overload\n"
        "def f(x: int) -> int: ...\n"
        "@overload\n"
        "def f(x: str) -> str: ...\n"
        "def f(x):\n"
        "    return x\n"
        "\n"
        "f(1)\n"
    )
    graph = make_analysis().materialize_all()
    f_decls = [n for n in graph.nodes if n.type == "function" and n.fqname == "mod.f"]
    assert len(f_decls) == 3, [d.position.start.line for d in f_decls]
    # Ideally two of these would carry NodeFlags.OVERLOAD; currently none do.
    assert sum(1 for d in f_decls if d.flags & NodeFlags.OVERLOAD) == 0


def test_peer_pyi_is_not_filtered_when_py_twin_exists(tmp_path, make_analysis):
    """Ideal: ``mod.py`` next to ``mod.pyi`` causes the ``.pyi`` to be
    dropped at enumeration -- a peer stub never contributes nodes when
    a real source twin exists.

    Current rust behaviour: both files are ingested. The graph carries
    two ``mod`` module nodes (one per path) and any decl unique to the
    stub (``stub_only``) is present even though Python's importer
    would prefer the ``.py`` at runtime. ``find_module(\"mod\")`` could
    resolve to either node.
    """
    (tmp_path / "mod.py").write_text("def f(x):\n    return x\n")
    (tmp_path / "mod.pyi").write_text("def stub_only(x: int) -> int: ...\n")
    graph = make_analysis().materialize_all()
    module_paths = {n.path.name for n in graph.nodes if n.type == "module" and n.fqname == "mod"}
    assert module_paths == {"mod.py", "mod.pyi"}  # ideal: {"mod.py"}
    function_fqnames = {n.fqname for n in graph.nodes if n.type == "function"}
    assert "mod.stub_only" in function_fqnames  # ideal: only ``mod.f``


def test_orphan_pyi_stub_decls_are_seeded_as_entrypoints(tmp_path, make_analysis):
    """Ideal: an orphan ``.pyi`` stub (compiled-extension layout, no
    ``.py`` twin) declares an API surface but isn't itself an
    entrypoint -- when nothing imports it the codemod removes the
    ``.pyi`` and its decls.

    Current rust behaviour: every decl sourced from a ``.pyi`` is
    flagged ``ENTRYPOINT``, which keeps the stub alive even when no
    first-party code references it. ``pkg_view.remove_dead_code()``
    therefore leaves the orphan stub file untouched. Plugins that
    expect to surface unused stubs see them as alive.
    """
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_native.pyi").write_text(
        "def compute(x: int) -> int: ...\ndef other(x: int) -> int: ...\n"
    )
    (pkg / "live.py").write_text("def keep():\n    return 1\n")

    a = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["mypkg.live.keep"])])
    graph = a.materialize_all()
    stub_decls = [n for n in graph.nodes if n.fqname.startswith("mypkg._native.")]
    # Ideal: zero of these would carry ENTRYPOINT (the codemod would
    # then strip the orphan stub).
    assert all(d.flags & NodeFlags.ENTRYPOINT for d in stub_decls)

    a.package(tmp_path).remove_dead_code()
    # Ideal: not exists. The codemod sees the stubs as alive (from
    # the auto-entrypoint flag above) and leaves the file in place.
    assert (pkg / "_native.pyi").exists()


def test_malformed_notebook_does_not_emit_unparseable_synthetic(tmp_path, make_analysis):
    """Ideal: a ``.ipynb`` that isn't valid JSON falls through to the
    ``[unparseable] <module>`` synthetic that ``dead-cst`` uses for
    libcst parse errors, so ``why-alive`` can still report on it.

    Current rust behaviour: the rust backend treats the malformed
    notebook as an empty module (``module`` node only, no decls, no
    ``[unparseable]`` synthetic). Diagnostics about the JSON failure
    aren't surfaced at all -- callers see a silently empty file.
    """
    bad = tmp_path / "broken.ipynb"
    bad.write_text("{this is not json}")
    graph = make_analysis().materialize_all()
    unparseable = [n for n in graph.nodes if n.fqname.startswith("[unparseable]")]
    assert unparseable == []  # ideal: one entry pointing at ``bad``
    # The module node exists but is silent about the parse failure.
    assert any(n.type == "module" and n.path == bad for n in graph.nodes)


def test_notebook_fqname_keeps_ipynb_suffix(write_notebook, write_files, make_analysis):
    """Ideal: a notebook ``nb.ipynb`` mounts as the module ``nb``, so
    ``from nb import secret`` from a sibling ``.py`` *would* be able
    to resolve (and we'd then explicitly drop the edge to honour
    notebooks-are-not-importable). Today we get that behaviour for
    free through a different mechanism: the notebook's fqname carries
    the ``.ipynb`` suffix so cross-module imports can't match it.

    The downside is that the fqname doesn't match what users would
    type to reference the notebook (``why-alive nb.secret`` fails;
    they'd need ``why-alive nb.ipynb.secret``). Pinning the current
    shape so a future fix that drops the suffix is forced to also
    add the explicit cross-module-import exclusion.
    """
    write_notebook("nb.ipynb", ["def secret():\n    return 1\n"])
    write_files({"caller.py": "from nb import secret\nsecret()\n"})
    graph = make_analysis().materialize_all()
    notebook_decls = [n for n in graph.nodes if n.flags & NodeFlags.NOTEBOOK]
    fqnames = {n.fqname for n in notebook_decls}
    # Ideal: {"nb", "nb.secret"}.
    assert fqnames == {"nb.ipynb", "nb.ipynb.secret"}
