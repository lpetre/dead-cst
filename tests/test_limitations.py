"""Negative tests that document known gaps in the analysis.

Each case asserts the *current* edge set and includes a comment about
the ideal behaviour. When the analyser is improved these tests will
start producing the commented-out edges and will begin to fail -- that
is the signal to promote them into ``test_declarations`` or
``test_imports``.

Most cases document *libcst-specific* gaps the rust backend doesn't
share (per-access edges through star imports, source-order star
resolution, ...); they're tagged ``skip="rust"`` so the rust suite
stays clean. Cases tagged ``skip="libcst"`` are the inverse — gaps
unique to the rust backend (typically blocked on an upstream ty
TODO) where libcst handles the shape correctly.
"""

import pytest

from tests.conftest import case


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        # ------------------------------------------------------------------
        # Dynamic / runtime features
        # ------------------------------------------------------------------
        case(
            {
                "other.py": "def g(): pass\n",
                "mod.py": """
                from other import *
                def a(): g()
                """,
            },
            # ``from other import *`` is fanned out at the module level
            # and re-export-materialized as ``mod.g`` (an ``"import"``
            # node pointing at ``other.g``), so cross-module
            # ``from mod import g`` resolves correctly. Per-access
            # resolution inside ``a`` is still missing: ideally
            # ``mod.a -> other.g`` would also be present, but
            # ScopeProvider cannot bind the bare ``g`` reference back to
            # the star import, so the visitor never emits anything for
            # the call site. The rust backend resolves through star
            # imports via ty's type hierarchy and does emit per-access
            # edges.
            {
                "mod -> mod.g",
                "mod -> other",
                "mod -> other.g",
                "mod.a -> mod",
                "mod.g -> mod",
                "mod.g -> other",
                "mod.g -> other.g",
                "other.g -> other",
            },
            skip="rust",
            id="star-import-fans-out-but-misses-per-access-edge",
        ),
        case(
            {
                "mod.py": """
                x = 1
                del x
                """,
            },
            # ``del x`` removes the binding at runtime, so ``x`` is
            # effectively dead. The visitor does not model ``del`` and
            # keeps the declaration in the graph.
            {
                "mod -> mod.x",
                "mod.x -> mod",
            },
            skip="rust",
            id="del-does-not-remove-declaration",
        ),
        case(
            {
                "mod.py": """
                x = 1
                def f():
                    global x
                    x = 2
                f()
                print(x)
                """,
            },
            # ``global x`` means the inner ``x = 2`` writes through to
            # the module-level binding, so the module-level ``print(x)``
            # read at line 6 should produce ``mod -> mod.x``.
            #
            # Instead, two things conspire to break this:
            #   1. ``ScopeProvider`` reports BOTH the module-level
            #      ``x = 1`` and the inner ``x = 2`` as referents of the
            #      outer ``print(x)`` access (libcst attaches the inner
            #      assignment to the global scope's chain because of
            #      ``global x``).
            #   2. The flow filter's forward walk over the module body
            #      treats ``def f(): ...`` as a single statement and
            #      asks ``_referents_in(stmt)`` for any referents nested
            #      in it -- which finds the inner ``x = 2`` and (line
            #      151 of ``_flow.py``) replaces the live set with just
            #      that inner assignment, killing the module-level
            #      ``x = 1`` referent.
            # The inner ``x = 2`` is attributed to its enclosing
            # top-level decl ``mod.f``, so the ``print(x)`` access
            # produces a spurious ``mod -> mod.f`` edge that collapses
            # into the existing one from the ``f()`` call, and the real
            # ``mod -> mod.x`` edge never gets emitted.
            #
            # See ``dead_cst/_flow.py``'s module docstring:
            # ``global`` / ``nonlocal`` rebindings are explicitly listed
            # as not-yet-modelled.
            {
                "mod -> mod.f",
                "mod.f -> mod",
                "mod.x -> mod",
            },
            skip="rust",
            id="global-rebind-misattributes-outer-read",
        ),
        case(
            {
                "a.py": 'def g(): return "a"\n',
                "b.py": 'def g(): return "b"\n',
                "mod.py": """
                from a import *
                from b import *
                """,
                "consumer.py": "from mod import g\ng()\n",
            },
            # Python's runtime: the second ``from b import *`` overwrites
            # the first, so ``mod.g`` resolves to ``b.g``. The ideal edge
            # set would include ``consumer.g -> b.g`` and omit
            # ``consumer.g -> a.g``.
            #
            # ``_materialize_star_reexports`` is "first writer wins": it
            # sorts star records by ``(importer_fqname, target_module,
            # line, col)`` and creates a synthetic ``mod.g`` re-export
            # pointing at the first one to claim the name. Since stars
            # sort by target-module string (``"a" < "b"``), ``mod.g``
            # ends up pointing at ``a.g`` rather than ``b.g``. The
            # consumer's ``from mod import g`` routes through that
            # synthetic, so ``consumer.g -> a.g`` instead of
            # ``consumer.g -> b.g``.
            #
            # Both ``a.g`` and ``b.g`` are still kept alive via the
            # module-level fan-out (``mod -> a.g``, ``mod -> b.g``), so
            # this gap doesn't affect dead-code detection. Fixing it
            # requires propagating each star's source position into
            # ``star_records`` and replacing-on-collision in source
            # order rather than alphabetic-by-target order.
            #
            # The rust backend mints one node per star statement
            # rather than per name, so this libcst-specific bug
            # doesn't apply — there's no synthetic ``mod.g`` to
            # mis-route through.
            {
                "a.g -> a",
                "b.g -> b",
                "consumer -> a.g",
                "consumer -> consumer.g",
                "consumer -> mod",
                "consumer -> mod.g",
                "consumer.g -> a.g",
                "consumer.g -> consumer",
                "consumer.g -> mod",
                "consumer.g -> mod.g",
                "mod -> a",
                "mod -> a.g",
                "mod -> b",
                "mod -> b.g",
                "mod -> mod.g",
                "mod.g -> a",
                "mod.g -> a.g",
                "mod.g -> mod",
            },
            skip="rust",
            id="last-star-wins-not-implemented",
        ),
        # ------------------------------------------------------------------
        # Rust-specific gaps (blocked on upstream ty TODOs).
        # ------------------------------------------------------------------
        case(
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
            #
            # When ty grows the binding-in-enclosing-scope handling,
            # ``ingest_decls`` will pick the leaked binding up for
            # free and this test should be dropped (the libcst-side
            # already pins the ideal shape via
            # ``test_declarations`` ``walrus-comprehension-toplevel-leak-captured``).
            {
                "mod.nums -> mod",
                "mod.result -> mod",
                "mod.result -> mod.nums",
                "mod.use -> mod",
            },
            skip="libcst",
            id="comprehension-walrus-doesnt-leak-to-enclosing-scope",
        ),
    ],
)
def test_limitation(build_decl_graph, assert_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)
