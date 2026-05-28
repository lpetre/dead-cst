"""Negative tests that document known gaps in the analysis.

Each case asserts the *current* edge set and includes a comment about
the ideal behaviour. When the analyser is improved these tests will
start producing the commented-out edges and will begin to fail -- that
is the signal to promote them into ``test_declarations`` or
``test_imports``.
"""

import pytest


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


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        pytest.param(
            # Uses inside an assignment whose *target* is a subscript /
            # slice (``x[...] = ...``) are dropped entirely -- neither the
            # subscripted object on the LHS nor any name on the RHS emits
            # a use edge. Here ``os`` (loaded to build ``os.environ[...]``),
            # ``f`` (loaded to build ``f[:]``), and ``SomeClass`` (called
            # on the RHS) all end up with ZERO in-edges, so each looks
            # dead and the codemod would happily delete ``import os``,
            # ``from a import SomeClass``, and ``f = []`` -- breaking the
            # module at runtime.
            #
            # Ideal: the use sites should add
            #     "mod -> mod.os@1:7",          # os.environ[...] = ...
            #     "mod -> mod.f@4:0",           # f[:] = ...
            #     "mod -> mod.SomeClass@3:14",  # ... = [SomeClass()]
            #     "mod -> a",
            #     "mod -> a.SomeClass@1:0",
            {
                "a.py": "class SomeClass: pass\n",
                "mod.py": (
                    "import os\n"
                    'os.environ["k"] = "v"\n'
                    "from a import SomeClass\n"
                    "f = []\n"
                    "f[:] = [SomeClass()]\n"
                ),
            },
            {
                "a.SomeClass@1:0 -> a",
                "mod.SomeClass@3:14 -> a",
                "mod.SomeClass@3:14 -> a.SomeClass@1:0",
                "mod.SomeClass@3:14 -> mod",
                "mod.f@4:0 -> mod",
                "mod.os@1:7 -> mod",
            },
            id="subscript-assignment-target-drops-uses",
        ),
        pytest.param(
            # Two sibling submodule imports that share the same root
            # binding (``import a.foo`` then ``import a.bar`` both bind the
            # local name ``a``). The use sites ``a.foo.x()`` and
            # ``a.bar.z()`` resolve their upstream module/decl edges
            # correctly, but the *alias* edge from each use lands only on
            # the LAST ``a`` binding (``mod.a@2:7``). The first binding
            # (``mod.a@1:7`` -- the ``import a.foo`` statement) gets ZERO
            # in-edges, so the codemod would drop ``import a.foo`` and
            # break ``a.foo.x()`` (importing ``a.bar`` does not import the
            # ``a.foo`` submodule).
            #
            # Ideal: ``a.foo.x()`` should keep its own binding alive --
            #     "mod -> mod.a@1:7",
            # (and, symmetrically, both uses arguably edge to both
            # reaching ``a`` bindings).
            {
                "a/__init__.py": "",
                "a/foo.py": "def x(): pass\n",
                "a/bar.py": "def z(): pass\n",
                "mod.py": "import a.foo\nimport a.bar\na.foo.x()\na.bar.z()\n",
            },
            {
                "a.bar -> a",
                "a.bar.z@1:0 -> a.bar",
                "a.foo -> a",
                "a.foo.x@1:0 -> a.foo",
                "mod -> a.bar",
                "mod -> a.bar.z@1:0",
                "mod -> a.foo",
                "mod -> a.foo.x@1:0",
                "mod -> mod.a@2:7",
                "mod.a@1:7 -> a.foo",
                "mod.a@1:7 -> mod",
                "mod.a@2:7 -> a.bar",
                "mod.a@2:7 -> mod",
            },
            id="sibling-submodule-imports-share-root-binding",
        ),
        pytest.param(
            # ``if TYPE_CHECKING: from a import X`` / ``else: from b
            # import X`` -- the two branches bind ``SomeClass`` to
            # different upstreams. A use of ``SomeClass`` has two reaching
            # defs, but only the if-branch import (``mod.SomeClass@3:18``,
            # from ``a``) picks up the use edge. The else-branch import
            # (``mod.SomeClass@5:18``, from ``b``) gets ZERO in-edges.
            # Since ``TYPE_CHECKING`` is False at runtime, the else branch
            # is the one that actually executes -- so the codemod would
            # delete the live import and leave ``SomeClass`` undefined.
            #
            # Ideal: the use should reach BOTH bindings --
            #     "mod -> mod.SomeClass@5:18",
            #     "mod -> b",
            #     "mod -> b.SomeClass@1:0",
            {
                "a.py": "class SomeClass: pass\n",
                "b.py": "class SomeClass: pass\n",
                "mod.py": (
                    "from typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n"
                    "    from a import SomeClass\n"
                    "else:\n"
                    "    from b import SomeClass\n"
                    "SomeClass()\n"
                ),
            },
            {
                "a.SomeClass@1:0 -> a",
                "b.SomeClass@1:0 -> b",
                "mod -> a",
                "mod -> a.SomeClass@1:0",
                "mod -> mod.SomeClass@3:18",
                "mod -> mod.TYPE_CHECKING@1:19",
                "mod.SomeClass@3:18 -> a",
                "mod.SomeClass@3:18 -> a.SomeClass@1:0",
                "mod.SomeClass@3:18 -> mod",
                "mod.SomeClass@5:18 -> b",
                "mod.SomeClass@5:18 -> b.SomeClass@1:0",
                "mod.SomeClass@5:18 -> mod",
                "mod.TYPE_CHECKING@1:19 -> mod",
            },
            id="type-checking-else-branch-import-dropped",
        ),
    ],
)
def test_limitation_positional(build_decl_graph, assert_positional_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_positional_edges(graph, expected_edges)
