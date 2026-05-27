"""Tests for import resolution in the symbol graph.

Every case adds a ``p/x.py`` file on top of the shared package fixture
below and asserts the complete set of edges the graph contains.
``IMPORT_BASE_EDGES`` captures the edges that are always present from
the fixture so individual cases only list the edges they introduce.
"""

import logging

import pytest

from dead_cst.plugins._core import EXTERNAL_PREFIXES, STDLIB_PREFIX, UNRESOLVED_PREFIX

IMPORT_TEST_FILES = {
    "p/__init__.py": "",
    "p/functions.py": "def f(): pass\ndef g(): pass",
    "p/classes.py": "class C(): pass",
    "p/chain.py": "from . import functions",
}

# Edges always produced by IMPORT_TEST_FILES plus an (empty-or-populated)
# p/x.py file: module-hierarchy edges, the `p.chain` re-import, and the
# parent edges for p.x itself.
IMPORT_BASE_EDGES = frozenset(
    {
        "p.chain -> p",
        "p.chain.functions -> p.chain",
        "p.chain.functions -> p.functions",
        "p.classes -> p",
        "p.classes.C -> p.classes",
        "p.functions -> p",
        "p.functions.f -> p.functions",
        "p.functions.g -> p.functions",
        "p.x -> p",
    }
)

# Edges from materializing ``from p.functions import *`` (or a synonym)
# into ``p/x.py``: one synthetic re-export per name plus the
# ``module -> synthetic -> target`` chain. Used by every star-shaped
# test case in this file.
STAR_REEXPORT_EDGES = frozenset(
    {
        "p.x -> p.x.f",
        "p.x -> p.x.g",
        "p.x.f -> p.functions",
        "p.x.f -> p.functions.f",
        "p.x.f -> p.x",
        "p.x.g -> p.functions",
        "p.x.g -> p.functions.g",
        "p.x.g -> p.x",
    }
)


@pytest.mark.parametrize(
    "src, expected_extra_edges",
    [
        pytest.param(
            "import p.functions\ndef a(): p.functions.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.p",
                "p.x.p -> p.functions",
                "p.x.p -> p.x",
            },
            id="cst.Import-dotted-module",
        ),
        pytest.param(
            "from p.functions import f\ndef a(): f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="cst.ImportFrom-function",
        ),
        pytest.param(
            # Bare-name reference inside a quoted annotation. ty parses
            # the string in annotation position and re-runs name
            # resolution; the rust ingest hands the parsed sub-AST to a
            # sub-collector so each Name inside contributes the same
            # alias + upstream edges it would unquoted. Edges match the
            # ``f()`` call case above exactly — annotations are uses.
            "from p.functions import f\ndef a(x: 'f') -> None: pass",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="string-annotation-emits-use-edge",
        ),
        pytest.param(
            # Quoted name *inside* a non-quoted annotation
            # (``Container['f']``). ty registers the inner string as an
            # annotation; we walk it just like the standalone case.
            (
                "from typing import List\n"
                "from p.functions import f\n"
                "def a(x: List['f']) -> None: pass\n"
            ),
            {
                "p.x.List -> p.x",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.List",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="string-annotation-inside-subscript",
        ),
        pytest.param(
            # Belt-and-braces: a bare string literal in a *non*-annotation
            # position must NOT emit use edges. ty never marks it as a
            # string annotation, so ``enter_string_annotation`` returns
            # ``None`` and the walker leaves it alone. The expected
            # edge set lists no ``p.x.label -> p.x.f`` — pinning that
            # absence is the point of this case.
            "from p.functions import f\nlabel = 'f'\n",
            {
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
                "p.x.label -> p.x",
            },
            id="regular-string-literal-not-walked",
        ),
        pytest.param(
            "from p.classes import C\ndef a(): C.f()",
            {
                "p.x.C -> p.classes",
                "p.x.C -> p.classes.C",
                "p.x.C -> p.x",
                "p.x.a -> p.classes",
                "p.x.a -> p.classes.C",
                "p.x.a -> p.x",
                "p.x.a -> p.x.C",
            },
            id="cst.ImportFrom-class-attribute-access",
        ),
        pytest.param(
            "import p.functions as f\ndef a(): f.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.x",
            },
            id="cst.Import-with-alias",
        ),
        pytest.param(
            "from p import functions as f\ndef a(): f.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.x",
            },
            id="cst.ImportFrom-with-alias",
        ),
        pytest.param(
            "def a(): import p.functions; p.functions.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.Import",
        ),
        pytest.param(
            "def a(): from p.functions import f; f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.ImportFrom",
        ),
        pytest.param(
            # Nested import with no use of the bound name still
            # creates a dep on the upstream module — the import
            # statement itself is a side effect attributed to the
            # enclosing decl.
            "def a():\n    import p.functions\n",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
            },
            id="nested-cst.Import-no-use",
        ),
        pytest.param(
            # `from X import Y` with no use still emits both the
            # parent module and the upstream decl edges, just like
            # the module-scope ImportFrom alias does.
            "def a():\n    from p.functions import f\n",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.ImportFrom-no-use",
        ),
        pytest.param(
            # Bare-`import p` then dotted access through it inside
            # the function body. The chain `.functions.f` walks
            # submodule then decl from the bound name `p`.
            "def a():\n    import p\n    p.functions.f()\n",
            {
                "p.x.a -> p",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.Import-bare-then-dotted",
        ),
        pytest.param(
            # Nested import shadows a module-scope import of the
            # same root name. Inside `a` the use resolves to the
            # nested binding; outside it the module-scope alias
            # still points at its own upstream.
            "import p.functions\ndef a():\n    import p.classes\n    p.classes.C()\n",
            {
                "p.x.a -> p.classes",
                "p.x.a -> p.classes.C",
                "p.x.a -> p.x",
                "p.x.p -> p.functions",
                "p.x.p -> p.x",
            },
            id="nested-cst.Import-shadows-module-scope",
        ),
        pytest.param(
            # Import in a class body. The class is the owner;
            # the chain walks normally.
            "class A:\n    import p.functions\n    p.functions.f()\n",
            {
                "p.x.A -> p.functions",
                "p.x.A -> p.functions.f",
                "p.x.A -> p.x",
            },
            id="class-body-cst.Import",
        ),
        pytest.param(
            "class A:\n    from p.functions import f\n    f()\n",
            {
                "p.x.A -> p.functions",
                "p.x.A -> p.functions.f",
                "p.x.A -> p.x",
            },
            id="class-body-cst.ImportFrom",
        ),
        pytest.param(
            # Imports inside method bodies attribute to the
            # enclosing class — methods are not separate top-level
            # nodes, per the project's "nested defs are folded
            # into the enclosing top-level decl" convention.
            "class A:\n    def m(self):\n        import p.functions\n        p.functions.f()\n",
            {
                "p.x.A -> p.functions",
                "p.x.A -> p.functions.f",
                "p.x.A -> p.x",
            },
            id="method-body-cst.Import",
        ),
        pytest.param(
            # `try: import X as A; except: import Y as A` — both
            # branches reach end-of-scope with `A` bound, so a use
            # of `A` (or the import statements alone) attributes
            # edges to *both* upstreams (Principle 3 — every
            # reaching def gets edges).
            "def a():\n"
            "    try:\n"
            "        import p.functions as src\n"
            "    except ImportError:\n"
            "        import p.classes as src\n"
            "    src\n",
            {
                "p.x.a -> p.classes",
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
            },
            id="nested-try-except-cst.Import",
        ),
        pytest.param(
            # Nested star import fans out to every name `p.functions`
            # exports, attributed to the enclosing function. No alias
            # node is minted.
            "def a():\n    from p.functions import *\n    f()\n",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.functions.g",
                "p.x.a -> p.x",
            },
            id="nested-star-import",
        ),
        pytest.param(
            "from .functions import f\ndef a(): f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="relative-import",
        ),
        # libcst follows the reexport chain at the use site, so `g.f()`
        # picks up parallel `p.x.a -> p.chain.functions` /
        # `p.x.a -> p.functions.f` edges (in addition to the alias
        # edges on `p.x.g`). The rust backend stops at the alias edge
        # — `p.x.g` already points at `p.chain.functions`, but no
        # parallel use-site edges fan through that chain.
        pytest.param(
            "from p.chain import functions as g\ndef a(): g.f()",
            {
                "p.x.a -> p.chain",
                "p.x.a -> p.chain.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.g",
                "p.x.g -> p.chain",
                "p.x.g -> p.chain.functions",
                "p.x.g -> p.x",
            },
            id="import-chain-via-reexport-rust",
        ),
        pytest.param(
            "from p.functions import f\nfrom p.classes import C\ndef a(): f(); C()",
            {
                "p.x.C -> p.classes",
                "p.x.C -> p.classes.C",
                "p.x.C -> p.x",
                "p.x.a -> p.classes",
                "p.x.a -> p.classes.C",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.C",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="multiple-from-imports",
        ),
        pytest.param(
            "import p.functions\n",
            {
                "p.x.p -> p.functions",
                "p.x.p -> p.x",
            },
            id="bare-cst.Import-keeps-module-alive",
        ),
        pytest.param(
            "from p.functions import f\n",
            {
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="bare-cst.ImportFrom-keeps-module-alive",
        ),
        pytest.param(
            "from p.functions import f\nf()",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="module-level-call-of-imported-symbol",
        ),
        # Same reexport-chain divergence as `import-chain-via-reexport`
        # above: libcst emits parallel use-site edges through the
        # chain (`p.x.a -> p.chain.functions`, `p.x.a -> p.functions.f`),
        # rust stops at the alias edge on `p.x.functions`.
        pytest.param(
            "from p.chain import functions\ndef a(): functions.f()",
            {
                "p.x.a -> p.chain",
                "p.x.a -> p.chain.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.functions",
                "p.x.functions -> p.chain",
                "p.x.functions -> p.chain.functions",
                "p.x.functions -> p.x",
            },
            id="reexport-through-package-init-rust",
        ),
        pytest.param(
            "import p\ndef a(): p.functions.f()",
            # The access ``p.functions.f`` synthesizes an
            # ``Import(module="p", decl="functions.f")``; the stitcher
            # canonicalizes it to ``module="p.functions"`` because the
            # whole prefix resolves as a submodule, so the edge points
            # at the deepest reached module rather than at ``p``.
            # Reachability of ``p`` itself is still preserved through
            # ``p.x.p -> p`` plus the ``p.functions -> p`` parent-edge.
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.p",
                "p.x.p -> p",
                "p.x.p -> p.x",
            },
            id="import-package-then-dotted-access",
        ),
        # libcst minted per-name synthetic aliases as a workaround for
        # its inability to resolve uses *through* a star import; the
        # rust backend leans on ty and mints one node per
        # `from X import *` statement (named `<mod>.*<src>`) with a
        # single outgoing edge to the upstream module. Use sites
        # route through this node and emit the standard parallel
        # upstream module/decl edges via ty's name resolution
        # (Principle 2).
        pytest.param(
            "from p.functions import *\ndef a(): f()",
            {
                "p.x.*p.functions -> p.x",
                "p.x.*p.functions -> p.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.*p.functions",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
            },
            id="star-import-fans-out-to-all-decls-rust",
        ),
        # ------------------------------------------------------------------
        # Dynamic imports (`__import__` / `importlib.import_module`).
        #
        # libcst minted per-name synthetic aliases that fanned out to
        # every export of the target module. The rust backend leans on
        # ty: one edge per *explicit symbol* the call mentions, tagged
        # with `EdgeFlags.DYNAMIC_IMPORT` so a contrib plugin can fan
        # out further if a project wants the old semantic.
        # ------------------------------------------------------------------
        pytest.param(
            "__import__('p.functions')",
            {"p.x -> p.functions"},
            id="dunder-import-call-fans-out-like-star-rust",
        ),
        pytest.param(
            "def a(): getattr(__import__('p.functions'), 'f')()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
            },
            id="dunder-import-call-inside-function-attributes-to-enclosing-decl-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('p.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-fans-out-like-star-rust",
        ),
        pytest.param(
            "import importlib\ndef a(): importlib.import_module('p.functions')",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-inside-function-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions', 'p')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-positional-package-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions', package='p')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-keyword-package-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-uses-enclosing-package-rust",
        ),
        pytest.param(
            "__import__('functions', globals(), locals(), [], 1)",
            {"p.x -> p.functions"},
            id="dunder-import-positional-level-resolves-relative-rust",
        ),
        pytest.param(
            "__import__('functions', level=1)",
            {"p.x -> p.functions"},
            id="dunder-import-keyword-level-resolves-relative-rust",
        ),
        # ------------------------------------------------------------------
        # ``__import__`` fromlist: rust emits one edge per explicit symbol
        # the call mentions (base module plus each literal fromlist entry
        # that resolves as a submodule or global-scope decl), all tagged
        # ``EdgeFlags.DYNAMIC_IMPORT``. Attribute-style entries that don't
        # resolve drop silently — no warning.
        # ------------------------------------------------------------------
        pytest.param(
            "__import__('p', None, None, ['functions'])",
            {
                "p.x -> p",
                "p.x -> p.functions",
            },
            id="dunder-import-fromlist-positional-resolves-submodule",
        ),
        pytest.param(
            "__import__('p', fromlist=['functions'])",
            {
                "p.x -> p",
                "p.x -> p.functions",
            },
            id="dunder-import-fromlist-keyword-resolves-submodule",
        ),
        pytest.param(
            # Mixed: ``f`` resolves as a global-scope decl in p.functions
            # (one edge, no fan-out to ``g``); the empty string entry
            # drops silently with no warning.
            "__import__('p.functions', fromlist=['f', ''])",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
            },
            id="dunder-import-fromlist-attribute-entries-silent",
        ),
        # ------------------------------------------------------------------
        # ``__all__`` is followed when assigned a list/tuple of string
        # literals; each named entry becomes an outgoing edge from the
        # ``__all__`` synthetic node to the local decl (or import alias).
        # Unknown names are ignored silently.
        # ------------------------------------------------------------------
        pytest.param(
            'from p.functions import f\n__all__ = ["f"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-references-import",
        ),
        pytest.param(
            'from p.functions import f\nfrom p.classes import C\n__all__ = ("f", "C")',
            {
                "p.x.C -> p.classes",
                "p.x.C -> p.classes.C",
                "p.x.C -> p.x",
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.C",
                "p.x.__all__ -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-tuple-of-imports",
        ),
        pytest.param(
            'def g(): pass\nfrom p.functions import f\n__all__ = ["f", "g"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.f",
                "p.x.__all__ -> p.x.g",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
                "p.x.g -> p.x",
            },
            id="dunder-all-mixes-local-and-imported",
        ),
        pytest.param(
            'from p.functions import f\n__all__: list[str] = ["f"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-annotated-assignment",
        ),
        pytest.param(
            'from p.functions import f\n__all__ = ["missing"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-unknown-name-is-ignored",
        ),
        # ------------------------------------------------------------------
        # Common real-world shapes.
        # ------------------------------------------------------------------
        pytest.param(
            # Stdlib import used via a subscript. ``os`` doesn't surface a
            # synthetic node (stdlib is silent — see
            # ``test_stdlib_imports_are_silent``), so the only edges are
            # the local alias and the module-level use of it. The
            # ``environ["foo"]`` access doesn't add anything past the
            # alias.
            'import os\nos.environ["foo"]\n',
            {
                "p.x -> p.x.os",
                "p.x.os -> p.x",
            },
            id="stdlib-import-subscript-access",
        ),
        pytest.param(
            # External-dist import combined with a submodule import that
            # binds the same root name. Both ``import anyio`` and
            # ``import anyio.to_thread`` bind ``anyio`` locally and
            # resolve to the same ``[external dist] anyio`` synthetic;
            # the second statement's module-level side-effect contributes
            # the ``p.x -> [external dist] anyio`` edge directly.
            "import anyio\nimport anyio.to_thread\nanyio.run()\n",
            {
                "p.x -> [external dist] anyio",
                "p.x -> p.x.anyio",
                "p.x.anyio -> [external dist] anyio",
                "p.x.anyio -> p.x",
            },
            id="external-import-with-submodule",
        ),
        pytest.param(
            # Quoted attribute-style annotation (``"futures.Future[str]"``).
            # The walker re-parses the string and resolves ``futures``
            # to the local alias. ``concurrent.futures`` is stdlib, so
            # the alias has no upstream edges — only the alias use
            # propagates to ``p.x.f``.
            'from concurrent import futures\ndef f() -> "futures.Future[str]": ...\n',
            {
                "p.x.f -> p.x",
                "p.x.f -> p.x.futures",
                "p.x.futures -> p.x",
            },
            id="quoted-attribute-annotation-on-stdlib-import",
        ),
        pytest.param(
            # ``@dataclass`` decorator + ``field(default_factory=list)``
            # default-value call inside the class body. Both bind to the
            # enclosing class ``C`` (decorators and class-body
            # expressions attribute to the top-level decl per the
            # "nested defs fold into the enclosing top-level decl"
            # convention). ``dataclasses`` is stdlib so no upstream
            # edges; ``list`` and the undefined ``T`` are builtins or
            # missing names and contribute nothing.
            (
                "from dataclasses import dataclass, field\n"
                "@dataclass\n"
                "class C:\n"
                "    a: list[T] = field(default_factory=list)\n"
            ),
            {
                "p.x.C -> p.x",
                "p.x.C -> p.x.dataclass",
                "p.x.C -> p.x.field",
                "p.x.dataclass -> p.x",
                "p.x.field -> p.x",
            },
            id="dataclass-decorator-and-field-factory",
        ),
    ],
)
def test_imports(build_decl_graph, assert_edges, src, expected_extra_edges):
    graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    assert_edges(graph, IMPORT_BASE_EDGES | expected_extra_edges)


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        pytest.param(
            # Regression guard for the visitor-level dunder strip:
            # ``Cls.__name__`` / ``Cls.__doc__`` drop the dunder and
            # everything after it, so an access through an imported
            # symbol still resolves to that symbol (not past it).
            {
                "pkg/__init__.py": "",
                "pkg/lib.py": "class Cls:\n    pass\n",
                "pkg/uses.py": "from pkg.lib import Cls\nWHO = Cls.__name__\nDOCSTR = Cls.__doc__\n",
            },
            {
                "pkg.lib -> pkg",
                "pkg.lib.Cls -> pkg.lib",
                "pkg.uses -> pkg",
                "pkg.uses.Cls -> pkg.lib",
                "pkg.uses.Cls -> pkg.lib.Cls",
                "pkg.uses.Cls -> pkg.uses",
                "pkg.uses.DOCSTR -> pkg.lib",
                "pkg.uses.DOCSTR -> pkg.lib.Cls",
                "pkg.uses.DOCSTR -> pkg.uses",
                "pkg.uses.DOCSTR -> pkg.uses.Cls",
                "pkg.uses.WHO -> pkg.lib",
                "pkg.uses.WHO -> pkg.lib.Cls",
                "pkg.uses.WHO -> pkg.uses",
                "pkg.uses.WHO -> pkg.uses.Cls",
            },
            id="dunder-on-imported-symbol-strips-tail",
        ),
        pytest.param(
            # Re-export cycle terminates without spinning. The rust
            # backend resolves each alias once and stops — no
            # reexport-chain self-edges (``A.x -> A.x``), no transitive
            # walk through the cycle (no ``main -> B.x``). The cycle
            # itself is still represented (``A.x -> B.x``,
            # ``B.x -> A.x``) so reachability from ``main`` can walk
            # it. The use-site emits the standard Principle 2
            # parallel-upstream edge to ``A.x`` (the decl in A's
            # namespace that the import resolved to) — same shape as
            # ``main -> A`` for the upstream module.
            {
                "A.py": "from B import x",
                "B.py": "from A import x",
                "main.py": "from A import x\nx()",
            },
            {
                "A.x -> A",
                "A.x -> B",
                "A.x -> B.x",
                "B.x -> A",
                "B.x -> A.x",
                "B.x -> B",
                "main -> A",
                "main -> A.x",
                "main -> main.x",
                "main.x -> A",
                "main.x -> A.x",
                "main.x -> main",
            },
            id="cyclic-reexport-terminates",
        ),
        pytest.param(
            # ``from pkg import g`` resolves through the star node to
            # ``pkg._internal.g``. The rust backend mints one
            # ``*pkg._internal`` node per ``from pkg._internal import *``
            # statement rather than a per-name ``pkg.g`` alias; ty
            # resolves ``from pkg import g`` straight to
            # ``pkg._internal.g``, so the consumer alias edges directly
            # to the upstream decl. The absence of any
            # ``consumer.g -> pkg.g`` edge from the set below codifies
            # the "no per-name alias is minted" invariant.
            {
                "pkg/__init__.py": "from pkg._internal import *\n",
                "pkg/_internal.py": "def g(): pass\n",
                "consumer.py": "from pkg import g\ng()\n",
            },
            {
                "consumer -> consumer.g",
                "consumer -> pkg",
                "consumer -> pkg.*pkg._internal",
                "consumer.g -> consumer",
                "consumer.g -> pkg",
                "consumer.g -> pkg._internal.g",
                "pkg.*pkg._internal -> pkg",
                "pkg.*pkg._internal -> pkg._internal",
                "pkg._internal -> pkg",
                "pkg._internal -> pkg._internal",
                "pkg._internal.g -> pkg._internal",
            },
            id="import-resolves-through-star-reexport",
        ),
        pytest.param(
            # ``from A import g`` follows the ``A -> B -> C`` star
            # chain to ``C.g``.
            {
                "A.py": "from B import *\n",
                "B.py": "from C import *\n",
                "C.py": "def g(): pass\n",
                "consumer.py": "from A import g\ng()\n",
            },
            {
                "A.*B -> A",
                "A.*B -> B",
                "B.*C -> B",
                "B.*C -> C",
                "C.g -> C",
                "consumer -> A",
                "consumer -> A.*B",
                "consumer -> consumer.g",
                "consumer.g -> A",
                "consumer.g -> C.g",
                "consumer.g -> consumer",
            },
            id="import-resolves-through-chained-star-reexports",
        ),
        pytest.param(
            # Mutual ``from B import *`` / ``from A import *``
            # terminates without spinning; consumer's ``b`` still
            # resolves to ``B.b``.
            {
                "A.py": "from B import *\ndef a(): pass\n",
                "B.py": "from A import *\ndef b(): pass\n",
                "consumer.py": "from A import b\n",
            },
            {
                "A.*B -> A",
                "A.*B -> B",
                "A.a -> A",
                "B.*A -> A",
                "B.*A -> B",
                "B.b -> B",
                "consumer.b -> A",
                "consumer.b -> B.b",
                "consumer.b -> consumer",
            },
            id="star-reexport-cycle-terminates",
        ),
        pytest.param(
            # A real decl in the importing module wins over a star
            # re-export of the same name. The absence of
            # ``consumer.g -> other.g`` from the set below codifies
            # that the consumer's ``g`` resolves to ``mod.g`` (not
            # through the star re-export to ``other.g``).
            {
                "other.py": "def g(): pass\n",
                "mod.py": "from other import *\ndef g(): return 1\n",
                "consumer.py": "from mod import g\ng()\n",
            },
            {
                "consumer -> consumer.g",
                "consumer -> mod",
                "consumer -> mod.g",
                "consumer.g -> consumer",
                "consumer.g -> mod",
                "consumer.g -> mod.g",
                "mod.*other -> mod",
                "mod.*other -> other",
                "mod.g -> mod",
                "other.g -> other",
            },
            id="star-reexport-shadowed-by-real-decl",
        ),
        pytest.param(
            # ``from .sub import X, Y`` inside an ``__init__.py`` mints
            # both a normal ``ImportFrom`` alias for each name and an
            # ``ImportFromSubmodule`` alias for the side-effect
            # attribute ``sub`` on the current package — Python rebinds
            # ``pkg.sub`` to the submodule whenever the statement runs.
            # The two are one inseparable syntactic unit, so
            # reachability must drag the submodule alias alive whenever
            # any sibling alias is — the ``pkg.X -> pkg.sub`` and
            # ``pkg.Y -> pkg.sub`` edges below are those sibling-rescue
            # edges. (Both the submodule attribute alias and the module
            # itself share the fqname ``pkg.sub`` — the
            # ``pkg.sub -> pkg.sub`` edge is the attribute alias edging
            # to the module it shadows.)
            {
                "pkg/__init__.py": "from .sub import X, Y\n",
                "pkg/sub.py": "class X: pass\nclass Y: pass\n",
            },
            {
                "pkg.X -> pkg",
                "pkg.X -> pkg.sub",
                "pkg.X -> pkg.sub.X",
                "pkg.Y -> pkg",
                "pkg.Y -> pkg.sub",
                "pkg.Y -> pkg.sub.Y",
                "pkg.sub -> pkg",
                "pkg.sub -> pkg.sub",
                "pkg.sub.X -> pkg.sub",
                "pkg.sub.Y -> pkg.sub",
            },
            id="from-relative-import-in-init-keeps-submodule-alias-alive",
        ),
        pytest.param(
            # Composition test: an import that lives under a dead
            # ``if TYPE_CHECKING:`` branch *and* is referenced only
            # from a quoted annotation in live code must still pick up
            # the use edge. ``mod.f -> mod.Helper`` below is that use
            # edge. The basic mechanics (string annotations are uses;
            # regular string literals are not) are pinned by
            # ``test_imports`` cases above.
            {
                "mod.py": (
                    "from __future__ import annotations\n"
                    "from typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n"
                    "    from helpers import Helper\n"
                    "def f(x: 'Helper') -> 'Helper':\n"
                    "    return x\n"
                ),
                "helpers.py": "class Helper: pass\n",
            },
            {
                "helpers.Helper -> helpers",
                "mod -> mod.TYPE_CHECKING",
                "mod.Helper -> helpers",
                "mod.Helper -> helpers.Helper",
                "mod.Helper -> mod",
                "mod.TYPE_CHECKING -> mod",
                "mod.annotations -> mod",
                "mod.f -> helpers",
                "mod.f -> helpers.Helper",
                "mod.f -> mod",
                "mod.f -> mod.Helper",
            },
            id="type-checking-import-used-only-in-string-annotation",
        ),
        pytest.param(
            # Module-level ``isinstance(foo, SomeClass)``. The use sits
            # at module scope so the enclosing module owns the edges
            # (``mod -> mod.SomeClass`` plus the parallel upstream
            # edges via Principle 2). ``foo`` is undefined and
            # contributes nothing.
            {
                "a.py": "class SomeClass: pass\n",
                "mod.py": "from a import SomeClass\nisinstance(foo, SomeClass)\n",
            },
            {
                "a.SomeClass -> a",
                "mod -> a",
                "mod -> a.SomeClass",
                "mod -> mod.SomeClass",
                "mod.SomeClass -> a",
                "mod.SomeClass -> a.SomeClass",
                "mod.SomeClass -> mod",
            },
            id="isinstance-uses-imported-class",
        ),
        pytest.param(
            # ``if TYPE_CHECKING: from a import SomeClass`` paired with
            # ``else: SomeClass = None`` — both branches mint a
            # ``mod.SomeClass`` decl (one import, one variable) that
            # share an fqname. ``def f(x: SomeClass)`` resolves
            # *unquoted* against the reaching defs; the import path
            # carries the upstream edges to ``a.SomeClass`` and the
            # variable path adds nothing past the parent edge, so the
            # dedup'd edge set shows the import's upstream resolution
            # plus the single shared ``mod.f -> mod.SomeClass`` alias
            # edge. The else-branch assignment is live at runtime
            # (TYPE_CHECKING is False), but its presence doesn't
            # suppress the import's upstream resolution — both reaching
            # defs contribute.
            {
                "a.py": "class SomeClass: pass\n",
                "mod.py": (
                    "from typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n"
                    "    from a import SomeClass\n"
                    "else:\n"
                    "    SomeClass = None\n"
                    "def f(x: SomeClass): ...\n"
                ),
            },
            {
                "a.SomeClass -> a",
                "mod -> mod.TYPE_CHECKING",
                "mod.SomeClass -> a",
                "mod.SomeClass -> a.SomeClass",
                "mod.SomeClass -> mod",
                "mod.TYPE_CHECKING -> mod",
                "mod.f -> a",
                "mod.f -> a.SomeClass",
                "mod.f -> mod",
                "mod.f -> mod.SomeClass",
            },
            id="type-checking-import-shadowed-by-else-assignment",
        ),
    ],
)
def test_full_graph_edges(build_decl_graph, assert_edges, files, expected_edges):
    """Bespoke-scaffold edge tests. Each case provides a complete
    ``{filename: contents}`` mapping (rather than overlaying onto
    :data:`IMPORT_TEST_FILES` like :func:`test_imports` does) and the
    full edge set the graph produces. The case IDs document what each
    scenario proves.
    """
    graph = build_decl_graph(files)
    assert_edges(graph, expected_edges)


def test_third_party_import_creates_synthetic_node(build_decl_graph):
    graph = build_decl_graph(
        {
            "p/__init__.py": "",
            "p/uses_rx.py": "import click as rx\ndef build(): return rx.click()",
        }
    )
    rx_nodes = {
        n
        for n in graph.nodes()
        if n.kind == "synthetic" and n.fqname.startswith(EXTERNAL_PREFIXES) and "click" in n.fqname
    }
    assert rx_nodes, (
        "expected an external-dep synthetic node for click, got "
        f"{[n.fqname for n in graph.nodes() if n.kind == 'synthetic']}"
    )

    edge_srcs = {
        graph.nodes()[u].fqname for u, v, _ in graph.edges() if graph.nodes()[v] in rx_nodes
    }
    assert {"p.uses_rx.rx", "p.uses_rx.build"} <= edge_srcs


def test_third_party_import_uses_canonical_dist_name(build_decl_graph):
    """``import yaml`` lands on ``[external dist] pyyaml`` (PEP 503).

    The distribution's top-level module (``yaml``) and its canonical
    PyPI name (``PyYAML`` → ``pyyaml``) differ. Plugins query by the
    canonical name so the synthetic must match what
    :data:`importlib.metadata`'s ``Name:`` header normalizes to.
    """
    graph = build_decl_graph(
        {
            "p/__init__.py": "",
            "p/uses_yaml.py": "import yaml\nDATA = yaml.safe_load('a: 1')\n",
        }
    )
    fqnames = {n.fqname for n in graph.nodes() if n.kind == "synthetic"}
    assert "[external dist] pyyaml" in fqnames, fqnames


def test_stdlib_imports_are_silent(build_decl_graph, caplog):
    """Stdlib imports drop without surfacing a synthetic node or a warning."""
    with caplog.at_level(logging.WARNING, logger="dead_cst._edges"):
        graph = build_decl_graph(
            {
                "p/__init__.py": "",
                "p/uses_stdlib.py": (
                    "import datetime\n"
                    "import os\n"
                    "from pathlib import Path\n"
                    "from collections.abc import Iterable\n"
                ),
            }
        )

    assert [r.getMessage() for r in caplog.records] == []
    synthetics = {n.fqname for n in graph.nodes() if n.kind == "synthetic"}
    # No stdlib ever surfaces as a graph node, and ``collections.abc``
    # must not fall through to ``[unresolved] collections`` (regression
    # against the synthesized-submodule parent-fallback).
    assert not [fq for fq in synthetics if fq.startswith(STDLIB_PREFIX)]
    assert f"{UNRESOLVED_PREFIX}collections" not in synthetics


def test_unresolved_import_emits_synthetic_silently(build_decl_graph, caplog):
    """A genuinely-missing top-level import gets a ``[unresolved]`` node, no warning."""
    with caplog.at_level(logging.WARNING, logger="dead_cst._edges"):
        graph = build_decl_graph(
            {
                "p/__init__.py": "",
                "p/uses_missing.py": "from unknown_pkg_xyz import thing\n",
            }
        )

    assert [r.getMessage() for r in caplog.records] == []
    assert any(
        n.kind == "synthetic" and n.fqname == f"{UNRESOLVED_PREFIX}unknown_pkg_xyz"
        for n in graph.nodes()
    )


def test_module_runtime_dunder_access_is_module_dep(build_decl_graph, assert_edges, caplog):
    """``pkg.__file__`` collapses to a plain ``Import(module=pkg, decl=None)``.

    The import machinery injects ``__file__`` / ``__name__`` / ``__spec__`` /
    etc. onto every module object at runtime -- attribute access past one of
    these is a path / string op, not a symbol reference. The visitor truncates
    the access chain at the dunder so the edge stitcher sees a clean module-
    level dependency (no speculative ``decl="__file__"`` that it would warn
    about, no synthetic node for the missing attribute).
    """
    with caplog.at_level(logging.WARNING, logger="dead_cst._edges"):
        graph = build_decl_graph(
            {
                "pkg/__init__.py": "",
                "pkg/config.py": (
                    "from pathlib import Path\n"
                    "import pkg as pkg_alias\n"
                    "FILE_PATH = Path(pkg_alias.__file__).parent\n"
                    "NAME = pkg_alias.__name__\n"
                    "SPEC = pkg_alias.__spec__\n"
                ),
            }
        )

    assert [r.getMessage() for r in caplog.records] == []
    # No synthetic was minted for the missing-dunder lookup.
    assert not [n.fqname for n in graph.nodes() if n.fqname.endswith(".__file__")]
    # The module-level dependency edges remain intact for each user of a dunder
    # (FILE_PATH / NAME / SPEC all edge to ``pkg`` directly).
    assert_edges(
        graph,
        {
            "pkg.config -> pkg",
            "pkg.config.FILE_PATH -> pkg",
            "pkg.config.FILE_PATH -> pkg.config",
            "pkg.config.FILE_PATH -> pkg.config.Path",
            "pkg.config.FILE_PATH -> pkg.config.pkg_alias",
            "pkg.config.NAME -> pkg",
            "pkg.config.NAME -> pkg.config",
            "pkg.config.NAME -> pkg.config.pkg_alias",
            "pkg.config.Path -> pkg.config",
            "pkg.config.SPEC -> pkg",
            "pkg.config.SPEC -> pkg.config",
            "pkg.config.SPEC -> pkg.config.pkg_alias",
            "pkg.config.pkg_alias -> pkg",
            "pkg.config.pkg_alias -> pkg.config",
        },
    )


def test_from_import_prefers_namespace_binding_over_submodule(build_decl_graph):
    """``from p import q`` where ``p/__init__.py`` binds ``q`` (e.g. to an
    int) and ``p/q.py`` *also* exists: CPython's ``_handle_fromlist``
    binds ``q`` to the int via the namespace, the submodule is never
    imported, and dead code in ``p/q.py`` stays dead.

    Pins the rust backend's matching behavior (``resolve_from_imported``
    probes ``globals_by_name`` before falling back to the submodule
    lookup). The libcst path canonicalizes the import the other way
    (`_edges.py` pushes ``q`` into the module name as long as it
    resolves as a submodule in the trie) and is wrong for this corner
    case — when libcst is updated this test should drop the
    backend-skip and start enforcing the assertion everywhere.
    """
    graph = build_decl_graph(
        {
            "p/__init__.py": "q = 42\n",
            "p/q.py": "def dead(): pass\n",
            "consumer.py": "from p import q\nprint(q)\n",
        }
    )
    # Same fqname appears twice (the `q` variable in p/__init__.py and
    # the `q` module from p/q.py), so we have to disambiguate on type.
    consumer_q_alias = next(
        n for n in graph.nodes() if n.fqname == "consumer.q" and n.kind == "import"
    )
    targets = {
        (graph.nodes()[v].fqname, graph.nodes()[v].kind)
        for u, v, _ in graph.edges()
        if graph.nodes().index(consumer_q_alias) == u
    }
    assert ("p.q", "variable") in targets, targets
    # Crucial: the submodule must NOT be linked. The old submodule-first
    # order would wrongly add ("p.q", "module"); the namespace-first
    # order matches CPython and skips it.
    assert ("p.q", "module") not in targets, targets


def test_star_reexport_is_skipped_by_codemod_rust(build_decl_graph):
    """The rust backend mints one node per ``from X import *`` named
    ``<mod>.*<src>`` with ``kind="import"`` and an ``imports.star=True``
    payload — the ``*`` prefix is the codemod's marker."""
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "from pkg._internal import *\n",
            "pkg/_internal.py": "def g(): pass\n",
        }
    )
    star_nodes = [
        n for n in graph.nodes() if n.fqname == "pkg.*pkg._internal" and n.kind == "import"
    ]
    assert len(star_nodes) == 1, [n.fqname for n in graph.nodes() if "pkg" in n.fqname]
    assert star_nodes[0].imports is not None
    assert star_nodes[0].imports.star is True
    assert star_nodes[0].imports.module == "pkg._internal"


def test_cross_dep_submodule_import(tmp_path, make_analysis, assert_edges):
    """Importing a submodule from a dep base resolves through the dep's exported trie.

    Layout:
        pkg_a/A/__init__.py  -- empty package
        pkg_a/A/sub.py       -- def f(): ...
        pkg_b/B/__init__.py  -- from A import sub; sub.f()

    Packages: pkg_b(deps=("pkg_a",)), pkg_a(deps=())

    ``A.sub`` lives in the dep's exported trie, not the consumer's own
    trie. ``resolve_edges`` must find it via the merged ``symbol_lookup``
    and follow the submodule through ``cur.children`` (since ``A/__init__.py``
    has no explicit ``import sub`` declaration -- the name resolves as a
    real submodule path).
    """
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    (pkg_a / "A").mkdir(parents=True)
    (pkg_b / "B").mkdir(parents=True)
    (pkg_a / "A" / "__init__.py").write_text("")
    (pkg_a / "A" / "sub.py").write_text("def f(): ...\n")
    (pkg_b / "B" / "__init__.py").write_text("from A import sub\nsub.f()\n")

    graph = make_analysis(["pkg_b:pkg_a", "pkg_a"]).materialize_all()
    assert_edges(
        graph,
        {
            "A.sub -> A",
            "A.sub.f -> A.sub",
            "B -> A.sub",
            "B -> A.sub.f",
            "B -> B.sub",
            "B.sub -> A.sub",
            "B.sub -> B",
        },
    )


# ---------------------------------------------------------------------------
# Submodule attribute alias kept alive by sibling aliases
# ---------------------------------------------------------------------------


def _find_node(graph, *, fqname, kind, path_substring):
    """Disambiguate nodes that share an fqname (e.g. the submodule
    attribute alias ``pkg.sub`` in ``__init__.py`` vs. the module
    ``pkg.sub`` itself in ``pkg/sub.py``).
    """
    matches = [
        n
        for n in graph.nodes()
        if n.fqname == fqname and n.kind == kind and path_substring in str(n.path)
    ]
    assert len(matches) == 1, f"expected unique match for {fqname}/{kind}/{path_substring}"
    return matches[0]


def test_submodule_alias_dies_when_all_siblings_dead(build_decl_graph):
    """When every sibling alias from a ``from .sub import X`` statement
    is dead, the submodule-attribute alias dies too — there's no
    longer any reason to keep the import line. The sibling edges only
    rescue the submodule alias *from being singled out as unique
    noise*, not from a legitimate sweep of the whole statement.
    """
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "from .sub import X\n",
            "pkg/sub.py": "class X: pass\n",
        }
    )
    sub_alias = _find_node(graph, fqname="pkg.sub", kind="import", path_substring="pkg/__init__.py")
    x_alias = _find_node(graph, fqname="pkg.X", kind="import", path_substring="pkg/__init__.py")

    # Nothing references X (the only sibling), so neither the sibling
    # nor the submodule-attribute alias is reachable.
    reachable = set(graph.reachable())
    assert x_alias not in reachable
    assert sub_alias not in reachable


def test_submodule_alias_not_minted_in_non_init_file(build_decl_graph):
    """``from .sub import X`` inside a non-``__init__`` module does NOT
    create the ``ImportFromSubmodule`` side-effect binding (the
    ``pkg.sub`` attribute is only rebound when the containing module
    *is* ``pkg``). Verify we don't emit a spurious sibling edge or
    a spurious submodule alias for the ordinary case.
    """
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/sub.py": "class X: pass\n",
            "pkg/consumer.py": "from .sub import X\nuse = X\n",
        }
    )
    # Only ONE node with fqname=pkg.consumer.X exists (the regular
    # alias) — no extra "submodule attribute" alias on pkg.consumer.
    consumer_aliases = [
        n for n in graph.nodes() if n.kind == "import" and "pkg/consumer.py" in str(n.path)
    ]
    fqnames = {n.fqname for n in consumer_aliases}
    assert fqnames == {"pkg.consumer.X"}
