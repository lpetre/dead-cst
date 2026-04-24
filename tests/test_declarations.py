"""Tests for top-level declaration and assignment tracking.

Each case is a single-module Python source. ``expected_edges`` lists
*every* edge the resulting symbol graph should contain, so any stray or
missing edge fails the test.
"""

import pytest


@pytest.mark.parametrize(
    "src, expected_edges",
    [
        # ------------------------------------------------------------------
        # Functions
        # ------------------------------------------------------------------
        pytest.param(
            """
            def a(): pass
            def b(): a()
            b()
            """,
            {
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.b -> mod.a",
                "mod -> mod.b",
            },
            id="function-call-and-module-entry",
        ),
        pytest.param(
            """
            async def a(): pass
            async def b(): await a()
            """,
            {
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.b -> mod.a",
            },
            id="async-def",
        ),
        pytest.param(
            """
            def dec(f): return f
            @dec
            def f(): pass
            """,
            {
                "mod.dec -> mod",
                "mod.f -> mod",
                "mod.f -> mod.dec",
            },
            id="bare-decorator",
        ),
        pytest.param(
            """
            def dec(x): return lambda f: f
            P = 1
            @dec(P)
            def f(): pass
            """,
            {
                "mod.P -> mod",
                "mod.dec -> mod",
                "mod.f -> mod",
                "mod.f -> mod.P",
                "mod.f -> mod.dec",
            },
            id="decorator-with-argument-reference",
        ),
        pytest.param(
            """
            def d1(f): return f
            def d2(f): return f
            @d1
            @d2
            def f(): pass
            """,
            {
                "mod.d1 -> mod",
                "mod.d2 -> mod",
                "mod.f -> mod",
                "mod.f -> mod.d1",
                "mod.f -> mod.d2",
            },
            id="stacked-decorators",
        ),
        pytest.param(
            """
            D = 1
            def f(x=D): return x
            """,
            {
                "mod.D -> mod",
                "mod.f -> mod",
                "mod.f -> mod.D",
            },
            id="default-argument-reference",
        ),
        pytest.param(
            """
            D = 1
            def f(x=D, /): return x
            """,
            {
                "mod.D -> mod",
                "mod.f -> mod",
                "mod.f -> mod.D",
            },
            id="positional-only-default-reference",
        ),
        pytest.param(
            """
            D = 1
            def f(*, x=D): return x
            """,
            {
                "mod.D -> mod",
                "mod.f -> mod",
                "mod.f -> mod.D",
            },
            id="keyword-only-default-reference",
        ),
        pytest.param(
            """
            T = int
            def f(x: T) -> T: return x
            """,
            {
                "mod.T -> mod",
                "mod.f -> mod",
                "mod.f -> mod.T",
            },
            id="annotation-references-argument-and-return",
        ),
        pytest.param(
            """
            x = 1
            f = lambda: x
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.x",
                "mod.x -> mod",
            },
            id="lambda-closes-over-module-var",
        ),
        # ------------------------------------------------------------------
        # Classes
        # ------------------------------------------------------------------
        pytest.param(
            """
            class A: pass
            class B(A): pass
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.B -> mod.A",
            },
            id="single-inheritance",
        ),
        pytest.param(
            """
            class A: pass
            class B: pass
            class C(A, B): pass
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.C -> mod",
                "mod.C -> mod.A",
                "mod.C -> mod.B",
            },
            id="multiple-inheritance",
        ),
        pytest.param(
            """
            class Meta(type): pass
            class C(metaclass=Meta): pass
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.Meta",
                "mod.Meta -> mod",
            },
            id="metaclass-keyword",
        ),
        pytest.param(
            """
            def dec(c): return c
            @dec
            class C: pass
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.dec",
                "mod.dec -> mod",
            },
            id="class-decorator",
        ),
        pytest.param(
            """
            X = 1
            class C:
                y = X
                def m(self): return self.y
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.X",
                "mod.X -> mod",
            },
            id="class-body-references-module-var",
        ),
        pytest.param(
            """
            def helper(): return 1
            class C:
                v = helper()
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.helper",
                "mod.helper -> mod",
            },
            id="class-body-calls-module-function",
        ),
        pytest.param(
            """
            class A: pass
            Base = A
            class C(Base): pass
            """,
            {
                "mod.A -> mod",
                "mod.Base -> mod",
                "mod.Base -> mod.A",
                "mod.C -> mod",
                "mod.C -> mod.Base",
            },
            id="class-base-is-variable-alias",
        ),
        pytest.param(
            """
            class E(Exception): pass
            def f():
                try: pass
                except E: pass
            """,
            {
                "mod.E -> mod",
                "mod.f -> mod",
                "mod.f -> mod.E",
            },
            id="exception-class-in-except-clause",
        ),
        pytest.param(
            """
            class CM:
                def __enter__(self): return self
                def __exit__(self, *a): ...
            def f():
                with CM() as c:
                    pass
            """,
            {
                "mod.CM -> mod",
                "mod.f -> mod",
                "mod.f -> mod.CM",
            },
            id="context-manager-class-in-with",
        ),
        # ------------------------------------------------------------------
        # Variables and assignments
        # ------------------------------------------------------------------
        pytest.param(
            """
            a = 1
            b = a
            """,
            {
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.b -> mod.a",
            },
            id="simple-variable-copy",
        ),
        pytest.param(
            """
            T = int
            x: T = 1
            """,
            {
                "mod.T -> mod",
                "mod.x -> mod",
                "mod.x -> mod.T",
            },
            id="annotated-assign-with-value",
        ),
        pytest.param(
            """
            T = int
            x: T
            """,
            {
                "mod.T -> mod",
                "mod.x -> mod",
                "mod.x -> mod.T",
            },
            id="annotated-assign-without-value",
        ),
        pytest.param(
            """
            a, b = 1, 2
            c, d = a, b
            """,
            {
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.c -> mod",
                "mod.c -> mod.a",
                "mod.d -> mod",
                "mod.d -> mod.b",
            },
            id="tuple-unpacking-pairwise",
        ),
        pytest.param(
            """
            def f(): return 1
            def g(): return 2
            a, b = f(), g()
            """,
            {
                "mod.a -> mod",
                "mod.a -> mod.f",
                "mod.b -> mod",
                "mod.b -> mod.g",
                "mod.f -> mod",
                "mod.g -> mod",
            },
            id="tuple-of-calls-pairwise",
        ),
        pytest.param(
            """
            xs = [1, 2, 3]
            a, *b, c = xs
            """,
            {
                "mod.a -> mod",
                "mod.a -> mod.xs",
                "mod.b -> mod",
                "mod.b -> mod.xs",
                "mod.c -> mod",
                "mod.c -> mod.xs",
                "mod.xs -> mod",
            },
            id="starred-target-in-tuple-unpacking",
        ),
        pytest.param(
            """
            def f(): pass
            b = c = f
            """,
            {
                "mod.b -> mod",
                "mod.b -> mod.f",
                "mod.c -> mod",
                "mod.c -> mod.f",
                "mod.f -> mod",
            },
            id="chained-assignment-shares-rhs",
        ),
        pytest.param(
            """
            def f(): return 1, 2
            a, b = f()
            """,
            {
                "mod.a -> mod",
                "mod.a -> mod.f",
                "mod.b -> mod",
                "mod.b -> mod.f",
                "mod.f -> mod",
            },
            id="tuple-unpack-from-call-broadcasts-rhs",
        ),
        pytest.param(
            "[a, b] = 1, 2\n",
            {
                "mod.a -> mod",
                "mod.b -> mod",
            },
            id="list-target-pattern-produces-decls",
        ),
        pytest.param(
            "(a, (b, c)) = 1, (2, 3)\n",
            {
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.c -> mod",
            },
            id="nested-tuple-target-descends",
        ),
        pytest.param(
            """
            def f(): return 1
            def g(): return 2
            if True: x = f()
            else: x = g()
            """,
            {
                "mod.f -> mod",
                "mod.g -> mod",
                "mod.x -> mod",
                "mod.x -> mod.f",
                "mod.x -> mod.g",
            },
            id="conditional-assignment-unifies-both-branches",
        ),
        pytest.param(
            """
            a = 1
            b = a
            a = 2
            """,
            {
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.b -> mod.a",
            },
            id="reassignment-does-not-duplicate-decl",
        ),
        pytest.param(
            """
            a = 1
            b = 1
            a += b
            """,
            {
                "mod -> mod.b",
                "mod.a -> mod",
                "mod.b -> mod",
            },
            id="augmented-assign-rhs-is-module-level-read",
        ),
        # ------------------------------------------------------------------
        # Expressions in RHS / control flow
        # ------------------------------------------------------------------
        pytest.param(
            """
            A = 1
            B = 2
            items = [A, B]
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.items -> mod",
                "mod.items -> mod.A",
                "mod.items -> mod.B",
            },
            id="list-literal-references",
        ),
        pytest.param(
            """
            K = 1
            V = 2
            d = {K: V}
            """,
            {
                "mod.K -> mod",
                "mod.V -> mod",
                "mod.d -> mod",
                "mod.d -> mod.K",
                "mod.d -> mod.V",
            },
            id="dict-literal-references",
        ),
        pytest.param(
            """
            A = [1]
            B = [x for x in A if x]
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.B -> mod.A",
            },
            id="list-comprehension-reference",
        ),
        pytest.param(
            """
            A = [1]
            B = {x: x for x in A}
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.B -> mod.A",
            },
            id="dict-comprehension-reference",
        ),
        pytest.param(
            """
            NAME = 'x'
            def f(): return f'{NAME}'
            """,
            {
                "mod.NAME -> mod",
                "mod.f -> mod",
                "mod.f -> mod.NAME",
            },
            id="fstring-interpolation-reference",
        ),
        pytest.param(
            """
            def f(): return 1
            x = (y := f())
            """,
            {
                "mod.f -> mod",
                "mod.x -> mod",
                "mod.x -> mod.f",
            },
            id="walrus-in-assignment-rhs",
        ),
        pytest.param(
            """
            data = [1]
            for x in data: pass
            """,
            {
                "mod -> mod.data",
                "mod.data -> mod",
            },
            id="module-level-for-loop",
        ),
        pytest.param(
            """
            data = [1]
            def f():
                for x in data: pass
            """,
            {
                "mod.data -> mod",
                "mod.f -> mod",
                "mod.f -> mod.data",
            },
            id="for-loop-inside-function",
        ),
        pytest.param(
            """
            a = [1,2]
            def gen():
                for x in a: yield x
            """,
            {
                "mod.a -> mod",
                "mod.gen -> mod",
                "mod.gen -> mod.a",
            },
            id="generator-function-body-reference",
        ),
        pytest.param(
            """
            def f(): pass
            f()
            """,
            {
                "mod -> mod.f",
                "mod.f -> mod",
            },
            id="module-level-call-is-entry-edge",
        ),
        pytest.param(
            """
            X = 1
            def outer():
                y = X
                def inner():
                    return y
                return inner
            """,
            {
                "mod.X -> mod",
                "mod.outer -> mod",
                "mod.outer -> mod.X",
            },
            id="closure-reference-folds-into-outer-decl",
        ),
        # ------------------------------------------------------------------
        # Local shadowing of module-level names does not create an edge
        # ------------------------------------------------------------------
        pytest.param(
            """
            X = 1
            def f():
                X = 2
                return X
            """,
            {
                "mod.X -> mod",
                "mod.f -> mod",
            },
            id="local-var-shadows-module-var",
        ),
        pytest.param(
            """
            X = 1
            def f(X): return X
            """,
            {
                "mod.X -> mod",
                "mod.f -> mod",
            },
            id="parameter-shadows-module-var",
        ),
        pytest.param(
            """
            X = 1
            def f():
                for X in [1, 2]:
                    pass
            """,
            {
                "mod.X -> mod",
                "mod.f -> mod",
            },
            id="for-loop-target-shadows-module-var",
        ),
        pytest.param(
            """
            E = 1
            def f():
                try: pass
                except Exception as E: pass
            """,
            {
                "mod.E -> mod",
                "mod.f -> mod",
            },
            id="except-as-shadows-module-var",
        ),
        pytest.param(
            """
            X = 1
            def f():
                with open('a') as X:
                    pass
            """,
            {
                "mod.X -> mod",
                "mod.f -> mod",
            },
            id="with-as-shadows-module-var",
        ),
        pytest.param(
            """
            def helper(): pass
            def outer():
                def helper(): pass
                helper()
            """,
            {
                "mod.helper -> mod",
                "mod.outer -> mod",
            },
            id="nested-function-shadows-outer-name",
        ),
        pytest.param(
            """
            X = 1
            class C:
                X = 2
                def m(self): return C.X
            """,
            {
                "mod.C -> mod",
                "mod.X -> mod",
            },
            id="class-attribute-shadows-module-var",
        ),
        pytest.param(
            """
            X = 1
            class C:
                def m(self):
                    X = 2
                    return X
            """,
            {
                "mod.C -> mod",
                "mod.X -> mod",
            },
            id="method-local-shadows-module-var",
        ),
        pytest.param(
            """
            X = [1, 2]
            def f():
                return [X for X in X]
            """,
            # Python evaluates the outermost iterable of a comprehension
            # in the enclosing scope, so the ``X`` in ``for X in X``
            # still resolves to the module-level ``X``. The comprehension
            # target then shadows it for the element expression.
            {
                "mod.X -> mod",
                "mod.f -> mod",
                "mod.f -> mod.X",
            },
            id="comprehension-iterable-uses-enclosing-scope",
        ),
    ],
)
def test_declarations(build_decl_graph, assert_edges, src, expected_edges):
    graph = build_decl_graph({"mod.py": src})
    assert_edges(graph, expected_edges)


@pytest.mark.parametrize(
    "src, expected_edges",
    [
        # ------------------------------------------------------------------
        # Same-kind redeclaration creates one node per textual decl.
        # The fqname-only edge view collapses these cases; here we assert
        # the per-position structure directly.
        # ------------------------------------------------------------------
        pytest.param(
            """
            a = 1
            a = a + 1
            """,
            # Two distinct ``mod.a`` variable nodes at different
            # positions. The second assignment's RHS genuinely references
            # the first binding -- no cycle in the underlying graph.
            # Both decls -- including the shadowed one -- carry a
            # parent-module edge so the graph stays well-formed.
            {
                "mod.a@1:0 -> mod",
                "mod.a@2:0 -> mod",
                "mod.a@2:0 -> mod.a@1:0",
            },
            id="reassignment-creates-two-nodes",
        ),
        pytest.param(
            """
            def f(): pass
            def f(): pass
            f()
            """,
            # The flow filter drops the shadowed line-1 def at the call
            # site, so only the surviving ``mod.f@2:0`` is reachable
            # there. The shadowed ``mod.f@1:0`` still belongs to the
            # module, so it carries the parent-module edge.
            {
                "mod -> mod.f@2:0",
                "mod.f@1:0 -> mod",
                "mod.f@2:0 -> mod",
            },
            id="function-redefined-creates-two-nodes",
        ),
        pytest.param(
            """
            class C: pass
            class C: pass
            C()
            """,
            {
                "mod -> mod.C@2:0",
                "mod.C@1:0 -> mod",
                "mod.C@2:0 -> mod",
            },
            id="class-redefined-creates-two-nodes",
        ),
        pytest.param(
            """
            def a(): pass
            def f(): pass
            def f(): a()
            f()
            """,
            # ``f()`` only reaches the last ``def f`` after filtering.
            # The shadowed ``mod.f@2:0`` keeps its parent-module edge so
            # it stays in the graph as a (now-orphan-from-the-call-site)
            # decl.
            {
                "mod -> mod.f@3:0",
                "mod.a@1:0 -> mod",
                "mod.f@2:0 -> mod",
                "mod.f@3:0 -> mod",
                "mod.f@3:0 -> mod.a@1:0",
            },
            id="function-redefined-second-body-has-own-edges",
        ),
        pytest.param(
            """
            def dec(f): return f
            def f(): pass
            @dec
            def f(): pass
            f()
            """,
            # Decorator edge lives on the decorated (second) decl only.
            {
                "mod -> mod.f@4:0",
                "mod.dec@1:0 -> mod",
                "mod.f@2:0 -> mod",
                "mod.f@4:0 -> mod",
                "mod.f@4:0 -> mod.dec@1:0",
            },
            id="function-redefined-with-decorator-on-second-copy",
        ),
        pytest.param(
            """
            def f(): pass
            def g(): pass
            if True: f = g
            f()
            """,
            # ``f = g`` creates a second ``mod.f`` variable node at
            # column 9 (after ``if True: ``). The alias edge lives on
            # that node.
            {
                "mod -> mod.f@1:0",
                "mod -> mod.f@3:9",
                "mod.f@1:0 -> mod",
                "mod.f@3:9 -> mod",
                "mod.f@3:9 -> mod.g@2:0",
                "mod.g@2:0 -> mod",
            },
            id="conditional-rebind-to-alias",
        ),
        pytest.param(
            """
            def a(): pass
            def b(): pass
            if True:
                def f(): a()
            else:
                def f(): b()
            f()
            """,
            # Each branch's ``def f`` is its own node with its own body
            # edge. ``f()`` resolves to both.
            {
                "mod -> mod.f@4:4",
                "mod -> mod.f@6:4",
                "mod.a@1:0 -> mod",
                "mod.b@2:0 -> mod",
                "mod.f@4:4 -> mod",
                "mod.f@4:4 -> mod.a@1:0",
                "mod.f@6:4 -> mod",
                "mod.f@6:4 -> mod.b@2:0",
            },
            id="if-else-function-redefinition",
        ),
        pytest.param(
            """
            def f(): return 1
            def g(): return 2
            try:
                x = f()
            except Exception:
                x = g()
            """,
            {
                "mod.f@1:0 -> mod",
                "mod.g@2:0 -> mod",
                "mod.x@4:4 -> mod",
                "mod.x@4:4 -> mod.f@1:0",
                "mod.x@6:4 -> mod",
                "mod.x@6:4 -> mod.g@2:0",
            },
            id="try-except-assignment-creates-two-x-nodes",
        ),
    ],
)
def test_redeclarations(build_decl_graph, assert_positional_edges, src, expected_edges):
    graph = build_decl_graph({"mod.py": src})
    assert_positional_edges(graph, expected_edges)


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        # ------------------------------------------------------------------
        # Cross-kind shadowing: the flow filter drops the shadowed
        # referent at the access, so the shadowed decl loses its only
        # incoming edge and no longer drags its upstream along.
        # ------------------------------------------------------------------
        pytest.param(
            {
                "other.py": "def f(): pass\n",
                "mod.py": """
                from other import f
                def f(): pass
                f()
                """,
            },
            # ``f()`` resolves only to the line-2 def; the shadowed
            # import node at col 18 is still in the graph (its own
            # import edges wire it to ``other``), but nothing points at
            # it so ``other`` is no longer kept alive by this module.
            {
                "mod -> mod.f@2:0",
                "mod.f@1:18 -> mod",
                "mod.f@1:18 -> other",
                "mod.f@1:18 -> other.f@1:0",
                "mod.f@2:0 -> mod",
                "other.f@1:0 -> other",
            },
            id="import-shadowed-by-function",
        ),
        pytest.param(
            {
                "other.py": "def f(): pass\n",
                "mod.py": """
                def f(): pass
                from other import f
                f()
                """,
            },
            # The shadowing import wins at the call; the shadowed def
            # at line 1 has no incoming edges and stays in the graph
            # only via its parent-module edge.
            {
                "mod -> mod.f@2:18",
                "mod -> other",
                "mod -> other.f@1:0",
                "mod.f@1:0 -> mod",
                "mod.f@2:18 -> mod",
                "mod.f@2:18 -> other",
                "mod.f@2:18 -> other.f@1:0",
                "other.f@1:0 -> other",
            },
            id="function-shadowed-by-import",
        ),
        pytest.param(
            {
                "other.py": "def f(): pass\n",
                "mod.py": """
                from other import f
                f = 1
                print(f)
                """,
            },
            {
                "mod -> mod.f@2:0",
                "mod.f@1:18 -> mod",
                "mod.f@1:18 -> other",
                "mod.f@1:18 -> other.f@1:0",
                "mod.f@2:0 -> mod",
                "other.f@1:0 -> other",
            },
            id="import-rebound-to-constant",
        ),
        pytest.param(
            {
                "a.py": "def x(): pass\n",
                "b.py": "def x(): pass\n",
                "mod.py": """
                from a import x
                from b import x
                x()
                """,
            },
            # Only the second import is live at the call, so ``a`` is
            # unreachable from ``mod`` (the line-1 import node lingers
            # as an orphan but contributes no new incoming edges).
            {
                "a.x@1:0 -> a",
                "b.x@1:0 -> b",
                "mod -> b",
                "mod -> b.x@1:0",
                "mod -> mod.x@2:14",
                "mod.x@1:14 -> a",
                "mod.x@1:14 -> a.x@1:0",
                "mod.x@1:14 -> mod",
                "mod.x@2:14 -> b",
                "mod.x@2:14 -> b.x@1:0",
                "mod.x@2:14 -> mod",
            },
            id="two-imports-same-alias-last-wins",
        ),
        # ------------------------------------------------------------------
        # Same-kind redeclaration: the dead body is no longer kept alive
        # through the shadowed decl.
        # ------------------------------------------------------------------
        pytest.param(
            {
                "mod.py": """
                def dead_helper(): pass
                def f(): dead_helper()
                def f(): pass
                f()
                """,
            },
            # ``f()`` only reaches the line-3 def now, so the dead body
            # at line 2 becomes an orphan -- ``dead_helper`` is no
            # longer reachable.
            {
                "mod -> mod.f@3:0",
                "mod.dead_helper@1:0 -> mod",
                "mod.f@2:0 -> mod",
                "mod.f@2:0 -> mod.dead_helper@1:0",
                "mod.f@3:0 -> mod",
            },
            id="dead-function-body-is-orphaned",
        ),
        pytest.param(
            {
                "mod.py": """
                def dead_helper(): pass
                class C:
                    def m(self): dead_helper()
                class C: pass
                C()
                """,
            },
            {
                "mod -> mod.C@4:0",
                "mod.C@2:0 -> mod",
                "mod.C@2:0 -> mod.dead_helper@1:0",
                "mod.C@4:0 -> mod",
                "mod.dead_helper@1:0 -> mod",
            },
            id="dead-method-body-is-orphaned",
        ),
        pytest.param(
            {
                "mod.py": """
                def a(): pass
                def b(): pass
                def f(): a()
                def f(): b()
                def f(): pass
                f()
                """,
            },
            # Only the last ``def f`` is live at the call; the two
            # earlier bodies keep their body edges but nothing reaches
            # them.
            {
                "mod -> mod.f@5:0",
                "mod.a@1:0 -> mod",
                "mod.b@2:0 -> mod",
                "mod.f@3:0 -> mod",
                "mod.f@3:0 -> mod.a@1:0",
                "mod.f@4:0 -> mod",
                "mod.f@4:0 -> mod.b@2:0",
                "mod.f@5:0 -> mod",
            },
            id="chain-of-shadowed-functions-orphans-earlier-bodies",
        ),
    ],
)
def test_shadowed_declarations(build_decl_graph, assert_positional_edges, files, expected_edges):
    graph = build_decl_graph(files)
    assert_positional_edges(graph, expected_edges)


def test_module_hierarchy_edges(build_decl_graph, assert_edges):
    """Submodules point at their parent package to keep __init__.py alive."""
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/leaf.py": "",
        }
    )
    assert_edges(
        graph,
        {
            "pkg.sub -> pkg",
            "pkg.sub.leaf -> pkg.sub",
        },
    )
