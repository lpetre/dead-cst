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
            a = a + 1
            """,
            {
                "mod.a -> mod",
            },
            id="self-reference-drops-self-edge",
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
        # Redeclarations / shadowing (same-kind collapses cleanly)
        # ------------------------------------------------------------------
        pytest.param(
            """
            def f(): pass
            def f(): pass
            f()
            """,
            {
                "mod -> mod.f",
                "mod.f -> mod",
            },
            id="function-redefined-collapses-to-one-node",
        ),
        pytest.param(
            """
            class C: pass
            class C: pass
            C()
            """,
            {
                "mod -> mod.C",
                "mod.C -> mod",
            },
            id="class-redefined-collapses-to-one-node",
        ),
        pytest.param(
            """
            def a(): pass
            def f(): pass
            def f(): a()
            f()
            """,
            {
                "mod -> mod.f",
                "mod.a -> mod",
                "mod.f -> mod",
                "mod.f -> mod.a",
            },
            id="function-redefined-second-body-references-collapse",
        ),
        pytest.param(
            """
            def dec(f): return f
            def f(): pass
            @dec
            def f(): pass
            f()
            """,
            {
                "mod -> mod.f",
                "mod.dec -> mod",
                "mod.f -> mod",
                "mod.f -> mod.dec",
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
            {
                "mod -> mod.f",
                "mod.f -> mod",
                "mod.f -> mod.g",
                "mod.g -> mod",
            },
            id="conditional-rebind-to-alias-adds-edge",
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
            {
                "mod -> mod.f",
                "mod.a -> mod",
                "mod.b -> mod",
                "mod.f -> mod",
                "mod.f -> mod.a",
                "mod.f -> mod.b",
            },
            id="if-else-function-redefinition-unifies-edges",
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
                "mod.f -> mod",
                "mod.g -> mod",
                "mod.x -> mod",
                "mod.x -> mod.f",
                "mod.x -> mod.g",
            },
            id="try-except-assignment-unifies-both-branches",
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
