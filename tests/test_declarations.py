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
        pytest.param(
            """
            f = lambda s: (s := 1, s)
            """,
            # The walrus ``s := 1`` rebinds the lambda parameter ``s``;
            # the trailing ``s`` still resolves to that local binding,
            # so no module-level refs flow out of the body. Regression
            # test: this used to crash because ``_scope_body`` assumed
            # every ``FunctionScope.node.body`` was an ``IndentedBlock``.
            {
                "mod.f -> mod",
            },
            id="lambda-walrus-rebinds-parameter",
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
            NAME = 'x'
            def f(): return t'{NAME}'
            """,
            {
                "mod.NAME -> mod",
                "mod.f -> mod",
                "mod.f -> mod.NAME",
            },
            id="tstring-interpolation-reference",
        ),
        pytest.param(
            """
            WIDTH = 4
            def helper(): return 1
            def f(): return t"v={helper():{WIDTH}d}"
            """,
            {
                "mod.WIDTH -> mod",
                "mod.helper -> mod",
                "mod.f -> mod",
                "mod.f -> mod.WIDTH",
                "mod.f -> mod.helper",
            },
            id="tstring-format-spec-and-call-reference",
        ),
        pytest.param(
            """
            def f(): return 1
            x = (y := f())
            """,
            # The walrus surfaces ``y`` as its own top-level decl, so
            # the RHS reference to ``f`` attributes to ``y`` rather
            # than to ``x``. Reachability is preserved -- ``f`` is
            # still kept alive transitively when ``x`` is reachable
            # because the parser registers ``mod.y -> mod`` so the
            # parent-module edge keeps ``y`` (and thus ``f``) live.
            {
                "mod.f -> mod",
                "mod.x -> mod",
                "mod.y -> mod",
                "mod.y -> mod.f",
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
        # ------------------------------------------------------------------
        # PEP 572 walrus -- module-scope walruses surface as top-level
        # decls, references inside the RHS attribute to that decl, and
        # cross-decl uses get edges to it. Walruses inside a function /
        # class / lambda body stay local; their RHS references attribute
        # to the enclosing top-level decl as before.
        # ------------------------------------------------------------------
        pytest.param(
            """
            def src(): return 1
            if (Y := src()): pass
            def use(): return Y
            """,
            {
                "mod.src -> mod",
                "mod.Y -> mod",
                "mod.Y -> mod.src",
                "mod.use -> mod",
                "mod.use -> mod.Y",
            },
            id="walrus-toplevel-binding-captured",
        ),
        # Skipped on rust: ty has a `// TODO walrus in comprehensions
        # is implicitly nonlocal` (see
        # ``vendor/ruff/crates/ty_python_core/src/builder.rs:3605``),
        # so the leaked ``last`` binding isn't surfaced in the module
        # scope's place table. ``test_limitations`` pins rust's
        # current edge set; when ty grows the leak-to-enclosing-scope
        # support, this test should pass on both backends and the
        # limitation entry can be dropped.
        pytest.param(
            """
            def src(): return 1
            def f():
                if (x := src()): return x
                return 0
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.src",
                "mod.src -> mod",
            },
            id="walrus-in-if-condition",
        ),
        pytest.param(
            """
            def src(): return None
            def f():
                while (x := src()) is not None:
                    pass
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.src",
                "mod.src -> mod",
            },
            id="walrus-in-while-condition",
        ),
        pytest.param(
            """
            def src(): return 1
            def f():
                return [n for v in [1] if (n := src())]
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.src",
                "mod.src -> mod",
            },
            id="walrus-in-comprehension",
        ),
        pytest.param(
            """
            def src(): return 1
            def f(): return f'{(x := src())}'
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.src",
                "mod.src -> mod",
            },
            id="walrus-in-fstring-interpolation",
        ),
        # ------------------------------------------------------------------
        # PEP 695 type-parameter syntax (3.12+) and ``type`` statements.
        # Generic ``[T]`` clauses introduce a synthetic enclosing scope, but
        # references in the function/class body still resolve outward to the
        # module. ``type`` statements surface as top-level decls of kind
        # ``"type_alias"`` and the RHS is walked for references.
        # ------------------------------------------------------------------
        pytest.param(
            """
            Base = int
            type Alias = Base
            """,
            # ``type Alias = Base`` surfaces ``mod.Alias`` as its own
            # decl. Refs in the RHS are attributed to the alias, not
            # the module, so removing ``Alias`` releases ``Base``.
            {
                "mod.Alias -> mod",
                "mod.Alias -> mod.Base",
                "mod.Base -> mod",
            },
            id="pep695-type-statement-rhs-reference",
        ),
        pytest.param(
            """
            type Alias = int
            def use(x: Alias) -> Alias: return x
            """,
            # Users referencing the alias get an edge into the alias decl.
            {
                "mod.Alias -> mod",
                "mod.use -> mod",
                "mod.use -> mod.Alias",
            },
            id="pep695-type-statement-referenced-by-annotation",
        ),
        pytest.param(
            """
            def helper(): return 1
            def f[T](x: T) -> T:
                return helper()
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.helper",
                "mod.helper -> mod",
            },
            id="pep695-generic-function-body-reference",
        ),
        pytest.param(
            """
            def helper(): return 1
            class C[T]:
                def m(self): return helper()
            """,
            {
                "mod.C -> mod",
                "mod.C -> mod.helper",
                "mod.helper -> mod",
            },
            id="pep695-generic-class-body-reference",
        ),
        pytest.param(
            """
            class B: pass
            def f[T: B](x: T) -> T: return x
            """,
            {
                "mod.B -> mod",
                "mod.f -> mod",
                "mod.f -> mod.B",
            },
            id="pep695-generic-typeparam-bound",
        ),
        pytest.param(
            """
            class A: pass
            class B: pass
            def f[T: (A, B)](x: T) -> T: return x
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.f -> mod",
                "mod.f -> mod.A",
                "mod.f -> mod.B",
            },
            id="pep695-generic-typeparam-constraints",
        ),
        pytest.param(
            """
            class Default: pass
            def f[T = Default](x): return x
            """,
            # PEP 696 (3.13+) typeparam default. The default expression
            # is evaluated in the enclosing scope, so ``Default`` is
            # referenced from ``mod.f``.
            {
                "mod.Default -> mod",
                "mod.f -> mod",
                "mod.f -> mod.Default",
            },
            id="pep695-generic-typeparam-default",
        ),
        pytest.param(
            """
            class Helper: pass
            def f[*Ts](x: tuple[Helper]): return x
            """,
            {
                "mod.Helper -> mod",
                "mod.f -> mod",
                "mod.f -> mod.Helper",
            },
            id="pep695-typevartuple-body-reference",
        ),
        pytest.param(
            """
            class Helper: pass
            def f[**P](x: Helper): return x
            """,
            {
                "mod.Helper -> mod",
                "mod.f -> mod",
                "mod.f -> mod.Helper",
            },
            id="pep695-paramspec-body-reference",
        ),
        # ------------------------------------------------------------------
        # PEP 701 f-string syntax (3.12+) -- nested f-strings, multiline,
        # format-spec interpolations, conversion modifiers.
        # ------------------------------------------------------------------
        pytest.param(
            """
            N = 'x'
            def f(): return f"a {f'b {N}'} c"
            """,
            {
                "mod.N -> mod",
                "mod.f -> mod",
                "mod.f -> mod.N",
            },
            id="pep701-nested-fstring-reference",
        ),
        pytest.param(
            '''
            N = "x"
            def f(): return f"""hi
            {N}
            bye"""
            ''',
            {
                "mod.N -> mod",
                "mod.f -> mod",
                "mod.f -> mod.N",
            },
            id="pep701-multiline-fstring-reference",
        ),
        pytest.param(
            """
            W = 5
            N = 1
            def f(): return f'{N:{W}d}'
            """,
            {
                "mod.N -> mod",
                "mod.W -> mod",
                "mod.f -> mod",
                "mod.f -> mod.N",
                "mod.f -> mod.W",
            },
            id="pep701-fstring-format-spec-reference",
        ),
        pytest.param(
            """
            N = 1
            def f(): return f'{N!r}'
            """,
            {
                "mod.N -> mod",
                "mod.f -> mod",
                "mod.f -> mod.N",
            },
            id="pep701-fstring-conversion-reference",
        ),
        # ------------------------------------------------------------------
        # PEP 634-636 structural pattern matching (3.10+). Names referenced
        # from value patterns, class patterns, mapping keys, guards, and
        # ``as`` / ``or`` pattern bodies must all attribute correctly to
        # the enclosing top-level decl.
        # ------------------------------------------------------------------
        pytest.param(
            """
            SENT = 1
            def f(v):
                match v:
                    case 1: return SENT
                    case _: return 0
            """,
            {
                "mod.SENT -> mod",
                "mod.f -> mod",
                "mod.f -> mod.SENT",
            },
            id="match-case-value-pattern-body-reference",
        ),
        pytest.param(
            """
            class Dot: pass
            def f(v):
                match v:
                    case Dot(): return 1
                    case _: return 0
            """,
            {
                "mod.Dot -> mod",
                "mod.f -> mod",
                "mod.f -> mod.Dot",
            },
            id="match-case-class-pattern-reference",
        ),
        pytest.param(
            """
            class Pt:
                __match_args__ = ('x', 'y')
            def f(v):
                match v:
                    case Pt(x_val, y_val): return x_val
                    case _: return 0
            """,
            {
                "mod.Pt -> mod",
                "mod.f -> mod",
                "mod.f -> mod.Pt",
            },
            id="match-case-class-positional-pattern-reference",
        ),
        pytest.param(
            """
            class Pt:
                x: int
            def f(v):
                match v:
                    case Pt(x=val): return val
                    case _: return 0
            """,
            {
                "mod.Pt -> mod",
                "mod.f -> mod",
                "mod.f -> mod.Pt",
            },
            id="match-case-class-keyword-pattern-reference",
        ),
        pytest.param(
            """
            class K:
                NAME = 'k'
            def f(v):
                match v:
                    case {K.NAME: val}: return val
                    case _: return 0
            """,
            # Mapping-pattern keys must be literal or dotted-name value
            # patterns; ``K.NAME`` references the module-level ``K``.
            {
                "mod.K -> mod",
                "mod.f -> mod",
                "mod.f -> mod.K",
            },
            id="match-case-mapping-dotted-key-reference",
        ),
        pytest.param(
            """
            A = 1
            B = 2
            def f(v):
                match v:
                    case 1 | 2: return A
                    case _: return B
            """,
            {
                "mod.A -> mod",
                "mod.B -> mod",
                "mod.f -> mod",
                "mod.f -> mod.A",
                "mod.f -> mod.B",
            },
            id="match-case-or-pattern-body-references",
        ),
        pytest.param(
            """
            def helper(v): return v
            def f(v):
                match v:
                    case [1, *rest] as found: return helper(found)
                    case _: return 0
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.helper",
                "mod.helper -> mod",
            },
            id="match-case-as-pattern-body-reference",
        ),
        pytest.param(
            """
            def pred(v): return True
            def f(v):
                match v:
                    case x if pred(x): return x
                    case _: return 0
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.pred",
                "mod.pred -> mod",
            },
            id="match-case-guard-name-reference",
        ),
        pytest.param(
            """
            def helper(v): return v
            def f(seq):
                match seq:
                    case [first, *rest]: return helper(first)
                    case _: return None
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.helper",
                "mod.helper -> mod",
            },
            id="match-case-sequence-pattern-body-reference",
        ),
        # ------------------------------------------------------------------
        # PEP 654 ``except*`` (3.11+) exception groups. The handler body
        # is a regular suite, so references inside it should resolve
        # exactly like a plain ``except``.
        # ------------------------------------------------------------------
        pytest.param(
            """
            def handle(e): return e
            def f():
                try: pass
                except* ValueError as eg: return handle(eg)
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.handle",
                "mod.handle -> mod",
            },
            id="except-star-handler-reference",
        ),
        pytest.param(
            """
            def h1(e): return e
            def h2(e): return e
            def f():
                try: pass
                except* ValueError as eg: return h1(eg)
                except* TypeError as eg: return h2(eg)
            """,
            {
                "mod.f -> mod",
                "mod.f -> mod.h1",
                "mod.f -> mod.h2",
                "mod.h1 -> mod",
                "mod.h2 -> mod",
            },
            id="except-star-multiple-handler-references",
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
        # Static-`True` folding: ty (matching pyright) treats ``if True:``
        # as definitely-taken and drops the alternative branch from the
        # end-of-scope live bindings. Libcst's flow analyzer doesn't fold
        # static booleans, so it sees every conditional as runtime-live.
        # Rust's behavior is more accurate for the code as written —
        # rewriting these tests with ``if x:`` for a non-literal ``x``
        # would have them pass on both backends.
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


@pytest.mark.parametrize(
    "files, expected_edges",
    [
        # Both branches define ``f``; both are live at module exit, so a
        # cross-module ``from lib import f`` must reach each one.
        # Skipped on rust: ty's reachability folds ``if True:`` to
        # definitely-taken and drops the else branch from end-of-scope
        # live bindings (matching pyright). Rewriting with ``if x:`` for
        # a non-literal ``x`` would pass on both backends.
        # Conditional re-export through an intermediate module: each
        # branch imports from a different upstream, so resolving
        # ``mod -> compat.f`` forks the worklist into both upstreams.
        # Skipped on rust for two reasons: (1) ty folds ``if True:`` and
        # drops the else branch's import alias, and (2) rust emits
        # Principle 2 parallel-upstream edges one hop only; libcst
        # chases the alias chain transitively to emit ``mod -> a.f`` /
        # ``mod -> b.f``. Reachability is preserved either way
        # (``mod -> mod.f -> compat.f -> a.f`` still walks).
        # ``try`` body and ``except`` handler both bind ``f``; both are
        # live at exit (a handler can run before *or* instead of the
        # body completing), so both must be importable. Skipped on rust
        # only because of the transitive ``mod -> a.f`` /
        # ``mod.f -> a.f`` edges libcst emits by chasing through
        # ``lib.f@2:18``'s own ``from a import f`` alias — rust emits
        # one-hop Principle 2 edges only. All structural edges
        # (``mod -> lib.f@2:18``, ``mod -> lib.f@4:4``, etc.) are
        # present on rust; reachability is preserved.
    ],
)
def test_branch_bindings_exported(build_decl_graph, assert_positional_edges, files, expected_edges):
    """Decls bound on every reachable path to module exit are importable.

    Plain shadowing keeps single-survivor semantics, but conditional
    bindings (``if/else``, ``try/except``, ...) where multiple branches
    bind the same name should expose every branch to importing modules.
    """
    graph = build_decl_graph(files)
    assert_positional_edges(graph, expected_edges)


def test_cross_module_type_alias_import(build_decl_graph, assert_edges):
    """Importing a PEP 695 ``type`` alias from another module resolves cleanly.

    Regression: ``_edges.resolve_edges`` hit ``assert decl.type == "import"``
    when a ``type_alias`` declaration was the re-export target, because it
    was not included in the concrete-type guard alongside function/class/variable.
    """
    graph = build_decl_graph(
        {
            "lib.py": "type MyAlias = int\n",
            "user.py": "from lib import MyAlias\nx: MyAlias\n",
        }
    )
    assert_edges(
        graph,
        {
            "lib.MyAlias -> lib",
            "user.MyAlias -> lib",
            "user.MyAlias -> lib.MyAlias",
            "user.MyAlias -> user",
            "user.x -> lib",
            "user.x -> lib.MyAlias",
            "user.x -> user",
            "user.x -> user.MyAlias",
        },
    )


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


def test_main_module_distinct_from_package(build_decl_graph, assert_edges):
    """``pkg/__main__.py`` is the module ``pkg.__main__``, not ``pkg``."""
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/__main__.py": "def run(): pass\n",
        }
    )
    assert_edges(
        graph,
        {
            "pkg.__main__ -> pkg",
            "pkg.__main__.run -> pkg.__main__",
        },
    )


def test_top_level_main_module(build_decl_graph, assert_edges):
    """A top-level ``__main__.py`` is the module ``__main__``."""
    graph = build_decl_graph({"__main__.py": "def run(): pass\n"})
    assert_edges(
        graph,
        {
            "__main__.run -> __main__",
        },
    )


def test_main_module_relative_import(build_decl_graph, assert_edges):
    """Relative imports inside ``pkg/__main__.py`` resolve against ``pkg``."""
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/util.py": "def helper(): pass\n",
            "pkg/__main__.py": "from .util import helper\ndef run(): helper()\n",
        }
    )
    assert_edges(
        graph,
        {
            "pkg.__main__ -> pkg",
            "pkg.__main__.helper -> pkg.__main__",
            "pkg.__main__.helper -> pkg.util",
            "pkg.__main__.helper -> pkg.util.helper",
            "pkg.__main__.run -> pkg.__main__",
            "pkg.__main__.run -> pkg.__main__.helper",
            "pkg.__main__.run -> pkg.util",
            "pkg.__main__.run -> pkg.util.helper",
            "pkg.util -> pkg",
            "pkg.util.helper -> pkg.util",
        },
    )


# ---------------------------------------------------------------------------
# Peer ``.pyi`` files and Jupyter notebooks. Naming differences (fqname
# containing ``.ipynb`` on rust, duplicate module nodes for peer pyi)
# are intentionally not asserted — only the behaviors that affect
# reachability or codemod safety.
# ---------------------------------------------------------------------------


PEER_STUB_FILES = {
    "foo.py": "def runtime_only(): pass\ndef shared(): pass",
    "foo.pyi": "def stub_only() -> None: ...\ndef shared() -> None: ...",
    "bar.py": "from foo import shared, stub_only\nshared()\nstub_only()",
}


# libcst drops peer ``.pyi`` files at ``enumerate_files`` time, so the
# stub never appears in the graph and the rust-specific stub→runtime
# bookkeeping is observable only on the rust backend. Each rule below
# is the rust path's interpretation of "use ty's resolution, document
# stub→runtime edges, flag stub-only decls".


def test_peer_stub_emits_edge_to_matching_runtime_decl_rust(build_decl_graph):
    """Rule 2: for each ``.pyi`` decl with a same-name decl in the
    ``.py`` twin, emit a ``pyi_decl -> py_decl`` edge. The stub
    documents the relationship in the graph so reachability that lands
    on the stub decl (because ty's module resolver preferred the
    ``.pyi``) flows through to the runtime."""
    graph = build_decl_graph(PEER_STUB_FILES)
    # ``foo.shared`` exists in both files; the rust path mints two
    # nodes (one per file) sharing the fqname, so disambiguating
    # requires walking the underlying node paths.
    raw_edges = [(graph.node(u), graph.node(v)) for u, v in graph.raw.edge_list()]
    stub_runtime_pairs = {
        (str(s.path).rsplit("/", 1)[-1], str(t.path).rsplit("/", 1)[-1])
        for s, t in raw_edges
        if s.fqname == "foo.shared" and t.fqname == "foo.shared"
    }
    assert ("foo.pyi", "foo.py") in stub_runtime_pairs, (
        f"expected foo.pyi:foo.shared -> foo.py:foo.shared edge; got: {stub_runtime_pairs}"
    )
    # The stub-only decl has no matching runtime, so no stub->runtime
    # edge for ``stub_only`` — only the self-module edge.
    stub_only_outgoing = [(s, t) for s, t in raw_edges if s.fqname == "foo.stub_only"]
    assert {(s.fqname, t.fqname) for s, t in stub_only_outgoing} == {("foo.stub_only", "foo")}, (
        f"unexpected outgoing edges from foo.stub_only: {stub_only_outgoing}"
    )


def test_stub_only_decl_flagged_entrypoint_rust(build_decl_graph):
    """Rule 3: a ``.pyi`` decl with no matching ``.py`` decl is
    ``ENTRYPOINT``-flagged so it stays alive even when no consumer
    references it. Covers native-extension stubs (``_native.pyi`` next
    to ``_native.so``) and protobuf ``_pb2.pyi`` (peer ``.py`` is
    opaque-generated, has no static decls). Decls that DO have a
    matching runtime stay un-flagged — their liveness flows through
    the stub→runtime edge instead."""
    from dead_cst.graph import NodeFlags

    graph = build_decl_graph(PEER_STUB_FILES)
    by_fqname = {}
    for n in graph.raw.nodes():
        if str(n.path).endswith("foo.pyi") and n.type == "function":
            by_fqname[n.fqname] = NodeFlags(int(n.flags))
    assert by_fqname.get("foo.stub_only", NodeFlags.NONE) & NodeFlags.ENTRYPOINT, (
        f"stub-only decl missing ENTRYPOINT; flags = {by_fqname.get('foo.stub_only')!r}"
    )
    assert not (by_fqname.get("foo.shared", NodeFlags.NONE) & NodeFlags.ENTRYPOINT), (
        f"stub decl with matching runtime unexpectedly ENTRYPOINT-flagged; "
        f"flags = {by_fqname.get('foo.shared')!r}"
    )


def test_notebook_decls_carry_notebook_and_entrypoint_flags(write_notebook, build_decl_graph):
    """Every node minted from an ``.ipynb`` carries both
    ``NodeFlags.NOTEBOOK`` and ``NodeFlags.ENTRYPOINT``:

    * ``ENTRYPOINT`` keeps the cells alive (notebooks are executed
      top-to-bottom, not imported).
    * ``NOTEBOOK`` tells the codemod to skip these nodes — it can't
      rewrite cell JSON envelopes safely.

    The libcst pipeline ORs the flags via ``default_flags`` in
    ``_refresh._process_one_file``; the rust path mirrors that via
    ``file_default_flags`` consulted by every per-file node mint.
    """
    from dead_cst.graph import NodeFlags

    write_notebook("analysis.ipynb", ["def helper():\n    return 42", "helper()"])
    graph = build_decl_graph({})

    required = NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT
    notebook_nodes = [n for n in graph.raw.nodes() if str(n.path).endswith(".ipynb")]
    assert notebook_nodes, "expected at least one decl minted from the .ipynb file"
    for n in notebook_nodes:
        missing = required & ~NodeFlags(int(n.flags))
        assert not missing, f"{n.fqname!r} missing flags: {missing!r}"
