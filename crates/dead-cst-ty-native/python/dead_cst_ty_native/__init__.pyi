"""Type stubs for the ty-backed native graph builder.

Surfaces the pyo3 classes the rust crate exposes so callers in
``dead_cst/`` and ``tests/prototype/`` get static information without
having to introspect the binary module. Kept hand-written — pyo3
doesn't generate stubs and the public API is small.

The docstrings on each class / method are the contract: each one
mirrors the rust-side rustdoc, and a behavior change in the crate
should land in both places at once.
"""

from typing import Any, Iterable, Literal, Protocol

# The set of stable kind strings ``NativeNode.kind`` can carry. Use a
# ``Literal`` rather than ``str`` so the type checker catches typos at
# plugin author time.
NodeKind = Literal[
    "function",
    "class",
    "variable",
    "import",
    "type_alias",
    "module",
    "synthetic",
]

class NodeFlags:
    """Bit values stamped into ``NativeNode.flags``.

    Mirrors ``dead_cst.graph.NodeFlags`` exactly — same values, same
    semantics — so plugin code can interoperate with libcst-emitted
    nodes. Exposed as plain ``int`` class attributes; combine with
    bitwise OR (``NodeFlags.ENTRYPOINT | NodeFlags.NOQA``) the same way
    you would the libcst ``IntFlag``. The ``.name`` / ``.value``
    introspection surface of ``enum.IntFlag`` is intentionally not
    replicated — at the rust boundary the bits are plain ``u32``.
    """

    NONE: int
    SHADOWED: int
    """Decl rebound by a later assignment in the same file. Kept in
    the graph (with its parent-module edge) but excluded from the
    cross-module lookup so consumers of an exported name route to the
    live binding."""

    ENTRYPOINT: int
    """Reachability seed. BFS for "what's live" starts from every node
    carrying this bit."""

    OVERLOAD: int
    """``typing.overload`` stub (or any same-name decl anchored to a
    matching impl). Excluded from the lookup trie like ``SHADOWED``;
    kept alive by an explicit ``impl -> overload`` edge."""

    TESTCASE: int
    """Tags an entrypoint as test-only (pytest / unittest fixtures and
    test methods). Layered on top of ``ENTRYPOINT`` so the
    ``kept_alive_by_flags_only(TESTCASE)`` blast-radius query can ask
    "what's only alive because of tests"."""

    NOQA: int
    """Tags an entrypoint as preserved by an explicit user noqa
    directive (bare ``# noqa``, ``# noqa: F401``, multi-rule
    ``# noqa: E501, F401``, or the file-level ``# ruff: noqa`` /
    ``# flake8: noqa``)."""

    NOTEBOOK: int
    """Every node sourced from a Jupyter ``.ipynb`` file. Combined with
    ``ENTRYPOINT`` because cells run top-to-bottom rather than being
    imported, and the codemod skips notebook nodes (it can't rewrite
    the cell JSON envelope)."""

    EXPORTED: int
    """Every node sourced from a file under the package's ``exported``
    glob. Used by the cross-package merge to filter to entries the
    owning package opts into exposing."""

    STAR_REEXPORT: int
    """Import decl synthesized from ``from X import *`` — one per name
    the star statement brought in. Set so the cross-module trie can
    distinguish "real" import aliases from per-name star fan-out."""

class EdgeFlags:
    """Bit values stamped into the third slot of each
    ``NativeGraph.edges`` tuple.

    Mirrors ``dead_cst.graph.EdgeFlags``.
    """

    NONE: int
    DEAD_BRANCH: int
    """Reference originated inside a statically-dead region (the body
    of ``if False:``, the else of ``if True:``, after an unconditional
    ``return`` / ``raise`` / ``break`` / ``continue``, …). Metadata
    only — the edge still participates in default reachability; pass
    ``skip_flags=EdgeFlags.DEAD_BRANCH`` to traversal queries to
    compute the strict-reachability set that excludes dead branches."""

    DYNAMIC_IMPORT: int
    """Edge emitted from a runtime-import call (``__import__('X')`` /
    ``importlib.import_module('X')``). Lets plugins read which edges
    the visitor produced from dynamic-import shapes and choose to fan
    out / specialize."""

class Import:
    """Raw record of one cross-file import reference, attached to a
    ``kind="import"`` node. Mirrors ``dead_cst.graph.Import``.

    ``module`` is the import's *absolute* dotted target — relative
    dots are resolved by ty before this field is populated. ``decl``
    is the from-style imported name (``None`` for plain ``import`` and
    for the per-name nodes minted from ``from X import *``). ``star``
    flags the implicit-from-star case.
    """

    module: str
    decl: str | None
    star: bool

    def __init__(
        self,
        module: str,
        decl: str | None = ...,
        star: bool = ...,
    ) -> None: ...

class NativeNode:
    """A single node in a ``NativeGraph``.

    ``fqname`` is the dotted absolute name (``pkg.mod.MyClass``);
    ``kind`` is the static ``NodeKind`` literal. Position fields
    (``start_line`` / ``start_column`` / ``end_line`` /
    ``end_column``) point at the decl's target range — the identifier
    being bound — not the full body. ``flags`` is a bitmask of
    ``NodeFlags`` values. ``imports`` is populated only for
    ``kind="import"`` nodes (one per alias, plus one per name brought
    in by ``from X import *``); all other kinds carry ``None``.
    """

    fqname: str
    kind: NodeKind
    path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    flags: int
    imports: Import | None

# ----- Graph operations (yielded from plugin.run) ------------------------

class AddEdge:
    """Add an edge between two already-interned nodes.

    ``flags`` carries ``DEAD_BRANCH`` / future edge classifications.
    Plugins yield this from ``run(ctx)`` instead of mutating the graph
    directly so the apply pass is a single atomic step on the rust
    side.
    """

    src: NativeNode
    dst: NativeNode
    flags: int

    def __init__(self, src: NativeNode, dst: NativeNode, *, flags: int = 0) -> None: ...

class AddEntrypoint:
    """Mark ``decl`` as an entrypoint.

    ``marker`` is a self-documenting label (``"<celery-worker>"``,
    ``"<external-execution>:alembic"``, …) shown in ``why-alive`` to
    explain *why* the decl is alive without minting a synthetic graph
    node for the reason.

    Sugar for the single-target case; for multi-target
    (``marker -> [t1, t2, t3]``) or intermediate
    (``source -> marker -> targets``) markers use ``AddNode`` with
    ``edges_to`` / ``edges_from``.
    """

    decl: NativeNode
    marker: str

    def __init__(self, decl: NativeNode, *, marker: str) -> None: ...

class AddNode:
    """Mint a synthetic intermediate node, optionally wiring it with
    edges in the same op.

    Every element of ``edges_from`` becomes a ``source -> this`` edge;
    every element of ``edges_to`` becomes a ``this -> target`` edge —
    so a plugin doesn't need a separate handle to reference the
    freshly-minted node from subsequent ops. Set
    ``flags = NodeFlags.ENTRYPOINT`` to make the node a reachability
    seed; for the common single-target entrypoint pattern, prefer
    ``AddEntrypoint(decl, marker=...)``.
    """

    fqname: str
    kind: NodeKind
    path: str
    flags: int
    edges_from: list[NativeNode]
    edges_to: list[NativeNode]

    def __init__(
        self,
        fqname: str,
        *,
        path: str,
        kind: NodeKind = "synthetic",
        flags: int = 0,
        edges_from: Iterable[NativeNode] = ...,
        edges_to: Iterable[NativeNode] = ...,
    ) -> None: ...

GraphOp = AddEdge | AddEntrypoint | AddNode

class NativeGraph:
    """The project-wide graph snapshot returned by ``Project.build()``
    and ``ProjectContext.materialize()``.

    ``nodes`` is the interned node list (positional — edges index into
    it). ``edges`` is ``(src_idx, dst_idx, flags)`` triples.
    """

    nodes: list[NativeNode]
    edges: list[tuple[int, int, int]]

class Project:
    """Plugin-free project graph builder.

    Use ``ProjectContext`` if you need to register plugins; ``Project``
    is the bare-bones path for callers that just want the visitor's
    graph without plugin synthesis.
    """

    def __init__(
        self,
        root: str,
        *,
        src_roots: Iterable[str] | None = ...,
        extra_paths: Iterable[str] | None = ...,
        python_env: str | None = ...,
        python_version: str | None = ...,
        typeshed: str | None = ...,
    ) -> None: ...
    def build(self) -> NativeGraph:
        """Build the project-wide symbol graph in one pass."""
        ...

class _ProjectPluginLike(Protocol):
    def run(self, ctx: "ProjectContext") -> Iterable[GraphOp] | None: ...

class ProjectContext:
    """Plugin-aware project graph builder.

    Python instantiates a ``ProjectContext``, registers Python plugins
    via :meth:`add_plugin`, then calls :meth:`materialize`. Each
    plugin's ``run(ctx)`` is invoked with this same instance; the
    plugin yields ``GraphOp`` values that are applied to the in-progress
    graph, and may call any of the ``find_*`` / ``decls_*`` /
    ``descendants`` / ``ancestors`` / ``reachable`` queries below.

    Queries answer against the live in-progress graph (so an op
    yielded earlier in the same plugin is visible to later queries).
    They go through ty's semantic index when possible — subclass
    closure via ``type_hierarchy_subtypes``, method-defines through
    each class's ``DefinitionKind::Class``, etc. — and fall back to
    ruff AST walks only for the handful of shapes ty hasn't surfaced
    yet (see the per-method ``Notes`` below).
    """

    project_root: str
    """Absolute project root passed at construction. Plugins use it to
    compute paths relative to the project."""

    def __init__(
        self,
        root: str,
        *,
        src_roots: Iterable[str] | None = ...,
        extra_paths: Iterable[str] | None = ...,
        python_env: str | None = ...,
        python_version: str | None = ...,
        typeshed: str | None = ...,
    ) -> None: ...
    def add_plugin(self, plugin: _ProjectPluginLike | Any) -> None:
        """Register a plugin. Order of registration is order of
        invocation during :meth:`materialize`."""
        ...

    def materialize(self) -> NativeGraph:
        """Build the project-wide graph, run each registered plugin's
        ``run(ctx)``, then snapshot the final state.

        Borrows are released between phases so plugin ``run`` methods
        can re-enter queries through the same ``ctx`` without aliasing
        violations.
        """
        ...

    # ----- Decorator / construction queries ------------------------------

    def find_decorated(self, decorator_fqn: str) -> list[NativeNode]:
        """Decls decorated by ``@<decorator_fqn>`` or
        ``@<decorator_fqn>(...)``.

        Resolves through the file's local imports — aliased / dotted /
        module-prefixed forms all match. ``decorator_fqn`` is the
        upstream callable's absolute fqn
        (``celery.shared_task``, ``pytest.fixture``). For
        instance-method decorators (``@app.route(...)`` where ``app``
        is a ``flask.Flask``) use :meth:`find_decorations_on`.
        """
        ...

    def find_constructions(
        self, class_fqn: str, *, include_subclasses: bool = False
    ) -> list[NativeNode]:
        """Module-level variables assigned an instance of ``class_fqn``.

        e.g. ``find_constructions("flask.Flask")`` → every
        ``app = Flask(...)`` variable node.
        ``include_subclasses=True`` also matches direct constructions
        of any class that subclasses ``class_fqn`` (works for both
        project subclasses and external ones via ty's type hierarchy).
        """
        ...

    def find_decorations_on(
        self, instance: NativeNode, method_names: list[str]
    ) -> list[NativeNode]:
        """Decls decorated by ``@<instance>.<method>(...)`` for
        ``method`` in ``method_names``, where ``<instance>`` resolves
        to the given decl in the same file.

        Cross-file owners (where ``app = imported_factory()`` and
        ``@app.route`` is in a different file) aren't matched — same
        limitation the rust dispatch-app path has today.
        """
        ...

    def find_subclasses(self, base_fqn: str, *, transitive: bool = True) -> list[NativeNode]:
        """Subclasses of the class addressed by ``base_fqn``.

        Works for both project classes (where the fqn resolves to a
        graph node) and external classes (``unittest.TestCase``,
        ``pydantic.BaseModel``) via ty's module resolver +
        ``type_hierarchy_subtypes``. ``transitive=True`` (default)
        walks the full subclass closure; ``transitive=False`` returns
        only direct subclasses.
        """
        ...

    def find_subclasses_of(self, class_node: NativeNode) -> list[NativeNode]:
        """Transitive subclasses of a class already in the graph.

        Like :meth:`find_subclasses` but takes a ``NativeNode`` rather
        than a fqn — useful when you already have the seed in hand and
        don't want to round-trip through a string. Direct subtypes
        come from ty's ``type_hierarchy_subtypes``; results that don't
        land in the project (stdlib / external classes) are dropped.
        """
        ...

    # ----- FQN resolution ------------------------------------------------

    def resolve(self, fqname: str) -> NativeNode | None:
        """Resolve a dotted FQN to either a declaration or a module
        node.

        Tries an exact decl match first, then an exact module match,
        then walks back through dotted segments looking for an
        enclosing decl (``pkg.lib.Cls.method`` resolves to
        ``pkg.lib.Cls`` because methods don't get their own graph
        nodes). Returns ``None`` when the fqname can't be found
        anywhere — never raises.
        """
        ...

    def find_declarations(self, fqname: str) -> list[NativeNode]:
        """Every declaration matching ``fqname``, walking back through
        dotted segments to find the enclosing top-level decl when the
        exact name doesn't match.

        ``pkg.lib.Cls.method`` returns ``pkg.lib.Cls`` because methods
        aren't represented as their own graph nodes — same rule the
        libcst :func:`find_declarations` follows. Modules are never
        returned; use :meth:`find_module` for that.
        """
        ...

    def find_module(self, fqname: str) -> NativeNode | None:
        """Return the module node for the given dotted fqname, if one
        exists in the project graph."""
        ...

    def module_for(self, path: str) -> NativeNode | None:
        """Return the module node owning ``path``, if any.

        O(1) — backed by the same ``module_nodes_by_file`` index
        :meth:`find_main_blocks` uses, so plugins don't have to scan
        :meth:`nodes` per call.
        """
        ...

    def module_surface(self, module_fqn: str) -> list[NativeNode]:
        """Module node + every transitive decl whose fqname lives
        under ``module_fqn``.

        Models ``importlib.import_module(module_fqn)``: the module's
        whole top-level surface plus everything its submodules expose.
        Empty list when ``module_fqn`` doesn't resolve to a project
        module.
        """
        ...

    def find_module_top_level_decls(self, module_fqn: str) -> list[NativeNode]:
        """``module_fqn``'s immediate top-level decls — every
        function / class / variable / import bound at its module scope.

        Models ``from module_fqn import *``: only the names that
        statement would bind into the importing scope. Unlike
        :meth:`module_surface`, submodules and their decls are
        excluded — a ``from p.functions import *`` doesn't pull in
        ``p.functions.sub.x``. Empty list when ``module_fqn`` doesn't
        resolve to a project module.
        """
        ...

    def find_module_dunder_all_exports(self, module_fqn: str) -> list[NativeNode] | None:
        """Decls listed in ``module_fqn``'s ``__all__``, or ``None``
        when the module doesn't declare ``__all__``.

        The visitor's ``emit_dunder_all_edges`` already wires
        ``__all__`` → each string-listed decl as a regular edge; this
        query walks those successor edges and filters out the default
        ``decl -> parent_module`` edge. The distinction between "no
        ``__all__``" (``None``) and "empty ``__all__``" (``[]``)
        matters: callers that want CPython's ``from X import *``
        semantics should fall back to the non-underscore decl list
        only in the ``None`` case.
        """
        ...

    # ----- Path / name filters ------------------------------------------

    def decls_under(self, path_prefix: str) -> list[NativeNode]:
        """Every node whose ``path`` starts with the given prefix."""
        ...

    def decls_matching(self, substring: str) -> list[NativeNode]:
        """Every node whose ``path`` contains ``substring`` anywhere.

        Useful for path-pattern plugins (``alembic/versions/``,
        ``.ignore.py``).
        """
        ...

    def decls_matching_name(self, pattern: str) -> list[NativeNode]:
        """Every top-level decl whose simple name matches ``pattern``
        (a regex).

        Used by plugins that key on naming conventions —
        :class:`ModuleDundersPlugin` (``__xxx__`` names),
        :class:`PytestPlugin` (``test_*`` / ``Test*``), etc.
        """
        ...

    # ----- Traversal -----------------------------------------------------

    def descendants(self, root: NativeNode, *, skip_flags: int = 0) -> list[NativeNode]:
        """Forward closure: every node reachable from ``root`` by
        following graph edges.

        ``skip_flags`` filters out edges whose flag mask intersects —
        pass ``EdgeFlags.DEAD_BRANCH`` to compute strict reachability
        excluding dead branches.
        """
        ...

    def ancestors(self, decl: NativeNode, *, skip_flags: int = 0) -> list[NativeNode]:
        """Reverse closure: every node that can reach ``decl`` by
        following graph edges.

        Used for ``why-alive`` and blast-radius scoping. ``skip_flags``
        works the same as in :meth:`descendants`.
        """
        ...

    def reachable(self, *, skip_flags: int = 0) -> list[NativeNode]:
        """Forward closure from every entrypoint-flagged node. The set
        of dead decls is the complement against :meth:`nodes`."""
        ...

    # ----- Pure scans over the in-progress graph ------------------------

    def find_module_dunders(self) -> list[NativeNode]:
        """Every top-level variable node whose name matches ``__xxx__``.

        Pure scan over already-interned nodes — no ty re-query needed
        — because the visitor's decl pass already minted one node per
        global-scope variable binding.
        """
        ...

    def find_imports_of(self, module_name: str) -> list[NativeNode]:
        """Every import-kind node whose upstream ``module`` matches.

        Covers both ``import <module_name>`` and
        ``from <module_name> import ...`` styles — both bind
        import-kind nodes whose ``Import.module`` is the absolute
        dotted name. Star re-exports synthesized from
        ``from <module_name> import *`` are also included.
        """
        ...

    def find_main_blocks(self) -> list[tuple[NativeNode, list[NativeNode]]]:
        """``(module_node, [decls inside the block])`` for every file
        with a top-level ``if __name__ == "__main__":`` block.

        The decls list contains the file's class / function / variable
        / import nodes whose source position falls inside the block's
        range — same shape ``MainBlockPlugin``'s libcst path computes
        from the visitor's payload.
        """
        ...

    def find_classes_defining_method(self, method_name: str) -> list[NativeNode]:
        """Every class that defines a method with the given name.

        Walks each class's ``DefinitionKind::Class`` body for an
        ``Stmt::FunctionDef`` whose name matches. ty's
        ``parsed_module`` is Salsa-cached, so this is just a body scan
        per class.
        """
        ...

    # ----- Decorator / construction shapes (syntactic walks) ------------
    #
    # These three queries (``find_decorated_decls``,
    # ``find_instance_constructions``, ``find_calls_to_imported``)
    # match upstream callables / classes by their *bare name* combined
    # with a per-file import check — they deliberately do **not** route
    # through ty's module resolver.
    #
    # The reason is contract-level: these queries underpin plugins for
    # third-party frameworks (``celery``, ``flask``, ``fastapi``,
    # ``click``, ``typer``, ``cyclopts``, ``pytest``, ``unittest``,
    # ``discord.py``, …). ``dead-cst`` users routinely run the analyzer
    # against codebases where those frameworks are imported but not
    # *installed* — the consumer's venv may be a slim runtime image,
    # the analyzer may be invoked from CI without the project's deps,
    # or the resolver may not be configured to pick up the right venv.
    # In every case ty's module resolver returns ``None`` for
    # ``celery.shared_task`` etc., which would silently make every
    # framework plugin no-op.
    #
    # A syntactic match keyed on ``from <module> import <name>`` (and
    # the aliased / dotted variants — see ``collect_module_imports_local``)
    # works regardless of whether ty can resolve the target, mirrors
    # the libcst pipeline's existing behavior bit-for-bit, and matches
    # what users actually mean when they write
    # ``@app.route("/")`` — they're keying on the framework's *name*,
    # not on whether their venv can resolve the type.
    #
    # When the per-file ``decorator_fqn`` chain is consistent across
    # the project (the common case — projects don't usually re-export
    # ``celery.shared_task`` under a different name), the syntactic
    # path produces identical results to the ty path. The re-export
    # shape (``from myapp.tasks import shared_task`` where
    # ``shared_task = celery.shared_task``) is the only place the ty
    # path would catch something the syntactic path misses; today no
    # plugin needs that.

    def find_decorated_decls(
        self, decorator_module: str, decorator_names: list[str]
    ) -> list[NativeNode]:
        """Every top-level function decorated with
        ``@<decorator_module>.<name>`` or ``@<name>`` for any ``name``
        in ``decorator_names``.

        Both ``@<name>`` (bare) and ``@<name>(...)`` (called) forms
        match — the function call is unwrapped before the pattern is
        checked. Identity for the attribute prefix is literal
        (``@pytest.fixture`` matches; ``@p.fixture`` with
        ``import pytest as p`` does not, but the import-aliased
        ``@fixture`` does because the file's ``from pytest import
        fixture`` brings ``fixture`` into local scope).

        Syntactic — see the section comment above for why this query
        does not route through ty's module resolver.
        """
        ...

    def find_instance_constructions(
        self, module: str, ctor_names: list[str]
    ) -> list[tuple[NativeNode, str]]:
        """Top-level ``<var> = <Ctor>(...)`` constructions where
        ``Ctor`` is imported from ``module`` and is one of
        ``ctor_names``.

        Recognized shapes (mirroring the libcst plugin helpers):

        * ``from <module> import <Ctor>; X = Ctor(...)``
        * ``from <module> import <Ctor> as A; X = A(...)``
        * ``import <module>; X = <module>.Ctor(...)``
        * ``import <module> as m; X = m.Ctor(...)``
        * ``X: T = Ctor(...)`` annotated form

        Returns ``[(var_node, ctor_name)]``; ``ctor_name`` is the
        upstream constructor's bare name (``"Flask"`` even when
        imported as ``F``).

        Syntactic — see the section comment above for why this query
        does not route through ty's module resolver.
        """
        ...

    def find_calls_to_imported(
        self, module: str, name: str, arg_index: int
    ) -> list[tuple[NativeNode, str]]:
        """Calls to a callable imported from ``module`` with the name
        ``name``. Returns ``(owning_decl, string_literal_arg)`` pairs
        where the call resolves through the file's local imports and
        the positional arg at ``arg_index`` is a string literal.

        The owning decl is the top-level ``FunctionDef`` / ``ClassDef``
        the call lives under (including its decorator subtree); calls
        at module scope attribute to the module node.

        Syntactic — see the section comment above for why this query
        does not route through ty's module resolver.
        """
        ...

    # ----- Decorator / call patterns keyed on attribute names -----------

    def find_handler_decorators(self, decorator_attrs: list[str]) -> list[tuple[str, NativeNode]]:
        """Top-level functions decorated with ``@<owner>.<attr>(...)``
        where ``attr`` is in ``decorator_attrs``.

        Returns ``[(owner_name, function_node)]``. ``owner_name`` is
        the raw textual prefix of the decorator (``"app"`` for
        ``@app.route``), not resolved to a graph node — the caller
        decides which owners correspond to real framework instances.
        Multiple decorators on the same function emit multiple
        entries.
        """
        ...

    def find_handler_decorators_via(
        self, via_attr: str, decorator_attrs: list[str]
    ) -> list[tuple[str, NativeNode]]:
        """Like :meth:`find_handler_decorators` but matches the
        two-level form ``@<owner>.<via_attr>.<attr>(...)``
        (e.g. ``@bot.tree.command()`` for discord.py's slash commands).

        Returns the same ``[(owner_name, function_node)]`` shape,
        where ``owner_name`` is the leftmost ``Name`` in the
        decorator chain.
        """
        ...

    def find_calls_on_attr(self, attr: str, arg_index: int) -> list[tuple[NativeNode, str]]:
        """Calls of the form ``<expr>.<attr>(...)`` regardless of
        receiver, where the positional arg at ``arg_index`` is either
        a string literal **or** a list/tuple of string literals.

        Returns ``[(owning_decl, captured_string)]`` — one row per
        captured string, so ``load_extensions(["a", "b"])`` yields two
        rows.

        Unlike :meth:`find_calls_on_var`, this matches any receiver
        shape: ``bot.load_extension(...)``,
        ``self.bot.load_extension(...)``,
        ``get_bot().load_extension(...)``, etc. Use this when the call
        pattern is keyed on the method name and the receiver is the
        plugin's concern (typically gated by a per-file import check).
        """
        ...

    def find_calls_on_var(
        self,
        owner: str,
        attr: str,
        arg_index: int,
        *,
        required_positional: int | None = ...,
    ) -> list[tuple[NativeNode, str]]:
        """``<owner>.<attr>(...)`` calls where ``owner`` is the
        textual prefix (no import resolution — covers pytest fixture
        conventions like ``mocker.patch`` / ``monkeypatch.setattr``).

        ``required_positional`` disambiguates fqname-form calls from
        object-form calls when the same method name is overloaded:
        ``monkeypatch.setattr("X.Y", v)`` has 2 positional args
        (fqname + value) while
        ``monkeypatch.setattr(obj, "name", v)`` has 3. Pass ``None``
        to accept any positional-arg count.

        Returns ``(owning_decl, string_literal_arg)`` pairs.
        """
        ...

    def find_factory_decls(
        self, module: str, ctor_names: list[str]
    ) -> list[tuple[NativeNode, list[str]]]:
        """Top-level functions / classes whose body constructs one of
        ``ctor_names`` imported from ``module``.

        Recursively walks each candidate's body looking for
        ``<Ctor>(...)`` or ``<module>.<Ctor>(...)`` call expressions.
        Returns ``[(decl_node, [kind, ...])]`` where ``kind`` is the
        matched constructor's bare name; multiple kinds appear when a
        single factory constructs more than one (e.g. a function that
        returns a ``Flask`` after mounting several ``Blueprint``\\ s).
        """
        ...

    # ----- Comment-driven patterns --------------------------------------

    def find_comment_patterns(self, pattern: str) -> list[tuple[NativeNode, str]]:
        """``(decl_node, comment_text)`` for every comment in the
        project matching ``pattern`` (a regex), paired with the next
        declaration that follows it in the same file.

        Comments are scanned from the parser's ``Tokens`` stream (no
        re-lexing); regex matching is full-text against the comment
        content (leading ``#`` included).
        """
        ...

    # ----- Read-only graph accessors ------------------------------------

    def nodes(self) -> list[NativeNode]:
        """Live nodes in the in-progress graph. Cheap, no copy."""
        ...

    def edges(self) -> list[tuple[int, int, int]]:
        """Live edges as ``(src_idx, dst_idx, flags)`` triples."""
        ...
