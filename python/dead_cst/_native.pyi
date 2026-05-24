"""Type stubs for the ty-backed native graph builder.

Surfaces the pyo3 classes the rust crate exposes so callers in
``dead_cst/`` and ``tests/prototype/`` get static information without
having to introspect the binary module. Kept hand-written — pyo3
doesn't generate stubs and the public API is small.

The docstrings on each class / method are the contract: each one
mirrors the rust-side rustdoc, and a behavior change in the crate
should land in both places at once.
"""

from typing import Any, Iterable, Iterator, Literal, Protocol

# The set of stable kind strings ``SymbolNode.kind`` can carry. Use a
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
    """Bit values stamped into ``SymbolNode.flags``.

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
    """Explicit entrypoint — plugin-emitted seeds, the CLI's ``-e`` flag,
    ``[project.scripts]`` targets, factory-app synthetics, etc. One of
    the keepalive bits ORed into ``KEEPALIVE_DEFAULT`` on the Python
    side, so reachability seeds from ``ENTRYPOINT``-flagged nodes by
    default."""

    OVERLOAD: int
    """``typing.overload`` stub (or any same-name decl anchored to a
    matching impl). Excluded from the lookup trie like ``SHADOWED``;
    kept alive by an explicit ``impl -> overload`` edge."""

    TESTCASE: int
    """Pytest / unittest test discoveries. One of the keepalive bits in
    ``KEEPALIVE_DEFAULT``, so tests are alive by default. The
    ``kept_alive_by_flags_only(TESTCASE)`` blast-radius query isolates
    "what's only alive because of tests" by computing the diff against
    ``reachable(seed_flags=KEEPALIVE_DEFAULT & ~TESTCASE)``."""

    NOQA: int
    """Import alias preserved by a user noqa directive (bare
    ``# noqa``, ``# noqa: F401``, multi-rule ``# noqa: E501, F401``, or
    the file-level ``# ruff: noqa`` / ``# flake8: noqa``). One of the
    keepalive bits in ``KEEPALIVE_DEFAULT``."""

    NOTEBOOK: int
    """Every node sourced from a Jupyter ``.ipynb`` file. Cells run
    top-to-bottom rather than being imported, so the bit alone keeps
    the node alive (no ``ENTRYPOINT`` overlay needed — ``NOTEBOOK`` is
    in ``KEEPALIVE_DEFAULT``). The codemod also reads the bit to skip
    notebook nodes (it can't rewrite the cell JSON envelope)."""

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

class SymbolNode:
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

    def __init__(
        self,
        fqname: str,
        kind: NodeKind,
        path: str,
        *,
        start_line: int = ...,
        start_column: int = ...,
        end_line: int = ...,
        end_column: int = ...,
        flags: int = ...,
        imports: Import | None = ...,
    ) -> None: ...

# ----- Graph operations (yielded from plugin.run) ------------------------

class AddEdge:
    """Add an edge between two already-interned nodes.

    ``flags`` carries ``DEAD_BRANCH`` / future edge classifications.
    Plugins yield this from ``run(ctx)`` instead of mutating the graph
    directly so the apply pass is a single atomic step on the rust
    side.
    """

    src: SymbolNode
    dst: SymbolNode
    flags: int

    def __init__(self, src: SymbolNode, dst: SymbolNode, *, flags: int = 0) -> None: ...

class AddEntrypoint:
    """Mark ``decl`` as an entrypoint.

    ``marker`` is a self-documenting label (``"<celery-worker>"``,
    ``"<external-execution>:alembic"``, …) used in the
    :meth:`Analysis.ancestors` chain to explain *why* the decl is
    alive without minting a synthetic graph node for the reason.

    Sugar for the single-target case; for multi-target
    (``marker -> [t1, t2, t3]``) or intermediate
    (``source -> marker -> targets``) markers use ``AddNode`` with
    ``edges_to`` / ``edges_from``.
    """

    decl: SymbolNode
    marker: str

    def __init__(self, decl: SymbolNode, *, marker: str) -> None: ...

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
    edges_from: list[SymbolNode]
    edges_to: list[SymbolNode]

    def __init__(
        self,
        fqname: str,
        *,
        path: str,
        kind: NodeKind = "synthetic",
        flags: int = 0,
        edges_from: Iterable[SymbolNode] = ...,
        edges_to: Iterable[SymbolNode] = ...,
    ) -> None: ...

GraphOp = AddEdge | AddEntrypoint | AddNode

class NativeGraph:
    """The project-wide graph snapshot returned by ``Project.build()``
    and ``ProjectContext.materialize()``.

    ``nodes`` is the interned node list (positional — edges index into
    it). ``edges`` is ``(src_idx, dst_idx, flags)`` triples.
    """

    nodes: list[SymbolNode]
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
        show_progress: bool = ...,
    ) -> None: ...
    @property
    def stack_size(self) -> int | None:
        """Rayon worker stack size override (bytes) for the populate
        phase, or ``None`` if no override is set (in which case the
        populate phase runs on rayon's global pool with rayon's own
        default stack — 2 MiB unless ``RAYON_STACK_SIZE`` /
        ``RUST_MIN_STACK`` are set process-wide)."""

    def set_stack_size(self, bytes_: int) -> None:
        """Override the rayon worker stack size (bytes) used by the
        populate phase. Call BEFORE :meth:`materialize`; calls after
        the graph is materialized have no effect on the already-
        built graph. Raises :class:`ValueError` if ``bytes_`` is
        not positive.

        Set this on projects with deeply-nested generated code
        (protobuf modules, ML-generated ASTs, big literal dicts)
        that overflow rayon's default 2 MiB stack."""

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

    def query(self) -> QueryBuilder:
        """Open a chainable query builder against this context.

        Sugar for ``query(ctx)`` — same return value, picked up by
        ``__init__.py`` from the rust module surface."""
        ...

    # Decorator / construction / call queries route through
    # :meth:`query` and the builder API (``DecoratorQuery`` /
    # ``ConstructionQuery`` / ``CallQuery`` below). The legacy
    # ``find_decorated`` / ``find_constructions`` / ``find_decorations_on``
    # / ``find_decorated_decls`` / ``find_instance_constructions`` /
    # ``find_handler_decorators`` / ``find_handler_decorators_via`` /
    # ``find_calls_on_attr`` / ``find_calls_to_imported`` /
    # ``find_calls_on_var`` methods are no longer part of the public
    # type-stub contract; use ``query(ctx)`` instead.

    def find_subclasses(self, base_fqn: str, *, transitive: bool = True) -> list[SymbolNode]:
        """Subclasses of the class addressed by ``base_fqn``.

        Works for both project classes (where the fqn resolves to a
        graph node) and external classes (``unittest.TestCase``,
        ``pydantic.BaseModel``) via ty's module resolver +
        ``type_hierarchy_subtypes``. ``transitive=True`` (default)
        walks the full subclass closure; ``transitive=False`` returns
        only direct subclasses.
        """
        ...

    def find_subclasses_of(self, class_node: SymbolNode) -> list[SymbolNode]:
        """Transitive subclasses of a class already in the graph.

        Like :meth:`find_subclasses` but takes a ``SymbolNode`` rather
        than a fqn — useful when you already have the seed in hand and
        don't want to round-trip through a string. Direct subtypes
        come from ty's ``type_hierarchy_subtypes``; results that don't
        land in the project (stdlib / external classes) are dropped.
        """
        ...

    # ----- FQN resolution ------------------------------------------------

    def resolve(self, fqname: str) -> SymbolNode | None:
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

    def find_declarations(self, fqname: str) -> list[SymbolNode]:
        """Every declaration matching ``fqname``, walking back through
        dotted segments to find the enclosing top-level decl when the
        exact name doesn't match.

        ``pkg.lib.Cls.method`` returns ``pkg.lib.Cls`` because methods
        aren't represented as their own graph nodes — same rule the
        libcst :func:`find_declarations` follows. Modules are never
        returned; use :meth:`find_module` for that.
        """
        ...

    def find_module(self, fqname: str) -> SymbolNode | None:
        """Return the module node for the given dotted fqname, if one
        exists in the project graph."""
        ...

    def module_for(self, path: str) -> SymbolNode | None:
        """Return the module node owning ``path``, if any.

        O(1) — backed by the same ``module_nodes_by_file`` index
        :meth:`find_main_blocks` uses, so plugins don't have to scan
        :meth:`nodes` per call.
        """
        ...

    def module_surface(self, module_fqn: str) -> list[SymbolNode]:
        """Module node + every transitive decl whose fqname lives
        under ``module_fqn``.

        Models ``importlib.import_module(module_fqn)``: the module's
        whole top-level surface plus everything its submodules expose.
        Empty list when ``module_fqn`` doesn't resolve to a project
        module.
        """
        ...

    def find_module_top_level_decls(self, module_fqn: str) -> list[SymbolNode]:
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

    def find_module_dunder_all_exports(self, module_fqn: str) -> list[SymbolNode] | None:
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

    def find_literal_list_entries(self, var_fqn: str) -> list[str] | None:
        """Read the literal-list value of a top-level variable
        assignment (``X = ["a", "b"]`` / ``X: tuple[str, ...] = (...)``)
        and return the entries as strings.

        Returns ``None`` when the variable isn't found, when its
        assignment value isn't a list / tuple of string literals, or
        when any element is a non-literal (``[*BASE, "c"]``,
        ``list(...)``, etc.). Targeted read used by
        :class:`dead_cst.plugins.decl_shapes.LiteralListPlugin` to
        stay independent of the visitor's ``__all__``-only string-list
        edge emission.
        """
        ...

    # ----- Path / name filters ------------------------------------------

    def decls_under(self, path_prefix: str) -> list[SymbolNode]:
        """Every node whose ``path`` starts with the given prefix."""
        ...

    def decls_matching(self, substring: str) -> list[SymbolNode]:
        """Every node whose ``path`` contains ``substring`` anywhere.

        Useful for path-pattern plugins (``alembic/versions/``,
        ``.ignore.py``).
        """
        ...

    def decls_matching_name(self, pattern: str) -> list[SymbolNode]:
        """Every top-level decl whose simple name matches ``pattern``
        (a regex).

        Used by plugins that key on naming conventions —
        :class:`ModuleDundersPlugin` (``__xxx__`` names),
        :class:`PytestPlugin` (``test_*`` / ``Test*``), etc.
        """
        ...

    def find_nodes_matching_specs(
        self,
        project_root: str,
        regexes: list[str],
        str_specs: list[str],
        abs_paths: list[str],
    ) -> list[SymbolNode]:
        """Match nodes against pre-classified path / fqname specs.

        Avoids the per-node FFI hop + ``pathlib.Path`` allocation that
        a ``for node in ctx.nodes(): ...`` loop pays. ``regexes`` are
        anchored at the start of input (``re.Pattern.match`` semantics).
        ``str_specs`` match exactly against the relative path OR the
        node's fqname. ``abs_paths`` match exactly against the
        absolute path.
        """
        ...

    # ----- Traversal -----------------------------------------------------

    def descendants(self, root: SymbolNode, *, skip_flags: int = 0) -> list[SymbolNode]:
        """Forward closure: every node reachable from ``root`` by
        following graph edges.

        ``skip_flags`` filters out edges whose flag mask intersects —
        pass ``EdgeFlags.DEAD_BRANCH`` to compute strict reachability
        excluding dead branches.
        """
        ...

    def ancestors(self, decl: SymbolNode, *, skip_flags: int = 0) -> list[SymbolNode]:
        """Reverse closure: every node that can reach ``decl`` by
        following graph edges.

        Used for predecessor-chain walks and blast-radius scoping.
        ``skip_flags`` works the same as in :meth:`descendants`.
        """
        ...

    def reachable(self, *, skip_flags: int = 0, seed_flags: int = ...) -> list[SymbolNode]:
        """Forward closure from every node carrying any bit in
        ``seed_flags`` (defaults to :data:`NodeFlags.ENTRYPOINT`). The
        set of dead decls is the complement against :meth:`nodes`."""
        ...

    # ----- Pure scans over the in-progress graph ------------------------

    def find_module_dunders(self) -> list[SymbolNode]:
        """Every top-level variable node whose name matches ``__xxx__``.

        Pure scan over already-interned nodes — no ty re-query needed
        — because the visitor's decl pass already minted one node per
        global-scope variable binding.
        """
        ...

    def find_imports_of(self, module_name: str) -> list[SymbolNode]:
        """Every import-kind node whose upstream ``module`` matches.

        Covers both ``import <module_name>`` and
        ``from <module_name> import ...`` styles — both bind
        import-kind nodes whose ``Import.module`` is the absolute
        dotted name. Star re-exports synthesized from
        ``from <module_name> import *`` are also included.
        """
        ...

    def find_main_blocks(self) -> list[tuple[SymbolNode, list[SymbolNode]]]:
        """``(module_node, [decls inside the block])`` for every file
        with a top-level ``if __name__ == "__main__":`` block.

        The decls list contains the file's class / function / variable
        / import nodes whose source position falls inside the block's
        range — same shape ``MainBlockPlugin``'s libcst path computes
        from the visitor's payload.
        """
        ...

    def find_classes_defining_method(self, method_name: str) -> list[SymbolNode]:
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

    # The decorator / construction / call queries that used to live
    # here (``find_decorated_decls``, ``find_instance_constructions``,
    # ``find_handler_decorators``, ``find_handler_decorators_via``,
    # ``find_calls_on_attr``, ``find_calls_to_imported``,
    # ``find_calls_on_var``) are no longer part of the public type-stub
    # contract. Use the builder API instead:
    # ``query(ctx).decorators().where_module(...).where_name(...)``,
    # ``query(ctx).constructions().where_module(...).where_name(...)``,
    # ``query(ctx).calls().where_module(...).where_name(...).string_arg_at(...)``,
    # etc. See :class:`DecoratorQuery` / :class:`ConstructionQuery` /
    # :class:`CallQuery` below for the full predicate vocabulary.
    #
    # The rust pyo3 methods still exist at runtime — they're the
    # underlying impl the builder dispatches into — but they aren't
    # exposed in this stub. A follow-up cleanup will move them out of
    # ``#[pymethods]`` so they're rust-internal too.

    def find_factory_decls(
        self, module: str, ctor_names: list[str]
    ) -> list[tuple[SymbolNode, list[str]]]:
        """Top-level functions / classes whose body constructs one of
        ``ctor_names`` imported from ``module``.

        Recursively walks each candidate's body looking for
        ``<Ctor>(...)`` or ``<module>.<Ctor>(...)`` call expressions.
        Returns ``[(decl_node, [kind, ...])]`` where ``kind`` is the
        matched constructor's bare name; multiple kinds appear when a
        single factory constructs more than one (e.g. a function that
        returns a ``Flask`` after mounting several ``Blueprint``\\ s).

        Not yet wrapped by the builder API — the only caller is
        :class:`DispatchAppPlugin` and it uses this shape directly.
        Likely candidate for a ``ConstructionQuery.inside_factory()``
        predicate in a follow-up.
        """
        ...

    # ----- Comment-driven patterns --------------------------------------

    def find_comment_patterns(self, pattern: str) -> list[tuple[SymbolNode, str]]:
        """``(decl_node, comment_text)`` for every comment in the
        project matching ``pattern`` (a regex), paired with the next
        declaration that follows it in the same file.

        Comments are scanned from the parser's ``Tokens`` stream (no
        re-lexing); regex matching is full-text against the comment
        content (leading ``#`` included).
        """
        ...

    # ----- Read-only graph accessors ------------------------------------

    def nodes(self) -> list[SymbolNode]:
        """Live nodes in the in-progress graph. Cheap, no copy."""
        ...

    def edges(self) -> list[tuple[int, int, int]]:
        """Live edges as ``(src_idx, dst_idx, flags)`` triples."""
        ...

# ---------- Builder query API ---------------------------------------------
#
# Phase 3 surface: rust pyclasses with the chainable predicate API.
# ``collect()`` walks ``project_files`` via the underlying ``find_*``
# pyo3 helpers (now internal — removed from the public stub above) and
# ``where_path(regex)`` is fused into each helper's file-iteration
# loop so unrelated files skip parsing entirely. A follow-up will
# also move the ``find_*`` methods out of ``#[pymethods]`` and inline
# their bodies into each query's ``collect()`` directly.

class DecoratorRef:
    """One decorator application on a top-level function or class.

    Field nullability follows the query shape that produced the ref:

    * ``where_module + where_name`` populates ``decorated`` only;
      ``decorator_name`` is ``None`` because the underlying walk
      doesn't surface which of the queried names matched.
    * ``where_owner_attr`` fills ``decorator_owner`` (the textual
      ``@<owner>.<attr>`` prefix).
    * ``where_owner_attr_via`` additionally fills ``decorator_via``
      with the middle attribute name.

    ``args`` and ``kwargs`` are populated from the decorator's
    ``Call`` form (``@dec(a, b, k=v)``). Bare-attribute decorators
    (``@app.route`` without ``()``) get empty containers. Each value
    is a Python literal (str / int / float / bool / None / list /
    tuple), a :class:`SymbolNode` when the expression statically
    resolves to a project decl, or ``None`` for any other non-literal
    expression.
    """

    decorated: SymbolNode
    decorator_name: str | None
    decorator_owner: str | None
    decorator_via: str | None
    args: list[Any]
    kwargs: dict[str, Any]

    @property
    def path(self) -> str: ...

class ConstructionRef:
    """One ``<var> = <Ctor>(...)`` construction at module scope.

    ``class_name`` is the upstream constructor's bare name
    (``"Flask"`` even when imported as ``F``).

    ``args`` and ``kwargs`` carry the construction's full positional /
    keyword argument shape. Each value is a Python literal (str / int
    / float / bool / None / list / tuple), a :class:`SymbolNode` when
    the expression statically resolves to a project decl, or ``None``
    for any other non-literal expression.
    """

    var: SymbolNode
    class_name: str
    args: list[Any]
    kwargs: dict[str, Any]

    @property
    def path(self) -> str: ...

class CallRef:
    """One matched call site.

    ``string_arg`` is the positional string literal at the index
    passed to :meth:`CallQuery.string_arg_at`.

    ``args`` and ``kwargs`` carry the call's full positional /
    keyword argument shape. Each value is a Python literal (str /
    int / float / bool / None / list / tuple), a :class:`SymbolNode`
    when the expression statically resolves to a project decl, or
    ``None`` for any other non-literal expression.
    """

    owner: SymbolNode
    string_arg: str
    args: list[Any]
    kwargs: dict[str, Any]

    @property
    def path(self) -> str: ...

def query(ctx: ProjectContext) -> QueryBuilder:
    """Open a chainable query builder against ``ctx``.

    Equivalent to :meth:`ProjectContext.query`; exists so plugins can
    write ``query(ctx).decorators()...`` without first dereferencing
    the method.
    """
    ...

class QueryBuilder:
    """Entry point for the chainable query API.

    Filtered streams (terminated by ``.collect()``):
    :meth:`decorators` / :meth:`constructions` / :meth:`calls` /
    :meth:`subclasses` / :meth:`imports` / :meth:`classes` /
    :meth:`factories` / :meth:`edges`.

    Point lookups (e.g. :meth:`ProjectContext.find_module`,
    :meth:`ProjectContext.find_declarations`,
    :meth:`ProjectContext.find_main_blocks`,
    :meth:`ProjectContext.find_comment_patterns`,
    :meth:`ProjectContext.find_module_dunders`,
    :meth:`ProjectContext.find_literal_list_entries`) live directly on
    :class:`ProjectContext`.
    """

    def decorators(self) -> DecoratorQuery: ...
    def constructions(self) -> ConstructionQuery: ...
    def calls(self) -> CallQuery: ...
    def subclasses(self) -> SubclassQuery: ...
    def imports(self) -> ImportQuery: ...
    def classes(self) -> ClassQuery: ...
    def factories(self) -> FactoryQuery: ...
    def edges(self) -> EdgeQuery: ...
    def decls(self) -> DeclQuery: ...

class DecoratorQuery:
    """Find decorated top-level functions / classes. Pick exactly one
    of the four decorator-shape predicates per chain — mixing raises
    ``ValueError`` at ``collect()`` time."""

    def where_module(self, module: str | list[str]) -> DecoratorQuery:
        """Filter decorators to those resolved through an import of
        ``module``. Pass a list to match any of several modules (OR
        semantics — useful for framework-family predicates like
        ``["flask", "quart"]``). A single string is the common case
        and stays supported.
        """
        ...

    def where_callee(self, fqn: str) -> DecoratorQuery: ...
    def where_name(self, names: str | list[str] | tuple[str, ...]) -> DecoratorQuery: ...
    def where_owner_attr(self, attrs: str | list[str] | tuple[str, ...]) -> DecoratorQuery: ...
    def where_owner_attr_via(
        self, via: str, attrs: str | list[str] | tuple[str, ...]
    ) -> DecoratorQuery: ...
    def in_decl(self, node: SymbolNode) -> DecoratorQuery: ...
    def where_path(self, regex: str) -> DecoratorQuery: ...
    def where_kwarg(self, name: str, value: Any) -> DecoratorQuery:
        """Filter to decorator calls whose ``name=value`` kwarg matches.

        Multiple ``.where_kwarg`` calls AND together. ``value`` must
        be a Python literal (``None`` / ``bool`` / ``int`` /
        ``float`` / ``str`` / ``list`` / ``tuple``). A missing kwarg
        on the call never matches. A non-literal kwarg expression
        never matches.
        """
        ...

    def collect(self) -> list[DecoratorRef]: ...
    def first(self) -> DecoratorRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[DecoratorRef]: ...

class ConstructionQuery:
    """Find module-scope ``<var> = <Ctor>(...)`` sites."""

    def where_module(self, module: str | list[str]) -> ConstructionQuery:
        """Filter constructions to those whose constructor is imported
        from ``module``. Pass a list to match any of several modules
        (OR semantics).
        """
        ...

    def where_name(self, names: str | list[str] | tuple[str, ...]) -> ConstructionQuery: ...
    def where_class(self, fqn: str, *, include_subclasses: bool = False) -> ConstructionQuery: ...
    def where_path(self, regex: str) -> ConstructionQuery: ...
    def collect(self) -> list[ConstructionRef]: ...
    def first(self) -> ConstructionRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[ConstructionRef]: ...

class CallQuery:
    """Find call sites with a captured positional string-literal arg.

    :meth:`string_arg_at` is required — it picks the positional
    index. Pick one of the three receiver shapes per chain.
    """

    def where_module(self, module: str | list[str]) -> CallQuery:
        """Filter calls to those whose callee is imported from
        ``module``. Pass a list to match any of several modules (OR
        semantics).
        """
        ...

    def where_name(self, name: str) -> CallQuery: ...
    def where_owner(self, owner: str) -> CallQuery: ...
    def where_attr(self, attr: str) -> CallQuery: ...
    def string_arg_at(self, index: int) -> CallQuery: ...
    def where_required_positional(self, n: int | None = ...) -> CallQuery: ...
    def where_path(self, regex: str) -> CallQuery: ...
    def where_kwarg(self, name: str, value: Any) -> CallQuery:
        """Filter to call sites whose ``name=value`` kwarg matches.

        Multiple ``.where_kwarg`` calls AND together. ``value`` must
        be a Python literal (``None`` / ``bool`` / ``int`` /
        ``float`` / ``str`` / ``list`` / ``tuple``). A missing kwarg
        on the call never matches. A non-literal kwarg expression
        never matches.
        """
        ...

    def collect(self) -> list[CallRef]: ...
    def first(self) -> CallRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[CallRef]: ...

class SubclassQuery:
    """Walk the subclass closure of a class.

    Pick exactly one of :meth:`of_fqn` / :meth:`of_node`. The
    default :meth:`transitive` is ``True``; flip to ``False`` for
    direct subclasses only. Mirrors the union of
    ``ProjectContext.find_subclasses`` and ``find_subclasses_of``.
    """

    def of_fqn(self, fqn: str) -> SubclassQuery: ...
    def of_node(self, node: SymbolNode) -> SubclassQuery: ...
    def transitive(self, value: bool) -> SubclassQuery: ...
    def collect(self) -> list[SymbolNode]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[SymbolNode]: ...

class ImportQuery:
    """Enumerate the ``kind="import"`` nodes that bind a name from a
    given module. Requires :meth:`of` (the upstream module name).

    Mirrors :meth:`ProjectContext.find_imports_of`.
    """

    def of(self, module: str) -> ImportQuery: ...
    def collect(self) -> list[SymbolNode]: ...
    def count(self) -> int: ...
    def exists(self) -> bool:
        """O(1) presence probe — does any project file import the
        configured module? Short-circuits without materialising a
        Python list. Preferred over ``.count() > 0`` / ``.collect()``
        for plugin guards that just need a boolean.
        """
        ...

    def __iter__(self) -> Iterator[SymbolNode]: ...

class ClassQuery:
    """Enumerate classes by structural property. Today the only filter
    is :meth:`defining_method` (matches classes whose body has a
    ``FunctionDef`` with that name).

    Mirrors :meth:`ProjectContext.find_classes_defining_method`.
    """

    def defining_method(self, name: str) -> ClassQuery: ...
    def collect(self) -> list[SymbolNode]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[SymbolNode]: ...

class FactoryRef:
    """One result row from :class:`FactoryQuery`.

    ``decl`` is the owning top-level function or class; ``kinds`` is
    the sorted set of constructor bare-names matched inside its body
    (multiple kinds appear when a single factory constructs more than
    one — e.g. a function that returns a ``Flask`` after mounting
    several ``Blueprint``\\ s).
    """

    decl: SymbolNode
    kinds: list[str]

    @property
    def path(self) -> str: ...

class FactoryQuery:
    """Walk function / class bodies for ``<Ctor>(...)`` calls where
    ``Ctor`` is imported from :meth:`of_module` and matches one of
    :meth:`where_name`. Both filters are required.

    Mirrors :meth:`ProjectContext.find_factory_decls`.
    """

    def of_module(self, module: str | list[str]) -> FactoryQuery:
        """Filter to factories whose constructed type is imported from
        ``module``. Pass a list to match any of several modules (OR
        semantics).
        """
        ...

    def where_name(self, names: str | list[str] | tuple[str, ...]) -> FactoryQuery: ...
    def collect(self) -> list[FactoryRef]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[FactoryRef]: ...

class EdgeRef:
    """One graph edge with both endpoint nodes resolved.

    Avoids the ``nodes[src_idx]`` / ``nodes[dst_idx]`` ping-pong that a
    Python-side ``for src_idx, dst_idx, flags in ctx.edges()`` loop pays.
    """

    src: SymbolNode
    dst: SymbolNode
    flags: int

class EdgeQuery:
    """Filtered enumeration over the in-progress graph's edges.

    Predicates AND together; any unset predicate doesn't filter. The
    entire filter runs rust-side and only the surviving rows are
    materialized into ``Py<SymbolNode>``.
    """

    def with_flags(self, mask: int) -> EdgeQuery:
        """Keep edges where ``flags & mask != 0``. Pass an
        :class:`EdgeFlags` constant (or OR of constants) to filter to
        a specific edge classification.
        """
        ...

    def with_src_kind(self, kind: str) -> EdgeQuery:
        """Keep edges whose ``src`` node has the given ``kind``
        (``"module"``, ``"function"``, ``"import"``, …). Matches by
        exact string compare against :attr:`SymbolNode.kind`.
        """
        ...

    def with_dst_kind(self, kind: str) -> EdgeQuery:
        """Keep edges whose ``dst`` node has the given ``kind``."""
        ...

    def collect(self) -> list[EdgeRef]: ...
    def first(self) -> EdgeRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[EdgeRef]: ...

class DeclQuery:
    """Generic filter over every interned node in the in-progress graph.

    Folds the per-node Python filter loops that show up in plugins
    (filter on ``kind``, basename, simple-name, flag mask, path set,
    path regex) down into one rust pass. All configured predicates are
    AND-ed; an empty predicate set yields every node.
    """

    def with_kind(self, kind: str) -> DeclQuery: ...
    def with_kinds(self, kinds: str | list[str] | tuple[str, ...]) -> DeclQuery: ...
    def with_filename(self, name: str) -> DeclQuery: ...
    def with_filenames(self, names: str | list[str] | tuple[str, ...]) -> DeclQuery: ...
    def with_simple_name(self, name: str) -> DeclQuery: ...
    def with_simple_names(self, names: str | list[str] | tuple[str, ...]) -> DeclQuery: ...
    def with_paths(self, paths: str | list[str] | tuple[str, ...]) -> DeclQuery: ...
    def with_path_regex(self, regex: str) -> DeclQuery: ...
    def with_flags(self, mask: int) -> DeclQuery:
        """Restrict to nodes whose ``flags & mask == mask`` (all bits set)."""
        ...

    def with_any_flag(self, mask: int) -> DeclQuery:
        """Restrict to nodes whose ``flags & mask != 0`` (any bit set)."""
        ...

    def with_fqname_prefix(self, prefix: str) -> DeclQuery: ...
    def collect(self) -> list[SymbolNode]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[SymbolNode]: ...

# ---------- Graph persistence --------------------------------------------

class GraphMetadata:
    """Header block returned by :func:`read_graph` alongside the graph.

    ``created_at`` is unix-epoch seconds at write time. ``user_meta``
    is the list of ``(key, value)`` pairs the writer passed (from
    ``dead-cst build --meta key=value``). The counts mirror the
    graph stamped into the file and double as a sanity check against
    the deserialized payload.
    """

    format_version: int
    created_at: int
    node_count: int
    edge_count: int
    file_count: int
    line_count: int
    user_meta: list[tuple[str, str]]

def write_graph(
    path: str,
    nodes: list[SymbolNode],
    edges: list[tuple[int, int, int]],
    meta: list[tuple[str, str]],
) -> None:
    """Persist ``nodes`` + ``edges`` to ``path`` as a bincode-encoded
    graph file. ``meta`` is the user-supplied list of ``(key, value)``
    pairs (from ``--meta`` on the CLI); they are stored verbatim in
    the file's metadata block.
    """
    ...

def read_graph(path: str) -> tuple[NativeGraph, GraphMetadata]:
    """Load a graph file written by :func:`write_graph`.

    Raises ``ValueError`` when the magic bytes or the on-disk format
    version don't match the one this build knows how to read — the
    library never silently migrates an older file, since rebuilding
    a graph is cheap.
    """
    ...
