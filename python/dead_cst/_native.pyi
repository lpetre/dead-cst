"""Type stubs for the ty-backed native graph builder.

Surfaces the pyo3 classes the rust crate exposes so callers in
``dead_cst/`` and ``tests/prototype/`` get static information without
having to introspect the binary module. Kept hand-written — pyo3
doesn't generate stubs and the public API is small.

The docstrings on each class / method are the contract: each one
mirrors the rust-side rustdoc, and a behavior change in the crate
should land in both places at once.
"""

import re
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Protocol, Sequence

if TYPE_CHECKING:
    from dead_cst.analyze import ProgressSnapshot

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

class AddEdgeByIdx:
    """Index-keyed variant of :class:`AddEdge`. Accepts positional
    indices into ``ctx.nodes()`` instead of ``SymbolNode`` references.

    Lets plugins that already work in index space (paired with the
    ``.indices()`` query terminals or
    :meth:`ProjectContext.indices_where`) emit edges without ever
    round-tripping through ``Py<SymbolNode>``. The apply pass treats
    it identically to :class:`AddEdge` once the indices land in the
    builder. Raises :class:`IndexError` at apply time when either
    endpoint is out of range.
    """

    src_idx: int
    dst_idx: int
    flags: int

    def __init__(self, src_idx: int, dst_idx: int, *, flags: int = 0) -> None: ...

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

class AddEntrypointByIdx:
    """Index-keyed variant of :class:`AddEntrypoint`. Takes a positional
    index into :meth:`ProjectContext.nodes` instead of a ``SymbolNode``
    reference; the apply pass reads the decl's ``fqname`` / ``path``
    on the rust side to compose the marker, so plugins working in
    idx-space don't pay the ``Py<SymbolNode>`` allocation just to flag
    a seed.

    Raises :class:`IndexError` at apply time when ``decl_idx`` is out
    of range.
    """

    decl_idx: int
    marker: str

    def __init__(self, decl_idx: int, *, marker: str) -> None: ...

class AddNode:
    """Mint a synthetic intermediate node, optionally wiring it with
    edges in the same op.

    Every element of ``edges_from`` becomes a ``source -> this`` edge;
    every element of ``edges_to`` becomes a ``this -> target`` edge —
    so a plugin doesn't need a separate handle to reference the
    freshly-minted node from subsequent ops. Set
    ``flags = NodeFlags.ENTRYPOINT`` to make the node a reachability
    seed; for the common single-target entrypoint pattern, prefer
    :class:`AddEntrypoint` (or :class:`AddEntrypointByIdx` from
    idx-space).
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

class AddNodeByIdx:
    """Index-keyed variant of :class:`AddNode`. Wires the freshly-minted
    synthetic node with positional indices into ``ctx.nodes()`` instead
    of :class:`SymbolNode` references for ``edges_from`` / ``edges_to``.

    Pairs with the ``.indices()`` query terminals and
    :meth:`ProjectContext.indices_where` so plugins that already work
    in index space don't round-trip through ``Py<SymbolNode>`` just to
    wire their synthetic markers. The apply pass treats it identically
    to :class:`AddNode` once the indices land in the builder.

    Raises :class:`IndexError` at apply time when any endpoint is out
    of range. The bounds check runs *before* the new node is interned,
    so a bad index never leaves an unconnected synthetic behind.
    """

    fqname: str
    kind: NodeKind
    path: str
    flags: int
    edges_from_idx: list[int]
    edges_to_idx: list[int]

    def __init__(
        self,
        fqname: str,
        *,
        path: str,
        kind: NodeKind = "synthetic",
        flags: int = 0,
        edges_from_idx: Iterable[int] = ...,
        edges_to_idx: Iterable[int] = ...,
    ) -> None: ...

GraphOp = AddEdge | AddEdgeByIdx | AddEntrypoint | AddEntrypointByIdx | AddNode | AddNodeByIdx

class PerFileEdge:
    """Add an edge between two file-local nodes.

    File-local analogue of :class:`AddEdgeByIdx`; ``src`` and ``dst``
    are indices into the file's per-file payload (the ints returned by
    :class:`FileScope` query methods). The fan-in pass translates them
    to global graph indices at assemble time.
    """

    src: int
    dst: int
    flags: int

    def __init__(self, src: int, dst: int, *, flags: int = 0) -> None: ...

class PerFileEntrypoint:
    """Mark a file-local node as an entrypoint.

    File-local analogue of :class:`AddEntrypointByIdx`. ``target`` is
    an index into the file's per-file payload.
    """

    target: int
    marker: str

    def __init__(self, target: int, *, marker: str) -> None: ...

class PerFileNode:
    """Mint a synthetic node and wire it to file-local nodes.

    File-local analogue of :class:`AddNodeByIdx`. ``edges_from`` and
    ``edges_to`` are lists of file-local indices.
    """

    fqname: str
    kind: NodeKind
    path: str
    flags: int
    edges_from: list[int]
    edges_to: list[int]

    def __init__(
        self,
        fqname: str,
        *,
        path: str,
        kind: NodeKind = "synthetic",
        flags: int = 0,
        edges_from: Iterable[int] = ...,
        edges_to: Iterable[int] = ...,
    ) -> None: ...

PerFileOp = PerFileEdge | PerFileEntrypoint | PerFileNode

class FileScope:
    """Narrow per-file query handle handed to :class:`PerFilePlugin`.

    Fully pre-computed once per file before any plugin runs. All
    methods return plain ``int`` (file-local indices into the file's
    per-file payload) or lists thereof; the assemble-time fan-in
    translates the indices to global graph indices.

    Per-file plugins can only reach file-local state — there is no
    way to query other files from here, by design.
    """

    @property
    def path(self) -> str:
        """Absolute path of the file this scope wraps."""
        ...

    @property
    def module_fqname(self) -> str:
        """Fully-qualified dotted module name."""
        ...

    @property
    def module_idx(self) -> int:
        """Local index of the synthetic module node (always 0)."""
        ...

    def main_block(self) -> tuple[int, list[int]] | None:
        """`(module_idx, decl_idxs)` for the file's ``if __name__ == "__main__":``
        block, or :class:`None` when the file has none.
        """
        ...

    def dunder_decls(self) -> list[int]:
        """Local indices of variable/function decls whose fqname is a
        Python dunder name."""
        ...

    def imports_of(self, module_name: str) -> list[int]:
        """Local indices of import nodes targeting ``module_name``."""
        ...

class CollectedOps:
    """Opaque handle to one plugin's collected graph ops.

    Returned by :meth:`ProjectContext.run_plugin_collect`; consumed by
    :meth:`ProjectContext.apply_ops_batched`. The handle is single-use:
    a second ``apply_ops_batched`` on the same instance raises
    :class:`ValueError`.

    There is no Python-side constructor or accessor — the type exists
    solely as a transport for the (pure-rust) prepared ops between
    the collect and apply phases of plugin execution.
    """

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

    def add_per_file_plugin(self, plugin: Any) -> None:
        """Register a per-file plugin. Per-file plugins implement
        ``run_per_file(file)`` against a :class:`FileScope` and run
        during the populate phase via a salsa-tracked function;
        the output is cached keyed by file revision.
        """
        ...

    def materialize(self) -> NativeGraph:
        """Build the project-wide graph, run each registered plugin's
        ``run(ctx)``, then snapshot the final state.

        Borrows are released between phases so plugin ``run`` methods
        can re-enter queries through the same ``ctx`` without aliasing
        violations.
        """
        ...

    def build_only(self) -> None:
        """Run only the project-wide build pass, without invoking any
        registered plugins. Used by :class:`dead_cst.Analysis` to
        split build from the plugin pass so the latter can run on a
        Python :class:`concurrent.futures.ThreadPoolExecutor`."""
        ...

    def run_plugin(self, plugin: _ProjectPluginLike | Any) -> None:
        """Invoke ``plugin.run(ctx)`` once, collecting every yielded
        op and applying the batch under one write-lock window at
        the end. The plugin's own emissions are invisible to its
        own queries — the graph is frozen for the duration of
        ``run``. Prefer :meth:`run_plugin_collect` +
        :meth:`apply_ops_batched` when driving multiple plugins
        concurrently so the apply pass runs once for the full
        cohort."""
        ...

    def run_plugin_collect(self, plugin: _ProjectPluginLike | Any) -> CollectedOps:
        """Invoke ``plugin.run(ctx)`` once and return its yielded ops
        as an opaque :class:`CollectedOps` handle without mutating
        the graph. Safe to call concurrently from multiple Python
        threads — the graph is read-only for the duration. The
        handle is passed (alongside other plugins' handles) to
        :meth:`apply_ops_batched`, which folds them all into the
        graph under one write-lock window."""
        ...

    def apply_ops_batched(self, ops: list[CollectedOps]) -> None:
        """Apply a list of :class:`CollectedOps` handles to the graph
        in list order under a single write-lock window. Each handle
        is consumed; re-applying the same handle raises
        :class:`ValueError`. Ops within a handle apply in the order
        the plugin yielded them; ops across handles apply in
        ``ops`` order."""
        ...

    def snapshot_graph(self) -> NativeGraph:
        """Snapshot the current graph (post-:meth:`build_only` +
        any :meth:`run_plugin` calls) as a :class:`NativeGraph`.
        Used by :class:`dead_cst.Analysis` to return the final graph
        without re-running :meth:`materialize`."""
        ...

    def read_progress_snapshot(self) -> "ProgressSnapshot":
        """Atomic snapshot of the build-progress counters as a
        :class:`dead_cst.analyze.ProgressSnapshot` ``TypedDict``.
        Drives :class:`dead_cst.Analysis`'s polling thread; user code
        should prefer the structured ``progress_callback`` API on
        :class:`dead_cst.Analysis`.

        ``materialize`` holds the context's ``borrow_mut`` for the
        whole build, so a concurrent polling thread calling this
        method will see ``RuntimeError("Already mutably borrowed")``
        until the build releases. For race-free polling, use
        :meth:`progress_handle` instead — the returned handle reads
        the same atomic counters without touching pyo3's borrow flag."""
        ...

    def progress_handle(self) -> "ProgressHandle":
        """Mint a borrow-free handle over the progress counters.
        The handle holds an :class:`Arc` over the rust-side atomics
        so the polling thread can call :meth:`ProgressHandle.snapshot`
        concurrently with a long-running ``materialize`` call (which
        keeps the context's ``borrow_mut`` token held for the entire
        build)."""
        ...

    def mark_progress_finished(self) -> None:
        """Force the build-progress ``finished`` atomic to ``True``.
        Used by :class:`dead_cst.Analysis` after a build error so the
        polling thread exits cleanly. Idempotent."""
        ...

    def progress_plugin_done(self) -> None:
        """Bump the plugins-done counter by one. Used by the Python
        :class:`concurrent.futures.ThreadPoolExecutor` plugin pass to
        signal per-plugin completion to the polling thread."""
        ...

    def progress_plugins_start(self, names: list[str]) -> None:
        """Stamp the plugins phase as started + allocate per-plugin
        counter slabs keyed by registration order. ``names`` is
        the plugin list (``type(plugin).__qualname__`` per entry);
        indices passed to :meth:`progress_plugin_started` /
        :meth:`progress_plugin_finished` match. Called by
        :class:`dead_cst.Analysis` before launching the
        :class:`concurrent.futures.ThreadPoolExecutor`."""
        ...

    def progress_plugin_started(self, idx: int) -> None:
        """Stamp the indexed plugin's start time. Called by the
        :class:`concurrent.futures.ThreadPoolExecutor` worker on
        entry so the per-plugin slot snapshot reflects the actual
        start order (not the registration order)."""
        ...

    def progress_plugin_finished(self, idx: int) -> None:
        """Stamp the indexed plugin's finish time. Called by the
        :class:`concurrent.futures.ThreadPoolExecutor` worker on
        exit (both success and exception paths)."""
        ...

    def progress_plugins_finish(self) -> None:
        """Stamp the plugins phase as finished + mark the whole
        build pipeline finished. Called by :class:`dead_cst.Analysis`
        once every plugin future has resolved."""
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

    # The subclass-walk surface lives entirely on
    # :class:`SubclassQuery`. The point-lookup ``ctx.find_subclasses(fqn)``
    # / ``find_subclasses_of(node)`` / ``find_subclasses_of_idx(idx)``
    # methods that used to mirror these queries are now rust-only
    # (called by :class:`SubclassQuery` internally). Plugin authors:
    # use the DSL — ``native.query(ctx).subclasses().of_fqn(fqn)`` /
    # ``.of_idx(idx)`` / ``.of_node(node)`` (legacy node-form input)
    # with ``.collect()`` or ``.indices()`` terminals.

    # ----- FQN resolution ------------------------------------------------
    #
    # Plugins use the DSL: ``query(ctx).declarations()`` for fqname
    # → decl lookup, ``query(ctx).modules()`` for module fqname /
    # path lookup, surface / top-level / dunder-all transforms, and
    # ``query(ctx).literal_lists().for_fqn(fqn).entries()`` for
    # module-scope string-literal lists. The flat idx-form helpers
    # below back those DSL terminals and stay on :class:`ProjectContext`
    # as a low-level escape hatch.

    def resolve_idx(self, fqname: str) -> int | None:
        """First decl matching ``fqname``, falling back to the module
        match (same walk-back rules as :meth:`find_declarations_indices`
        but module nodes are included). Returns ``None`` when the
        fqname can't be found anywhere.
        """
        ...

    def find_declarations_indices(self, fqname: str) -> list[int]:
        """Every decl matching ``fqname`` (walk-back through dotted
        segments included). Modules are never returned; use
        :meth:`find_module_idx` for that. Returns positional indices
        into :meth:`nodes`.
        """
        ...

    def find_module_idx(self, fqname: str) -> int | None:
        """O(1) module-by-fqname lookup. Returns the positional index
        into :meth:`nodes`, or ``None`` if ``fqname`` doesn't name a
        project module.
        """
        ...

    def module_for_indices(self, path: str) -> int | None:
        """O(1) path-to-module lookup. Returns the positional index
        into :meth:`nodes`, or ``None`` if ``path`` doesn't name a
        project module.
        """
        ...

    def modules_for_paths(self, paths: list[str]) -> list[int | None]:
        """Bulk form of :meth:`module_for_indices`. One ``materialize``
        check + one O(1) lookup per path; missing paths map to ``None``.
        Lets plugins that call ``module_for(path)`` once per ref row
        collapse N FFI hops into one.
        """
        ...

    def module_surface_indices(self, module_fqn: str) -> list[int]:
        """BFS over the fqname tree from ``module_fqn``: the module
        idx + every transitive descendant idx. Models
        ``importlib.import_module(module_fqn)``: the module's whole
        top-level surface plus everything its submodules expose.
        Empty list when ``module_fqn`` doesn't resolve to a project
        module.
        """
        ...

    def module_surfaces_indices(self, module_fqns: list[str]) -> dict[str, list[int]]:
        """Bulk form: resolve every fqname in ``module_fqns`` in a
        single scan instead of one scan per fqname. Returns a dict
        keyed by input fqname; modules that don't resolve map to empty
        lists. Duplicate inputs share the same result list.
        """
        ...

    def find_module_top_level_decls_indices(self, module_fqn: str) -> list[int]:
        """``module_fqn``'s immediate top-level decls — every function
        / class / variable / import bound at its module scope.
        Submodules are excluded. Empty list when ``module_fqn`` isn't
        a project module.
        """
        ...

    def find_module_dunder_all_exports_indices(self, module_fqn: str) -> list[int] | None:
        """Decls listed in ``module_fqn``'s ``__all__``. ``None`` means
        "no ``__all__``"; ``[]`` means "empty / unresolvable
        ``__all__``" — callers wanting CPython's
        ``from X import *`` semantics fall back to the non-underscore
        decl list only in the ``None`` case.
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
    #
    # Plugins use the DSL: ``query(ctx).decls()`` with the
    # ``with_path_prefix`` / ``with_path_contains`` / ``with_simple_name_regex``
    # predicates; ``query(ctx).matching_specs(...)`` for the
    # OR-form entrypoint matcher. The idx-form helpers below back
    # those DSL terminals as low-level escape hatches.

    def decls_under_indices(self, path_prefix: str) -> list[int]:
        """Every node whose ``path`` starts with ``path_prefix``.
        Returns positional indices into :meth:`nodes`.
        """
        ...

    def decls_matching_indices(self, substring: str) -> list[int]:
        """Every node whose ``path`` contains ``substring`` anywhere."""
        ...

    def decls_matching_name_indices(self, pattern: str) -> list[int]:
        """Every top-level decl (function / class / variable / import /
        type_alias) whose simple name matches ``pattern`` (a regex)."""
        ...

    def find_nodes_matching_specs_indices(
        self,
        project_root: str,
        regexes: list[str],
        str_specs: list[str],
        abs_paths: list[str],
    ) -> list[int]:
        """OR-form spec matcher. ``regexes`` are anchored at the start
        of input (``re.Pattern.match`` semantics) against the relative
        path. ``str_specs`` match exactly against the relative path OR
        the node's fqname. ``abs_paths`` match exactly against the
        absolute path. Returns positional indices into :meth:`nodes`.
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

    def descendants_indices(self, root_idx: int, *, skip_flags: int = 0) -> list[int]:
        """Idx-keyed variant of :meth:`descendants`. Takes a positional
        index into :meth:`nodes` and returns descendant indices rather
        than allocating ``SymbolNode`` clones. Raises
        :class:`IndexError` when ``root_idx`` is out of range.
        """
        ...

    def ancestors(self, decl: SymbolNode, *, skip_flags: int = 0) -> list[SymbolNode]:
        """Reverse closure: every node that can reach ``decl`` by
        following graph edges.

        Used for predecessor-chain walks and blast-radius scoping.
        ``skip_flags`` works the same as in :meth:`descendants`.
        """
        ...

    def ancestors_indices(self, decl_idx: int, *, skip_flags: int = 0) -> list[int]:
        """Idx-keyed variant of :meth:`ancestors`. Takes a positional
        index into :meth:`nodes` and returns ancestor indices rather
        than allocating ``SymbolNode`` clones. Raises
        :class:`IndexError` when ``decl_idx`` is out of range.
        """
        ...

    def direct_predecessors_idx(self, idx: int, *, skip_flags: int = 0) -> list[int]:
        """One-hop reverse step: every node with an edge directly into
        ``idx``. Dedups by source idx, so a pair of parallel edges
        with different :class:`EdgeFlags` between the same two nodes
        only produces one entry. ``skip_flags`` filters edges by
        intersecting flag mask — same semantics as
        :meth:`ancestors_indices`. Plugins use the DSL:
        ``native.query(ctx).from_idx(idx).direct_predecessors()``.
        """
        ...

    def reachable(self, *, skip_flags: int = 0, seed_flags: int = ...) -> list[SymbolNode]:
        """Forward closure from every node carrying any bit in
        ``seed_flags`` (defaults to :data:`NodeFlags.ENTRYPOINT`). The
        set of dead decls is the complement against :meth:`nodes`."""
        ...

    def reachable_indices(self, *, skip_flags: int = 0, seed_flags: int = ...) -> list[int]:
        """Idx-only sibling of :meth:`reachable`. Same semantics, but
        returns positional indices into :meth:`nodes` instead of
        materialising ``SymbolNode`` clones. Use when you only need
        set-membership / counting on the reached set — pair with
        :meth:`nodes_at` to revive specific nodes on demand.
        """
        ...

    def indices_where(
        self,
        *,
        kind: str | None = None,
        kinds: list[str] | None = None,
        filename: str | None = None,
        filenames: list[str] | None = None,
        simple_name: str | None = None,
        simple_names: list[str] | None = None,
        paths: list[str] | None = None,
        path_regex: str | None = None,
        flags: int | None = None,
        flags_any: int | None = None,
        fqname_prefix: str | None = None,
    ) -> list[int]:
        """Flat-form predicate filter returning positional indices into
        :meth:`nodes`. Mirrors the :class:`DeclQuery` predicate
        vocabulary but skips the builder construction — useful when
        you only have one filter step and just need a ``list[int]``.

        Every parameter is keyword-only and optional; unset arguments
        don't filter. ``kind`` / ``kinds`` (and similarly the
        ``filename`` / ``simple_name`` pairs) are merged; pass either
        form. All set predicates AND together. ``flags`` is the
        all-bits-set form (``node.flags & mask == mask``);
        ``flags_any`` is the any-bit form (``node.flags & mask != 0``).
        """
        ...

    def nodes_at(self, indices: Sequence[int]) -> list[SymbolNode]:
        """Inverse of the ``.indices()`` terminals: materialize
        specific nodes by their positional indices into :meth:`nodes`.
        Validates bounds and raises :class:`IndexError` when any index
        is out of range.
        """
        ...

    def node_attrs(self, indices: Sequence[int]) -> list[NodeAttrs]:
        """Batched snapshot of ``(kind, path, fqname, flags)`` per
        index. One FFI hop instead of N per-attribute ``borrow``
        round-trips — lets plugins that filter or partition by these
        four fields stay GIL-free in the inner loop, which matters
        once the plugin pass runs concurrently.

        When the plugin only needs ``path`` (the common "bucket by
        file" case), prefer :meth:`node_paths` — it skips the per-row
        ``kind`` / ``fqname`` / ``flags`` clones that ``node_attrs``
        would allocate but the plugin would throw away.

        Validates bounds the same way :meth:`nodes_at` does and raises
        :class:`IndexError` when any index is out of range.
        """
        ...

    def node_paths(self, indices: Sequence[int]) -> list[str]:
        """Batched ``path``-only snapshot for each index. Same FFI shape
        as :meth:`node_attrs` but allocates only one ``str`` per row
        instead of a 4-tuple — roughly 3× fewer Python allocations on
        the common "bucket by file" path.

        Validates bounds and raises :class:`IndexError` when any index
        is out of range.
        """
        ...

    def function_parameters(self, indices: Sequence[int]) -> list[list[str]]:
        """Batched parameter-name snapshot for each function-kind
        index. Walks the parsed AST once per distinct path; matches
        each top-level ``FunctionDef`` against the rust-side
        ``decl_by_name_range`` by its name's byte range.

        Returns the union of positional-only, positional-or-keyword,
        and keyword-only parameter names in declaration order;
        ``*args`` and ``**kwargs`` are skipped. Indices that don't
        resolve to a top-level ``FunctionDef`` (non-function kinds,
        nested functions, decorator-only stubs) surface an empty list
        at the same position.

        Used by :class:`dead_cst.contrib.PytestPlugin` to discover
        ``test_foo(my_fixture)`` → ``test_foo → my_fixture`` edges.

        Validates bounds and raises :class:`IndexError` when any
        index is out of range.
        """
        ...

    def class_method_parameters(self, indices: Sequence[int]) -> list[list[str]]:
        """Batched class-method parameter-name snapshot for each
        class-kind index. For each input class, walks its body's
        top-level ``FunctionDef`` statements and returns the union
        of their parameter names (positional-only + positional-or-
        keyword + keyword-only), deduped in first-seen order, with
        ``self`` and ``cls`` excluded. ``*args`` and ``**kwargs``
        are skipped.

        Indices that don't resolve to a top-level ``ClassDef`` in the
        AST surface an empty list at the same position.

        Used by :class:`dead_cst.contrib.PytestPlugin` to wire
        ``class → fixture`` edges for ``Test*`` classes — class
        methods aren't represented as their own graph nodes, so the
        class itself is the rendezvous point for any fixture any
        method uses.

        Validates bounds and raises :class:`IndexError` when any
        index is out of range.
        """
        ...

    # ----- Pure scans over the in-progress graph ------------------------
    #
    # Plugins use the DSL: ``query(ctx).modules().with_dunders().indices()``
    # for module-dunder enumeration, ``query(ctx).main_blocks().index_pairs()``
    # for ``if __name__ == "__main__":`` blocks. The idx-form helpers
    # below back those DSL terminals.

    def find_module_dunders_indices(self) -> list[int]:
        """Every top-level variable / function node whose name matches
        ``__xxx__``. Pure scan over already-interned nodes — no ty
        re-query needed."""
        ...

    # The import-of-module surface lives entirely on
    # :class:`ImportQuery`. Use ``native.query(ctx).imports().of(m)``
    # then ``.collect()`` / ``.indices()`` / ``.count()`` / ``.exists()``.

    def find_main_blocks_indices(self) -> list[tuple[int, list[int]]]:
        """``(module_idx, [decl_idx])`` pairs into :meth:`nodes` for
        every file with a top-level ``if __name__ == "__main__":``
        block. The decls list contains the file's class / function /
        variable / import nodes whose source position falls inside the
        block's range.
        """
        ...

    # The class-defining-method surface lives on :class:`ClassQuery`.
    # Use ``native.query(ctx).classes().defining_method(name).collect()`` /
    # ``.indices()``.

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

class ArgLiteral:
    """A literal arg / kwarg value. ``value`` is a native Python
    primitive: ``str`` / ``int`` / ``float`` / ``bool`` / ``None`` /
    ``bytes`` / ``list`` / ``tuple``. Nested ``list`` / ``tuple``
    elements are themselves :class:`ArgLiteral` / :class:`ArgNodeRef`
    / :class:`ArgOpaque` instances (the discriminated union recurses).
    """

    value: Any

class ArgNodeRef:
    """A decl reference inside an arg / kwarg position.

    Materialised when the expression at that position resolves through
    the file's imports to a project decl — e.g. ``func(SomeClass)``
    where ``SomeClass`` is imported. ``idx`` is a positional index
    into :meth:`ProjectContext.nodes`; pair with
    :meth:`ProjectContext.node_attrs` (or :class:`AddEdgeByIdx`) to
    consume it without leaving idx-space.
    """

    idx: int

class ArgOpaque:
    """An arg / kwarg expression that's neither a recognised literal
    nor a statically-resolvable decl reference. Callers who need the
    source text should fall back to ty's parsed module — the rust
    extractor doesn't preserve unresolved expressions verbatim.
    """

    def __init__(self) -> None: ...

class NodeAttrs:
    """Tuple-like row returned by :meth:`ProjectContext.node_attrs`
    and every query's ``.attrs()`` terminal.

    Supports both attribute access (``attr.fqname``) and tuple
    semantics (``kind, path, fqname, flags = attr``; ``attr[2]``;
    ``len(attr) == 4``; ``list(attr)``). Not a `typing.NamedTuple`
    instance, but drop-in compatible for unpacking and subscript.
    Frozen — fields are immutable once constructed.
    """

    kind: NodeKind
    path: str
    fqname: str
    flags: int

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...

CallArg = ArgLiteral | ArgNodeRef | ArgOpaque
"""Discriminated-union type for entries inside the lazy ``args`` /
``kwargs`` getters on :class:`DecoratorIdxRef` / :class:`ConstructionIdxRef`
/ :class:`CallIdxRef`. Use ``isinstance`` or ``match`` to dispatch:

.. code-block:: python

    for arg in row.kwargs.values():
        match arg:
            case native.ArgLiteral(value=str() as s):
                ...
            case native.ArgNodeRef(idx=i):
                ...
            case native.ArgOpaque():
                ...
"""

class DecoratorIdxRef:
    """One decorator application on a top-level function or class.

    Field nullability follows the query shape that produced the row:

    * ``where_module + where_name`` populates ``decorated_idx`` only.
    * ``where_owner_attr`` fills ``decorator_owner`` (the textual
      ``@<owner>.<attr>`` prefix).
    * ``where_owner_attr_via`` additionally fills ``decorator_via``
      with the middle attribute name.

    ``args`` and ``kwargs`` are **lazy** — accessing them walks the
    row's rust-side ``CallArgs`` and materialises a Python ``list`` /
    ``dict`` of :class:`ArgLiteral` / :class:`ArgNodeRef` /
    :class:`ArgOpaque`. Plugins that never touch them pay zero Python
    allocation cost for the args payload. Bare-attribute decorators
    (``@app.route`` without ``()``) get empty containers.

    Returned by :meth:`DecoratorQuery.collect`.
    """

    decorated_idx: int
    path: str
    decorator_name: str | None
    decorator_owner: str | None
    decorator_via: str | None

    @property
    def args(self) -> list[CallArg]: ...
    @property
    def kwargs(self) -> dict[str, CallArg]: ...

class ConstructionIdxRef:
    """One ``<var> = <Ctor>(...)`` construction at module scope.

    ``class_name`` is the upstream constructor's bare name (``"Flask"``
    even when imported as ``F``). ``args`` / ``kwargs`` are the
    constructor call's positional / keyword arguments — lazy getters,
    same discriminated-union shape as :class:`DecoratorIdxRef`.

    Returned by :meth:`ConstructionQuery.collect`.
    """

    var_idx: int
    path: str
    class_name: str

    @property
    def args(self) -> list[CallArg]: ...
    @property
    def kwargs(self) -> dict[str, CallArg]: ...

class CallIdxRef:
    """One matched call site.

    ``string_arg`` is the positional string literal at the index passed
    to :meth:`CallQuery.string_arg_at`. ``args`` / ``kwargs`` are the
    call's full positional / keyword arguments — lazy getters, same
    discriminated-union shape as :class:`DecoratorIdxRef`.

    Returned by :meth:`CallQuery.collect`.
    """

    owner_idx: int
    path: str
    string_arg: str

    @property
    def args(self) -> list[CallArg]: ...
    @property
    def kwargs(self) -> dict[str, CallArg]: ...

class FactoryIdxRef:
    """One factory-function / class hit. ``decl_idx`` is the owning
    top-level decl's positional index into :meth:`ProjectContext.nodes`;
    ``kinds`` is the sorted set of constructor bare-names matched
    inside its body. ``path`` is the decl's source file as a cheap
    bucket key for per-file fan-out.

    Returned by :meth:`FactoryQuery.collect`.
    """

    decl_idx: int
    path: str
    kinds: list[str]

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
    def modules(self) -> ModuleQuery: ...
    def classes(self) -> ClassQuery: ...
    def factories(self) -> FactoryQuery: ...
    def edges(self) -> EdgeQuery: ...
    def decls(self) -> DeclQuery: ...
    def declarations(self) -> DeclarationsQuery: ...
    def main_blocks(self) -> MainBlockQuery: ...
    def literal_lists(self) -> LiteralListQuery: ...
    def from_idx(self, seed_idx: int) -> TraverseQuery:
        """Anchor a closure walk on a single seed. Returns a
        :class:`TraverseQuery` whose terminals (``descendants`` /
        ``ancestors`` / ``direct_predecessors``) return positional
        indices into :meth:`ProjectContext.nodes`.
        """
        ...

    def reachable(self, *, skip_flags: int = 0, seed_flags: int | None = None) -> list[int]:
        """Seedless reachability terminal. Forward closure from every
        node carrying any bit in ``seed_flags`` (default:
        ``NodeFlags.ENTRYPOINT``), filtering edges by ``skip_flags``.
        Returns positional indices into :meth:`ProjectContext.nodes`.
        """
        ...

    def matching_specs(
        self,
        project_root: str,
        *,
        regexes: list[str] = ...,
        str_specs: list[str] = ...,
        abs_paths: list[str] = ...,
    ) -> list[int]:
        """OR-form spec matcher (used by
        :class:`ExplicitEntrypointPlugin`). A node matches if any of:

        * ``regexes`` contains a pattern matching the node's path
          relative to ``project_root`` (anchored, ``re.match`` style);
        * ``str_specs`` contains the node's relative path or fqname;
        * ``abs_paths`` contains the node's absolute path.

        Returns positional indices into :meth:`ProjectContext.nodes`.
        """
        ...

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
    def in_decl_idx(self, idx: int) -> DecoratorQuery:
        """Idx-form sibling of :meth:`in_decl`. Pass a positional index
        into :meth:`ProjectContext.nodes` directly so plugins working
        in idx-space don't round-trip through a ``SymbolNode``.
        """
        ...
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

    def with_args(self, value: bool) -> DecoratorQuery:
        """Opt in to rust-side ``args`` / ``kwargs`` extraction.

        Defaults to ``False`` — the per-row
        :fn:`extract_call_args_kwargs` walk is skipped, and the
        row's ``args`` / ``kwargs`` getters surface empty containers.
        Pass ``True`` when a plugin actually reads ``args`` /
        ``kwargs`` off the matched rows.

        Auto-forced back to ``True`` at row-collection time when any
        ``where_kwarg`` is set (kwarg filtering needs the data).
        """
        ...

    def collect(self) -> list[DecoratorIdxRef]: ...
    def first(self) -> DecoratorIdxRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[DecoratorIdxRef]: ...
    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched ``decorated_idx`` values by their
        owning file path. ``dict[path, list[int]]``; reads ``path``
        straight off each row.
        """
        ...

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
    def with_args(self, value: bool) -> ConstructionQuery:
        """Opt out of rust-side ``args`` / ``kwargs`` extraction. See
        :meth:`DecoratorQuery.with_args`.
        """
        ...

    def collect(self) -> list[ConstructionIdxRef]: ...
    def first(self) -> ConstructionIdxRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[ConstructionIdxRef]: ...
    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched ``var_idx`` values by their owning
        file path.
        """
        ...

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

    def with_args(self, value: bool) -> CallQuery:
        """Opt out of rust-side ``args`` / ``kwargs`` extraction. See
        :meth:`DecoratorQuery.with_args`. Auto-forced back to ``True``
        when any ``where_kwarg`` is set.
        """
        ...

    def collect(self) -> list[CallIdxRef]: ...
    def first(self) -> CallIdxRef | None: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[CallIdxRef]: ...
    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched ``owner_idx`` values by their
        owning file path.
        """
        ...

class SubclassQuery:
    """Walk the subclass closure of a class.

    Pick exactly one of :meth:`of_fqn` / :meth:`of_node`. The
    default :meth:`transitive` is ``True``; flip to ``False`` for
    direct subclasses only. Mirrors the union of
    ``ProjectContext.find_subclasses`` and ``find_subclasses_of``.
    """

    def of_fqn(self, fqn: str) -> SubclassQuery: ...
    def of_node(self, node: SymbolNode) -> SubclassQuery: ...
    def of_idx(self, idx: int) -> SubclassQuery:
        """Idx-form sibling of :meth:`of_node`. Pass a positional
        index into :meth:`ProjectContext.nodes` directly so plugins
        working in idx-space don't round-trip through a ``SymbolNode``.
        """
        ...

    def transitive(self, value: bool) -> SubclassQuery: ...
    def collect(self) -> list[SymbolNode]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[SymbolNode]: ...
    def indices(self) -> list[int]:
        """Index-returning terminal. Same lookup as :meth:`collect`,
        but emits each subclass's positional index into
        :meth:`ProjectContext.nodes` instead of allocating
        ``SymbolNode`` clones.
        """
        ...

    def attrs(self) -> list[NodeAttrs]:
        """Terminal — :class:`NodeAttrs` for every matched subclass,
        in the same order :meth:`indices` returns. Avoids the
        boilerplate of ``ctx.node_attrs(q.indices())``.
        """
        ...

    def first_idx(self) -> int | None:
        """Terminal — first matched subclass's positional index, or
        ``None`` when no subclass exists.
        """
        ...

    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched indices by their owning file
        path.
        """
        ...

class ImportQuery:
    """Enumerate the ``kind="import"`` nodes that bind a name from a
    given module. Requires :meth:`of` (the upstream module name).

    Mirrors :meth:`ProjectContext.find_imports_of`.
    """

    def of(self, module: str) -> ImportQuery: ...
    def collect(self) -> list[SymbolNode]: ...
    def indices(self) -> list[int]:
        """Index-returning terminal. Reads positional indices straight
        out of the pre-built ``imports_by_module`` index — no Python
        allocation per row.
        """
        ...

    def count(self) -> int: ...
    def exists(self) -> bool:
        """O(1) presence probe — does any project file import the
        configured module? Short-circuits without materialising a
        Python list. Preferred over ``.count() > 0`` / ``.collect()``
        for plugin guards that just need a boolean.
        """
        ...

    def attrs(self) -> list[NodeAttrs]:
        """Terminal — :class:`NodeAttrs` for every matched import
        node, in the same order :meth:`indices` returns.
        """
        ...

    def first_idx(self) -> int | None:
        """Terminal — first matched import node's positional index,
        or ``None`` when no project file imports the configured
        module.
        """
        ...

    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched indices by their owning file
        path.
        """
        ...

    def __iter__(self) -> Iterator[SymbolNode]: ...

class ModuleQuery:
    """Enumerate / inspect project module nodes.

    Pick exactly one filter (:meth:`with_fqn` / :meth:`with_path` /
    :meth:`with_dunders`), optionally follow with one transform
    (:meth:`surface` / :meth:`top_level` / :meth:`dunder_all`), then
    drop into a terminal (:meth:`indices` / :meth:`first_idx` /
    :meth:`dunder_all`).

    Idx-only terminals — :class:`ModuleQuery` doesn't have a
    ``.collect()`` SymbolNode form; plugins consume the idxs and
    fetch attrs via :meth:`ProjectContext.node_attrs` /
    :meth:`node_paths` as needed.
    """

    def with_fqn(self, fqn: str) -> ModuleQuery:
        """Narrow to a single module by dotted fqname."""
        ...

    def with_path(self, path: str) -> ModuleQuery:
        """Narrow to the module owning ``path``. O(1) — backed by the
        same ``module_nodes_by_file`` index :meth:`find_main_blocks`
        uses."""
        ...

    def with_dunders(self) -> ModuleQuery:
        """Project-wide scan: every module-level variable named
        ``__xxx__`` plus every PEP 562 dunder function
        (``__getattr__`` / ``__dir__``). Terminal-friendly with
        :meth:`indices`. Pairs with no transform.
        """
        ...

    def surface(self) -> ModuleQuery:
        """Transform: module + every transitive decl whose fqname
        lives under the filtered module's fqname. Models
        ``importlib.import_module(...)`` reachability — submodules
        are recursed into, but a decl's sub-fqnames are not.
        """
        ...

    def top_level(self) -> ModuleQuery:
        """Transform: the filtered module's immediate top-level decls.
        Models ``from <module> import *`` semantics — submodules and
        their decls are excluded.
        """
        ...

    def indices(self) -> list[int]:
        """Terminal: list of matching positional indices into
        :meth:`ProjectContext.nodes`.

        * No transform + ``with_fqn`` / ``with_path``: a 0- or
          1-element list with the matched module idx.
        * ``surface()`` / ``top_level()``: the module + relevant
          decls.
        * ``with_dunders()``: every module-level dunder name in the
          project.
        """
        ...

    def first_idx(self) -> int | None:
        """Terminal: first matching idx or ``None``. Convenience for
        single-value lookups (``with_fqn`` / ``with_path`` without a
        transform).
        """
        ...

    def dunder_all(self) -> list[int] | None:
        """Terminal: decls listed in the module's ``__all__``.

        Returns ``None`` when the module doesn't declare ``__all__``;
        returns ``[]`` when ``__all__`` exists but resolves to no
        in-project decls. Requires :meth:`with_fqn`; other filters /
        transforms are ignored. The distinction between ``None`` and
        ``[]`` matters: CPython's ``from X import *`` semantics fall
        back to the non-underscore decl list only in the ``None``
        case.
        """
        ...

    def count(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def attrs(self) -> list[NodeAttrs]:
        """Terminal — :class:`NodeAttrs` for every matched index, in
        the same order :meth:`indices` returns. Avoids the
        boilerplate of ``ctx.node_attrs(q.indices())``.
        """
        ...

    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched indices by their owning file
        path.
        """
        ...

class TraverseQuery:
    """Closure walks anchored at a single seed node. Built via
    :meth:`QueryBuilder.from_idx`. All terminals return positional
    indices into :meth:`ProjectContext.nodes`; revive rows via
    :meth:`ProjectContext.nodes_at` if a plugin needs full
    :class:`SymbolNode` objects. For the seedless "alive from
    entrypoints" walk, use :meth:`QueryBuilder.reachable` instead.
    """

    def descendants(self, *, skip_flags: int = 0) -> list[int]:
        """Terminal: forward closure from the seed. ``skip_flags`` is
        an :class:`EdgeFlags` mask — edges whose flag mask intersects
        are filtered out (e.g. ``EdgeFlags.DEAD_BRANCH.value``).
        """
        ...

    def ancestors(self, *, skip_flags: int = 0) -> list[int]:
        """Terminal: reverse closure to the seed. ``skip_flags``
        filters edges the same way as :meth:`descendants`.
        """
        ...

    def direct_predecessors(self, *, skip_flags: int = 0) -> list[int]:
        """Terminal: one-hop reverse — the immediate predecessors of
        the seed (deduped by source idx, so parallel edges with
        different flags collapse to a single entry).
        """
        ...

class DeclarationsQuery:
    """Look up declarations by fully-qualified name. Built via
    :meth:`QueryBuilder.declarations`. Requires :meth:`with_fqname`;
    terminals are :meth:`indices` (all matching decls) and
    :meth:`resolve_idx` (first match, with module fallback).

    The walk-back rule: when the exact fqname doesn't match, dotted
    segments are stripped from the right until an enclosing top-level
    decl is found (``pkg.lib.Cls.method`` resolves to ``pkg.lib.Cls``
    because methods aren't graph nodes).
    """

    def with_fqname(self, fqname: str) -> DeclarationsQuery: ...
    def indices(self) -> list[int]:
        """Terminal: every decl matching ``fqname`` (walk-back
        included). Modules are never returned; use
        :meth:`QueryBuilder.modules` for module lookup.
        """
        ...

    def resolve_idx(self) -> int | None:
        """Terminal: first decl matching ``fqname``, falling back to a
        module match. Returns ``None`` when the fqname can't be found
        anywhere.
        """
        ...

    def count(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def attrs(self) -> list[NodeAttrs]:
        """Terminal — :class:`NodeAttrs` for every matched decl, in
        the same order :meth:`indices` returns. Modules are excluded
        (same rule as :meth:`indices`).
        """
        ...

    def first_idx(self) -> int | None:
        """Terminal — first matched decl's positional index, or
        ``None`` when no decl matches. Distinct from
        :meth:`resolve_idx`: this one skips the module fallback.
        """
        ...

    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched indices by their owning file
        path.
        """
        ...

class MainBlockQuery:
    """Enumerate every ``if __name__ == "__main__":`` block in the
    project. Built via :meth:`QueryBuilder.main_blocks`. The only
    terminal is :meth:`index_pairs`.
    """

    def index_pairs(self) -> list[tuple[int, list[int]]]:
        """Terminal: ``(module_idx, [decl_idx, ...])`` pairs for every
        file with a top-level ``if __name__ == "__main__":`` block.
        One entry per file; ``decl_idx`` lists the top-level decls
        whose source position falls inside the block's range.
        """
        ...

    def count(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[int, list[int]]]: ...

class LiteralListQuery:
    """Read the entries of a module-level string-literal list / tuple
    (typical use: ``__all__``, but works for any name). Built via
    :meth:`QueryBuilder.literal_lists`. Requires :meth:`for_fqn`; the
    only terminal is :meth:`entries`.
    """

    def for_fqn(self, fqn: str) -> LiteralListQuery: ...
    def entries(self) -> list[str] | None:
        """Terminal: the string entries bound to the configured
        ``fqn`` at module scope (concatenated across multiple decls in
        declaration order), or ``None`` when the name isn't a
        module-level decl or doesn't bind a string-literal list /
        tuple.
        """
        ...

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
    def indices(self) -> list[int]:
        """Index-returning terminal. Same per-file parallel walk as
        :meth:`collect`, but emits positional indices into
        :meth:`ProjectContext.nodes` instead of allocating
        ``SymbolNode`` clones.
        """
        ...

    def attrs(self) -> list[NodeAttrs]:
        """Terminal — :class:`NodeAttrs` for every matched class, in
        the same order :meth:`indices` returns.
        """
        ...

    def first_idx(self) -> int | None:
        """Terminal — first matched class's positional index, or
        ``None`` when no class defines the configured method.
        """
        ...

    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched indices by their owning file
        path.
        """
        ...

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
    def collect(self) -> list[FactoryIdxRef]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[FactoryIdxRef]: ...
    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched ``decl_idx`` values by their
        owning file path.
        """
        ...

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
    def index_triples(self) -> list[tuple[int, int, int]]:
        """Index-returning terminal for edges. Same per-edge predicate
        pipeline as :meth:`collect`, but emits ``(src_idx, dst_idx,
        flags)`` triples instead of materialising one :class:`EdgeRef`
        (and two ``SymbolNode`` clones) per row.

        Faster than :meth:`collect` when you only need set-membership
        / counting over edge endpoints; pair with
        :meth:`ProjectContext.nodes_at` to revive the surviving
        endpoints on demand.
        """
        ...

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
    def with_path_prefix(self, prefix: str) -> DeclQuery:
        """Restrict to nodes whose absolute path starts with
        ``prefix``. Cheaper than :meth:`with_path_regex` for simple
        directory scoping.
        """
        ...

    def with_path_contains(self, substring: str) -> DeclQuery:
        """Restrict to nodes whose absolute path contains ``substring``
        anywhere. Useful for path-pattern plugins like
        ``alembic/versions/`` or ``.ignore.py``.
        """
        ...

    def with_simple_name_regex(self, pattern: str) -> DeclQuery:
        """Restrict to nodes whose trailing fqname segment matches
        ``pattern`` (a regex). Combine with :meth:`with_kind` /
        :meth:`with_kinds` to drop modules when you only want
        top-level decls.
        """
        ...
    def with_flags(self, mask: int) -> DeclQuery:
        """Restrict to nodes whose ``flags & mask == mask`` (all bits set)."""
        ...

    def with_any_flag(self, mask: int) -> DeclQuery:
        """Restrict to nodes whose ``flags & mask != 0`` (any bit set)."""
        ...

    def with_fqname_prefix(self, prefix: str) -> DeclQuery:
        """Restrict to nodes whose ``fqname`` starts with ``prefix`` —
        a raw string prefix, not segment-bounded. ``prefix="foo"``
        matches both ``foo.bar`` and ``foobar``. Use
        :meth:`with_fqname_under` for the segment-bounded
        "descendants of this fqname" predicate that walks the fqname
        tree via the ``children_by_parent`` index.
        """
        ...

    def with_fqname_under(self, parent_fqn: str) -> DeclQuery:
        """Restrict to nodes whose ``fqname`` equals ``parent_fqn``
        or is a transitive descendant of it in the fqname tree.

        Segment-bounded: ``parent_fqn="pkg.foo"`` matches ``pkg.foo``,
        ``pkg.foo.bar``, ``pkg.foo.bar.baz`` — but **not** ``pkg.foobar``.
        Backed by the project's ``children_by_parent`` index, so this
        is O(matches) instead of the O(all_nodes) scan
        :meth:`with_fqname_prefix` performs.
        """
        ...

    def where_fqname(
        self,
        value: str | re.Pattern[str] | Sequence[str | re.Pattern[str]],
    ) -> DeclQuery:
        """Restrict to nodes whose ``fqname`` matches the predicate.

        Accepts any combination of ``str`` (literal equality) and
        ``re.Pattern`` (regex match). A sequence value matches when
        the node's ``fqname`` matches any element. ``re.Pattern``
        instances are recompiled rust-side using rust's ``regex``
        crate, so any PCRE-only syntax raises ``ValueError`` at the
        call site (not later at ``collect``).
        """
        ...

    def collect(self) -> list[SymbolNode]: ...
    def count(self) -> int: ...
    def __iter__(self) -> Iterator[SymbolNode]: ...
    def indices(self) -> list[int]:
        """Index-returning terminal. Same predicate semantics as
        :meth:`collect`, but emits each surviving node's positional
        index into :meth:`ProjectContext.nodes` (a plain
        ``list[int]``) instead of allocating one ``SymbolNode`` per
        row.

        Use when you only need set membership / counting on the
        surviving nodes (or want to feed an index-keyed
        :class:`AddEdgeByIdx`); call
        :meth:`ProjectContext.nodes_at` to materialize back to
        ``SymbolNode`` later.
        """
        ...

    def attrs(self) -> list[NodeAttrs]:
        """Terminal — :class:`NodeAttrs` for every surviving node, in
        the same order :meth:`indices` returns. Avoids the
        boilerplate of ``ctx.node_attrs(q.indices())``.
        """
        ...

    def first_idx(self) -> int | None:
        """Terminal — first matching node's positional index, or
        ``None`` when no node matches. Convenience for single-value
        lookups.
        """
        ...

    def indices_by_path(self) -> dict[str, list[int]]:
        """Terminal — group matched indices by their owning file
        path. One :meth:`ProjectContext.node_paths` call internally.
        Lets plugins fan out per-file work without re-querying.
        """
        ...

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

class ProgressHandle:
    """Borrow-free view over the build-progress atomic counters.

    Created by :meth:`ProjectContext.progress_handle` — the handle
    holds a shared ``Arc`` over the rust counters, so calling
    :meth:`snapshot` is GIL-bound but doesn't go through the pyo3
    borrow flag that ``materialize`` holds for the entire build. The
    Python polling thread driving
    :class:`dead_cst.Analysis`'s ``progress_callback`` API uses this
    handle to read counters concurrently with a long-running build.
    """

    def snapshot(self) -> "ProgressSnapshot":
        """Atomic snapshot of every counter as a plain Python dict
        (``int`` values; ``finished`` is ``bool``). Same key set as
        :meth:`ProjectContext.read_progress_snapshot`."""
        ...

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
