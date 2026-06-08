"""Type stubs for the ty-backed native graph builder.

Surfaces the pyo3 classes the rust crate exposes so callers in
``dead_cst/`` and ``tests/prototype/`` get static information without
having to introspect the binary module. Kept hand-written — pyo3
doesn't generate stubs and the public API is small.

The docstrings on each class / method are the contract: each one
mirrors the rust-side rustdoc, and a behavior change in the crate
should land in both places at once.
"""

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Sequence

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
    "external",
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

    UNRESOLVED: int
    """Import alias whose upstream module ty could not resolve (a bad
    relative-dots target, a missing dependency, …). Stamped on the local
    ``kind="import"`` node in place of the old ``[unresolved] X``
    sink node. Metadata only (not a seed); registered
    ``engine/unresolved``."""

    ENTRYPOINT: int
    """Explicit entrypoint — plugin-emitted seeds, the CLI's ``-e`` flag,
    ``[project.scripts]`` targets, factory-app entrypoints, etc. A seed
    flag (registered ``engine/entrypoint``), so reachability seeds from
    ``ENTRYPOINT``-flagged nodes by default."""

    NOQA: int
    """Import alias preserved by a user noqa directive (bare
    ``# noqa``, ``# noqa: F401``, multi-rule ``# noqa: E501, F401``, or
    the file-level ``# ruff: noqa`` / ``# flake8: noqa``). A seed flag
    (registered ``engine/noqa``)."""

    NOTEBOOK: int
    """Every node sourced from a Jupyter ``.ipynb`` file. Cells run
    top-to-bottom rather than being imported, so the bit alone keeps
    the node alive — a seed flag (registered ``engine/notebook``). The
    codemod also reads the bit to skip notebook nodes (it can't rewrite
    the cell JSON envelope)."""

    DEAD_BRANCH: int
    """Decl that sits in a statically-dead region (the body of
    ``if False:``, the else of ``if True:``, code after an
    unconditional ``return`` / ``raise``, …) — the node-level companion
    to ``EdgeFlags.DEAD_BRANCH``. Metadata only (not a seed); registered
    ``engine/dead_branch``."""

    STAR_REEXPORT: int
    """Per-name import decl synthesized from ``from X import *`` — one
    node per name the star statement brought in, keyed on ty's
    per-name ``StarImport`` definition so uses resolve straight to it.
    Each edges to the kept ``<mod>.*<src>`` statement node (which
    carries the single upstream-module edge and stays the unit the
    codemod removes). The codemod skips ``STAR_REEXPORT`` nodes
    themselves: they share the ``*`` statement's source range and have
    no removable span of their own."""

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

    INIT_SUBCLASS: int
    """Edge from a base class to a subclass discovered via
    ``__init_subclass__`` (the built-in init-subclass plugin emits one
    ``parent -> subclass`` edge per registered subclass). Keeps the
    subclass alive whenever the base is, without a separate anchor
    node. Metadata only — the edge participates in default reachability."""

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

class NativePlugin:
    """A rust-backed plugin — the only plugin mechanism.

    Every built-in plugin is a ``NativePlugin``, constructed via a
    static factory (one per bundled impl, e.g. :meth:`main_block`); the
    default constructor is intentionally not exposed. Pass instances to
    :class:`Analysis` in its ``plugins=`` list; the harness runs each
    plugin's rust impl directly during ``materialize()``.

    Out-of-tree plugins are *external* native plugins compiled against
    the shipped runtime dylib and loaded via
    :func:`load_native_plugins` (see ``NATIVE_PLUGINS.md``); they arrive
    as ``NativePlugin`` instances too. The underlying rust trait and op
    types are crate-private with no stability commitment — the stable
    surface is this class plus its factories.
    """

    @property
    def name(self) -> str:
        """Plugin name (e.g. ``"MainBlockPlugin"``). Used in harness
        logs and ``progress_callback`` events.
        """
        ...

    def prepare(self, project_root: Any) -> None:
        """Pre-graph hook. The harness calls it on every plugin before
        graph construction; it is forwarded to the underlying rust impl
        (a project-wide builtin or an external dylib plugin's
        ``ExternalPlugin::prepare``), so an external plugin can scan
        ``project_root`` for config up front. Per-file plugins are pure
        functions of their file and take no prepare step.
        """
        ...

    @staticmethod
    def main_block() -> NativePlugin:
        """``MainBlockPlugin``. For every file with a top-level
        ``if __name__ == "__main__":`` block, keeps the containing module
        node and every top-level decl inside the block alive as entrypoints.
        Implemented as a per-file (salsa-cached) plugin.
        """
        ...

    @staticmethod
    def init_subclass() -> NativePlugin:
        """``InitSubclassPlugin``. Keeps transitive subclasses of
        ``__init_subclass__``-defining classes alive via a marker node.
        """
        ...

    @staticmethod
    def server_config(filenames: Sequence[str] | None = None) -> NativePlugin:
        """Native ``ServerConfigPlugin``. Marks conventional WSGI/ASGI
        server-config modules (``gunicorn.conf.py``, ``hypercorn.conf.py``,
        …) as entrypoints, keeping each matched file's whole top-level
        surface alive.

        ``filenames`` defaults to the conventional Gunicorn/Hypercorn set;
        pass a custom sequence to match other server-config basenames.
        Per-file (salsa-cached): a file matches purely on its own basename,
        so an unchanged file's ops are reused across ``re_materialize``.
        Identical filename sets intern to one cache key.
        """
        ...

    @staticmethod
    def unittest() -> NativePlugin:
        """``UnittestPlugin``. Keeps stdlib ``unittest`` test classes
        (transitive subclasses of ``TestCase`` /
        ``IsolatedAsyncioTestCase``) and module lifecycle hooks alive.
        """
        ...

    @staticmethod
    def flask() -> NativePlugin:
        """Native Flask dispatch-app plugin. Finds ``app = Flask(...)``
        instances (and ``Blueprint``/``Api``/factory functions returning
        them), wires ``@app.route`` / ``@bp.cli.command`` / … handlers
        through each instance, and seeds the app itself as an entrypoint.
        """
        ...

    @staticmethod
    def fastapi() -> NativePlugin:
        """Native FastAPI dispatch-app plugin. Finds ``app = FastAPI(...)``
        / ``APIRouter(...)`` instances and wires ``@app.get`` / ``@app.post``
        / … route handlers through them. The ``FastAPI`` app seeds itself;
        a bare ``APIRouter`` only goes live once mounted/reached.
        """
        ...

    @staticmethod
    def typer() -> NativePlugin:
        """Native Typer dispatch-app plugin. Finds ``app = Typer(...)``
        instances and wires ``@app.command`` / ``@app.callback`` handlers
        through them. The app does **not** seed itself — it goes live only
        via a main block, ``[project.scripts]``, or an explicit entrypoint.
        """
        ...

    @staticmethod
    def cyclopts() -> NativePlugin:
        """Native Cyclopts dispatch-app plugin. Finds ``app = App(...)``
        instances and wires ``@app.command`` / ``@app.default`` handlers
        through them. Like Typer, the app does not seed itself.
        """
        ...

    @staticmethod
    def slack_bolt() -> NativePlugin:
        """Native Slack Bolt dispatch-app plugin. Finds ``App(...)`` /
        ``AsyncApp(...)`` instances and wires ``@app.event`` / ``@app.command``
        / ``@app.action`` / … listener handlers through them. Seeds the app.
        """
        ...

    @staticmethod
    def fastmcp() -> NativePlugin:
        """Native FastMCP dispatch-app plugin. Finds ``FastMCP(...)`` /
        ``Server(...)`` instances and wires ``@mcp.tool`` / ``@mcp.resource``
        / ``@mcp.prompt`` / ``@mcp.completion`` handlers through them. Seeds
        the app.
        """
        ...

    @staticmethod
    def celery() -> NativePlugin:
        """Native Celery dispatch-app plugin. Finds ``app = Celery(...)``
        instances and wires ``@app.task`` handlers through them, plus a
        module-wide ``@shared_task`` fan-out (shared tasks bind to no
        specific app instance). Seeds the app.
        """
        ...

    @staticmethod
    def dispatch_app(
        name: str,
        app_classes: Sequence[str],
        registration_decorators: Sequence[str],
        seed_as_entrypoint: bool,
    ) -> NativePlugin:
        """Build a dispatch-app plugin for a framework ``dead-cst`` doesn't
        bundle — the generalized form behind :meth:`flask` … :meth:`celery`.

        ``name`` labels the plugin in progress logs. ``app_classes`` are
        dotted fqnames of the application classes (e.g.
        ``["myframework.App"]``) whose instances — and transitive subclasses —
        anchor handler wiring; ``registration_decorators`` are the bare method
        names a handler is decorated with on such an instance
        (``@app.route`` → ``"route"``).

        When ``seed_as_entrypoint`` is true the discovered app instances (and
        factory functions returning them) are kept alive — the web/task
        default (Flask/FastAPI/Celery). Pass ``False`` for a pure-dispatch CLI
        (Typer/Cyclopts), where an unused app surfaces as dead. The
        Celery-style appless ``@shared_task`` fan-out is not exposed here; use
        :meth:`celery`.
        """
        ...

    @staticmethod
    def click() -> NativePlugin:
        """Native Click CLI plugin. Finds Click groups (functions decorated
        ``@click.group`` / ``@click.Group`` or ``X = click.Group(...)``
        constructions) and wires ``@<group>.command`` / ``@<group>.group`` /
        ``@<group>.result_callback`` handlers to their owning group. A
        ``@<group>.group()`` handler is itself promoted to a group via a
        fixpoint, so nested sub-commands wire transitively. Groups are *not*
        seeded as entrypoints — reach them through ``[project.scripts]`` /
        ``__main__`` / ``add_command``.
        """
        ...

    @staticmethod
    def mock_patch() -> NativePlugin:
        """Native mock-patch plugin. Resolves the string-fqname target of
        ``unittest.mock.patch`` / ``mock.patch``, pytest-mock's
        ``mocker.patch``, and pytest's ``monkeypatch.setattr`` /
        ``monkeypatch.delattr`` to its target decl (and module), wiring a
        direct keep-alive edge from each enclosing call site to every
        resolved target. An unresolved fqname keeps nothing alive.
        """
        ...

    @staticmethod
    def discordpy() -> NativePlugin:
        """Native discord.py plugin. Gated on a ``discord`` import. Seeds
        ``Bot`` / ``Client`` constructions as ``<discordpy-app>`` entrypoints,
        wires ``@<bot>.<verb>`` and ``@<bot>.tree.<verb>`` handler decorators
        to their owning instance, keeps ``commands.Cog`` subclasses plus their
        module-level ``setup`` / ``teardown`` hooks alive
        (``<discordpy-cog>:<file>``), and keeps the module surface of every
        ``load_extension`` / ``load_extensions`` string-literal target alive
        (``<discordpy-extension>:<fqname>``).
        """
        ...

    @staticmethod
    def pytest() -> NativePlugin:
        """Native pytest plugin. Flags every ``test_*`` function / ``Test*``
        class in ``test_*.py`` / ``*_test.py`` with the ``test/testcase`` node
        flag, and flags every ``@pytest.fixture`` function plus every
        top-level ``conftest.py`` decl with the provisional ``test/fixture``
        node flag (a conservative keepalive pending precise fixture/conftest
        usage modeling). Both are default-on seed flags. On top of the flags it
        wires ``test → fixture`` / ``class → fixture`` edges by parameter-name
        matching (honoring the ``name=`` kwarg alias), so the dependency graph
        is queryable. Unmatched parameter names (pytest builtins, plugin
        fixtures) are silently ignored.
        """
        ...

    @staticmethod
    def project_scripts(pyproject_path: str | None = None) -> NativePlugin:
        """Native ``[project.scripts]`` plugin. Reads ``pyproject.toml`` (from
        ``pyproject_path`` if given, else ``<project_root>/pyproject.toml``)
        and, for each ``name = "pkg.mod:func"`` entry, keeps the resolved
        ``pkg.mod.func`` decl alive as an entrypoint (falling back to the
        ``pkg.mod`` module node). A missing file or unresolved target is
        skipped silently.
        """
        ...

    @staticmethod
    def dynamic_import_fallback(
        *,
        include_underscore: bool = False,
        respect_dunder_all: bool = True,
        exclude_sources: Sequence[str] | None = None,
        exclude_targets: Sequence[str] | None = None,
        include_sources: Sequence[str] | None = None,
        include_targets: Sequence[str] | None = None,
    ) -> NativePlugin:
        """Native dynamic-import fan-out plugin. Fans every
        ``EdgeFlags.DYNAMIC_IMPORT`` edge that targets a module out to that
        module's exports (``__all__`` when ``respect_dunder_all`` and present,
        else top-level decls; ``_``-prefixed names dropped unless
        ``include_underscore``).

        ``include_sources`` / ``include_targets`` allowlist call-site paths
        (matched via ``PurePosixPath.match`` against the project-relative path)
        and target module fqnames (``fnmatch.fnmatchcase``); ``exclude_*``
        denylist them. When both are set an edge must match an ``include_*``
        AND no ``exclude_*``.
        """
        ...

    @staticmethod
    def explicit(
        regexes: Sequence[str],
        str_specs: Sequence[str],
        abs_paths: Sequence[str],
    ) -> NativePlugin:
        """Native explicit-entrypoint plugin. Marks every node matching one of
        the pre-bucketed specs as a ``<entrypoint>``: ``regexes`` match the
        project-relative file path, ``str_specs`` match an exact fqname or a
        project-relative file path, and ``abs_paths`` match an exact absolute
        path. The CLI buckets its ``str | Path | re.Pattern`` specs into these
        three lists before calling.
        """
        ...

def _builtin_native_plugin(name: str) -> NativePlugin | None:
    """Resolve a built-in plugin name (e.g. ``"main_block"``) to its
    native implementation, or ``None`` if no native plugin owns that
    name yet. The CLI consults this before the Python builtin map; as
    plugins are ported to Rust they move into this registry.
    """
    ...

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

class ChangeEvent:
    """A file-system change event consumed by
    :meth:`ProjectContext.apply_changes`. Construct via the
    classmethods :meth:`changed` / :meth:`created` / :meth:`deleted` /
    :meth:`rescan`; or get a list back from
    :meth:`ProjectContext.detect_changes` to autodetect what changed
    on disk since the last build."""

    @property
    def kind(self) -> str:
        """One of ``"changed"`` / ``"created"`` / ``"deleted"`` /
        ``"rescan"``."""

    @property
    def path(self) -> str | None:
        """The path the event refers to, or ``None`` for
        :meth:`rescan`."""

    @classmethod
    def changed(cls, path: str) -> "ChangeEvent":
        """File at ``path`` was modified (content or metadata)."""
        ...

    @classmethod
    def created(cls, path: str) -> "ChangeEvent":
        """File or directory at ``path`` was created."""
        ...

    @classmethod
    def deleted(cls, path: str) -> "ChangeEvent":
        """File or directory at ``path`` was deleted."""
        ...

    @classmethod
    def rescan(cls) -> "ChangeEvent":
        """Full-project rescan sentinel."""
        ...

class ProjectContext:
    """Plugin-aware project graph builder.

    Python instantiates a ``ProjectContext``, registers
    :class:`NativePlugin` instances via :meth:`add_plugin`, then calls
    :meth:`materialize`. Each plugin's rust impl runs against this same
    instance, emitting graph mutations the apply pass folds into the
    in-progress graph; an impl may call any of the ``find_*`` /
    ``decls_*`` / ``descendants_indices`` / ``ancestors_indices`` /
    ``reachable`` queries below.

    Queries answer against the live in-progress graph (so an op
    emitted earlier in the same plugin is visible to later queries).
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

    def add_plugin(self, plugin: NativePlugin) -> None:
        """Register a plugin. Order of registration is order of
        invocation during :meth:`materialize`."""
        ...

    def clear_plugins(self) -> None:
        """Drop every plugin registered via :meth:`add_plugin`. Used by
        :class:`dead_cst.Analysis` so a re-materialize doesn't
        double-register plugins on the rust-serial path."""
        ...

    def apply_changes(self, events: Iterable[ChangeEvent]) -> bool:
        """Apply a batch of file-system change events to the salsa db.

        Forwards to ty_project's ``ProjectDatabase::apply_changes``,
        which handles each variant correctly:

        * ``Changed`` — bumps the file's revision iff mtime / size
          differ; otherwise no-op.
        * ``Created`` — registers the path with the project file set
          so brand-new files are visible on next enumeration.
        * ``Deleted`` — removes the file from the project set.
        * ``Rescan`` — triggers a full ``sync_all`` + project re-walk
          + metadata rediscovery.

        Project configuration files (``pyproject.toml``, ignore files,
        custom-stdlib ``VERSIONS``) are detected automatically and
        trigger a project reload.
        """
        ...

    def detect_changes(self) -> list[ChangeEvent]:
        """Return :class:`ChangeEvent`\\s that, when fed to
        :meth:`apply_changes`, bring the salsa db in sync with the
        current on-disk state.

        Today this returns a single ``ChangeEvent.rescan()``; the
        underlying ty rescan handler does an mtime-checked
        ``Files::sync_all`` (so per-file salsa caches survive when
        the file content hasn't changed), a project file re-walk
        (so new files are discovered and deleted ones dropped), and
        a metadata rediscovery (so config changes take effect).
        """
        ...

    def reset_progress(self) -> None:
        """Reset the progress counter state for a re-run.

        Replaces the rust-side ``ProgressCounters`` with a fresh
        instance so a subsequent :meth:`materialize` call starts from
        zero. Without it, a poller spun up for the second build would
        observe ``finished=true`` from the first build and exit
        immediately.
        """
        ...

    def tombstoned_indices(self) -> list[int]:
        """Sorted dense node indices tombstoned by incremental
        re-mints — slots whose file block was replaced by a later
        ``re_materialize``. The slots stay in place (live indices never
        remap) but are dead: blanked node data, zero flags, no edges,
        excluded from every query. Empty after a full build, which
        compacts the id space (and runs automatically once tombstones
        outnumber live nodes)."""
        ...

    def _last_resolve_counts(self) -> tuple[int, int]:
        """``(resolved, reused)`` cross-file resolution counts from the
        most recent build — the observability hook for the resolve
        cache's incremental reuse (a full build reports ``reused == 0``;
        a ``re_materialize`` after a small content-only edit should
        report ``resolved`` close to the edit's blast radius).
        Diagnostic only — not part of the supported surface.
        """
        ...

    def materialize(self) -> None:
        """Build the project-wide graph, run every registered plugin,
        and leave the live graph on this context (query it via
        ``nodes()`` / ``edges()`` / the index-returning queries — no
        ``NativeGraph`` snapshot is materialized; that cost one
        ``Py<SymbolNode>`` per node on every rebuild).
        """
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

    # ----- FQN resolution ------------------------------------------------
    #
    # Flat idx-form lookups: fqname → decl, module fqname / path →
    # module idx, surface / top-level / dunder-all transforms, and
    # module-scope string-literal lists. Plugins and
    # :class:`dead_cst.Analysis` call these directly.

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
        ``list(...)``, etc.). Targeted read for literal-list keep-alive
        idioms, independent of the visitor's ``__all__``-only
        string-list edge emission.
        """
        ...

    # ----- Path / name filters ------------------------------------------
    #
    # Idx-form path / name / spec matchers, consumed by the entrypoint
    # and discovery plugins and by :class:`dead_cst.Analysis`.

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

    def descendants_indices(self, root_idx: int, *, skip_flags: int = 0) -> list[int]:
        """Forward closure: every node reachable from ``root_idx`` by
        following graph edges. Takes a positional index into
        :meth:`nodes` and returns descendant indices.

        ``skip_flags`` filters out edges whose flag mask intersects —
        pass ``EdgeFlags.DEAD_BRANCH`` to compute strict reachability
        excluding dead branches. Raises :class:`IndexError` when
        ``root_idx`` is out of range.
        """
        ...

    def ancestors_indices(self, decl_idx: int, *, skip_flags: int = 0) -> list[int]:
        """Reverse closure: every node that can reach ``decl_idx`` by
        following graph edges. Takes a positional index into
        :meth:`nodes` and returns ancestor indices. Used for
        predecessor-chain walks and blast-radius scoping.

        ``skip_flags`` works the same as in
        :meth:`descendants_indices`. Raises :class:`IndexError` when
        ``decl_idx`` is out of range.
        """
        ...

    def direct_predecessors_idx(self, idx: int, *, skip_flags: int = 0) -> list[int]:
        """One-hop reverse step: every node with an edge directly into
        ``idx``. Dedups by source idx, so a pair of parallel edges
        with different :class:`EdgeFlags` between the same two nodes
        only produces one entry. ``skip_flags`` filters edges by
        intersecting flag mask — same semantics as
        :meth:`ancestors_indices`.
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
        :meth:`nodes`. A single-call attribute predicate — compose as
        many of the parameters as you need when you just want a
        ``list[int]``.

        Every parameter is keyword-only and optional; unset arguments
        don't filter. ``kind`` / ``kinds`` (and similarly the
        ``filename`` / ``simple_name`` pairs) are merged; pass either
        form. All set predicates AND together. ``flags`` is the
        all-bits-set form (``node.flags & mask == mask``);
        ``flags_any`` is the any-bit form (``node.flags & mask != 0``).
        """
        ...

    def nodes_at(self, indices: Sequence[int]) -> list[SymbolNode]:
        """Inverse of the ``*_indices`` queries: materialize specific
        nodes by their positional indices into :meth:`nodes`.
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

        Used by the native pytest plugin (``NativePlugin.pytest()``) to
        discover ``test_foo(my_fixture)`` → ``test_foo → my_fixture`` edges.

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

        Used by the native pytest plugin (``NativePlugin.pytest()``) to
        wire ``class → fixture`` edges for ``Test*`` classes — class
        methods aren't represented as their own graph nodes, so the
        class itself is the rendezvous point for any fixture any
        method uses.

        Validates bounds and raises :class:`IndexError` when any
        index is out of range.
        """
        ...

    # ----- Pure scans over the in-progress graph ------------------------
    #
    # Idx-form scans over already-interned nodes — module-dunder
    # enumeration and ``if __name__ == "__main__":`` block discovery.

    def find_module_dunders_indices(self) -> list[int]:
        """Every top-level variable / function node whose name matches
        ``__xxx__``. Pure scan over already-interned nodes — no ty
        re-query needed."""
        ...

    def find_main_blocks_indices(self) -> list[tuple[int, list[int]]]:
        """``(module_idx, [decl_idx])`` pairs into :meth:`nodes` for
        every file with a top-level ``if __name__ == "__main__":``
        block. The decls list contains the file's class / function /
        variable / import nodes whose source position falls inside the
        block's range.
        """
        ...

    # ----- Decorator / construction shapes (syntactic walks) ------------
    #
    # ``find_factory_decls`` here, plus the rust-internal decorator /
    # construction / call walks the native plugins drive, match upstream
    # callables / classes by their *bare name* combined with a per-file
    # import check — they deliberately do **not** route through ty's
    # module resolver.
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
    #
    # The decorator / construction / call walks (``find_decorated_decls``,
    # ``find_instance_constructions``, ``find_handler_decorators``,
    # ``find_calls_on_attr``, ``find_calls_to_imported``,
    # ``find_calls_on_var``) are rust-internal — the native plugins call
    # them directly, so they're not part of this stub. ``find_factory_decls``
    # is the one shape still surfaced to Python.

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

        The only caller is the native dispatch-app impl
        (``NativePlugin.flask()`` and friends), which uses this shape
        directly.
        """
        ...

    # ----- Read-only graph accessors ------------------------------------

    def nodes(self) -> list[SymbolNode]:
        """Live nodes in the in-progress graph. Cheap, no copy."""
        ...

    def edges(self) -> list[tuple[int, int, int]]:
        """Live edges as ``(src_idx, dst_idx, flags)`` triples."""
        ...

    # ----- Flag registries ----------------------------------------------
    #
    # The engine seeds both registries with its built-in flags at
    # construction; plugins extend the node / edge registries during
    # :meth:`materialize`. The getters decode bits by name and feed the
    # registry-derived keepalive seed the Python layer uses.

    def node_flag_registry(self) -> list[tuple[str, int, bool, bool, str]]:
        """Every registered node flag as
        ``(name, bit, seed, default_on, description)`` tuples, sorted by
        bit. Includes the engine built-ins plus any flag a registered
        plugin declared via ``declare_node_flags``."""
        ...

    def edge_flag_registry(self) -> list[tuple[str, int, bool, bool, str]]:
        """Every registered edge flag, same tuple shape as
        :meth:`node_flag_registry`."""
        ...

    def default_seed_mask(self) -> int:
        """OR of every node-flag bit whose flag both seeds reachability
        and is on by default — the registry-derived replacement for the
        old ``KEEPALIVE_DEFAULT`` constant. The default ``seed_flags`` for
        :class:`dead_cst.Analysis` reachability queries."""
        ...

    def node_flag(self, name: str) -> int | None:
        """Resolve a node-flag bit by its registered ``owner/name`` (e.g.
        ``"engine/entrypoint"``, ``"test/testcase"``), or ``None`` when no
        flag with that name is registered."""
        ...

    def edge_flag(self, name: str) -> int | None:
        """Resolve an edge-flag bit by its registered ``owner/name``, or
        ``None`` when no flag with that name is registered."""
        ...

    # ----- Topic / fact registry --------------------------------------
    #
    # Topics are a plugin-only channel for a per-file plugin to publish
    # facts to its project-wide reader. Plugins declare topics via
    # ``declare_topics``; the host assigns handles during
    # :meth:`materialize`. Unlike the flag registries there are no engine
    # built-ins, and the registry is not serialized into the graph file —
    # facts are an in-memory build-time side channel.

    def topic_registry(self) -> list[tuple[str, int, str]]:
        """Every registered topic as ``(name, handle, description)`` tuples
        in handle (registration) order: the topics registered plugins
        declared via ``declare_topics`` (empty before :meth:`materialize`)."""
        ...

    def facts_for_topic(self, name: str) -> list[tuple[str, int | None, str]]:
        """Every fact published under topic ``name`` across the project, as
        ``(path, decl_idx, value)`` tuples. ``path`` is the publishing file;
        ``decl_idx`` is the global node index the fact was pinned to (or
        ``None``); ``value`` is the plugin-defined payload. Empty for an
        unknown topic or one nothing published under. Raises if called
        before :meth:`materialize`."""
        ...

# ---------- Node-attribute rows ------------------------------------------

class NodeAttrs:
    """Tuple-like row returned by :meth:`ProjectContext.node_attrs`.

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

# ---------- Graph persistence --------------------------------------------

class GraphMetadata:
    """Header block returned by :func:`read_graph` alongside the graph.

    ``created_at`` is unix-epoch seconds at write time. ``user_meta``
    is the list of ``(key, value)`` pairs the writer passed (from
    ``dead-cst build --meta key=value``). ``node_flag_registry`` /
    ``edge_flag_registry`` carry the flag tables stamped into the file
    (``(name, bit, seed, default_on, description)`` tuples) so a reader
    can decode flag bits by name. The counts mirror the graph stamped
    into the file and double as a sanity check against the deserialized
    payload.
    """

    format_version: int
    created_at: int
    node_count: int
    edge_count: int
    file_count: int
    line_count: int
    user_meta: list[tuple[str, str]]
    node_flag_registry: list[tuple[str, int, bool, bool, str]]
    edge_flag_registry: list[tuple[str, int, bool, bool, str]]

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
    node_flag_registry: list[tuple[str, int, bool, bool, str]],
    edge_flag_registry: list[tuple[str, int, bool, bool, str]],
) -> None:
    """Persist ``nodes`` + ``edges`` to ``path`` as a bincode-encoded
    graph file. ``meta`` is the user-supplied list of ``(key, value)``
    pairs (from ``--meta`` on the CLI); they are stored verbatim in
    the file's metadata block. ``node_flag_registry`` /
    ``edge_flag_registry`` are the flag tables
    (``(name, bit, seed, default_on, description)`` tuples) so a reader
    can decode flag bits by name.
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

def _main_block_run_count() -> int:
    """Test helper. Total executions of the per-file ``MainBlockPlugin``
    impl since the last reset — a salsa cache *miss* counter. Lets the
    test suite assert an unchanged main-block file isn't re-run on
    ``re_materialize``. Not part of the supported surface.
    """
    ...

def _reset_main_block_run_count() -> None:
    """Test helper. Zero the :func:`_main_block_run_count` counter."""
    ...

def _server_config_run_count() -> int:
    """Test helper. Total executions of the configured per-file
    ``ServerConfigPlugin`` impl since the last reset — a salsa cache *miss*
    counter. Lets the test suite assert an unchanged server-config file
    isn't re-run on ``re_materialize`` and that distinct filename configs
    key separately. Not part of the supported surface.
    """
    ...

def _reset_server_config_run_count() -> None:
    """Test helper. Zero the :func:`_server_config_run_count` counter."""
    ...
