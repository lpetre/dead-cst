"""Type stubs for the ty-backed native graph builder.

Surfaces the pyo3 classes the rust crate exposes so callers in
``dead_cst/`` and ``tests/prototype/`` get static information without
having to introspect the binary module. Kept hand-written — pyo3
doesn't generate stubs and the public API is small.
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

class Import:
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
    """Edge from ``src`` to ``dst``. ``flags`` carries DEAD_BRANCH /
    future edge classifications."""

    src: NativeNode
    dst: NativeNode
    flags: int

    def __init__(self, src: NativeNode, dst: NativeNode, *, flags: int = 0) -> None: ...

class AddEntrypoint:
    """Mark ``decl`` as an entrypoint. ``marker`` is a self-documenting
    label for ``why-alive`` (e.g. ``"<celery-worker>"``)."""

    decl: NativeNode
    marker: str

    def __init__(self, decl: NativeNode, *, marker: str) -> None: ...

class AddNode:
    """Mint a synthetic intermediate node. Use rarely — ``AddEntrypoint``
    covers the common "this is alive because of X" pattern without
    needing a graph node for the reason."""

    fqname: str
    kind: NodeKind
    path: str

    def __init__(self, fqname: str, *, path: str, kind: NodeKind = "synthetic") -> None: ...

GraphOp = AddEdge | AddEntrypoint | AddNode

class NativeGraph:
    nodes: list[NativeNode]
    edges: list[tuple[int, int, int]]

class Project:
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
    def build(self) -> NativeGraph: ...

class _ProjectPluginLike(Protocol):
    def run(self, ctx: "ProjectContext") -> Iterable[GraphOp] | None: ...

class ProjectContext:
    project_root: str

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
    def add_plugin(self, plugin: _ProjectPluginLike | Any) -> None: ...
    def materialize(self) -> NativeGraph: ...
    def add_node(
        self,
        fqname: str,
        path: str,
        *,
        kind: str = ...,
        start_line: int = ...,
        start_column: int = ...,
        end_line: int = ...,
        end_column: int = ...,
        flags: int = ...,
    ) -> NativeNode: ...
    def add_edge(self, src: NativeNode, dst: NativeNode) -> None: ...
    def resolve(self, fqname: str) -> NativeNode | None: ...
    def module_surface(self, module_fqn: str) -> list[NativeNode]: ...
    def decls_under(self, path_prefix: str) -> list[NativeNode]: ...
    def decls_matching(self, substring: str) -> list[NativeNode]: ...
    def decls_matching_name(self, pattern: str) -> list[NativeNode]: ...
    def descendants(self, root: NativeNode, *, skip_flags: int = 0) -> list[NativeNode]: ...
    def ancestors(self, decl: NativeNode, *, skip_flags: int = 0) -> list[NativeNode]: ...
    def reachable(self, *, skip_flags: int = 0) -> list[NativeNode]: ...
    def find_module_dunders(self) -> list[NativeNode]: ...
    def find_imports_of(self, module_name: str) -> list[NativeNode]: ...
    def find_declarations(self, fqname: str) -> list[NativeNode]: ...
    def find_module(self, fqname: str) -> NativeNode | None: ...
    def module_for(self, path: str) -> NativeNode | None: ...
    def find_main_blocks(self) -> list[tuple[NativeNode, list[NativeNode]]]: ...
    def find_classes_defining_method(self, method_name: str) -> list[NativeNode]: ...
    def find_decorated_decls(
        self, decorator_module: str, decorator_names: list[str]
    ) -> list[NativeNode]: ...
    def find_instance_constructions(
        self, module: str, ctor_names: list[str]
    ) -> list[tuple[NativeNode, str]]: ...
    def find_handler_decorators(
        self, decorator_attrs: list[str]
    ) -> list[tuple[str, NativeNode]]: ...
    def find_factory_decls(
        self, module: str, ctor_names: list[str]
    ) -> list[tuple[NativeNode, list[str]]]: ...
    def find_calls_to_imported(
        self, module: str, name: str, arg_index: int
    ) -> list[tuple[NativeNode, str]]: ...
    def find_calls_on_var(
        self,
        owner: str,
        attr: str,
        arg_index: int,
        *,
        required_positional: int | None = ...,
    ) -> list[tuple[NativeNode, str]]: ...
    def find_subclasses_of(self, class_node: NativeNode) -> list[NativeNode]: ...
    def find_comment_patterns(self, pattern: str) -> list[tuple[NativeNode, str]]: ...
    def nodes(self) -> list[NativeNode]: ...
    def edges(self) -> list[tuple[int, int, int]]: ...
