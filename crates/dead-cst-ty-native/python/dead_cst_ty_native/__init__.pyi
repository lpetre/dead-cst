"""Type stubs for the ty-backed native graph builder.

Surfaces the pyo3 classes the rust crate exposes so callers in
``dead_cst/`` and ``tests/prototype/`` get static information without
having to introspect the binary module. Kept hand-written — pyo3
doesn't generate stubs and the public API is small.
"""

from typing import Any, Iterable, Protocol

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
    kind: str
    path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    flags: int
    imports: Import | None

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
    def run(self, ctx: "ProjectContext") -> None: ...

class ProjectContext:
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
    def find_module_dunders(self) -> list[NativeNode]: ...
    def find_imports_of(self, module_name: str) -> list[NativeNode]: ...
    def find_classes_defining_method(self, method_name: str) -> list[NativeNode]: ...
    def find_subclasses_of(self, class_node: NativeNode) -> list[NativeNode]: ...
    def find_comment_patterns(self, pattern: str) -> list[tuple[NativeNode, str]]: ...
    def nodes(self) -> list[NativeNode]: ...
    def edges(self) -> list[tuple[int, int, int]]: ...
