"""Plugin: keep stdlib ``unittest`` test classes and lifecycle hooks alive.

Two-phase: ``observe`` walks every class def in every file, resolves
each base expression to a fqname (using the file's local imports +
same-file decls), and emits a ``<unittest:base-of>:<base_fqname>``
bucket marker pointing at the subclass. ``finalize`` walks the graph
from ``unittest.TestCase`` / ``IsolatedAsyncioTestCase`` -- and every
import alias of those -- through the bucket chain to collect the
transitive subclass closure. Each discovered subclass becomes a test
entrypoint.

This handles three real-world patterns the prior single-phase plugin
missed:

* **Direct subclass:** ``class MyTest(unittest.TestCase)`` -- caught by
  the ``observe`` bucket emission and finalize's BFS from
  ``unittest.TestCase``.
* **Project-local mixin:** ``class ProjectTC(unittest.TestCase)``,
  then ``class MyTest(ProjectTC)`` -- caught because ``ProjectTC``'s
  bucket points to it, and a second-level bucket keyed on
  ``ProjectTC``'s fqname points to ``MyTest``.
* **Re-exported base:** ``from unittest import TestCase`` in
  ``pkg.bases``, then ``from pkg.bases import TestCase; class
  MyTest(TestCase)`` -- caught because finalize iteratively expands
  the alias set through the graph's import-edge successors, so
  ``pkg.bases.TestCase`` is treated as a unittest base too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import libcst as cst

from ..graph import NodeFlags, SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    AddNode,
    GraphOp,
    ObserveContext,
    PluginContext,
    make_payload,
    module_node,
    payload_imports_module,
    simple_name,
    synthetic_node,
)
from ..plugins.init_subclass import (
    _local_class_decls,
    _local_import_targets,
    _resolve_bases,
)

if TYPE_CHECKING:
    from libcst.metadata import CodeRange

    from ..graph import VisitorPayload

UNITTEST_PREFIX = "<unittest>:"
UNITTEST_BASE_PREFIX = "<unittest:base-of>:"

_MODULE_HOOKS: frozenset[str] = frozenset({"setUpModule", "tearDownModule", "load_tests"})

_UNITTEST_BASE_FQNAMES: frozenset[str] = frozenset(
    {"unittest.TestCase", "unittest.IsolatedAsyncioTestCase"}
)


@dataclass
class UnittestPlugin:
    """Mark stdlib ``unittest`` discoveries as entrypoints.

    ``observe`` (per file):

    * ``setUpModule`` / ``tearDownModule`` / ``load_tests`` functions
      in any file that imports ``unittest`` are wired to a
      ``<unittest>:<module_fqname>`` entrypoint synth.
    * Every top-level class def in the project gets a
      ``<unittest:base-of>:<base_fqname>`` bucket marker per resolvable
      base, with an edge from the bucket to the subclass. Buckets share
      a project-root path so identical-fqname buckets dedupe across
      files into a single graph node.

    ``finalize`` (per package):

    * Compute the set of unittest aliases: start with
      ``unittest.TestCase`` / ``IsolatedAsyncioTestCase``, then expand
      iteratively using each import node's raw ``Import`` metadata --
      any import whose ``module.decl`` target is already an alias
      becomes one too. The raw metadata works across the stdlib
      boundary that ``unittest.TestCase`` sits on (it never appears as
      a graph node, so successor-walking would dead-end).
    * BFS the bucket graph from each alias, collecting transitive
      subclasses. Each subclass under the current package is wired to
      its module's ``<unittest>:<module_fqname>`` entrypoint synth
      (reused from the hooks branch if present, otherwise created).

    Limitations:

    * Bases written as calls (``class X(make_base())``) are skipped --
      same as :class:`InitSubclassPlugin`.
    * ``from unittest import *`` is invisible to the base resolver:
      the visitor doesn't bind individual names from a star import,
      so ``class X(TestCase)`` after ``from unittest import *`` does
      not resolve. Use ``from unittest import TestCase`` instead.
    """

    name: str = "unittest"
    version: int = 1778248994

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        module = module_node(ctx.payload)
        if module is None:
            return None

        new_nodes: list[SymbolNode] = []
        new_edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        if payload_imports_module(ctx.payload, "unittest", include_star=True):
            hook_decls = _find_module_hook_decls(ctx.module, ctx.payload.nodes)
            if hook_decls:
                synth = synthetic_node(
                    f"{UNITTEST_PREFIX}{module.fqname}",
                    ctx.path,
                    flags=NodeFlags.ENTRYPOINT | NodeFlags.TESTCASE,
                )
                new_nodes.append(synth)
                new_edges.extend((synth, h, SYNTHETIC_POSITION) for h in hook_decls)

        nodes_by_simple = _local_class_decls(ctx.payload.nodes)
        local_imports = _local_import_targets(ctx.payload.nodes)
        for stmt in ctx.module.body:
            if not isinstance(stmt, cst.ClassDef):
                continue
            class_decls = nodes_by_simple.get(stmt.name.value, [])
            if not class_decls:
                continue
            base_fqnames = _resolve_bases(stmt, local_imports, nodes_by_simple, module.fqname)
            for class_decl in class_decls:
                for base_fqname in base_fqnames:
                    bucket = synthetic_node(
                        f"{UNITTEST_BASE_PREFIX}{base_fqname}", ctx.project_root
                    )
                    new_nodes.append(bucket)
                    new_edges.append((bucket, class_decl, SYNTHETIC_POSITION))

        if not new_nodes:
            return None
        return make_payload(nodes=new_nodes, edges=new_edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        aliases = _expand_aliases(ctx, _UNITTEST_BASE_FQNAMES)

        buckets_by_base: dict[str, SymbolNode] = {}
        for node in ctx.graph.nodes:
            if node.type == "synthetic" and node.fqname.startswith(UNITTEST_BASE_PREFIX):
                # Buckets share project_root path, so identical-fqname
                # nodes dedupe to one entry; setdefault preserves the
                # first one seen even if duplicates slipped in.
                buckets_by_base.setdefault(node.fqname[len(UNITTEST_BASE_PREFIX) :], node)

        subclasses: set[SymbolNode] = set()
        stack: list[str] = list(aliases)
        while stack:
            fq = stack.pop()
            bucket = buckets_by_base.get(fq)
            if bucket is None:
                continue
            for sub in ctx.graph.successors(bucket):
                if sub.type != "class" or sub in subclasses:
                    continue
                subclasses.add(sub)
                # Recurse: a subclass of an alias is itself an alias --
                # its own subclasses are also test cases.
                stack.append(sub.fqname)

        package_path = ctx.package.path
        by_module_path: dict[Path, list[SymbolNode]] = {}
        for sub in subclasses:
            if not sub.path.is_relative_to(package_path):
                continue
            by_module_path.setdefault(sub.path, []).append(sub)

        if not by_module_path:
            return

        modules_by_path = {
            n.path: n for n in ctx.graph.nodes if n.type == "module" and n.path in by_module_path
        }
        existing_synths = {
            n.fqname: n
            for n in ctx.graph.nodes
            if n.type == "synthetic" and n.fqname.startswith(UNITTEST_PREFIX)
        }

        for path, subs in by_module_path.items():
            module = modules_by_path.get(path)
            if module is None:
                continue
            synth_fqname = f"{UNITTEST_PREFIX}{module.fqname}"
            synth = existing_synths.get(synth_fqname)
            if synth is None:
                synth = synthetic_node(
                    synth_fqname,
                    path,
                    flags=NodeFlags.ENTRYPOINT | NodeFlags.TESTCASE,
                )
                yield AddNode(synth, entrypoint=True, testcase=True)
            for sub in subs:
                yield AddEdge(synth, sub)


def _expand_aliases(ctx: PluginContext, seeds: frozenset[str]) -> set[str]:
    """Return ``seeds`` plus every import-node fqname that transitively
    resolves to one of them.

    Uses the raw :class:`Import` metadata on each import node (the
    dotted ``module``/``decl`` pair literally written in source), not
    graph successors -- the unittest base is stdlib and therefore
    never appears as a graph node, so successor-walking would dead-end
    at the first re-export. Iterates until quiescent so chains like
    ``a.X <- b.X <- unittest.TestCase`` fold every link into the
    alias set.
    """
    aliases: set[str] = set(seeds)
    import_nodes = [n for n in ctx.graph.nodes if n.type == "import" and n.imports is not None]
    changed = True
    while changed:
        changed = False
        for node in import_nodes:
            if node.fqname in aliases:
                continue
            assert node.imports is not None
            target = (
                f"{node.imports.module}.{node.imports.decl}"
                if node.imports.decl
                else node.imports.module
            )
            if target in aliases:
                aliases.add(node.fqname)
                changed = True
    return aliases


def _find_module_hook_decls(module: cst.Module, nodes: Iterable[SymbolNode]) -> list[SymbolNode]:
    hook_names = {
        stmt.name.value
        for stmt in module.body
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value in _MODULE_HOOKS
    }
    if not hook_names:
        return []
    return [n for n in nodes if n.type == "function" and simple_name(n.fqname) in hook_names]
