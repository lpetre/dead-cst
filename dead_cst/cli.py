"""Command-line interface for dead-cst."""

from __future__ import annotations

import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import networkx as nx
import typer

from . import build_symbol_graph, count_nodes, find_reachable, order_paths, remove_code
from ._branches import is_unreachable_node
from ._plugins import (
    EdgePlugin,
    ExplicitEntrypointPlugin,
    ModuleDundersPlugin,
    load_plugin,
)
from ._plugins._core import EXTERNAL_PREFIXES
from ._plugins.module_dunders import DUNDER_PREFIX
from ._resolvers import (
    ManualResolver,
    PathMap,
    PathResolver,
    load_resolver,
    merge_paths,
)
from ._symbols import SymbolNode


app = typer.Typer(help="Dead code analysis for Python using libcst.")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def parse_entrypoint(ep: str) -> str | re.Pattern[str]:
    if ep.startswith("re:"):
        return re.compile(ep[3:])
    return ep


def _rel_path(path: Path, root: Path) -> Path:
    """``path`` made relative to ``root`` if possible, else ``path`` unchanged."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def build_plugins(
    *,
    entrypoints: list[str],
    plugin_names: list[str],
) -> list[EdgePlugin]:
    """Compose the plugin list from CLI flags.

    Order: user-specified plugins first, then ``ModuleDundersPlugin``, then
    ``ExplicitEntrypointPlugin`` with the ``-e`` specs. ``-e`` runs last so
    it can hang entrypoints off any synthetic nodes contributed upstream.
    """
    plugins: list[EdgePlugin] = []
    for name in plugin_names:
        plugins.append(load_plugin(name))
    plugins.append(ModuleDundersPlugin())
    if entrypoints:
        specs = [parse_entrypoint(ep) for ep in entrypoints]
        plugins.append(ExplicitEntrypointPlugin(specs=specs))
    return plugins


def resolve_paths(
    root: Path, path_specs: list[str], resolver_names: list[str]
) -> tuple[PathMap, list[PathResolver]]:
    """Build the resolver chain from ``-p`` specs and ``--resolver`` names.

    ``-p`` specs become a :class:`ManualResolver` and slot into the
    chain alongside named resolvers, so explicit specs participate in
    :meth:`PathResolver.resolve_import` lookups too. Returns the
    merged :data:`PathMap` and the resolver instances; the analyzer
    threads the latter through to govern import resolution.
    """
    resolvers: list[PathResolver] = []
    if path_specs:
        resolvers.append(ManualResolver(specs=path_specs))
    resolvers.extend(load_resolver(name) for name in resolver_names)
    if not resolvers:
        return {root: []}, []
    return merge_paths(*[r.resolve(root) for r in resolvers]), resolvers


def version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"dead-cst {version('dead-cst')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Dead code analysis for Python using libcst."""


@app.command()
def analyze(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    entrypoint: Annotated[
        list[str] | None,
        typer.Option(
            "-e",
            "--entrypoint",
            help="Entrypoint: file path, FQN, or 're:pattern' for regex.",
        ),
    ] = None,
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        list[str] | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.text,
) -> None:
    """Analyze a Python codebase for dead code."""
    setup_logging(verbose)
    root = root.resolve()

    paths_dict, resolvers = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, resolvers=resolvers, project_root=root)
    reachable = find_reachable(graph)

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    if output_format == OutputFormat.json:
        _output_json(graph, unreachable_graph, root, paths_dict)
    else:
        _output_text(graph, unreachable_graph, root, paths_dict)

    if unreachable_graph.number_of_nodes() > 0:
        raise typer.Exit(1)


def _output_text(
    graph: nx.DiGraph,
    unreachable: nx.DiGraph,
    root: Path,
    paths_dict: PathMap,
) -> None:
    for base in order_paths(paths_dict):
        typer.echo(f"\n{base}:")
        total_counts = count_nodes(graph, base)
        unreachable_counts = count_nodes(unreachable, base)
        for kind in sorted(total_counts):
            # Synthetic nodes (entrypoint sentinels, dead-suite markers)
            # don't represent user-visible declarations; reporting them
            # as "dead" alongside functions/classes would be misleading.
            # Dead-suite nodes get their own section below.
            if kind == "synthetic":
                continue
            total = total_counts[kind]
            dead = unreachable_counts.get(kind, 0)
            if dead > 0:
                typer.echo(f"  {kind}: {total} total, {dead} dead")
            else:
                typer.echo(f"  {kind}: {total} total")

    dead_real, branches = _partition_unreachable(unreachable)
    if dead_real:
        typer.echo(f"\nDead symbols ({len(dead_real)}):")
        for node in sorted(dead_real, key=lambda n: (str(n.path), n.fqname)):
            typer.echo(f"  {node.fqname} ({node.type}) at {_rel_path(node.path, root)}")

    if branches:
        typer.echo(f"\nUnreachable branches ({len(branches)}):")
        for node in sorted(branches, key=lambda n: (str(n.path), n.position.start)):
            rel = _rel_path(node.path, root)
            start = node.position.start
            end = node.position.end
            typer.echo(f"  {rel}:{start.line}:{start.column}-{end.line}:{end.column}")


def _partition_unreachable(
    unreachable: nx.DiGraph,
) -> tuple[list[SymbolNode], list[SymbolNode]]:
    """Split ``unreachable.nodes`` into ``(dead_real, branches)`` in one pass.

    Synthetic dead-suite nodes are orphan sources -- never visited by
    reachability -- so they always land in ``unreachable``.
    """
    dead_real: list[SymbolNode] = []
    branches: list[SymbolNode] = []
    for n in unreachable.nodes:
        (branches if is_unreachable_node(n) else dead_real).append(n)
    return dead_real, branches


def _output_json(
    graph: nx.DiGraph,
    unreachable: nx.DiGraph,
    root: Path,
    paths_dict: PathMap,
) -> None:
    result: dict = {
        "summary": {},
        "dead_symbols": [],
        "unreachable_branches": [],
    }

    for base in order_paths(paths_dict):
        base_str = str(base)
        total_counts = count_nodes(graph, base)
        unreachable_counts = count_nodes(unreachable, base)
        # Same rationale as the text output: synthetic nodes are reported
        # via ``unreachable_branches`` (and entrypoint sentinels), not as
        # part of the per-kind summary.
        result["summary"][base_str] = {
            kind: {"total": total_counts[kind], "dead": unreachable_counts.get(kind, 0)}
            for kind in total_counts
            if kind != "synthetic"
        }

    dead_real, branches = _partition_unreachable(unreachable)
    for node in sorted(dead_real, key=lambda n: (str(n.path), n.fqname)):
        result["dead_symbols"].append(
            {
                "fqname": node.fqname,
                "type": node.type,
                "path": str(_rel_path(node.path, root)),
            }
        )

    for node in sorted(branches, key=lambda n: (str(n.path), n.position.start)):
        result["unreachable_branches"].append(
            {
                "path": str(_rel_path(node.path, root)),
                "start": {"line": node.position.start.line, "column": node.position.start.column},
                "end": {"line": node.position.end.line, "column": node.position.end.column},
            }
        )

    typer.echo(json.dumps(result, indent=2))


@app.command("why-alive")
def why_alive(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    fqname: Annotated[str, typer.Argument(help="Fully qualified name of the symbol to check.")],
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        list[str] | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Show why a symbol is considered alive (reachable)."""
    setup_logging(verbose)
    root = root.resolve()

    paths_dict, resolvers = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=[],
        plugin_names=plugin or [],
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, resolvers=resolvers, project_root=root)

    target_node: SymbolNode | None = None
    for node in graph.nodes:
        if node.fqname == fqname:
            target_node = node
            break

    if target_node is None:
        typer.echo(f"Symbol not found: {fqname}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nSymbol: {target_node.fqname} ({target_node.type})")
    typer.echo(f"Path: {_rel_path(target_node.path, root)}")
    typer.echo(f"In-degree: {graph.in_degree(target_node)}")
    typer.echo("\nPredecessor chain:")

    seen: set[SymbolNode] = set()
    stack = [target_node]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        typer.echo(f"  <- {node.fqname} ({node.type}) at {_rel_path(node.path, root)}")
        stack.extend(graph.predecessors(node))


def _is_dunder_all(node: SymbolNode) -> bool:
    return node.type == "variable" and node.fqname.endswith("__all__")


def _is_external_dep(node: SymbolNode) -> bool:
    return node.type == "synthetic" and node.fqname.startswith(EXTERNAL_PREFIXES)


@app.command()
def dependencies(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        list[str] | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.text,
) -> None:
    """List third-party dependencies imported by the codebase."""
    setup_logging(verbose)
    root = root.resolve()

    paths_dict, resolvers = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    graph = build_symbol_graph(paths_dict, resolvers=resolvers, project_root=root)

    deps_by_base: dict[Path, list[SymbolNode]] = {base: [] for base in order_paths(paths_dict)}
    for node in graph.nodes:
        if not _is_external_dep(node):
            continue
        if node.path in deps_by_base:
            deps_by_base[node.path].append(node)

    if output_format == OutputFormat.json:
        result = {
            str(base): sorted(n.fqname for n in nodes) for base, nodes in deps_by_base.items()
        }
        typer.echo(json.dumps(result, indent=2))
        return

    for base, nodes in deps_by_base.items():
        typer.echo(f"\n{base}:")
        if not nodes:
            typer.echo("  (no third-party dependencies found)")
            continue
        for node in sorted(nodes, key=lambda n: n.fqname):
            typer.echo(f"  {node.fqname}")


@app.command("unused-exports")
def unused_exports(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    entrypoint: Annotated[
        list[str] | None,
        typer.Option(
            "-e",
            "--entrypoint",
            help="Entrypoint: file path, FQN, or 're:pattern' for regex.",
        ),
    ] = None,
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        list[str] | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Report __all__ entries whose targets are only alive because of __all__."""
    setup_logging(verbose)
    root = root.resolve()

    paths_dict, resolvers = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, resolvers=resolvers, project_root=root)
    reachable = find_reachable(graph)

    # ModuleDundersPlugin keeps each ``__all__`` alive via a synthetic
    # entrypoint node ``<dunder>:<fqname>``. Cut the edge from each such
    # synthetic into an ``__all__`` variable and re-run reachability;
    # whatever drops out was alive only because of __all__.
    pruned = graph.copy()
    pruned.remove_edges_from(
        [
            (s, d)
            for s, d in graph.edges
            if _is_dunder_all(d) and s.type == "synthetic" and s.fqname.startswith(DUNDER_PREFIX)
        ]
    )
    reachable_without_all = find_reachable(pruned)
    only_via_all = reachable - reachable_without_all

    by_all: dict[SymbolNode, list[SymbolNode]] = {}
    for sym in only_via_all:
        if _is_dunder_all(sym):
            continue
        for pred in graph.predecessors(sym):
            if _is_dunder_all(pred):
                by_all.setdefault(pred, []).append(sym)

    if not by_all:
        typer.echo("No __all__ entries are kept alive only by __all__.")
        return

    for all_sym in sorted(by_all, key=lambda n: n.fqname):
        typer.echo(f"\n{all_sym.fqname} at {_rel_path(all_sym.path, root)}:")
        for sym in sorted(by_all[all_sym], key=lambda n: n.fqname):
            typer.echo(f"  {sym.fqname} ({sym.type})")


@app.command()
def remove(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    entrypoint: Annotated[
        list[str] | None,
        typer.Option(
            "-e",
            "--entrypoint",
            help="Entrypoint: file path, FQN, or 're:pattern' for regex.",
        ),
    ] = None,
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        list[str] | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be removed without making changes.")
    ] = False,
) -> None:
    """Remove dead code from a Python codebase."""
    setup_logging(verbose)
    root = root.resolve()

    paths_dict, resolvers = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, resolvers=resolvers, project_root=root)
    reachable = find_reachable(graph)

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    # Synthetic ``unreachable`` nodes are reported by ``analyze`` but
    # not yet removable by ``remove`` -- the codemod doesn't know how to
    # delete an arbitrary suite. Filter them out of the listing so we
    # don't promise something we don't deliver.
    removable = [n for n in unreachable_graph.nodes if not is_unreachable_node(n)]

    if not removable:
        typer.echo("No dead code found.")
        return

    typer.echo(f"\nDead symbols to remove ({len(removable)}):")
    for node in sorted(removable, key=lambda n: (str(n.path), n.fqname)):
        typer.echo(f"  {node.fqname} ({node.type}) at {_rel_path(node.path, root)}")

    if dry_run:
        typer.echo("\n--dry-run specified, no changes made.")
        return

    if not typer.confirm("\nProceed with removal?"):
        typer.echo("Aborted.")
        return

    for base in order_paths(paths_dict):
        remove_code(unreachable_graph, base)

    typer.echo("Dead code removed.")


def main_cli() -> None:
    app()


if __name__ == "__main__":
    main_cli()
