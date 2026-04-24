"""Command-line interface for dead-cst."""

from __future__ import annotations

import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import networkx as nx
import typer

from . import build_symbol_graph, count_nodes, find_reachable, order_paths, remove_code
from ._plugins import (
    CSTAwareEdgePlugin,
    DunderAllPlugin,
    EdgePlugin,
    ExplicitEntrypointPlugin,
    load_plugin,
)
from ._resolvers import load_resolver, merge_paths
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


def build_plugins(
    *,
    entrypoints: list[str],
    plugin_names: list[str],
    preserve_dunder_all: bool,
) -> list[EdgePlugin | CSTAwareEdgePlugin]:
    """Compose the plugin list from CLI flags.

    Order: user-specified plugins first, then ``DunderAllPlugin`` (if
    enabled), then ``ExplicitEntrypointPlugin`` with the ``-e`` specs.
    ``-e`` runs last so it can hang entrypoints off any synthetic nodes
    contributed upstream.
    """
    plugins: list[EdgePlugin | CSTAwareEdgePlugin] = []
    for name in plugin_names:
        plugins.append(load_plugin(name))
    if preserve_dunder_all:
        plugins.append(DunderAllPlugin())
    if entrypoints:
        specs = [parse_entrypoint(ep) for ep in entrypoints]
        plugins.append(ExplicitEntrypointPlugin(specs=specs))
    return plugins


def resolve_paths(
    root: Path, path_specs: list[str], resolver_names: list[str]
) -> dict[Path, list[Path]]:
    """Merge ``-p`` specs and ``--resolver`` outputs into a single path map."""
    explicit = parse_paths(root, path_specs) if path_specs else {}
    resolved_maps = [load_resolver(name).resolve(root) for name in resolver_names]
    if not explicit and not resolved_maps:
        return {root: []}
    return merge_paths(explicit, *resolved_maps)


def parse_paths(root: Path, paths_str: list[str]) -> dict[Path, list[Path]]:
    """Parse path specifications into the paths dict format.

    Format: "base:dep1,dep2,dep3" or just "base" for no dependencies.
    """
    if not paths_str:
        return {root: []}

    result = {}
    for spec in paths_str:
        if ":" in spec:
            base_str, deps_str = spec.split(":", 1)
            base = root / base_str
            deps = [root / d.strip() for d in deps_str.split(",") if d.strip()]
        else:
            base = root / spec
            deps = []
        result[base] = deps
    return result


def version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"dead-cst {version('dead-cst')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Dead code analysis for Python using libcst."""


@app.command()
def analyze(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    entrypoint: Annotated[
        Optional[list[str]],
        typer.Option(
            "-e",
            "--entrypoint",
            help="Entrypoint: file path, FQN, or 're:pattern' for regex.",
        ),
    ] = None,
    path: Annotated[
        Optional[list[str]],
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        Optional[list[str]],
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        Optional[list[str]],
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    preserve_dunder_all: Annotated[bool, typer.Option(help="Keep __all__ variables alive.")] = True,
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

    paths_dict = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
        preserve_dunder_all=preserve_dunder_all,
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, project_root=root)
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
    paths_dict: dict[Path, list[Path]],
) -> None:
    for base in order_paths(paths_dict):
        typer.echo(f"\n{base}:")
        total_counts = count_nodes(graph, base)
        unreachable_counts = count_nodes(unreachable, base)
        for kind in sorted(total_counts):
            total = total_counts[kind]
            dead = unreachable_counts.get(kind, 0)
            if dead > 0:
                typer.echo(f"  {kind}: {total} total, {dead} dead")
            else:
                typer.echo(f"  {kind}: {total} total")

    dead_count = unreachable.number_of_nodes()
    if dead_count > 0:
        typer.echo(f"\nDead symbols ({dead_count}):")
        for node in sorted(unreachable.nodes, key=lambda n: (str(n.path), n.fqname)):
            try:
                rel_path = node.path.relative_to(root)
            except ValueError:
                rel_path = node.path
            typer.echo(f"  {node.fqname} ({node.type}) at {rel_path}")


def _output_json(
    graph: nx.DiGraph,
    unreachable: nx.DiGraph,
    root: Path,
    paths_dict: dict[Path, list[Path]],
) -> None:
    result: dict = {
        "summary": {},
        "dead_symbols": [],
    }

    for base in order_paths(paths_dict):
        base_str = str(base)
        total_counts = count_nodes(graph, base)
        unreachable_counts = count_nodes(unreachable, base)
        result["summary"][base_str] = {
            kind: {"total": total_counts[kind], "dead": unreachable_counts.get(kind, 0)}
            for kind in total_counts
        }

    for node in sorted(unreachable.nodes, key=lambda n: (str(n.path), n.fqname)):
        try:
            rel_path = str(node.path.relative_to(root))
        except ValueError:
            rel_path = str(node.path)
        result["dead_symbols"].append(
            {
                "fqname": node.fqname,
                "type": node.type,
                "path": rel_path,
            }
        )

    typer.echo(json.dumps(result, indent=2))


@app.command("why-alive")
def why_alive(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    fqname: Annotated[str, typer.Argument(help="Fully qualified name of the symbol to check.")],
    path: Annotated[
        Optional[list[str]],
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        Optional[list[str]],
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        Optional[list[str]],
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    preserve_dunder_all: Annotated[bool, typer.Option(help="Keep __all__ variables alive.")] = True,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Show why a symbol is considered alive (reachable)."""
    setup_logging(verbose)
    root = root.resolve()

    paths_dict = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=[],
        plugin_names=plugin or [],
        preserve_dunder_all=preserve_dunder_all,
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, project_root=root)

    target_node: SymbolNode | None = None
    for node in graph.nodes:
        if node.fqname == fqname:
            target_node = node
            break

    if target_node is None:
        typer.echo(f"Symbol not found: {fqname}", err=True)
        raise typer.Exit(1)

    try:
        rel_path = target_node.path.relative_to(root)
    except ValueError:
        rel_path = target_node.path

    typer.echo(f"\nSymbol: {target_node.fqname} ({target_node.type})")
    typer.echo(f"Path: {rel_path}")
    typer.echo(f"In-degree: {graph.in_degree(target_node)}")
    typer.echo("\nPredecessor chain:")

    seen: set[SymbolNode] = set()
    stack = [target_node]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        try:
            node_rel = node.path.relative_to(root)
        except ValueError:
            node_rel = node.path
        typer.echo(f"  <- {node.fqname} ({node.type}) at {node_rel}")
        stack.extend(graph.predecessors(node))


@app.command()
def remove(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    entrypoint: Annotated[
        Optional[list[str]],
        typer.Option(
            "-e",
            "--entrypoint",
            help="Entrypoint: file path, FQN, or 're:pattern' for regex.",
        ),
    ] = None,
    path: Annotated[
        Optional[list[str]],
        typer.Option("-p", "--path", help="Search path spec: 'base:dep1,dep2' or 'base'."),
    ] = None,
    resolver: Annotated[
        Optional[list[str]],
        typer.Option("--resolver", help="Path resolver to run (e.g. venv, pyproject)."),
    ] = None,
    plugin: Annotated[
        Optional[list[str]],
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    preserve_dunder_all: Annotated[bool, typer.Option(help="Keep __all__ variables alive.")] = True,
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

    paths_dict = resolve_paths(root, path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
        preserve_dunder_all=preserve_dunder_all,
    )
    graph = build_symbol_graph(paths_dict, plugins=plugins, project_root=root)
    reachable = find_reachable(graph)

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    if unreachable_graph.number_of_nodes() == 0:
        typer.echo("No dead code found.")
        return

    typer.echo(f"\nDead symbols to remove ({unreachable_graph.number_of_nodes()}):")
    for node in sorted(unreachable_graph.nodes, key=lambda n: (str(n.path), n.fqname)):
        try:
            rel_path = node.path.relative_to(root)
        except ValueError:
            rel_path = node.path
        typer.echo(f"  {node.fqname} ({node.type}) at {rel_path}")

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
