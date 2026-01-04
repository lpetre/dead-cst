"""Command-line interface for dead-cst."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import click
import networkx as nx

from . import build_symbol_graph, count_nodes, find_reachable, order_paths, remove_code
from ._symbols import SymbolNode


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def parse_entrypoint(ep: str) -> str | re.Pattern:
    """Parse an entrypoint string, converting regex patterns."""
    if ep.startswith("re:"):
        return re.compile(ep[3:])
    return ep


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


@click.group()
@click.version_option()
def main() -> None:
    """Dead code analysis for Python using libcst."""
    pass


@main.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-p",
    "--path",
    "paths",
    multiple=True,
    help="Search path spec: 'base:dep1,dep2' or 'base'. Can be repeated.",
)
@click.option(
    "-e",
    "--entrypoint",
    "entrypoints",
    multiple=True,
    required=True,
    help="Entrypoint: file path, FQN, or 're:pattern' for regex. Can be repeated.",
)
@click.option(
    "--preserve-dunder-all/--no-preserve-dunder-all",
    default=True,
    help="Keep __all__ variables alive.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
def analyze(
    root: Path,
    paths: tuple[str, ...],
    entrypoints: tuple[str, ...],
    preserve_dunder_all: bool,
    verbose: bool,
    output_format: str,
) -> None:
    """Analyze a Python codebase for dead code.

    ROOT is the root directory to analyze.
    """
    setup_logging(verbose)
    root = root.resolve()

    # Parse paths
    paths_dict = parse_paths(root, list(paths))

    # Build graph
    click.echo(f"Building symbol graph for {root}...", err=True)
    graph = build_symbol_graph(paths_dict)

    # Parse entrypoints
    eps = [parse_entrypoint(ep) for ep in entrypoints]

    # Find reachable nodes
    reachable = find_reachable(graph, root, eps)

    # Optionally preserve __all__
    if preserve_dunder_all:
        for node in graph.nodes:
            if node.fqname.endswith("__all__") and node.type == "variable":
                reachable.add(node)

    # Get unreachable subgraph
    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    # Output results
    if output_format == "json":
        _output_json(graph, unreachable_graph, root, paths_dict)
    else:
        _output_text(graph, unreachable_graph, root, paths_dict)

    # Exit with error code if dead code found
    if unreachable_graph.number_of_nodes() > 0:
        sys.exit(1)


def _output_text(
    graph: nx.DiGraph,
    unreachable: nx.DiGraph,
    root: Path,
    paths_dict: dict[Path, list[Path]],
) -> None:
    """Output results in text format."""
    for base in order_paths(paths_dict):
        click.echo(f"\n{base}:")
        total_counts = count_nodes(graph, base)
        unreachable_counts = count_nodes(unreachable, base)
        for kind in sorted(total_counts):
            total = total_counts[kind]
            dead = unreachable_counts.get(kind, 0)
            if dead > 0:
                click.echo(f"  {kind}: {total} total, {dead} dead")
            else:
                click.echo(f"  {kind}: {total} total")

    # List dead symbols
    dead_count = unreachable.number_of_nodes()
    if dead_count > 0:
        click.echo(f"\nDead symbols ({dead_count}):")
        for node in sorted(unreachable.nodes, key=lambda n: (str(n.path), n.fqname)):
            try:
                rel_path = node.path.relative_to(root)
            except ValueError:
                rel_path = node.path
            click.echo(f"  {node.fqname} ({node.type}) at {rel_path}")


def _output_json(
    graph: nx.DiGraph,
    unreachable: nx.DiGraph,
    root: Path,
    paths_dict: dict[Path, list[Path]],
) -> None:
    """Output results in JSON format."""
    import json

    result = {
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

    click.echo(json.dumps(result, indent=2))


@main.command("why-alive")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("fqname")
@click.option(
    "-p",
    "--path",
    "paths",
    multiple=True,
    help="Search path spec: 'base:dep1,dep2' or 'base'. Can be repeated.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def why_alive(
    root: Path,
    fqname: str,
    paths: tuple[str, ...],
    verbose: bool,
) -> None:
    """Show why a symbol is considered alive (reachable).

    ROOT is the root directory to analyze.
    FQNAME is the fully qualified name of the symbol to check.
    """
    setup_logging(verbose)
    root = root.resolve()

    # Parse paths
    paths_dict = parse_paths(root, list(paths))

    # Build graph
    click.echo(f"Building symbol graph for {root}...", err=True)
    graph = build_symbol_graph(paths_dict)

    # Find the node
    target_node: SymbolNode | None = None
    for node in graph.nodes:
        if node.fqname == fqname:
            target_node = node
            break

    if target_node is None:
        click.echo(f"Symbol not found: {fqname}", err=True)
        sys.exit(1)

    try:
        rel_path = target_node.path.relative_to(root)
    except ValueError:
        rel_path = target_node.path

    click.echo(f"\nSymbol: {target_node.fqname} ({target_node.type})")
    click.echo(f"Path: {rel_path}")
    click.echo(f"In-degree: {graph.in_degree(target_node)}")
    click.echo("\nPredecessor chain:")

    # Walk predecessors
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
        click.echo(f"  <- {node.fqname} ({node.type}) at {node_rel}")
        stack.extend(graph.predecessors(node))


@main.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-p",
    "--path",
    "paths",
    multiple=True,
    help="Search path spec: 'base:dep1,dep2' or 'base'. Can be repeated.",
)
@click.option(
    "-e",
    "--entrypoint",
    "entrypoints",
    multiple=True,
    required=True,
    help="Entrypoint: file path, FQN, or 're:pattern' for regex. Can be repeated.",
)
@click.option(
    "--preserve-dunder-all/--no-preserve-dunder-all",
    default=True,
    help="Keep __all__ variables alive.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without making changes.")
def remove(
    root: Path,
    paths: tuple[str, ...],
    entrypoints: tuple[str, ...],
    preserve_dunder_all: bool,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Remove dead code from a Python codebase.

    ROOT is the root directory to analyze.
    """
    setup_logging(verbose)
    root = root.resolve()

    # Parse paths
    paths_dict = parse_paths(root, list(paths))

    # Build graph
    click.echo(f"Building symbol graph for {root}...", err=True)
    graph = build_symbol_graph(paths_dict)

    # Parse entrypoints
    eps = [parse_entrypoint(ep) for ep in entrypoints]

    # Find reachable nodes
    reachable = find_reachable(graph, root, eps)

    # Optionally preserve __all__
    if preserve_dunder_all:
        for node in graph.nodes:
            if node.fqname.endswith("__all__") and node.type == "variable":
                reachable.add(node)

    # Get unreachable subgraph
    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    if unreachable_graph.number_of_nodes() == 0:
        click.echo("No dead code found.")
        return

    # Show what would be removed
    click.echo(f"\nDead symbols to remove ({unreachable_graph.number_of_nodes()}):")
    for node in sorted(unreachable_graph.nodes, key=lambda n: (str(n.path), n.fqname)):
        try:
            rel_path = node.path.relative_to(root)
        except ValueError:
            rel_path = node.path
        click.echo(f"  {node.fqname} ({node.type}) at {rel_path}")

    if dry_run:
        click.echo("\n--dry-run specified, no changes made.")
        return

    # Confirm before removing
    if not click.confirm("\nProceed with removal?"):
        click.echo("Aborted.")
        return

    # Remove dead code
    for base in order_paths(paths_dict):
        remove_code(unreachable_graph, base)

    click.echo("Dead code removed.")


if __name__ == "__main__":
    main()
