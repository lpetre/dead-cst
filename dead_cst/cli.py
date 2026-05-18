"""Command-line interface for dead-cst."""

from __future__ import annotations

import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Sequence

import typer

from ._graphstore import SymbolGraph
from .analyze import Analysis, _count_nodes_by_prefix, _entrypoint_seeds, _find_reachable
from .codemod import generate_patch
from .graph import NodeFlags, SymbolNode
from .plugins import (
    EXTERNAL_PREFIXES,
    ExplicitEntrypointPlugin,
    ModuleDundersPlugin,
    load_plugin,
    simple_name,
)
from .plugins.module_dunders import DUNDER_PREFIX
from .resolvers import (
    ManualResolver,
    PathResolver,
    load_resolver,
)


app = typer.Typer(help="Dead code analysis for Python.")


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
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def build_plugins(
    *,
    entrypoints: list[str],
    plugin_names: list[str],
) -> list[object]:
    """Compose the plugin list from CLI flags."""
    plugins: list[object] = []
    for name in plugin_names:
        plugins.append(load_plugin(name))
    plugins.append(ModuleDundersPlugin())
    if entrypoints:
        specs = [parse_entrypoint(ep) for ep in entrypoints]
        plugins.append(ExplicitEntrypointPlugin(specs=specs))
    return plugins


def build_resolver(path_specs: list[str], resolver_name: str | None) -> PathResolver:
    if path_specs and resolver_name is not None:
        raise typer.BadParameter("`-p`/`--path` and `--resolver` are mutually exclusive.")
    if resolver_name is not None:
        return load_resolver(resolver_name)
    if path_specs:
        return ManualResolver(specs=path_specs)
    return ManualResolver(specs=["."])


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
    """Dead code analysis for Python."""


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
        typer.Option("-p", "--path", help="Search path spec: 'package:dep1,dep2' or 'package'."),
    ] = None,
    resolver: Annotated[
        str | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
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

    path_resolver = build_resolver(path or [], resolver)

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    analysis = Analysis(root, resolver=path_resolver, plugins=plugins)
    graph = analysis.materialize_all()
    reachable = _find_reachable(graph, _entrypoint_seeds(graph))

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    package_paths = [p.path for p in analysis.packages]
    if output_format == OutputFormat.json:
        _output_json(graph, unreachable_graph, root, package_paths)
    else:
        _output_text(graph, unreachable_graph, root, package_paths)

    if len(unreachable_graph) > 0:
        raise typer.Exit(1)


def _output_text(
    graph: SymbolGraph,
    unreachable: SymbolGraph,
    root: Path,
    package_paths: Sequence[Path],
) -> None:
    total_by_path = _count_nodes_by_prefix(graph.nodes, package_paths)
    unreachable_by_path = _count_nodes_by_prefix(unreachable.nodes, package_paths)
    for path in package_paths:
        typer.echo(f"\n{path}:")
        total_counts = total_by_path[path]
        unreachable_counts = unreachable_by_path[path]
        for kind in sorted(total_counts):
            if kind == "synthetic":
                continue
            total = total_counts[kind]
            dead = unreachable_counts.get(kind, 0)
            if dead > 0:
                typer.echo(f"  {kind}: {total} total, {dead} dead")
            else:
                typer.echo(f"  {kind}: {total} total")

    dead_real = _dead_real(unreachable)
    if dead_real:
        typer.echo(f"\nDead symbols ({len(dead_real)}):")
        for node in sorted(dead_real, key=lambda n: (str(n.path), n.fqname)):
            typer.echo(f"  {node.fqname} ({node.type}) at {_rel_path(node.path, root)}")


def _dead_real(unreachable: SymbolGraph) -> list[SymbolNode]:
    return [n for n in unreachable.nodes if n.type != "synthetic"]


def _output_json(
    graph: SymbolGraph,
    unreachable: SymbolGraph,
    root: Path,
    package_paths: Sequence[Path],
) -> None:
    result: dict = {
        "summary": {},
        "dead_symbols": [],
    }

    total_by_path = _count_nodes_by_prefix(graph.nodes, package_paths)
    unreachable_by_path = _count_nodes_by_prefix(unreachable.nodes, package_paths)
    for path in package_paths:
        path_str = str(path)
        total_counts = total_by_path[path]
        unreachable_counts = unreachable_by_path[path]
        result["summary"][path_str] = {
            kind: {"total": total_counts[kind], "dead": unreachable_counts.get(kind, 0)}
            for kind in total_counts
            if kind != "synthetic"
        }

    for node in sorted(_dead_real(unreachable), key=lambda n: (str(n.path), n.fqname)):
        result["dead_symbols"].append(
            {
                "fqname": node.fqname,
                "type": node.type,
                "path": str(_rel_path(node.path, root)),
            }
        )

    typer.echo(json.dumps(result, indent=2))


@app.command("why-alive")
def why_alive(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    fqname: Annotated[str, typer.Argument(help="Fully qualified name of the symbol to check.")],
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'package:dep1,dep2' or 'package'."),
    ] = None,
    resolver: Annotated[
        str | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
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

    path_resolver = build_resolver(path or [], resolver)

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=[],
        plugin_names=plugin or [],
    )
    graph = Analysis(root, resolver=path_resolver, plugins=plugins).materialize_all()

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
    typer.echo(f"In-degree: {graph.raw.in_degree(graph.index(target_node))}")
    typer.echo("\nPredecessor chain:")

    seen_idx: set[int] = set()
    stack: list[int] = [graph.index(target_node)]
    while stack:
        i = stack.pop()
        if i in seen_idx:
            continue
        seen_idx.add(i)
        node = graph.node(i)
        typer.echo(f"  <- {node.fqname} ({node.type}) at {_rel_path(node.path, root)}")
        stack.extend(graph.raw.predecessor_indices(i))


def _is_dunder_all(node: SymbolNode) -> bool:
    return node.type == "variable" and simple_name(node.fqname) == "__all__"


def _is_external_dep(node: SymbolNode) -> bool:
    return node.type == "synthetic" and node.fqname.startswith(EXTERNAL_PREFIXES)


@app.command()
def dependencies(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    path: Annotated[
        list[str] | None,
        typer.Option("-p", "--path", help="Search path spec: 'package:dep1,dep2' or 'package'."),
    ] = None,
    resolver: Annotated[
        str | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
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

    path_resolver = build_resolver(path or [], resolver)

    typer.echo(f"Building symbol graph for {root}...", err=True)
    analysis = Analysis(root, resolver=path_resolver)
    graph = analysis.materialize_all()

    deps_by_package: dict[Path, list[SymbolNode]] = {p.path: [] for p in analysis.packages}
    for node in graph.nodes:
        if not _is_external_dep(node):
            continue
        # Synthetic dep nodes carry an empty path. Attribute them to
        # each package that imports them by walking back through the
        # graph's predecessor edges.
        importer_paths: set[Path] = set()
        for j in graph.raw.predecessor_indices(graph.index(node)):
            importer_paths.add(graph.node(j).path)
        for pkg_path in deps_by_package:
            if any(p.is_relative_to(pkg_path) for p in importer_paths):
                if node not in deps_by_package[pkg_path]:
                    deps_by_package[pkg_path].append(node)

    if output_format == OutputFormat.json:
        result = {
            str(pkg_path): sorted(n.fqname for n in nodes)
            for pkg_path, nodes in deps_by_package.items()
        }
        typer.echo(json.dumps(result, indent=2))
        return

    for pkg_path, nodes in deps_by_package.items():
        typer.echo(f"\n{pkg_path}:")
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
        typer.Option("-p", "--path", help="Search path spec: 'package:dep1,dep2' or 'package'."),
    ] = None,
    resolver: Annotated[
        str | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
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

    path_resolver = build_resolver(path or [], resolver)

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    graph = Analysis(root, resolver=path_resolver, plugins=plugins).materialize_all()
    reachable = _find_reachable(graph, _entrypoint_seeds(graph))

    def _is_dunder_seed(node: SymbolNode) -> bool:
        return node.type == "synthetic" and node.fqname.startswith(DUNDER_PREFIX)

    visited_idx: set[int] = set()
    stack: list[int] = [graph.index(n) for n in graph.nodes if n.flags & NodeFlags.ENTRYPOINT]
    while stack:
        i = stack.pop()
        if i in visited_idx:
            continue
        visited_idx.add(i)
        is_seed = _is_dunder_seed(graph.node(i))
        for j in graph.raw.successor_indices(i):
            if is_seed and _is_dunder_all(graph.node(j)):
                continue
            stack.append(j)
    visited = {graph.node(i) for i in visited_idx}
    only_via_all = reachable - visited

    by_all: dict[SymbolNode, list[SymbolNode]] = {}
    for sym in only_via_all:
        if _is_dunder_all(sym):
            continue
        for j in graph.raw.predecessor_indices(graph.index(sym)):
            pred = graph.node(j)
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
        typer.Option("-p", "--path", help="Search path spec: 'package:dep1,dep2' or 'package'."),
    ] = None,
    resolver: Annotated[
        str | None,
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Write patch to this file instead of stdout.",
        ),
    ] = None,
) -> None:
    """Emit a unified diff that removes dead code; pipe to ``git apply``."""
    setup_logging(verbose)
    root = root.resolve()

    path_resolver = build_resolver(path or [], resolver)

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    analysis = Analysis(root, resolver=path_resolver, plugins=plugins)
    graph = analysis.materialize_all()
    reachable = _find_reachable(graph, _entrypoint_seeds(graph))

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    patch = generate_patch(unreachable_graph, root)

    if not patch:
        typer.echo("No dead code found.", err=True)
        return

    if output is not None:
        output.write_text(patch)
        typer.echo(f"Wrote patch to {output}. Apply with: git apply {output}", err=True)
    else:
        sys.stdout.write(patch)
        typer.echo("Apply with: dead-cst remove ... | git apply", err=True)


def main_cli() -> None:
    app()


if __name__ == "__main__":
    main_cli()
