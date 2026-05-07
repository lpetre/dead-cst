"""Command-line interface for dead-cst."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Iterator

import networkx as nx
import typer
from libcst.metadata import CodeRange

from .analyze import Analysis, _count_nodes, _find_reachable, _order_paths
from .cache import (
    GraphCache,
    clear_cache,
    default_cache_path,
)
from .codemod import remove_code
from .graph import SymbolNode
from .plugins import (
    EXTERNAL_PREFIXES,
    EdgePlugin,
    ExplicitEntrypointPlugin,
    ModuleDundersPlugin,
    load_plugin,
)
from .plugins.module_dunders import DUNDER_PREFIX
from .resolvers import (
    ManualResolver,
    PathMap,
    PathResolver,
    load_resolver,
)


app = typer.Typer(help="Dead code analysis for Python using libcst.")


WorkersOption = Annotated[
    int | None,
    typer.Option(
        "--workers",
        "-j",
        help="Run cache-miss visitor passes in this many worker processes (>=2 enables it).",
    ),
]


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


@contextlib.contextmanager
def _maybe_cache(
    root: Path,
    no_cache: bool,
) -> Iterator[GraphCache | None]:
    """Yield a per-run :class:`GraphCache`, or ``None`` when ``--no-cache`` is set.

    Per-base fingerprints (computed inside :class:`Analysis`) gate
    individual cache rows, so this just opens the database. A
    schema-version mismatch on open wipes ``file_cache`` automatically.
    The context manager closes the SQLite connection on exit, even
    when the analysis raises.
    """
    if no_cache:
        yield None
        return
    with GraphCache(default_cache_path(root)) as cache:
        yield cache


def build_resolvers(path_specs: list[str], resolver_names: list[str]) -> list[PathResolver]:
    """Build the resolver chain from ``-p`` specs and ``--resolver`` names.

    ``-p`` specs become a :class:`ManualResolver` and slot into the
    chain alongside named resolvers, so explicit specs participate in
    :meth:`PathResolver.resolve_import` lookups too. With neither flag
    supplied, falls back to a ``ManualResolver`` that treats the project
    root itself as the only base.
    """
    resolvers: list[PathResolver] = []
    if path_specs:
        resolvers.append(ManualResolver(specs=path_specs))
    resolvers.extend(load_resolver(name) for name in resolver_names)
    if not resolvers:
        resolvers.append(ManualResolver(specs=["."]))
    return resolvers


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
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the per-file VisitorPayload cache.")
    ] = False,
    workers: WorkersOption = None,
) -> None:
    """Analyze a Python codebase for dead code."""
    setup_logging(verbose)
    root = root.resolve()

    resolvers = build_resolvers(path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    with _maybe_cache(root, no_cache) as cache:
        analysis = Analysis(
            root,
            resolvers=resolvers,
            plugins=plugins,
            cache=cache,
            workers=workers,
        )
        graph = analysis.materialize_all()
    paths_dict = analysis.paths
    reachable = _find_reachable(graph)

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    if output_format == OutputFormat.json:
        _output_json(graph, unreachable_graph, root, paths_dict)
    else:
        _output_text(graph, unreachable_graph, root, paths_dict)

    if unreachable_graph.number_of_nodes() > 0:
        raise typer.Exit(1)


def _output_text(
    graph: nx.MultiDiGraph,
    unreachable: nx.MultiDiGraph,
    root: Path,
    paths_dict: PathMap,
) -> None:
    for base in _order_paths(paths_dict):
        typer.echo(f"\n{base}:")
        total_counts = _count_nodes(graph, base)
        unreachable_counts = _count_nodes(unreachable, base)
        for kind in sorted(total_counts):
            # Synthetic nodes (entrypoint sentinels, external-dist markers,
            # dunder-all stand-ins) don't represent user-visible
            # declarations; reporting them alongside functions/classes
            # would be misleading.
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

    branches = _dead_suite_locations(graph, paths_dict)
    if branches:
        typer.echo(f"\nUnreachable branches ({len(branches)}):")
        for path, pos in branches:
            rel = _rel_path(path, root)
            start = pos.start
            end = pos.end
            typer.echo(f"  {rel}:{start.line}:{start.column}-{end.line}:{end.column}")


def _dead_real(unreachable: nx.MultiDiGraph) -> list[SymbolNode]:
    """Return real (non-synthetic) dead nodes from ``unreachable``.

    Synthetic nodes -- entrypoint sentinels, external-dist markers,
    dunder-all stand-ins -- are excluded; they don't represent
    user-visible declarations and would confuse the dead-code report.
    """
    return [n for n in unreachable.nodes if n.type != "synthetic"]


def _dead_suite_locations(
    graph: nx.MultiDiGraph, paths_dict: PathMap
) -> list[tuple[Path, CodeRange]]:
    """Flatten ``graph.graph["dead_suites"]`` into a sorted ``(path, pos)`` list.

    Restricted to files under one of the analyzed bases so the report
    doesn't surface dead suites in workspace dependencies that the
    user isn't asking about.
    """
    bases = list(paths_dict)
    raw: dict = graph.graph.get("dead_suites", {})
    out: list[tuple[Path, CodeRange]] = []
    for path, suites in raw.items():
        if not any(path.is_relative_to(b) for b in bases):
            continue
        for pos in suites:
            out.append((path, pos))
    out.sort(key=lambda entry: (str(entry[0]), entry[1].start.line, entry[1].start.column))
    return out


def _output_json(
    graph: nx.MultiDiGraph,
    unreachable: nx.MultiDiGraph,
    root: Path,
    paths_dict: PathMap,
) -> None:
    result: dict = {
        "summary": {},
        "dead_symbols": [],
        "unreachable_branches": [],
    }

    for base in _order_paths(paths_dict):
        base_str = str(base)
        total_counts = _count_nodes(graph, base)
        unreachable_counts = _count_nodes(unreachable, base)
        # Same rationale as the text output: synthetic nodes are reported
        # via ``unreachable_branches`` (and entrypoint sentinels), not as
        # part of the per-kind summary.
        result["summary"][base_str] = {
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

    for path, pos in _dead_suite_locations(graph, paths_dict):
        result["unreachable_branches"].append(
            {
                "path": str(_rel_path(path, root)),
                "start": {"line": pos.start.line, "column": pos.start.column},
                "end": {"line": pos.end.line, "column": pos.end.column},
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
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the per-file VisitorPayload cache.")
    ] = False,
    workers: WorkersOption = None,
) -> None:
    """Show why a symbol is considered alive (reachable)."""
    setup_logging(verbose)
    root = root.resolve()

    resolvers = build_resolvers(path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=[],
        plugin_names=plugin or [],
    )
    with _maybe_cache(root, no_cache) as cache:
        graph = Analysis(
            root,
            resolvers=resolvers,
            plugins=plugins,
            cache=cache,
            workers=workers,
        ).materialize_all()

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
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.text,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the per-file VisitorPayload cache.")
    ] = False,
    workers: WorkersOption = None,
) -> None:
    """List third-party dependencies imported by the codebase."""
    setup_logging(verbose)
    root = root.resolve()

    resolvers = build_resolvers(path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    with _maybe_cache(root, no_cache) as cache:
        analysis = Analysis(
            root,
            resolvers=resolvers,
            cache=cache,
            workers=workers,
        )
        graph = analysis.materialize_all()
    paths_dict = analysis.paths

    deps_by_base: dict[Path, list[SymbolNode]] = {base: [] for base in _order_paths(paths_dict)}
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
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the per-file VisitorPayload cache.")
    ] = False,
    workers: WorkersOption = None,
) -> None:
    """Report __all__ entries whose targets are only alive because of __all__."""
    setup_logging(verbose)
    root = root.resolve()

    resolvers = build_resolvers(path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    with _maybe_cache(root, no_cache) as cache:
        graph = Analysis(
            root,
            resolvers=resolvers,
            plugins=plugins,
            cache=cache,
            workers=workers,
        ).materialize_all()
    reachable = _find_reachable(graph)

    # ModuleDundersPlugin keeps each ``__all__`` alive via a synthetic
    # entrypoint node ``<dunder>:<fqname>``. Cut the edge from each such
    # synthetic into an ``__all__`` variable and re-run reachability;
    # whatever drops out was alive only because of __all__.
    pruned = graph.copy()
    pruned.remove_edges_from(
        [
            (s, d, k)
            for s, d, k in graph.edges(keys=True)
            if _is_dunder_all(d) and s.type == "synthetic" and s.fqname.startswith(DUNDER_PREFIX)
        ]
    )
    reachable_without_all = _find_reachable(pruned)
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
        typer.Option("--resolver", help="Path resolver to run (e.g. uv)."),
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
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the per-file VisitorPayload cache.")
    ] = False,
    workers: WorkersOption = None,
) -> None:
    """Remove dead code from a Python codebase."""
    setup_logging(verbose)
    root = root.resolve()

    resolvers = build_resolvers(path or [], resolver or [])

    typer.echo(f"Building symbol graph for {root}...", err=True)
    plugins = build_plugins(
        entrypoints=entrypoint or [],
        plugin_names=plugin or [],
    )
    with _maybe_cache(root, no_cache) as cache:
        analysis = Analysis(
            root,
            resolvers=resolvers,
            plugins=plugins,
            cache=cache,
            workers=workers,
        )
        graph = analysis.materialize_all()
    paths_dict = analysis.paths
    reachable = _find_reachable(graph)

    unreachable_graph = graph.subgraph([n for n in graph.nodes if n not in reachable])

    # Synthetic nodes (entrypoint sentinels, external-dist markers,
    # dunder-all stand-ins) aren't user-visible declarations. The
    # codemod can't delete them, so they're filtered out of the
    # remove listing.
    removable = [n for n in unreachable_graph.nodes if n.type != "synthetic"]

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

    for base in _order_paths(paths_dict):
        remove_code(unreachable_graph, base)

    typer.echo("Dead code removed.")


cache_app = typer.Typer(help="Manage the on-disk analysis cache.")
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear(
    root: Annotated[
        Path,
        typer.Argument(help="Project root whose .dead-cst-cache directory should be removed."),
    ] = Path("."),
) -> None:
    """Delete the cached :class:`VisitorPayload` database for ``root``."""
    db = default_cache_path(root.resolve())
    removed = clear_cache(db)
    if removed:
        typer.echo(f"Removed {db}.")
    else:
        typer.echo(f"No cache found at {db}.")


def main_cli() -> None:
    app()


if __name__ == "__main__":
    main_cli()
