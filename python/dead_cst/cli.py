"""Command-line interface for dead-cst.

Three commands cover the surface:

* ``dead-cst build ROOT -o PATH`` — materialize the project graph and
  persist it to disk for reuse by later ``analyze`` / ``remove`` runs.
* ``dead-cst analyze ROOT`` — list dead code in text or JSON.
* ``dead-cst remove ROOT`` — emit a ``git apply``-compatible patch
  that drops dead code.

For multi-package monorepos, pass ``--venv PATH`` pointing at a venv
populated with ``uv sync --all-packages`` (or any equivalent editable
install layout). ty reads the venv's ``.pth`` files to discover
where each first-party member's published source lives and uses
them as additional module-resolution search paths.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterable, Sequence

import typer

from .analyze import Analysis
from .codemod import generate_patch
from .contrib.celery import CeleryPlugin
from .contrib.click import ClickPlugin
from .contrib.cyclopts import cyclopts_plugin
from .contrib.discordpy import DiscordPyPlugin
from .contrib.fastapi import fastapi_plugin
from .contrib.fastmcp import fastmcp_plugin
from .contrib.flask import flask_plugin
from .contrib.mock_patch import MockPatchPlugin
from .contrib.pytest import PytestPlugin
from .contrib.server_config import ServerConfigPlugin
from .contrib.typer import typer_plugin
from .contrib.unittest import UnittestPlugin
from .graph import (
    KEEPALIVE_DEFAULT,
    GraphMetadata,
    LoadedGraph,
    NodeFlags,
    SymbolNode,
    read_graph,
    write_graph,
)
from .plugins import (
    DynamicImportFallbackPlugin,
    ExplicitEntrypointPlugin,
    InitSubclassPlugin,
    MainBlockPlugin,
    ModuleDundersPlugin,
    Plugin,
    ProjectScriptsPlugin,
)

if TYPE_CHECKING:
    from dead_cst import _native as native

    GraphView = native.ProjectContext | LoadedGraph


app = typer.Typer(help="Dead code analysis for Python.")


_BUILTIN_PLUGINS: dict[str, Plugin] = {
    "main_block": MainBlockPlugin(),
    "project_scripts": ProjectScriptsPlugin(),
    "explicit": ExplicitEntrypointPlugin(),
    "module_dunders": ModuleDundersPlugin(),
    "pytest": PytestPlugin(),
    "unittest": UnittestPlugin(),
    "mock_patch": MockPatchPlugin(),
    "server_config": ServerConfigPlugin(),
    "fastapi": fastapi_plugin(),
    "fastmcp": fastmcp_plugin(),
    "flask": flask_plugin(),
    "typer": typer_plugin(),
    "click": ClickPlugin(),
    "cyclopts": cyclopts_plugin(),
    "celery": CeleryPlugin(),
    "discordpy": DiscordPyPlugin(),
    "init_subclass": InitSubclassPlugin(),
    "dynamic_import_fallback": DynamicImportFallbackPlugin(),
}


def _load_plugin(name: str) -> Plugin:
    builtin = _BUILTIN_PLUGINS.get(name)
    if builtin is not None:
        return builtin

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.plugins"):
        if ep.name == name:
            loaded = ep.load()
            if isinstance(loaded, Plugin):
                return loaded
            if callable(loaded):
                instance = loaded()
                if isinstance(instance, Plugin):
                    return instance
            raise TypeError(
                f"Plugin entry point {name!r} did not resolve to a Plugin instance "
                f"(got {type(loaded).__name__})"
            )
    raise KeyError(f"Unknown edge plugin: {name!r}")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class Query(str, Enum):
    dead = "dead"
    test_only = "test-only"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def parse_meta(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise typer.BadParameter(f"--meta expects 'key=value', got {spec!r}")
    key, _, value = spec.partition("=")
    key = key.strip()
    if not key:
        raise typer.BadParameter(f"--meta expects 'key=value', got {spec!r}")
    return key, value


def _rel_path(path: Path | str, root: Path) -> Path:
    p = Path(path) if isinstance(path, str) else path
    try:
        return p.relative_to(root)
    except ValueError:
        return p


def build_plugins(
    *,
    entrypoints: list[str],
    entrypoint_regexes: list[str],
    plugin_names: list[str],
) -> list[Plugin]:
    plugins: list[Plugin] = [_load_plugin(name) for name in plugin_names]
    plugins.append(ModuleDundersPlugin())
    specs: list[str | Path | re.Pattern[str]] = list(entrypoints)
    specs.extend(re.compile(p) for p in entrypoint_regexes)
    if specs:
        plugins.append(ExplicitEntrypointPlugin(specs=specs))
    return plugins


def _materialize(
    root: Path,
    *,
    venv: Path | None,
    plugin_names: list[str],
    entrypoints: list[str],
    entrypoint_regexes: list[str],
    show_progress: bool,
) -> native.ProjectContext:
    plugins = build_plugins(
        entrypoints=entrypoints,
        entrypoint_regexes=entrypoint_regexes,
        plugin_names=plugin_names,
    )
    analysis = Analysis(root, venv=venv, plugins=plugins, show_progress=show_progress)
    return analysis.materialize_all()


def _reject_build_inputs_with_graph(
    *,
    graph_path: Path | None,
    venv: Path | None,
    plugin_names: list[str],
    entrypoints: list[str],
    entrypoint_regexes: list[str],
) -> None:
    if graph_path is None:
        return
    offending: list[str] = []
    if venv is not None:
        offending.append("`--venv`")
    if plugin_names:
        offending.append("`--plugin`")
    if entrypoints:
        offending.append("`-e`/`--entrypoint`")
    if entrypoint_regexes:
        offending.append("`--entrypoint-regex`")
    if offending:
        raise typer.BadParameter(
            f"--graph loads a pre-built graph; build inputs are not allowed: {', '.join(offending)}"
        )


def _load_or_build(
    root: Path,
    *,
    graph_path: Path | None,
    venv: Path | None,
    plugin_names: list[str],
    entrypoints: list[str],
    entrypoint_regexes: list[str],
) -> tuple[GraphView, GraphMetadata | None]:
    _reject_build_inputs_with_graph(
        graph_path=graph_path,
        venv=venv,
        plugin_names=plugin_names,
        entrypoints=entrypoints,
        entrypoint_regexes=entrypoint_regexes,
    )
    if graph_path is not None:
        typer.echo(f"Loading symbol graph from {graph_path}...", err=True)
        loaded, meta = read_graph(graph_path)
        return loaded, meta
    typer.echo(f"Building symbol graph for {root}...", err=True)
    ctx = _materialize(
        root,
        venv=venv,
        plugin_names=plugin_names,
        entrypoints=entrypoints,
        entrypoint_regexes=entrypoint_regexes,
        show_progress=True,
    )
    return ctx, None


def _select_dead(view: GraphView, query: Query) -> list[SymbolNode]:
    if query is Query.dead:
        reachable = set(view.reachable(seed_flags=KEEPALIVE_DEFAULT))
        return [n for n in view.nodes() if n not in reachable]
    if query is Query.test_only:
        full = set(view.reachable(seed_flags=KEEPALIVE_DEFAULT))
        without_tests = set(view.reachable(seed_flags=KEEPALIVE_DEFAULT & ~NodeFlags.TESTCASE))
        return list(full - without_tests)
    raise typer.BadParameter(f"unknown --query value: {query!r}")


def _count_by_kind(nodes: Iterable[SymbolNode]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.kind] = counts.get(node.kind, 0) + 1
    return counts


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


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@app.command()
def build(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Write the graph to this file."),
    ],
    venv: Annotated[
        Path | None,
        typer.Option("--venv", help="Venv with editable .pth entries for first-party members."),
    ] = None,
    entrypoint: Annotated[
        list[str] | None,
        typer.Option("-e", "--entrypoint", help="Entrypoint as a file path or FQN."),
    ] = None,
    entrypoint_regex: Annotated[
        list[str] | None,
        typer.Option("--entrypoint-regex", help="Entrypoint as a regex over FQNs / file paths."),
    ] = None,
    plugin: Annotated[
        list[str] | None,
        typer.Option("--plugin", help="Edge plugin to run (e.g. main_block, project_scripts)."),
    ] = None,
    meta: Annotated[
        list[str] | None,
        typer.Option("--meta", help="Stash key=value metadata in the graph file (repeatable)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Build the project graph and persist it to disk."""
    setup_logging(verbose)
    root = root.resolve()
    ctx = _materialize(
        root,
        venv=venv,
        plugin_names=plugin or [],
        entrypoints=entrypoint or [],
        entrypoint_regexes=entrypoint_regex or [],
        show_progress=True,
    )
    meta_pairs = [parse_meta(s) for s in (meta or [])]
    write_graph(output, ctx, meta_pairs)
    typer.echo(
        f"Wrote graph to {output} ({len(ctx.nodes())} nodes, {len(ctx.edges())} edges).",
        err=True,
    )


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    graph: Annotated[
        Path | None,
        typer.Option("--graph", help="Load a pre-built graph instead of materializing."),
    ] = None,
    query: Annotated[
        Query,
        typer.Option("--query", help="Reachability question: 'dead' or 'test-only'."),
    ] = Query.dead,
    venv: Annotated[
        Path | None,
        typer.Option("--venv", help="Venv with editable .pth entries for first-party members."),
    ] = None,
    entrypoint: Annotated[
        list[str] | None,
        typer.Option("-e", "--entrypoint", help="Entrypoint as a file path or FQN."),
    ] = None,
    entrypoint_regex: Annotated[
        list[str] | None,
        typer.Option("--entrypoint-regex", help="Entrypoint as a regex over FQNs / file paths."),
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
    exit_zero: Annotated[
        bool,
        typer.Option("--exit-zero", help="Always exit 0 even when dead code is found."),
    ] = False,
) -> None:
    """Report dead code for a Python codebase."""
    setup_logging(verbose)
    root = root.resolve()

    view, _ = _load_or_build(
        root,
        graph_path=graph,
        venv=venv,
        plugin_names=plugin or [],
        entrypoints=entrypoint or [],
        entrypoint_regexes=entrypoint_regex or [],
    )
    all_nodes = view.nodes()
    dead_nodes = _select_dead(view, query)

    if output_format == OutputFormat.json:
        _output_json(all_nodes, dead_nodes, root)
    else:
        _output_text(all_nodes, dead_nodes, root)

    if not exit_zero and dead_nodes:
        raise typer.Exit(1)


def _output_text(
    all_nodes: Sequence[SymbolNode],
    dead_nodes: Sequence[SymbolNode],
    root: Path,
) -> None:
    total_counts = _count_by_kind(all_nodes)
    dead_counts = _count_by_kind(dead_nodes)
    typer.echo(f"\n{root}:")
    for kind in sorted(total_counts):
        if kind == "synthetic":
            continue
        total = total_counts[kind]
        dead = dead_counts.get(kind, 0)
        if dead > 0:
            typer.echo(f"  {kind}: {total} total, {dead} dead")
        else:
            typer.echo(f"  {kind}: {total} total")

    dead_real = _dead_real(dead_nodes)
    if dead_real:
        typer.echo(f"\nDead symbols ({len(dead_real)}):")
        for node in sorted(dead_real, key=lambda n: (str(n.path), n.fqname)):
            typer.echo(f"  {node.fqname} ({node.kind}) at {_rel_path(node.path, root)}")


def _dead_real(dead_nodes: Iterable[SymbolNode]) -> list[SymbolNode]:
    return [n for n in dead_nodes if n.kind != "synthetic"]


def _output_json(
    all_nodes: Sequence[SymbolNode],
    dead_nodes: Sequence[SymbolNode],
    root: Path,
) -> None:
    total_counts = _count_by_kind(all_nodes)
    dead_counts = _count_by_kind(dead_nodes)
    result: dict = {
        "summary": {
            kind: {"total": total_counts[kind], "dead": dead_counts.get(kind, 0)}
            for kind in total_counts
            if kind != "synthetic"
        },
        "dead_symbols": [],
    }
    for node in sorted(_dead_real(dead_nodes), key=lambda n: (str(n.path), n.fqname)):
        result["dead_symbols"].append(
            {
                "fqname": node.fqname,
                "type": node.kind,
                "path": str(_rel_path(node.path, root)),
            }
        )
    typer.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


@app.command()
def remove(
    root: Annotated[Path, typer.Argument(help="Root directory to analyze.")],
    graph: Annotated[
        Path | None,
        typer.Option("--graph", help="Load a pre-built graph instead of materializing."),
    ] = None,
    query: Annotated[
        Query,
        typer.Option("--query", help="Reachability question: 'dead' or 'test-only'."),
    ] = Query.dead,
    venv: Annotated[
        Path | None,
        typer.Option("--venv", help="Venv with editable .pth entries for first-party members."),
    ] = None,
    entrypoint: Annotated[
        list[str] | None,
        typer.Option("-e", "--entrypoint", help="Entrypoint as a file path or FQN."),
    ] = None,
    entrypoint_regex: Annotated[
        list[str] | None,
        typer.Option("--entrypoint-regex", help="Entrypoint as a regex over FQNs / file paths."),
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
        typer.Option("-o", "--output", help="Write patch to this file instead of stdout."),
    ] = None,
) -> None:
    """Emit a unified diff that removes dead code; pipe to ``git apply``."""
    setup_logging(verbose)
    root = root.resolve()

    view, _ = _load_or_build(
        root,
        graph_path=graph,
        venv=venv,
        plugin_names=plugin or [],
        entrypoints=entrypoint or [],
        entrypoint_regexes=entrypoint_regex or [],
    )
    dead_nodes = _select_dead(view, query)
    patch = generate_patch(dead_nodes, root)

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
