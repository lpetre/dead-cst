"""Command-line interface for dead-cst.

Three commands cover the surface:

* ``dead-cst build ROOT -o PATH`` — materialize the project graph and
  persist it to disk for reuse by later ``analyze`` / ``remove`` runs.
* ``dead-cst analyze ROOT`` — list dead code in text or JSON.
* ``dead-cst remove ROOT`` — emit a ``git apply``-compatible patch
  that drops dead code.

``analyze`` and ``remove`` each take either build inputs (resolver /
plugins / entrypoints) or a pre-built ``--graph PATH`` from a prior
``build`` invocation, and a ``--query`` selector that picks the
reachability question — ``dead`` (the default) or ``test-only`` (code
kept alive only by the test suite, via
:meth:`Analysis.kept_alive_by_flags_only` with ``NodeFlags.TESTCASE``).
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

from .analyze import Analysis, _count_nodes_by_prefix
from .codemod import generate_patch
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
    ExplicitEntrypointPlugin,
    ModuleDundersPlugin,
    Plugin,
    load_plugin,
)
from .resolvers import (
    ManualResolver,
    PathResolver,
    load_resolver,
)

if TYPE_CHECKING:
    from dead_cst import native

    # A graph view the CLI can query for "what's alive": either the
    # live rust context (built this run) or the in-memory wrapper
    # around a loaded graph file. Both expose ``nodes()`` and
    # ``reachable(seed_flags=...)``. Only used as a type alias, so it
    # can live entirely inside ``TYPE_CHECKING`` — references to it
    # in annotations are deferred by ``from __future__ import annotations``.
    GraphView = native.ProjectContext | LoadedGraph


app = typer.Typer(help="Dead code analysis for Python.")


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
    """Compose the plugin list from CLI flags.

    Plain ``-e`` values are passed through as strings (file paths or
    FQNs); ``--entrypoint-regex`` values are compiled to
    :class:`re.Pattern`. The explicit-entrypoint plugin runs last so
    it can pin nodes contributed by upstream plugins.
    """
    plugins: list[Plugin] = [load_plugin(name) for name in plugin_names]
    plugins.append(ModuleDundersPlugin())
    specs: list[str | Path | re.Pattern[str]] = list(entrypoints)
    specs.extend(re.compile(p) for p in entrypoint_regexes)
    if specs:
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


def _materialize(
    root: Path,
    *,
    path_specs: list[str],
    resolver_name: str | None,
    plugin_names: list[str],
    entrypoints: list[str],
    entrypoint_regexes: list[str],
    show_progress: bool,
) -> tuple[native.ProjectContext, Analysis]:
    """Build the project graph from CLI inputs."""
    path_resolver = build_resolver(path_specs, resolver_name)
    plugins = build_plugins(
        entrypoints=entrypoints,
        entrypoint_regexes=entrypoint_regexes,
        plugin_names=plugin_names,
    )
    analysis = Analysis(root, resolver=path_resolver, plugins=plugins, show_progress=show_progress)
    ctx = analysis.materialize_all()
    return ctx, analysis


def _reject_build_inputs_with_graph(
    *,
    graph_path: Path | None,
    path_specs: list[str],
    resolver_name: str | None,
    plugin_names: list[str],
    entrypoints: list[str],
    entrypoint_regexes: list[str],
) -> None:
    if graph_path is None:
        return
    offending: list[str] = []
    if path_specs:
        offending.append("`-p`/`--path`")
    if resolver_name is not None:
        offending.append("`--resolver`")
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
    path_specs: list[str],
    resolver_name: str | None,
    plugin_names: list[str],
    entrypoints: list[str],
    entrypoint_regexes: list[str],
) -> tuple[GraphView, list[Path], GraphMetadata | None]:
    _reject_build_inputs_with_graph(
        graph_path=graph_path,
        path_specs=path_specs,
        resolver_name=resolver_name,
        plugin_names=plugin_names,
        entrypoints=entrypoints,
        entrypoint_regexes=entrypoint_regexes,
    )
    if graph_path is not None:
        typer.echo(f"Loading symbol graph from {graph_path}...", err=True)
        loaded, meta = read_graph(graph_path)
        # Without resolver context, scope summaries to the project root.
        return loaded, [root], meta
    typer.echo(f"Building symbol graph for {root}...", err=True)
    ctx, analysis = _materialize(
        root,
        path_specs=path_specs,
        resolver_name=resolver_name,
        plugin_names=plugin_names,
        entrypoints=entrypoints,
        entrypoint_regexes=entrypoint_regexes,
        show_progress=True,
    )
    return ctx, [p.path for p in analysis.packages], None


def _select_dead(view: GraphView, query: Query) -> list[SymbolNode]:
    """Return the node set that the requested ``--query`` flags as dead.

    * ``dead`` — the complement of ``reachable(KEEPALIVE_DEFAULT)``.
    * ``test-only`` — the blast-radius diff between
      ``reachable(KEEPALIVE_DEFAULT)`` and the same query with the
      ``TESTCASE`` bit cleared from the seed mask. Test functions
      themselves are in the result (they carry ``TESTCASE``).
    """
    if query is Query.dead:
        reachable = set(view.reachable(seed_flags=KEEPALIVE_DEFAULT))
        return [n for n in view.nodes() if n not in reachable]
    if query is Query.test_only:
        full = set(view.reachable(seed_flags=KEEPALIVE_DEFAULT))
        without_tests = set(view.reachable(seed_flags=KEEPALIVE_DEFAULT & ~NodeFlags.TESTCASE))
        return list(full - without_tests)
    raise typer.BadParameter(f"unknown --query value: {query!r}")


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
    entrypoint: Annotated[
        list[str] | None,
        typer.Option("-e", "--entrypoint", help="Entrypoint as a file path or FQN."),
    ] = None,
    entrypoint_regex: Annotated[
        list[str] | None,
        typer.Option("--entrypoint-regex", help="Entrypoint as a regex over FQNs / file paths."),
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
    ctx, _ = _materialize(
        root,
        path_specs=path or [],
        resolver_name=resolver,
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
    entrypoint: Annotated[
        list[str] | None,
        typer.Option("-e", "--entrypoint", help="Entrypoint as a file path or FQN."),
    ] = None,
    entrypoint_regex: Annotated[
        list[str] | None,
        typer.Option("--entrypoint-regex", help="Entrypoint as a regex over FQNs / file paths."),
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
    exit_zero: Annotated[
        bool,
        typer.Option("--exit-zero", help="Always exit 0 even when dead code is found."),
    ] = False,
) -> None:
    """Report dead code for a Python codebase."""
    setup_logging(verbose)
    root = root.resolve()

    view, package_paths, _ = _load_or_build(
        root,
        graph_path=graph,
        path_specs=path or [],
        resolver_name=resolver,
        plugin_names=plugin or [],
        entrypoints=entrypoint or [],
        entrypoint_regexes=entrypoint_regex or [],
    )
    all_nodes = view.nodes()
    dead_nodes = _select_dead(view, query)

    if output_format == OutputFormat.json:
        _output_json(all_nodes, dead_nodes, root, package_paths)
    else:
        _output_text(all_nodes, dead_nodes, root, package_paths)

    if not exit_zero and dead_nodes:
        raise typer.Exit(1)


def _output_text(
    all_nodes: Sequence[SymbolNode],
    dead_nodes: Sequence[SymbolNode],
    root: Path,
    package_paths: Sequence[Path],
) -> None:
    total_by_path = _count_nodes_by_prefix(all_nodes, package_paths)
    unreachable_by_path = _count_nodes_by_prefix(dead_nodes, package_paths)
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
    package_paths: Sequence[Path],
) -> None:
    result: dict = {
        "summary": {},
        "dead_symbols": [],
    }

    total_by_path = _count_nodes_by_prefix(all_nodes, package_paths)
    unreachable_by_path = _count_nodes_by_prefix(dead_nodes, package_paths)
    for path in package_paths:
        path_str = str(path)
        total_counts = total_by_path[path]
        unreachable_counts = unreachable_by_path[path]
        result["summary"][path_str] = {
            kind: {"total": total_counts[kind], "dead": unreachable_counts.get(kind, 0)}
            for kind in total_counts
            if kind != "synthetic"
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
    entrypoint: Annotated[
        list[str] | None,
        typer.Option("-e", "--entrypoint", help="Entrypoint as a file path or FQN."),
    ] = None,
    entrypoint_regex: Annotated[
        list[str] | None,
        typer.Option("--entrypoint-regex", help="Entrypoint as a regex over FQNs / file paths."),
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

    view, _, _ = _load_or_build(
        root,
        graph_path=graph,
        path_specs=path or [],
        resolver_name=resolver,
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
