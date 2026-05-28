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
import os
import re
import shutil
import subprocess
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
from .contrib.slack_bolt import slack_bolt_plugin
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
    "slack_bolt": slack_bolt_plugin(),
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


# ---------------------------------------------------------------------------
# build-plugin
# ---------------------------------------------------------------------------


def _detect_source_root() -> Path | None:
    """The dead-cst source checkout, or None when running from a wheel.

    In an editable (``maturin develop``) layout this file lives at
    ``<root>/python/dead_cst/cli.py``; walk up to the dir that holds both the
    workspace ``Cargo.toml`` and the ``runtime`` crate.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (parent / "runtime" / "Cargo.toml").is_file():
            return parent
    return None


def _rustc_print(*args: str) -> str:
    return subprocess.run(
        ["rustc", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _host_std_lib() -> Path:
    """The toolchain dir holding the shared ``libstd-<hash>.dylib`` (everything
    is built ``-C prefer-dynamic``, so artifacts rpath here)."""
    sysroot = Path(_rustc_print("--print", "sysroot"))
    host = next(
        (
            line.split("host: ", 1)[1]
            for line in _rustc_print("-vV").splitlines()
            if line.startswith("host: ")
        ),
        None,
    )
    if host is None:
        raise typer.BadParameter("could not determine the host target triple from `rustc -vV`.")
    return sysroot / "lib" / "rustlib" / host / "lib"


def _plugin_host_bundle() -> Path | None:
    """A plugin-host bundle shipped inside the installed package
    (``libdead_cst_runtime.dylib`` + dep ``*.rmeta`` [+ the dynamic ``_native``]),
    as a future ``dead-cst[plugin-host]`` wheel would drop in. None if absent."""
    try:
        from dead_cst import _native
    except Exception:
        return None
    bundle = Path(_native.__file__).parent / "_plugin_host"
    return bundle if (bundle / "libdead_cst_runtime.dylib").is_file() else None


def _build_runtime_from_source(root: Path, *, release: bool, std_lib: Path) -> Path:
    """Build the runtime dylib + dep metadata (+ the dynamic ``_native``) from a
    source checkout into ``target/plugin-host``; return the deps dir."""
    if shutil.which("cargo") is None:
        raise typer.BadParameter(
            "cargo not found on PATH; needed to build the runtime from source."
        )
    manifest = root / "runtime" / "Cargo.toml"
    if not manifest.is_file():
        raise typer.BadParameter(f"no runtime crate at {manifest}; is --source-root correct?")
    original = manifest.read_text()
    # Build the runtime dylib-ONLY: under prefer-dynamic a dep available as both
    # rlib and dylib makes the cdylib bind the rlib's SVH while the loader
    # resolves the dylib. Restore the manifest afterward.
    dylib_only = original.replace('crate-type = ["rlib", "dylib"]', 'crate-type = ["dylib"]')
    if dylib_only == original:
        raise typer.BadParameter(
            "could not switch runtime crate-type to dylib-only "
            '(expected \'crate-type = ["rlib", "dylib"]\' in runtime/Cargo.toml).'
        )
    target_dir = root / "target" / "plugin-host"
    env = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target_dir),
        "RUSTFLAGS": " ".join(
            [
                "-C prefer-dynamic",
                "-C link-arg=-Wl,-undefined,dynamic_lookup",
                f"-C link-arg=-Wl,-rpath,{std_lib}",
                "-C link-arg=-Wl,-rpath,@loader_path",
            ]
        ),
    }
    cmd = ["cargo", "build", "-p", "dead-cst-native"] + (["--release"] if release else [])
    typer.echo(f"$ {' '.join(cmd)}  (prefer-dynamic, dylib-only runtime)", err=True)
    try:
        manifest.write_text(dylib_only)
        subprocess.run(cmd, cwd=root, env=env, check=True)
    finally:
        manifest.write_text(original)
    return target_dir / ("release" if release else "debug") / "deps"


@app.command(name="build-plugin")
def build_plugin(
    plugin_src: Annotated[
        Path | None,
        typer.Argument(
            help="Path to the plugin's Rust source (.rs). Defaults to the bundled "
            "example when run from a source checkout."
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Crate / output name (default: derived from the source)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Where to write the plugin .so."),
    ] = None,
    runtime_dir: Annotated[
        Path | None,
        typer.Option(
            "--runtime-dir",
            help="Directory with a prebuilt libdead_cst_runtime.dylib + dep *.rmeta. "
            "Overrides the bundle / source-checkout lookup.",
        ),
    ] = None,
    source_root: Annotated[
        Path | None,
        typer.Option(
            "--source-root", help="dead-cst source checkout (to build the runtime if needed)."
        ),
    ] = None,
    release: Annotated[
        bool, typer.Option("--release", help="Build optimized (release) artifacts.")
    ] = False,
    install: Annotated[
        bool,
        typer.Option(
            "--install/--no-install",
            help="Install the matching dynamic _native over the editable extension "
            "(so the host and plugin share one runtime).",
        ),
    ] = True,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Compile an external native plugin against the dead-cst runtime.

    The plugin is compiled with ``rustc --extern`` against a *prebuilt* runtime
    dylib + its metadata — no Cargo project, no runtime source, no ruff
    recompile. Runtime artifacts are resolved, in order, from: ``--runtime-dir``,
    a plugin-host bundle shipped in the installed package, or a source checkout
    (built on demand). Prints the plugin ``.so`` path on stdout; load it with
    ``native.load_native_plugins(<path>)``.

    Compiles a single ``.rs`` (multi-crate plugins are a follow-up). macOS only
    for now. Needs the rust toolchain at the version dead-cst was built with —
    the ABI fingerprint rejects a mismatch at load.
    """
    setup_logging(verbose)
    if sys.platform != "darwin":
        raise typer.BadParameter("build-plugin currently supports macOS only.")
    if shutil.which("rustc") is None:
        raise typer.BadParameter("rustc not found on PATH; the rust toolchain is required.")

    std_lib = _host_std_lib()
    root = source_root.resolve() if source_root is not None else _detect_source_root()

    # Resolve the runtime artifacts: a deps dir (dylib + dep *.rmeta), an
    # optional dynamic _native to install, and where to drop the output.
    native_to_install: Path | None
    if runtime_dir is not None:
        deps_dir = runtime_dir.resolve()
        cand = deps_dir / "libdead_cst_native.dylib"
        native_to_install = cand if cand.is_file() else None
        out_dir = Path.cwd()
    elif (bundle := _plugin_host_bundle()) is not None:
        deps_dir = bundle
        cand = bundle / "libdead_cst_native.dylib"
        native_to_install = cand if cand.is_file() else None
        out_dir = Path.cwd()
    elif root is not None:
        deps_dir = _build_runtime_from_source(root, release=release, std_lib=std_lib)
        native_to_install = deps_dir.parent / "libdead_cst_native.dylib"
        out_dir = deps_dir.parent
    else:
        raise typer.BadParameter(
            "no runtime artifacts: pass --runtime-dir, install a plugin-host bundle, "
            "or run from a dead-cst source checkout."
        )

    runtime_dylib = deps_dir / "libdead_cst_runtime.dylib"
    if not runtime_dylib.is_file():
        raise typer.BadParameter(f"runtime dylib not found at {runtime_dylib}.")

    # Resolve the plugin source.
    if plugin_src is None:
        if root is None:
            raise typer.BadParameter("no plugin source given (and no checkout for the example).")
        plugin_src = root / "examples" / "main_block_plugin" / "src" / "lib.rs"
    src = plugin_src.resolve()
    if not src.is_file():
        raise typer.BadParameter(f"plugin source not found: {src}")

    if name is None:
        name = src.parent.parent.name if src.stem in {"lib", "mod", "main"} else src.stem
    crate_name = name.replace("-", "_")
    out = output.resolve() if output is not None else out_dir / f"lib{crate_name}.dylib"

    # Compile against the prebuilt runtime — no cargo, no runtime source, no
    # ruff recompile. --extern reads the dylib's metadata; -L finds the dep
    # *.rmeta; dynamic_lookup defers Python symbols; the symbols resolve from
    # the (shared) runtime dylib at load.
    cmd = [
        "rustc",
        "--edition",
        "2021",
        "--crate-type",
        "cdylib",
        "--crate-name",
        crate_name,
        "-C",
        "prefer-dynamic",
        "--extern",
        f"dead_cst_runtime={runtime_dylib}",
        "-L",
        f"dependency={deps_dir}",
        "-C",
        "link-arg=-Wl,-undefined,dynamic_lookup",
        "-C",
        f"link-arg=-Wl,-rpath,{std_lib}",
        "-C",
        "link-arg=-Wl,-rpath,@loader_path",
        *(["-O"] if release else []),
        str(src),
        "-o",
        str(out),
    ]
    typer.echo(f"$ rustc --extern dead_cst_runtime=… {src.name} -o {out.name}", err=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode or 1) from exc
    if not out.is_file():
        typer.echo(f"error: plugin artifact not produced at {out}", err=True)
        raise typer.Exit(code=1)

    if install and native_to_install is not None and native_to_install.is_file():
        from dead_cst import _native

        installed = Path(_native.__file__)
        shutil.copyfile(native_to_install, installed)
        typer.echo(f"Installed dynamic _native -> {installed}", err=True)
        typer.echo("Restore the default static build with: uv run maturin develop --uv", err=True)
    elif install and native_to_install is None:
        typer.echo(
            "note: no dynamic _native found to install; ensure the active _native "
            "shares this runtime before loading the plugin.",
            err=True,
        )

    # Path on stdout so it can be captured: PLUGIN=$(dead-cst build-plugin ...).
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# bundle-plugin-host
# ---------------------------------------------------------------------------


def _otool_dep(dylib: Path, needle: str) -> str | None:
    """The first ``otool -L`` load-command path on ``dylib`` containing ``needle``."""
    out = subprocess.run(
        ["otool", "-L", str(dylib)], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines()[1:]:
        path = line.strip().split(" ", 1)[0]
        if needle in path:
            return path
    return None


@app.command(name="bundle-plugin-host")
def bundle_plugin_host(
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Bundle directory (default: dead_cst/_plugin_host)."),
    ] = None,
    source_root: Annotated[
        Path | None,
        typer.Option("--source-root", help="dead-cst source checkout (default: auto-detected)."),
    ] = None,
    release: Annotated[
        bool, typer.Option("--release", help="Build optimized (release) artifacts.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Assemble a relocatable plugin-host bundle.

    Builds the runtime and gathers ``libdead_cst_runtime.dylib`` + the dep
    ``*.rmeta`` closure + the toolchain's ``libstd`` + the dynamic ``_native``
    into one directory, with install names / rpaths rewritten to
    ``@rpath`` / ``@loader_path`` so it relocates. Dropped at
    ``dead_cst/_plugin_host`` by default, where ``build-plugin`` auto-detects it
    — this is the payload a future ``dead-cst[plugin-host]`` wheel would ship.
    macOS only; requires a source checkout + the Xcode command line tools.
    """
    setup_logging(verbose)
    if sys.platform != "darwin":
        raise typer.BadParameter("bundle-plugin-host currently supports macOS only.")
    for tool in ("cargo", "rustc", "install_name_tool", "codesign", "otool", "strip"):
        if shutil.which(tool) is None:
            raise typer.BadParameter(f"{tool} not found on PATH (need rust + Xcode CLT).")

    root = source_root.resolve() if source_root is not None else _detect_source_root()
    if root is None:
        raise typer.BadParameter(
            "a source checkout is required to build the bundle; pass --source-root."
        )

    std_lib = _host_std_lib()
    libstds = sorted(std_lib.glob("libstd-*.dylib"))
    if not libstds:
        raise typer.BadParameter(f"no libstd-*.dylib under {std_lib}.")
    libstd = libstds[0]

    deps_dir = _build_runtime_from_source(root, release=release, std_lib=std_lib)
    runtime_src = deps_dir / "libdead_cst_runtime.dylib"
    native_src = deps_dir / "libdead_cst_native.dylib"
    for artifact in (runtime_src, native_src):
        if not artifact.is_file():
            raise typer.BadParameter(f"missing build artifact: {artifact}")

    if output is not None:
        bundle = output.resolve()
    else:
        from dead_cst import _native

        bundle = Path(_native.__file__).parent / "_plugin_host"

    shutil.rmtree(bundle, ignore_errors=True)
    bundle.mkdir(parents=True)

    # Copy the dependency closure rustc needs to load dead_cst_runtime via
    # --extern: the .rlib of every dep + the proc-macro dylibs (required to
    # validate the crate graph) + the runtime dylib + the dynamic _native (all
    # under deps/). The .rmeta are skipped — they're redundant with the .rlib
    # (which embed metadata) and would nearly double the bundle. Then add libstd.
    n_files = 0
    for entry in deps_dir.iterdir():
        if entry.is_file() and entry.suffix in {".rlib", ".dylib"}:
            shutil.copy2(entry, bundle / entry.name)
            n_files += 1
    shutil.copy2(libstd, bundle / libstd.name)

    runtime_bundled = bundle / "libdead_cst_runtime.dylib"
    native_bundled = bundle / "libdead_cst_native.dylib"

    def _imt(*args: str) -> None:
        subprocess.run(["install_name_tool", *args], check=True)

    def _rpaths(dylib: Path) -> set[str]:
        out = subprocess.run(
            ["otool", "-l", str(dylib)], capture_output=True, text=True, check=True
        ).stdout
        lines = out.splitlines()
        found: set[str] = set()
        for i, line in enumerate(lines):
            if "LC_RPATH" in line:
                for follow in lines[i : i + 4]:
                    s = follow.strip()
                    if s.startswith("path "):
                        found.add(s[len("path ") :].split(" (offset")[0].strip())
        return found

    def _ensure_rpath(dylib: Path, rpath: str) -> None:
        # The build already bakes @loader_path; install_name_tool errors on a
        # duplicate, so only add what's missing.
        if rpath not in _rpaths(dylib):
            _imt("-add_rpath", rpath, str(dylib))

    def _drop_rpath(dylib: Path, rpath: str) -> None:
        # Strip the builder's absolute sysroot rpath so the bundle relocates;
        # libstd is found via the @loader_path rpaths instead.
        if rpath in _rpaths(dylib):
            _imt("-delete_rpath", rpath, str(dylib))

    # runtime: relocatable id + rpath to find its sibling libstd.
    _imt("-id", "@rpath/libdead_cst_runtime.dylib", str(runtime_bundled))
    _ensure_rpath(runtime_bundled, "@loader_path")
    _drop_rpath(runtime_bundled, str(std_lib))

    # dynamic _native: relocatable id; ref the runtime via @rpath; rpath into
    # _plugin_host/ for when it's installed one level up at
    # dead_cst/_native.abi3.so.
    _imt("-id", "@rpath/libdead_cst_native.dylib", str(native_bundled))
    old_runtime_ref = _otool_dep(native_bundled, "libdead_cst_runtime")
    if old_runtime_ref is not None:
        _imt("-change", old_runtime_ref, "@rpath/libdead_cst_runtime.dylib", str(native_bundled))
    _ensure_rpath(native_bundled, "@loader_path/_plugin_host")
    _drop_rpath(native_bundled, str(std_lib))

    # Strip local symbols, then ad-hoc re-sign every dylib. Both install_name_tool
    # and strip invalidate the code signature; dyld (and rustc, loading the
    # proc-macro dylibs) reject edited, unsigned dylibs on Apple Silicon.
    for dylib in sorted(bundle.glob("*.dylib")):
        subprocess.run(["strip", "-x", str(dylib)], check=True)
        subprocess.run(["codesign", "-s", "-", "-f", str(dylib)], check=True)

    # libstd's id is already @rpath; its copy keeps its valid signature.

    typer.echo(f"plugin-host bundle: {bundle}", err=True)
    typer.echo(
        f"  {n_files} dep artifacts + {libstd.name} (install names rewritten, ad-hoc signed)",
        err=True,
    )
    typer.echo(
        "  `dead-cst build-plugin <PLUGIN.rs>` now auto-detects it (no source needed).", err=True
    )
    typer.echo(str(bundle))


def main_cli() -> None:
    app()


if __name__ == "__main__":
    main_cli()
