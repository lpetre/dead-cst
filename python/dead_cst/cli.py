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

import atexit
import json
import logging
import lzma
import os
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterable, Sequence

import typer

from dead_cst import _native as native

from .analyze import Analysis
from .codemod import generate_patch
from .graph import (
    GraphMetadata,
    LoadedGraph,
    SymbolNode,
    read_graph,
    write_graph,
)

if TYPE_CHECKING:
    GraphView = native.ProjectContext | LoadedGraph


app = typer.Typer(help="Dead code analysis for Python.")


def _load_plugin(name: str) -> native.NativePlugin:
    """Resolve a ``--plugin`` name to a :class:`native.NativePlugin`.

    Built-in names (``main_block``, ``flask``, ``pytest``, …) resolve
    through ``_builtin_native_plugin``. Out-of-tree plugins register a
    ``dead_cst.plugins`` entry point whose target is (or returns) a
    :class:`native.NativePlugin` — typically a configured built-in or
    an external dylib plugin loaded via
    :func:`native.load_native_plugins`.
    """
    nat = native._builtin_native_plugin(name)
    if nat is not None:
        return nat

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.plugins"):
        if ep.name == name:
            loaded = ep.load()
            if isinstance(loaded, native.NativePlugin):
                return loaded
            if callable(loaded):
                instance = loaded()
                if isinstance(instance, native.NativePlugin):
                    return instance
            raise TypeError(
                f"Plugin entry point {name!r} did not resolve to a NativePlugin instance "
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
) -> list[native.NativePlugin]:
    plugins: list[native.NativePlugin] = [_load_plugin(name) for name in plugin_names]
    # ``entrypoints`` are exact fqnames / project-relative file paths;
    # ``entrypoint_regexes`` match the project-relative path. The native
    # ``explicit`` plugin takes the two buckets directly (no abs-path specs
    # from the CLI).
    if entrypoints or entrypoint_regexes:
        plugins.append(
            native.NativePlugin.explicit(list(entrypoint_regexes), list(entrypoints), [])
        )
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
    seed_mask = view.default_seed_mask()
    if query is Query.dead:
        reachable = set(view.reachable(seed_flags=seed_mask))
        return [n for n in view.nodes() if n not in reachable]
    if query is Query.test_only:
        testcase = view.node_flag("test/testcase") or 0
        fixture = view.node_flag("test/fixture") or 0
        full = set(view.reachable(seed_flags=seed_mask))
        without_tests = set(view.reachable(seed_flags=seed_mask & ~(testcase | fixture)))
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
        total = total_counts[kind]
        dead = dead_counts.get(kind, 0)
        if dead > 0:
            typer.echo(f"  {kind}: {total} total, {dead} dead")
        else:
            typer.echo(f"  {kind}: {total} total")

    if dead_nodes:
        typer.echo(f"\nDead symbols ({len(dead_nodes)}):")
        for node in sorted(dead_nodes, key=lambda n: (str(n.path), n.fqname)):
            typer.echo(f"  {node.fqname} ({node.kind}) at {_rel_path(node.path, root)}")


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
        },
        "dead_symbols": [],
    }
    for node in sorted(dead_nodes, key=lambda n: (str(n.path), n.fqname)):
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
    """The toolchain dir holding the shared ``libstd-<hash>`` dylib (everything
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


# --- platform abstraction for the shared-runtime plugin build ----------------
#
# The plumbing is the same on every Unix — build everything `-C prefer-dynamic`,
# defer the host (Python + runtime) symbols, and rpath the artifacts so they
# find their sibling dylibs + libstd — but the spellings differ:
#
#                          macOS                      Linux
#   shared-lib suffix      .dylib                     .so
#   defer undefined syms   -Wl,-undefined,            (default for -shared;
#                            dynamic_lookup            no flag needed)
#   loader-relative rpath  @loader_path               $ORIGIN
#   install-name / rpath   install_name_tool, otool   patchelf
#   re-sign after edit     codesign (Apple Silicon)   (n/a)

_IS_MACOS = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")


def _dylib_suffix() -> str:
    """The shared-library suffix for the host: ``.dylib`` / ``.so``."""
    return ".dylib" if _IS_MACOS else ".so"


def _dylib_name(stem: str) -> str:
    """``"dead_cst_runtime"`` -> ``libdead_cst_runtime.{dylib,so}``."""
    return f"lib{stem}{_dylib_suffix()}"


# Curated allowlist of runtime deps exposed to plugin authors via `rustc
# --extern` in `build-plugin`. Each must be a *direct* dependency of the runtime
# crate (so its `.rlib` is guaranteed in the compile closure). The rest of the
# runtime's transitive dep tree is intentionally NOT a stable surface — exposing
# it would leak ruff/ty's private deps as a de-facto public API.
_PLUGIN_EXTERN_CRATES = ("serde_json", "regex")


def _crate_key(filename: str) -> str:
    """Crate name from a cargo ``deps/`` artifact filename, dropping the SVH
    suffix: ``libserde_json-1a2b3c4d.rlib`` (or ``.dylib`` / ``.so`` for a
    proc-macro) -> ``serde_json``. Cargo names every dep artifact
    ``lib<crate>-<hash>.<ext>``, so strip the ``lib`` prefix, the extension, and
    the trailing ``-<hash>``."""
    stem = filename.rsplit(".", 1)[0]
    if stem.startswith("lib"):
        stem = stem[3:]
    return stem.rsplit("-", 1)[0]


def _prefer_dynamic_link_args(std_lib: Path) -> list[str]:
    """The ``-Wl,...`` linker args (without the ``-C link-arg=`` prefix) that
    make a ``prefer-dynamic`` artifact defer host symbols and find libstd + its
    sibling dylibs at load. Returns the raw ``-Wl`` tokens; callers wrap them
    for ``rustc`` argv or fold them into ``RUSTFLAGS``."""
    if _IS_MACOS:
        return [
            # Defer Python (and, for plugins, runtime) symbols to load time.
            "-Wl,-undefined,dynamic_lookup",
            f"-Wl,-rpath,{std_lib}",
            "-Wl,-rpath,@loader_path",
        ]
    # Linux `-shared` already allows undefined symbols (they resolve from the
    # host process / the runtime dylib at load), so no dynamic_lookup equivalent
    # is needed. `--enable-new-dtags` emits DT_RUNPATH so `$ORIGIN` is honored.
    return [
        "-Wl,--enable-new-dtags",
        f"-Wl,-rpath,{std_lib}",
        "-Wl,-rpath,$ORIGIN",
    ]


def _plugin_host_bundle() -> Path | None:
    """The compile-time payload shipped by the ``dead-cst[build-plugin]`` extra
    as the separate ``dead_cst_plugin_host`` package: the ``.rlib`` dependency
    closure + proc-macro dylibs ``rustc`` needs to compile a plugin against the
    runtime, each stored xz-compressed so the wheel clears PyPI's per-file size
    cap. (The runtime dylib + libstd ship in the base ``dead_cst`` wheel.) None
    if the extra isn't installed."""
    try:
        import dead_cst_plugin_host  # ty: ignore[unresolved-import]
    except Exception:
        return None
    bundle = Path(dead_cst_plugin_host.__file__).resolve().parent
    return bundle if any(bundle.glob("*.rlib.xz")) else None


def _materialize_dep_closure(dep_dir: Path) -> Path:
    """``rustc`` needs the dependency closure uncompressed under its original
    filenames, but the shipped ``dead_cst_plugin_host`` payload stores each
    artifact xz-compressed (to keep the wheel under PyPI's per-file size cap).
    Decompress any ``*.xz`` into a temp dir (cleaned at exit) and return it; a
    raw deps dir (e.g. a local ``--runtime-dir`` build) has no ``*.xz`` and is
    returned unchanged."""
    compressed = sorted(dep_dir.glob("*.xz"))
    if not compressed:
        return dep_dir
    staging = Path(tempfile.mkdtemp(prefix="dead-cst-plugin-host-"))
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    for src in compressed:
        with lzma.open(src, "rb") as fsrc, open(staging / src.name[: -len(".xz")], "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
    return staging


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
    deps_dir = target_dir / ("release" if release else "debug") / "deps"
    link_args = _prefer_dynamic_link_args(std_lib)
    env = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target_dir),
        "RUSTFLAGS": " ".join(["-C prefer-dynamic", *(f"-C link-arg={arg}" for arg in link_args)]),
    }
    cmd = ["cargo", "build", "-p", "dead-cst-native"] + (["--release"] if release else [])
    typer.echo(f"$ {' '.join(cmd)}  (prefer-dynamic, dylib-only runtime)", err=True)
    try:
        manifest.write_text(dylib_only)
        subprocess.run(cmd, cwd=root, env=env, check=True)
    finally:
        manifest.write_text(original)
    return deps_dir


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
        typer.Option("-o", "--output", help="Where to write the plugin dylib."),
    ] = None,
    runtime_dir: Annotated[
        Path | None,
        typer.Option(
            "--runtime-dir",
            help="Directory holding the rlib dependency closure (and, optionally, "
            "the runtime dylib). Overrides the installed-package lookup.",
        ),
    ] = None,
    release: Annotated[
        bool, typer.Option("--release", help="Build optimized (release) artifacts.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output.")
    ] = False,
) -> None:
    """Compile an external native plugin against the shipped dead-cst runtime.

    Compiles a single ``.rs`` with ``rustc --extern`` against the dynamic
    ``libdead_cst_runtime`` that ships *inside* the installed ``dead_cst``
    package (the published wheel) plus the ``.rlib`` dependency closure from the
    ``dead-cst[build-plugin]`` extra (``dead_cst_plugin_host``). The host already
    runs that shared runtime, so the built plugin loads with no ``_native`` swap.
    Prints the plugin path on stdout; load it with
    ``native.load_native_plugins(<path>)``.

    A curated allowlist of runtime deps (``serde_json``, ``regex``) is wired in
    via ``--extern`` for plugin authors (``use serde_json::Value;`` / ``use
    regex::Regex;`` work out of the box); the rest of the runtime's private
    dependency tree is intentionally not exposed.

    Needs the pinned rust toolchain (the ABI fingerprint rejects a mismatch at
    load) and the dynamic-runtime wheel (``pip install dead-cst[build-plugin]``).
    Compiles a single ``.rs`` (multi-crate plugins are a follow-up); macOS + Linux.
    """
    setup_logging(verbose)
    if not (_IS_MACOS or _IS_LINUX):
        raise typer.BadParameter("build-plugin currently supports macOS and Linux only.")
    if shutil.which("rustc") is None:
        raise typer.BadParameter("rustc not found on PATH; the rust toolchain is required.")

    std_lib = _host_std_lib()

    # The shared runtime dylib ships inside the installed dead_cst package; the
    # rlib dependency closure ships in the dead_cst_plugin_host extra.
    # --runtime-dir overrides both (e.g. a local prefer-dynamic build's deps dir).
    from dead_cst import _native

    runtime_name = _dylib_name("dead_cst_runtime")
    in_package_runtime = Path(_native.__file__).parent / runtime_name
    if runtime_dir is not None:
        dep_dir = runtime_dir.resolve()
        runtime_dylib = dep_dir / runtime_name
        if not runtime_dylib.is_file():
            runtime_dylib = in_package_runtime
    else:
        runtime_dylib = in_package_runtime
        bundle = _plugin_host_bundle()
        if bundle is None:
            raise typer.BadParameter(
                "dead_cst_plugin_host not found; install the extra: "
                "pip install dead-cst[build-plugin]."
            )
        dep_dir = bundle
    if not runtime_dylib.is_file():
        raise typer.BadParameter(
            f"shared runtime dylib not found at {runtime_dylib}; build-plugin needs the "
            "dynamic-runtime wheel (pip install dead-cst[build-plugin])."
        )

    # The shipped closure is xz-compressed (PyPI per-file size cap); rustc needs
    # it decompressed. No-op for a raw local deps dir.
    dep_dir = _materialize_dep_closure(dep_dir)

    # Resolve the plugin source (default: bundled example from a source checkout).
    if plugin_src is None:
        root = _detect_source_root()
        if root is None:
            raise typer.BadParameter("no plugin source given (and no checkout for the example).")
        plugin_src = root / "examples" / "main_block_plugin" / "src" / "lib.rs"
    src = plugin_src.resolve()
    if not src.is_file():
        raise typer.BadParameter(f"plugin source not found: {src}")

    if name is None:
        name = src.parent.parent.name if src.stem in {"lib", "mod", "main"} else src.stem
    crate_name = name.replace("-", "_")
    out = output.resolve() if output is not None else Path.cwd() / _dylib_name(crate_name)

    # Curated allowlist (`_PLUGIN_EXTERN_CRATES`): expose these direct runtime
    # deps to plugin authors via `--extern` (their rlibs are always in the
    # closure). Newest wins if a deps dir holds stale SVH-suffixed copies; an
    # absent crate (unexpected) is skipped rather than fatal.
    exposed_externs: list[str] = []
    for crate in _PLUGIN_EXTERN_CRATES:
        rlibs = sorted(dep_dir.glob(f"lib{crate}-*.rlib"), key=lambda p: p.stat().st_mtime)
        if rlibs:
            exposed_externs += ["--extern", f"{crate}={rlibs[-1]}"]
        elif verbose:
            typer.echo(
                f"note: {crate} rlib not found in dep dir; --extern {crate} skipped.", err=True
            )

    # --extern reads the runtime dylib's embedded metadata; -L finds the dep
    # rlibs/proc-macro dylibs; undefined Python + runtime symbols resolve from
    # the (already-loaded, shared) runtime at load.
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
        *exposed_externs,
        "-L",
        f"dependency={dep_dir}",
        *(
            arg
            for link_arg in _prefer_dynamic_link_args(std_lib)
            for arg in ("-C", f"link-arg={link_arg}")
        ),
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

    # Path on stdout so it can be captured: PLUGIN=$(dead-cst build-plugin ...).
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# bundle-plugin-host
# ---------------------------------------------------------------------------


@app.command(name="bundle-plugin-host")
def bundle_plugin_host(
    output: Annotated[
        Path | None,
        typer.Option(
            "-o", "--output", help="Output dir (default: the dead_cst_plugin_host package)."
        ),
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
    """Populate the ``dead_cst_plugin_host`` extra with the plugin-compile closure.

    Builds the runtime ``-C prefer-dynamic`` and copies the ``.rlib`` dependency
    closure + proc-macro dylibs that ``rustc`` needs to compile a plugin against
    the runtime into the ``dead_cst_plugin_host`` package (``plugin-host/``) — the
    payload the ``dead-cst[build-plugin]`` extra ships. The runtime dylib + libstd
    are **not** copied here: they ride in the base ``dead_cst`` wheel (the publish
    workflow repacks them in from this same build's ``deps/``, so their SVH
    matches this closure). Requires a source checkout + rust. Prints the populated
    package dir on stdout.
    """
    setup_logging(verbose)
    if not (_IS_MACOS or _IS_LINUX):
        raise typer.BadParameter("bundle-plugin-host currently supports macOS and Linux only.")
    for tool in ("cargo", "rustc"):
        if shutil.which(tool) is None:
            raise typer.BadParameter(f"{tool} not found on PATH (need rust).")

    root = source_root.resolve() if source_root is not None else _detect_source_root()
    if root is None:
        raise typer.BadParameter("a source checkout is required; pass --source-root.")

    std_lib = _host_std_lib()
    suffix = _dylib_suffix()
    deps_dir = _build_runtime_from_source(root, release=release, std_lib=std_lib)

    if output is not None:
        bundle = output.resolve()
    else:
        # Default: populate the separate `dead_cst_plugin_host` package that the
        # `dead-cst[build-plugin]` extra ships.
        bundle = root / "plugin-host" / "dead_cst_plugin_host"

    bundle.mkdir(parents=True, exist_ok=True)
    # Clean only prior build artifacts — keep package files like __init__.py.
    for stale in (*bundle.glob("*.rlib"), *bundle.glob(f"*{suffix}"), *bundle.glob("*.xz")):
        stale.unlink()

    # The plugin-compile closure: every dependency `.rlib` + the proc-macro
    # dylibs (rustc loads these to validate the crate graph). The runtime dylib,
    # the dynamic `_native`, and libstd are EXCLUDED — they ship in the base
    # `dead_cst` wheel. (.rmeta are skipped: redundant with the .rlib, which
    # embed metadata, and would nearly double the payload.)
    #
    # Dedup by (crate, kind): cargo's deps/ accumulates multiple SVH-suffixed
    # artifacts per crate across incremental rebuilds (this target dir is reused,
    # not cleaned, to keep rebuilds fast). Ship exactly one per (crate, kind) —
    # the newest, which is the set this build's runtime dylib actually binds
    # against (mtime is a faithful proxy right after a successful build).
    # Without this, stale copies (e.g. a second `regex` rlib) leak in and bloat
    # the wheel — and a `--extern <crate>` glob in `build-plugin` could pick the
    # wrong SVH.
    excluded = {_dylib_name("dead_cst_runtime"), _dylib_name("dead_cst_native")}
    newest: dict[tuple[str, str], Path] = {}
    for entry in deps_dir.iterdir():
        if not entry.is_file():
            continue
        is_proc_macro = (
            entry.suffix == suffix
            and entry.name not in excluded
            and not entry.name.startswith("libstd-")
        )
        if entry.suffix == ".rlib" or is_proc_macro:
            key = (_crate_key(entry.name), entry.suffix)
            cur = newest.get(key)
            if cur is None or entry.stat().st_mtime > cur.stat().st_mtime:
                newest[key] = entry
    # Store each artifact xz-compressed (`<name>.xz`). A wheel is a zip, and the
    # raw `.rlib` closure deflates to ~107 MB — over PyPI's 100 MB/file cap. The
    # bulk is `lib.rmeta` crate metadata embedded in each rlib, which can't be
    # stripped without breaking `rustc --extern`. xz packs far tighter than zip's
    # deflate (~70 MB here); `build-plugin` decompresses via `_materialize_dep_closure`
    # before handing paths to rustc. stdlib `lzma` keeps `[build-plugin]` dep-free.
    n_files = 0
    raw_bytes = 0
    xz_bytes = 0
    for entry in newest.values():
        dest = bundle / f"{entry.name}.xz"
        with (
            open(entry, "rb") as fsrc,
            lzma.open(dest, "wb", preset=9 | lzma.PRESET_EXTREME) as fdst,
        ):
            shutil.copyfileobj(fsrc, fdst)
        n_files += 1
        raw_bytes += entry.stat().st_size
        xz_bytes += dest.stat().st_size

    typer.echo(f"plugin-host closure: {bundle}  (deps build: {deps_dir})", err=True)
    typer.echo(
        f"  {n_files} rlib / proc-macro artifacts, xz-compressed "
        f"{raw_bytes / 1_000_000:.1f} MB -> {xz_bytes / 1_000_000:.1f} MB "
        f"(runtime dylib + libstd ship in dead_cst)",
        err=True,
    )
    typer.echo(str(bundle))


def main_cli() -> None:
    app()


if __name__ == "__main__":
    main_cli()
