"""Benchmark ``Analysis.materialize_all()`` over many packages and plugins.

Generates a synthetic workspace under a tempdir, runs the analysis
cold (visitor + compose) and warm (cache hit, compose only), and
optionally cProfiles the warm path to surface plugin / edge-stitching
costs. Defaults to 10 packages × 20 files (~210 files) with every
builtin plugin enabled. ``--no-plugins`` runs the same workload with
no plugins so you can subtract the plugin overhead.

Usage:
    uv run python scripts/profile_materialize.py
    uv run python scripts/profile_materialize.py --packages 20 --files 40 --profile
    uv run python scripts/profile_materialize.py --no-plugins
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import tempfile
import time
from pathlib import Path

from dead_cst import Analysis
from dead_cst.cache import GraphCache, default_cache_path
from dead_cst.plugins import BUILTIN_PLUGINS, ExplicitEntrypointPlugin
from dead_cst.resolvers import ManualResolver


def _generate_workspace(root: Path, num_packages: int, files_per_package: int) -> None:
    """Lay out ``num_packages`` packages with ``files_per_package`` modules each.

    Each package lives in its own ``base_p/`` directory under ``root``
    holding a subpackage ``pkg_p/`` -- libcst's FQN provider needs the
    file to be inside a real Python subpackage (not the package root)
    for relative imports to resolve. Each ``mod_i.py`` includes:

    * ``__main__`` block (seeds for :class:`MainBlockPlugin`)
    * a same-package relative import (intra-package edge)
    * a cross-package import to ``pkg_{p-1}`` for ``p > 0`` (real
      cross-package edge through :func:`resolve_edges`)
    """
    for p in range(num_packages):
        sub_dir = root / f"base_{p}" / f"pkg_{p}"
        sub_dir.mkdir(parents=True)
        (sub_dir / "__init__.py").write_text("")
        for f in range(files_per_package):
            lines: list[str] = []
            if f > 0:
                lines.append(f"from .mod_{f - 1} import C as Sibling")
            if p > 0:
                lines.append(f"from pkg_{p - 1}.mod_0 import C as Base")
            base_clause = "(Base)" if p > 0 else ""
            lines += [
                "",
                f"class C{base_clause}:",
                "    def method(self) -> int:",
                f"        return {f}",
                "",
                "",
                "def helper() -> int:",
                "    return C().method()",
                "",
                "",
                'if __name__ == "__main__":',
                "    helper()",
                "",
            ]
            (sub_dir / f"mod_{f}.py").write_text("\n".join(lines))


def _make_analysis(
    root: Path,
    num_packages: int,
    *,
    cache: GraphCache | None,
    with_plugins: bool,
) -> Analysis:
    specs: list[str] = ["base_0"]
    for p in range(1, num_packages):
        specs.append(f"base_{p}:base_{p - 1}")
    plugins: list = []
    if with_plugins:
        plugins = [p for p in BUILTIN_PLUGINS if not isinstance(p, ExplicitEntrypointPlugin)]
        plugins.append(ExplicitEntrypointPlugin(specs=["base_0.pkg_0"]))
    return Analysis(
        project_root=root,
        resolver=ManualResolver(specs=specs),
        plugins=plugins,
        cache=cache,
    )


def _run_materialize(root: Path, num_packages: int, *, with_plugins: bool) -> float:
    db = default_cache_path(root)
    with GraphCache(db) as cache:
        analysis = _make_analysis(root, num_packages, cache=cache, with_plugins=with_plugins)
        t0 = time.perf_counter()
        analysis.materialize_all()
        return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--packages", type=int, default=10)
    parser.add_argument("--files", type=int, default=20)
    parser.add_argument(
        "--no-plugins",
        action="store_true",
        help="Run with zero plugins (subtract from default run)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="cProfile the warm path and print hot functions"
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=5,
        help="Number of warm runs to profile when --profile is set",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _generate_workspace(root, args.packages, args.files)

        with_plugins = not args.no_plugins
        plugin_count = (
            len([p for p in BUILTIN_PLUGINS if not isinstance(p, ExplicitEntrypointPlugin)]) + 1
        )
        n_files = args.packages * (args.files + 1)
        print(f"workspace: {args.packages} packages × {args.files + 1} files = {n_files} files")
        print(f"plugins:   {plugin_count if with_plugins else 0}")

        cold = _run_materialize(root, args.packages, with_plugins=with_plugins)
        print(f"\ncold materialize_all: {cold * 1000:8.0f} ms (visitor + compose + finalize)")

        warm = _run_materialize(root, args.packages, with_plugins=with_plugins)
        print(f"warm materialize_all: {warm * 1000:8.0f} ms (cache hit, compose + finalize only)")
        print(f"visitor share:        {(cold - warm) * 1000:8.0f} ms")
        print(f"compose+finalize:     {warm * 1000:8.0f} ms")

        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(args.profile_repeats):
                _run_materialize(root, args.packages, with_plugins=with_plugins)
            profiler.disable()
            print(f"\nTop 25 functions by cumulative time (warm × {args.profile_repeats}):")
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
            print("\nTop 25 functions by internal (tottime):")
            pstats.Stats(profiler).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
