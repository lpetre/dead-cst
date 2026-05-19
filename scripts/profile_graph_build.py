"""Thorough perf profile of rust graph building (no plugins).

Measures ``ProjectContext.materialize()`` end-to-end and breaks it
down into the rust-side build phases that the
``DEAD_CST_TIMING=1`` env var exposes:

* ``enum``       — project-files enumeration + ``path_to_file`` /
                   peer-``.pyi`` registration
* ``phase1``     — per-file decl ingest (parse + ``SemanticIndex`` +
                   intern global-scope bindings as nodes)
* ``dist_lookup``— site-packages ``.dist-info`` walk to map third-party
                   files to canonical distribution names
* ``phase2``     — cross-file import-edge emission (alias → upstream)
* ``phase3``     — per-file reference-edge emission (Name → decl)
* ``fqname``     — post-pass ``fqname → idx`` index build

Two run modes per target:

* **cold** — a fresh ``ProjectContext`` per iteration. ty's Salsa db
  starts empty so every phase pays its real first-call cost. This is
  what the CLI's typical "run once on commit" path looks like.
* **warm** — the same ``ProjectContext`` reused. Salsa returns
  memoised parses / semantic indexes so the per-phase cost collapses
  to the work that doesn't go through Salsa (mostly node interning
  and Py allocation).

Each mode runs one warm-up iteration (the first call hits process-
global typeshed loading) plus ``--repeats`` steady-state iterations.
Steady-state reports best / mean / stdev so you can see how much
the numbers are jumping around.

Usage::

    uv run python scripts/profile_graph_build.py
    uv run python scripts/profile_graph_build.py --repeats 6
    uv run python scripts/profile_graph_build.py --targets flux0_workspace
    uv run python scripts/profile_graph_build.py --synthetic 50 200
    uv run python scripts/profile_graph_build.py --no-flux0

The rust extension must be importable; the script exits early
otherwise — measuring the python-only fallback isn't useful for a
rust-build profile.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

from _bench_common import (
    TargetConfig,
    add_common_target_args,
    file_count,
    gather_flux0_targets,
    require_native,
    stage_dead_cst,
)

# Silence the visitor's WARNING-level breadcrumbs.
logging.getLogger("dead_cst").setLevel(logging.ERROR)

require_native()
from dead_cst import native  # noqa: E402  (require_native gates this)


def _generate_synthetic(num_files: int) -> Path:
    """Generate a synthetic project with ``num_files`` modules in a
    single package, each importing a couple of siblings so the graph
    has realistic cross-file fan-out.

    Each module gets one class, one function, and ~2 cross-file
    from-imports — enough that every build phase has something to chew
    on, without standing up a real project or venv."""
    stage = Path(tempfile.mkdtemp(prefix=f"synthetic-{num_files}-"))
    pkg = stage / "synth"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for i in range(num_files):
        lines: list[str] = []
        for j in range(max(0, i - 2), i):
            lines.append(f"from .mod_{j} import C as Sib{j}")
        lines.extend(
            [
                "",
                f"class C{'(Sib%d)' % (i - 1) if i > 0 else ''}:",
                "    def method(self) -> int:",
                f"        return {i}",
                "",
                "",
                "def helper() -> int:",
                "    return C().method()",
                "",
                'if __name__ == "__main__":',
                "    helper()",
                "",
            ]
        )
        (pkg / f"mod_{i}.py").write_text("\n".join(lines))
    return stage


# ---------------------------------------------------------------------------
# Per-phase timing
# ---------------------------------------------------------------------------

# `[dead-cst-timing] files=8 nodes=175 edges=487 enum=2ms phase1=4ms dist_lookup=89ms phase2=4ms phase3=5ms fqname=84µs total=106ms`
_TIMING_RE = re.compile(r"\[dead-cst-timing\]\s+(.*)")
_PHASE_KEYS = ("enum", "phase1", "dist_lookup", "phase2", "phase3", "fqname", "total")


def _parse_duration(s: str) -> float:
    """Parse a rust ``{:?}`` Duration ("12.345ms" / "789µs" / "1.2s")
    into seconds."""
    units = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    for suffix, mul in units.items():
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mul
    raise ValueError(f"unrecognised duration {s!r}")


def _parse_timing_line(line: str) -> dict[str, float] | None:
    """Parse one `[dead-cst-timing]` line into a {phase: seconds} map
    (plus ``files`` / ``nodes`` / ``edges`` counts)."""
    m = _TIMING_RE.search(line)
    if not m:
        return None
    out: dict[str, float] = {}
    for token in m.group(1).strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if k in ("files", "nodes", "edges"):
            out[k] = float(v)
        elif k in _PHASE_KEYS:
            out[k] = _parse_duration(v)
    return out


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def _materialize(
    target: TargetConfig, *, reuse: native.ProjectContext | None
) -> tuple[float, native.ProjectContext]:
    """Run one materialize. If ``reuse`` is None build a fresh
    ProjectContext (cold). Returns ``(wall_seconds, ctx)`` so warm
    runs can reuse the ctx."""
    ctx = (
        reuse
        if reuse is not None
        else native.ProjectContext(str(target.root), **target.project_kwargs)
    )
    t0 = time.perf_counter()
    ctx.materialize()
    return time.perf_counter() - t0, ctx


def _bench(target: TargetConfig, repeats: int, *, mode: str) -> tuple[float, list[float]]:
    """Returns ``(warmup_seconds, steady_seconds_list)`` for either
    ``mode="cold"`` or ``mode="warm"``. Warm mode reuses one
    ``ProjectContext`` across all iterations (warm-up + steady)."""
    if mode == "warm":
        # Build once, prime, then measure the same ctx repeatedly.
        ctx = native.ProjectContext(str(target.root), **target.project_kwargs)
        warmup, ctx = _materialize(target, reuse=ctx)
        steady = [_materialize(target, reuse=ctx)[0] for _ in range(repeats)]
        return warmup, steady
    # cold
    warmup, _ = _materialize(target, reuse=None)
    steady = [_materialize(target, reuse=None)[0] for _ in range(repeats)]
    return warmup, steady


def _bench_with_phases(target: TargetConfig, repeats: int) -> list[dict[str, float]]:
    """Run ``repeats + 1`` cold iterations with DEAD_CST_TIMING=1 in a
    subprocess; collect the per-phase breakdown. Subprocess so the
    env var only affects this measurement and so the stderr output
    is captured cleanly.

    Returns ``repeats`` steady-state dicts (drops iter 1)."""
    code = (
        "from pathlib import Path\n"
        "from dead_cst import native\n"
        f"target_root = {str(target.root)!r}\n"
        f"kwargs = {target.project_kwargs!r}\n"
        f"for _ in range({repeats + 1}):\n"
        "    ctx = native.ProjectContext(target_root, **kwargs)\n"
        "    ctx.materialize()\n"
    )
    env = os.environ.copy()
    env["DEAD_CST_TIMING"] = "1"
    out = subprocess.run(
        ["uv", "run", "python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    rows: list[dict[str, float]] = []
    for line in out.stderr.splitlines():
        parsed = _parse_timing_line(line)
        if parsed is not None:
            rows.append(parsed)
    # Drop the warm-up iter — it includes process-global typeshed
    # / module-resolver init, which dwarfs steady-state.
    return rows[1:]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _stats(times: list[float]) -> tuple[float, float, float]:
    """Return ``(best_ms, mean_ms, stdev_ms)``."""
    if not times:
        return 0.0, 0.0, 0.0
    best = min(times) * 1000
    mean = statistics.fmean(times) * 1000
    stdev = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
    return best, mean, stdev


def _fmt_stats(times: list[float]) -> str:
    best, mean, stdev = _stats(times)
    return f"best {best:7.1f} ms  mean {mean:7.1f}±{stdev:.1f}"


def _run_target(target: TargetConfig, repeats: int, *, phases: bool) -> None:
    files = file_count(target.root)
    print(f"\n=== {target.name} === ({files} files)")
    print(f"  path: {target.root}")

    # Run cold and warm side-by-side, in-process for cleanest timing.
    cold_warmup, cold_steady = _bench(target, repeats, mode="cold")
    warm_warmup, warm_steady = _bench(target, repeats, mode="warm")

    # Best-of for ms/file rate.
    cold_best_ms = min(cold_steady) * 1000
    warm_best_ms = min(warm_steady) * 1000

    print(
        f"  cold (fresh ctx)  : {_fmt_stats(cold_steady)}  ({cold_best_ms / max(files, 1):.2f} ms/file)"
    )
    print(
        f"  warm (reused ctx) : {_fmt_stats(warm_steady)}  ({warm_best_ms / max(files, 1):.2f} ms/file)"
    )
    print(f"  cold warm-up iter : {cold_warmup * 1000:7.1f} ms — typeshed / module-resolver init")

    if phases:
        # Subprocess sweep with DEAD_CST_TIMING=1 to get per-phase
        # breakdown. Steady iters only.
        rows = _bench_with_phases(target, repeats)
        if not rows:
            print("  (per-phase timing unavailable — no rows captured)")
            return

        nodes = int(rows[0].get("nodes", 0))
        edges = int(rows[0].get("edges", 0))
        print(f"  nodes={nodes}  edges={edges}")
        print(f"  per-phase (cold, best of {len(rows)}):")
        for phase in _PHASE_KEYS:
            phase_vals = [r[phase] for r in rows if phase in r]
            if not phase_vals:
                continue
            best_ms = min(phase_vals) * 1000
            print(f"    {phase:14s} {best_ms:7.1f} ms")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _gather_targets(args: argparse.Namespace) -> Iterable[TargetConfig]:
    if "dead_cst" in args.targets:
        yield TargetConfig(name="dead_cst (self)", root=stage_dead_cst())
    if not args.no_flux0:
        yield from gather_flux0_targets(args.targets, args.cache_root)
    for n in args.synthetic or ():
        yield TargetConfig(name=f"synthetic ({n} files)", root=_generate_synthetic(n))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Steady-state iterations per (target, mode) (default 4). "
        "One extra warm-up iteration always runs and is reported separately.",
    )
    add_common_target_args(
        parser, default_targets=["dead_cst", "flux0_server", "flux0_cli", "flux0_workspace"]
    )
    parser.add_argument(
        "--synthetic",
        nargs="+",
        type=int,
        metavar="N",
        help="Generate synthetic projects with the given file counts "
        "(e.g. ``--synthetic 50 200 500``). Each file imports a couple "
        "of siblings so cross-file edge work is realistic.",
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Also dump the per-phase breakdown (enum / phase1 / "
        "dist_lookup / phase2 / phase3 / fqname) via a "
        "``DEAD_CST_TIMING=1`` subprocess sweep.",
    )
    args = parser.parse_args()

    print(f"python: {sys.version.split()[0]}")
    print(f"native: {native.__file__}")
    print(f"repeats: {args.repeats}")
    if args.phases:
        print("per-phase breakdown: ON (subprocess with DEAD_CST_TIMING=1)")

    targets = list(_gather_targets(args))
    if not targets:
        sys.exit("no targets to profile")

    for target in targets:
        _run_target(target, args.repeats, phases=args.phases)


if __name__ == "__main__":
    main()
