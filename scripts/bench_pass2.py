"""Microbenchmark for assemble_graph Pass 2 (edge translation).

Generates synthetic projects of varying file counts, runs
``ProjectContext.materialize()`` repeatedly with ``DEAD_CST_TIMING=1``,
and extracts the ``pass2=`` and ``assemble=`` timings. Toggles
``DEAD_CST_PASS2_SERIAL`` to compare the parallel implementation
against the original serial path inside the same binary.

Each variant runs in its own subprocess (fresh process, fresh salsa
DB) for ``warmup + iters`` cold iterations. The first iteration is
discarded to drop typeshed-load / module-resolver warmup; the
remaining iterations are reported as best / p50 / mean / stdev.

Usage::

    uv run python scripts/bench_pass2.py
    uv run python scripts/bench_pass2.py --sizes 200 1000 5000 --iters 7
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def _generate_synthetic(num_files: int) -> Path:
    """Lift directly from scripts/profile_graph_build.py. One package
    with `num_files` modules; each imports up to two siblings and
    exposes one class + one function."""
    stage = Path(tempfile.mkdtemp(prefix=f"synthetic-pass2-{num_files}-"))
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


_TIMING_RE = re.compile(r"\[dead-cst-timing\]\s+(.*)")


def _parse_duration(s: str) -> float:
    units = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    for suffix, mul in units.items():
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mul
    raise ValueError(f"unrecognised duration {s!r}")


def _run_iters(target_root: Path, iters: int, *, serial: bool) -> list[dict[str, float]]:
    """Run ``iters`` cold ``materialize()`` calls in a subprocess.
    Returns the list of `[dead-cst-timing]` line dicts captured from
    each iteration (concatenating multiple lines per iter where the
    pass2 sub-timing prints separately from the summary)."""
    code = (
        "from dead_cst import _native as native\n"
        f"target_root = {str(target_root)!r}\n"
        f"for _ in range({iters}):\n"
        "    ctx = native.ProjectContext(target_root)\n"
        "    ctx.materialize()\n"
    )
    env = os.environ.copy()
    env["DEAD_CST_TIMING"] = "1"
    if serial:
        env["DEAD_CST_PASS2_SERIAL"] = "1"
    else:
        env.pop("DEAD_CST_PASS2_SERIAL", None)
    out = subprocess.run(
        ["uv", "run", "python", "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    results: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for line in out.stderr.splitlines():
        m = _TIMING_RE.search(line)
        if not m:
            continue
        for token in m.group(1).strip().split():
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            if k in ("files", "nodes", "edges"):
                current[k] = float(v)
            elif k == "pass2_mode":
                current["pass2_mode"] = v  # type: ignore[assignment]
            elif k == "rss":
                # `12MB` — drop unit
                current[k] = float(v.rstrip("MB"))
            elif k.endswith("ms") or k.endswith("s") or k.endswith("µs") or k.endswith("us"):
                # the token already has a value with unit attached
                current[k] = _parse_duration(v)
            else:
                try:
                    current[k] = _parse_duration(v)
                except ValueError:
                    pass
        # The summary line containing `total=` closes one iteration.
        if "total" in current:
            results.append(current)
            current = {}
    return results


def _summary(values: list[float]) -> str:
    if not values:
        return "no data"
    best = min(values) * 1000
    median = statistics.median(values) * 1000
    mean = statistics.mean(values) * 1000
    stdev = (statistics.stdev(values) * 1000) if len(values) > 1 else 0.0
    return f"best={best:.3f}ms p50={median:.3f}ms mean={mean:.3f}ms stdev={stdev:.3f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[200, 1000, 5000],
        help="synthetic project sizes (default: 200 1000 5000)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=6,
        help="cold iterations per variant after warmup (default: 6)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warmup iterations to discard (default: 1)",
    )
    args = parser.parse_args()

    total_iters = args.warmup + args.iters

    for size in args.sizes:
        print(f"=== synthetic ({size} files) ===")
        target = _generate_synthetic(size)
        for variant in ("serial", "parallel"):
            serial = variant == "serial"
            rows = _run_iters(target, total_iters, serial=serial)
            rows = rows[args.warmup :]
            pass2 = [r["pass2"] for r in rows if "pass2" in r]
            assemble = [r["assemble"] for r in rows if "assemble" in r]
            total = [r["total"] for r in rows if "total" in r]
            mode_label = rows[0].get("pass2_mode", "?") if rows else "?"
            print(f"  {variant:9s} ({mode_label}):")
            print(f"    pass2    {_summary(pass2)}  n={len(pass2)}")
            print(f"    assemble {_summary(assemble)}  n={len(assemble)}")
            print(f"    total    {_summary(total)}  n={len(total)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
