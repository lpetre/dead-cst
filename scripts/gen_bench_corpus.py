#!/usr/bin/env python
"""Generate a synthetic Python corpus for at-scale materialize benchmarks.

The goal is a *reproducible*, *tunable* tree that drives the rust build
(parse -> ty resolve -> assemble -> fqname index) at a chosen node/edge
scale, so ``scripts/bench_materialize.py`` can time it and Claude can
hill-climb the numbers.

Layout: a flat set of importable top-level packages under one root::

    corpus/
      pkg_0000/__init__.py
      pkg_0000/mod_0000.py
      pkg_0000/mod_0001.py
      ...
      pkg_0599/mod_0332.py
      _manifest.json

The root is on ty's search path (``Analysis`` keeps ``project_root``
there), so every ``from pkg_XXXX import mod_YYYY`` resolves and each call
into an imported module mints a cross-file reachability edge. That is
what makes the edge count scale with ``--calls-per-decl`` rather than
collapsing to import-alias edges only.

Output is a pure function of the knobs (no RNG that affects structure),
so the same flags always produce byte-identical files -- a regenerate is
a no-op for salsa and for diffing.

Examples::

    # tiny, for calibrating the node/edge yield of the template
    uv run python scripts/gen_bench_corpus.py --out /tmp/c --packages 4 \\
        --modules-per-package 8

    # full target -- the defaults are calibrated to hit it: ~200k files
    # in 600 packages, ~4.0M nodes, ~20.8M edges (~19.8 nodes/file,
    # ~104 edges/file). Plan for ~15 GB of RAM and a few minutes.
    uv run python scripts/gen_bench_corpus.py \\
        --out ~/.cache/dead-cst-bench/corpus \\
        --packages 600 --modules-per-package 333

IMPORTANT: generate the corpus OUTSIDE the dead-cst repo (e.g. under
``~/.cache``, not ``target/``). ty's rescan-based re_materialize re-runs
project discovery and walks up to the nearest ``pyproject.toml``; a
corpus nested inside this repo gets re-rooted at the repo on rescan and
the incremental rebuild sees the wrong file set (silently empty / tiny).
A corpus outside any project rebuilds correctly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def _import_targets(
    p: int, m: int, packages: int, modules: int, count: int
) -> list[tuple[int, int]]:
    """Pick ``count`` distinct ``(pkg, module)`` import targets for module
    ``(p, m)``, never itself. Roughly half same-package siblings (cheap,
    in-tree resolution) and half cross-package (stresses ty's module
    resolver across the search path). Deterministic in the indices."""
    targets: list[tuple[int, int]] = []
    seen = {(p, m)}
    for k in range(count):
        if k % 2 == 0 and modules > 1:
            # same-package sibling, strided so it isn't always m+1
            tm = (m + 1 + k * 7) % modules
            tp = p
        else:
            # cross-package, spread across the whole corpus
            tp = (p + 1 + k * 13) % packages
            tm = (m + k * 5) % modules
        # nudge off any collision (incl. self) deterministically
        guard = 0
        while (tp, tm) in seen and guard < modules:
            tm = (tm + 1) % modules
            guard += 1
        seen.add((tp, tm))
        targets.append((tp, tm))
    return targets


def _module_source(
    p: int,
    m: int,
    packages: int,
    modules: int,
    decls: int,
    imports: int,
    calls: int,
) -> str:
    """Render one module: ``imports`` aliased module imports followed by
    ``decls`` top-level decls (functions, with ~1 in 5 a class). Every
    function/method body makes ``calls`` cross-module calls through the
    aliases, each to a function that is guaranteed to exist in the target
    (all modules share the same decl layout)."""
    fn_idx = [d for d in range(decls) if d % 5 != 4]  # the rest are classes
    if not fn_idx:
        fn_idx = [0]

    targets = _import_targets(p, m, packages, modules, imports)
    lines: list[str] = []
    for k, (tp, tm) in enumerate(targets):
        lines.append(f"from pkg_{tp:04d} import mod_{tm:04d} as _i{k}")
    lines.append("")

    n_imports = len(targets)
    for d in range(decls):
        if d % 5 == 4:
            lines.append(f"class C_{d}:")
            lines.append("    def run(self):")
            for c in range(calls):
                k = (d * 3 + c) % n_imports if n_imports else 0
                t = fn_idx[(d + c) % len(fn_idx)]
                lines.append(f"        _i{k}.fn_{t}()")
            if calls == 0:
                lines.append("        pass")
        else:
            lines.append(f"def fn_{d}():")
            for c in range(calls):
                k = (d * 3 + c) % n_imports if n_imports else 0
                t = fn_idx[(d + c) % len(fn_idx)]
                lines.append(f"    _i{k}.fn_{t}()")
            if calls == 0:
                lines.append("    pass")
        lines.append("")

    return "\n".join(lines) + "\n"


# Framework flavors woven into the corpus so the per-file plugins have real
# work to do — decorators to extract, fixtures / handlers / constructions to
# find, subclass bases to walk — instead of every plugin hitting its
# import/decorator early-out on a barren generic tree. Each framework file is
# *additive* (written alongside the generic modules, importing its generic
# sibling so it still resolves and mints an edge), so the generic import graph
# that carries the node/edge scale is untouched. Third-party frameworks
# (fastapi / flask / pytest / click) needn't be installed — the plugins match
# the import + decorator syntactically — while `unittest` / `__init_subclass__`
# resolve first-party, so their subclass walks fire too.
FRAMEWORK_FLAVORS = (
    "pytest",
    "fastapi",
    "flask",
    "unittest",
    "init_subclass",
    "mock_patch",
    "click",
)

# Flavors that need a `test_*.py` filename (pytest collection) — also the ones
# whose package gets a `conftest.py`.
_TEST_FLAVORS = frozenset({"pytest", "mock_patch"})


def _framework_file(flavor: str, p: int, m: int) -> tuple[str, str]:
    """``(filename, source)`` for one additive framework file for module
    ``(p, m)``. Deterministic in ``(flavor, p, m)``; imports the generic
    sibling so the body's call resolves to a real cross-file edge."""
    uid = f"{p:04d}_{m:04d}"
    head = [f"from pkg_{p:04d} import mod_{m:04d} as _i0", ""]
    call = ["    _i0.fn_0()"]
    meth = ["        _i0.fn_0()"]
    body: list[str] = []
    if flavor == "pytest":
        name = f"test_x{uid}.py"
        body += ["import pytest", ""]
        body += ["@pytest.fixture", f"def fix_{uid}():", f"    return {m}", ""]
        body += [
            f'@pytest.fixture(name="alias_{uid}")',
            f"def make_{uid}():",
            "    return object()",
            "",
        ]
        body += [f"def test_uses_{uid}(fix_{uid}, alias_{uid}):", *call, ""]
        body += [f"class TestGroup_{uid}:", f"    def test_method(self, fix_{uid}):", *meth, ""]
    elif flavor == "fastapi":
        name = f"app_{uid}.py"
        body += ["from fastapi import FastAPI", "", "app = FastAPI()", ""]
        for h, verb in enumerate(("get", "post", "put")):
            body += [f'@app.{verb}("/r_{uid}_{h}")', f"def handler_{uid}_{h}():", *call, ""]
    elif flavor == "flask":
        name = f"app_{uid}.py"
        body += ["from flask import Flask", "", "app = Flask(__name__)", ""]
        for h in range(2):
            body += [f'@app.route("/r_{uid}_{h}")', f"def view_{uid}_{h}():", *call, ""]
        body += ["def create_app():", "    return Flask(__name__)", "", "made = create_app()", ""]
        body += [f'@made.route("/m_{uid}")', f"def made_view_{uid}():", *call, ""]
    elif flavor == "unittest":
        name = f"tests_{uid}.py"
        body += ["import unittest", "", f"class Test_{uid}(unittest.TestCase):"]
        body += ["    def test_a(self):", *meth]
        body += ["    def test_b(self):", *meth, ""]
        body += ["def setUpModule():", "    pass", "", "def tearDownModule():", "    pass", ""]
    elif flavor == "init_subclass":
        name = f"models_{uid}.py"
        body += [
            f"class Base_{uid}:",
            "    registry = []",
            "    def __init_subclass__(cls, **kwargs):",
            "        super().__init_subclass__(**kwargs)",
            f"        Base_{uid}.registry.append(cls)",
            "",
        ]
        for s in range(2):
            body += [f"class Sub_{uid}_{s}(Base_{uid}):", "    def run(self):", *meth, ""]
    elif flavor == "mock_patch":
        name = f"test_p{uid}.py"
        body += [
            f"def test_patches_{uid}(monkeypatch):",
            f'    monkeypatch.setattr("pkg_{p:04d}.mod_{m:04d}.fn_0", None)',
            *call,
            "",
        ]
    elif flavor == "click":
        name = f"cli_{uid}.py"
        body += ["import click", "", "cli = click.Group()", ""]
        body += ["@cli.command()", f"def cmd_{uid}_0():", *call, ""]
        body += ["@cli.group()", f"def sub_{uid}():", "    pass", ""]
        body += [f"@sub_{uid}.command()", f"def cmd_{uid}_1():", *call, ""]
    else:  # pragma: no cover - guarded by FRAMEWORK_FLAVORS
        raise ValueError(flavor)
    return name, "\n".join(head + body) + "\n"


# conftest.py for a package containing pytest/mock_patch files: a couple of
# shared fixtures, so `@pytest.fixture` resolution + the conftest-decl rule fire.
_CONFTEST_SRC = (
    "import pytest\n\n"
    "@pytest.fixture\n"
    "def shared_fix():\n"
    "    return 1\n\n"
    '@pytest.fixture(name="shared_alias")\n'
    "def make_shared():\n"
    "    return object()\n\n"
    "def conftest_helper():\n"
    "    return 2\n"
)


def generate(
    out: Path,
    *,
    packages: int,
    modules_per_package: int,
    decls_per_module: int,
    imports_per_module: int,
    calls_per_decl: int,
    framework_fraction: float,
    clean: bool,
) -> dict[str, object]:
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Stride that selects ~`framework_fraction` of modules to also get an
    # additive framework file; flavors round-robin across the selected set so
    # every plugin gets a proportional share. Deterministic in the indices.
    stride = round(1 / framework_fraction) if framework_fraction > 0 else 0
    files = 0
    framework_files = 0
    flavor_counts: dict[str, int] = {f: 0 for f in FRAMEWORK_FLAVORS}
    selected = 0
    for p in range(packages):
        pkg = out / f"pkg_{p:04d}"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text("")
        files += 1
        pkg_needs_conftest = False
        for m in range(modules_per_package):
            src = _module_source(
                p,
                m,
                packages,
                modules_per_package,
                decls_per_module,
                imports_per_module,
                calls_per_decl,
            )
            (pkg / f"mod_{m:04d}.py").write_text(src)
            files += 1
            gi = p * modules_per_package + m
            if stride and gi % stride == 0:
                flavor = FRAMEWORK_FLAVORS[selected % len(FRAMEWORK_FLAVORS)]
                selected += 1
                name, fsrc = _framework_file(flavor, p, m)
                (pkg / name).write_text(fsrc)
                files += 1
                framework_files += 1
                flavor_counts[flavor] += 1
                if flavor in _TEST_FLAVORS:
                    pkg_needs_conftest = True
        if pkg_needs_conftest:
            (pkg / "conftest.py").write_text(_CONFTEST_SRC)
            files += 1

    manifest = {
        "packages": packages,
        "modules_per_package": modules_per_package,
        "decls_per_module": decls_per_module,
        "imports_per_module": imports_per_module,
        "calls_per_decl": calls_per_decl,
        "framework_fraction": framework_fraction,
        "framework_files": framework_files,
        "framework_flavors": flavor_counts,
        "files": files,
    }
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="Corpus output directory.")
    parser.add_argument("--packages", type=int, default=600)
    parser.add_argument("--modules-per-package", type=int, default=333)
    parser.add_argument(
        "--decls-per-module",
        type=int,
        default=13,
        help="Top-level decls per module (~1 in 5 is a class) -- the main "
        "node-count knob. nodes/file ~= 1 module + imports + decls. The "
        "default (13, with 6 imports) yields ~19.8 nodes/file.",
    )
    parser.add_argument("--imports-per-module", type=int, default=6)
    parser.add_argument(
        "--calls-per-decl",
        type=int,
        default=2,
        help="Cross-module calls per decl body -- the main edge-count knob. "
        "The default (2) yields ~103 edges/file (~5.2 edges/node).",
    )
    parser.add_argument(
        "--framework-fraction",
        type=float,
        default=0.0,
        help="Fraction (0..1) of modules that also get an additive framework "
        "file — a pytest fixture/test module, fastapi/flask app, unittest "
        "TestCase, __init_subclass__ base, monkeypatch test, or click group "
        "(round-robin) — plus a conftest.py per package with pytest files. "
        "These give the per-file plugins real work; the generic import graph "
        "(node/edge scale) is untouched. Default 0 reproduces the plain tree.",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Write into an existing dir instead of wiping it first.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    manifest = generate(
        args.out,
        packages=args.packages,
        modules_per_package=args.modules_per_package,
        decls_per_module=args.decls_per_module,
        imports_per_module=args.imports_per_module,
        calls_per_decl=args.calls_per_decl,
        framework_fraction=args.framework_fraction,
        clean=args.clean,
    )
    elapsed = time.perf_counter() - start
    print(f"wrote {manifest['files']:,} files to {args.out} in {elapsed:.1f}s")
    print(f"knobs: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()
