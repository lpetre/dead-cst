# uv-workspace

A two-member uv workspace that exercises `dead-cst`'s multi-base analysis via
the `-p / --path` flag. Layout:

```
uv-workspace/
  pyproject.toml             # workspace root
  packages/
    core/
      pyproject.toml
      src/core/api.py        # used_by_app (live), unused_old (DEAD)
    app/
      pyproject.toml         # depends on `core` via [tool.uv.sources]
      src/app/
        cli.py               # if __name__ == "__main__": run()
        helpers.py           # entire module DEAD
```

Why two `-p` specs are necessary: each member uses a `src/` layout, so
fully-qualified names should be rooted at `packages/<pkg>/src`, not at the
workspace root. We also need `core/src` to be a search path while analyzing
`app/src` so `from core.api import used_by_app` resolves.

## Run the analysis

```bash
uv run dead-cst analyze examples/uv-workspace \
    -p packages/core/src \
    -p packages/app/src:packages/core/src \
    --plugin main_block
```

Expected output:

```
.../examples/uv-workspace/packages/core/src:
  function: 2 total, 1 dead
  module: 2 total

.../examples/uv-workspace/packages/app/src:
  function: 2 total, 1 dead
  import: 1 total
  module: 3 total, 1 dead
  synthetic: 1 total

Dead symbols (3):
  app.helpers (module) at packages/app/src/app/helpers.py
  app.helpers.legacy_helper (function) at packages/app/src/app/helpers.py
  core.api.unused_old (function) at packages/core/src/core/api.py
```

`core.api.used_by_app` stays alive because `app.cli` -- kept alive by
`MainBlockPlugin` -- imports it. `core.api.unused_old`, on the other hand, has
no remaining callers in the workspace and is correctly flagged.

## Path-spec syntax

`-p` is repeatable. Each spec is one of:

- `BASE` -- analyze `BASE` with no extra search paths.
- `BASE:DEP1,DEP2,...` -- analyze `BASE` with the listed dependency paths
  available for import resolution.

Each `BASE` is resolved relative to the analysis root (here:
`examples/uv-workspace`). Paths form a small DAG inside
`build_symbol_graph` -- search paths are processed before the bases that
depend on them, so by the time `app/src` is visited, every `core` symbol is
already in the trie.

## Why this matters

In a real workspace you typically run `dead-cst` once per branch on the
*entire* code under your control. Without the cross-package search path, the
imports from `app` into `core` would fail to resolve and `dead-cst` would
warn (`Failed to resolve cst.Import: ...`); with it, the graph is unified and
genuine dead code surfaces in every member.
