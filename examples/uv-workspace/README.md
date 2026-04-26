# uv-workspace

A two-member uv workspace that exercises `dead-cst`'s multi-base analysis.
Layout:

```
uv-workspace/
  pyproject.toml             # workspace root
  uv.lock                    # source of truth for the resolver
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

Each member uses a `src/` layout, so fully-qualified names need to be rooted
at `packages/<pkg>/src` (not the workspace root). And while analyzing
`app/src`, `core/src` must be a search path so `from core.api import
used_by_app` resolves.

## Run the analysis

```bash
uv run dead-cst analyze examples/uv-workspace \
    --resolver uv_workspace --plugin main_block
```

The `uv_workspace` resolver reads `uv.lock`, treats every `[[package]]` with
`source = { editable = "..." }` as a workspace member, and wires each
member's `src/` directory together using uv's resolved dependency graph.
The above command is equivalent to:

```bash
uv run dead-cst analyze examples/uv-workspace \
    -p packages/core/src \
    -p packages/app/src:packages/core/src \
    --plugin main_block
```

...except you don't have to keep the `-p` list in sync with the workspace as
it grows.

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
`MainBlockPlugin` -- imports it. `core.api.unused_old` has no remaining
callers in the workspace and is correctly flagged.

## How `UvWorkspaceResolver` decides on a src root

For each workspace member directory, the resolver picks `<member>/src` if
that directory exists, otherwise the member directory itself. This matches
`PyprojectResolver`'s single-package convention and covers both the
`src/`-layout and the flat layout.

Direct dependency edges come from each member's `dependencies = [...]` list
in `uv.lock`; non-workspace deps (regular PyPI packages) are dropped because
they don't have a source tree under your control.

## Falling back to explicit paths

If you don't want to commit a `uv.lock`, the multi-`-p` invocation above
keeps working unchanged.
