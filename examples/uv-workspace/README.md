# uv-workspace

A two-member uv workspace that exercises `dead-cst`'s multi-base analysis,
including the per-member `tests/` directory pattern that's standard in
flat-layout monorepos.

```
uv-workspace/
  pyproject.toml             # workspace root
  uv.lock                    # source of truth for the resolver
  packages/
    core/
      pyproject.toml         # [tool.hatch.build.targets.wheel].packages = ["core"]
      core/api.py            # used_by_app (live), unused_old (DEAD)
      tests/conftest.py      # local pytest fixtures (DEAD: no entrypoint reaches them)
    app/
      pyproject.toml         # depends on `core`; packages = ["app"]
      app/
        cli.py               # if __name__ == "__main__": run()
        helpers.py           # entire module DEAD
      tests/conftest.py      # local pytest fixtures (DEAD)
```

Each member uses the flat layout: importable code lives in a top-level
package directory (`core/` or `app/`) and `tests/` sits next to it.
`exported_roots` reads each member's `[tool.hatch.build.targets.wheel].packages`
and treats only those dirs as visible to other members -- so when `app`
resolves an import, it never sees `core/tests/` (and vice-versa), even
though both members have a top-level `tests/` package.

## Run the analysis

```bash
uv run dead-cst analyze examples/uv-workspace \
    --resolver uv --plugin main_block
```

The `uv` resolver reads `uv.lock`, treats every `[[package]]` with
`source = { editable = "..." }` as a workspace member, and wires each
member's source root together using uv's resolved dependency graph. The
above command is roughly equivalent to:

```bash
uv run dead-cst analyze examples/uv-workspace \
    -p packages/core \
    -p packages/app:packages/core \
    --plugin main_block
```

...except you don't have to keep the `-p` list in sync with the workspace
as it grows, and the resolver also infers per-member exported roots so
internal directories like `tests/` stay scoped to their own member.

Expected output:

```
.../examples/uv-workspace/packages/core:
  function: 3 total, 2 dead
  import: 1 total, 1 dead
  module: 4 total, 2 dead

.../examples/uv-workspace/packages/app:
  function: 3 total, 2 dead
  import: 2 total, 1 dead
  module: 5 total, 3 dead

Dead symbols (13):
  [external dist] pytest (synthetic) at packages/app
  app.helpers (module) at packages/app/app/helpers.py
  app.helpers.legacy_helper (function) at packages/app/app/helpers.py
  tests (module) at packages/app/tests/__init__.py
  tests.conftest (module) at packages/app/tests/conftest.py
  tests.conftest.pytest (import) at packages/app/tests/conftest.py
  tests.conftest.runner (function) at packages/app/tests/conftest.py
  [external dist] pytest (synthetic) at packages/core
  core.api.unused_old (function) at packages/core/core/api.py
  tests (module) at packages/core/tests/__init__.py
  tests.conftest (module) at packages/core/tests/conftest.py
  tests.conftest.doubled (function) at packages/core/tests/conftest.py
  tests.conftest.pytest (import) at packages/core/tests/conftest.py
```

`core.api.used_by_app` stays alive because `app.cli` -- kept alive by
`MainBlockPlugin` -- imports it. `core.api.unused_old` has no remaining
callers in the workspace and is correctly flagged. Both members' `tests/`
trees show up as dead because no entrypoint plugin (e.g. `pytest`) is
enabled in this run -- pass `--plugin pytest` to keep them alive.

The two `tests` modules are distinct nodes (one per file path) and
coexist in the graph; `exported_roots` keeps them from colliding during
import resolution.

## How `UvResolver` decides on a source root

For each workspace member directory, the resolver picks `<member>/src` if
that directory exists, otherwise the member directory itself. This covers
both the `src/`-layout and the flat layout.

Direct dependency edges come from each member's `dependencies = [...]` list
in `uv.lock`; non-workspace deps (regular PyPI packages) are dropped because
they don't have a source tree under your control.

For each member, `dead_cst.resolvers.exported_roots` then reads its
`pyproject.toml` and returns the subdirs that are visible to other
members at import time. Discovery dispatches on `[build-system].build-backend`
(hatchling, setuptools, poetry, pdm, flit) with a `src/`-layout shortcut
and a name-match fallback for backends without explicit `packages` lists.

## Falling back to explicit paths

If you don't want to commit a `uv.lock`, the multi-`-p` invocation above
keeps working unchanged. Note that explicit `-p` paths skip
`exported_roots` discovery; if you have multiple flat-layout members with
overlapping internal packages (`tests/`, `scripts/`, etc.), prefer the
`uv` resolver so each member's exports are inferred from its
build backend.
