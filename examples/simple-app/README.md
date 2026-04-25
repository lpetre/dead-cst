# simple-app

A minimal package whose only entrypoint is the `if __name__ == "__main__":`
block in `simple_app/__main__.py`, runnable via `python -m simple_app`.
Demonstrates the `MainBlockPlugin`.

The package contains two obvious bits of dead code:

- `simple_app.core.legacy_greet` -- old API, no callers
- `simple_app.utils.stale_logger` -- old helper, no callers

## Run the analysis

From the repo root:

```bash
uv run dead-cst analyze examples/simple-app --plugin main_block
```

Expected output (abbreviated):

```
.../examples/simple-app:
  function: 6 total, 2 dead
  import: 2 total
  module: 4 total
  synthetic: 1 total

Dead symbols (2):
  simple_app.core.legacy_greet (function) at simple_app/core.py
  simple_app.utils.stale_logger (function) at simple_app/utils.py
```

Exit code is `1` whenever dead symbols are found, so this command slots
straight into a CI step.

## Ask why a symbol is alive

```bash
uv run dead-cst why-alive examples/simple-app simple_app.core.greet \
    --plugin main_block
```

## Remove the dead code

```bash
uv run dead-cst remove examples/simple-app --plugin main_block --dry-run
```

Drop `--dry-run` (and confirm at the prompt) to actually rewrite the files.
