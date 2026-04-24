# dead-cst examples

Three runnable mini-projects that exercise the analyzer's plugins, resolvers,
and multi-base path specs. Every command below assumes you're at the repo
root.

| Example | Demonstrates |
|---|---|
| [`simple-app/`](./simple-app) | `MainBlockPlugin` -- entrypoints discovered via `if __name__ == "__main__":` |
| [`scripts-and-all/`](./scripts-and-all) | `ProjectScriptsPlugin`, `DunderAllPlugin`, `PyprojectResolver` (auto-detects `src/` layout) |
| [`uv-workspace/`](./uv-workspace) | `UvWorkspaceResolver` -- multi-package analysis driven by `uv.lock` |

Each subdirectory has its own README with the exact commands to run and the
expected output. To poke at all three at once:

```bash
# 1. main-block entrypoints
uv run dead-cst analyze examples/simple-app --plugin main_block

# 2. project.scripts entrypoints + __all__ preservation, src/ auto-detected
uv run dead-cst analyze examples/scripts-and-all \
    --resolver pyproject --plugin project_scripts

# 3. multi-package workspace, members + dep edges read from uv.lock
uv run dead-cst analyze examples/uv-workspace \
    --resolver uv_workspace --plugin main_block
```

Each command exits non-zero when dead code is found, so they slot into CI
scripts as-is.

## Why the example uses `cli.py`, not `__main__.py`

libcst's `FullyQualifiedNameProvider` reports the same fully-qualified name
(`pkg`) for both `pkg/__init__.py` and `pkg/__main__.py`, which collides with
`dead-cst`'s "one module node per FQN" invariant. Until that's resolved
upstream or worked around in `dead-cst`, place your `if __name__ ==
"__main__":` block in any other module name (`cli.py`, `script.py`, etc.).
