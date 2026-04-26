# scripts-and-all

A `src/`-layout package whose live entrypoint is the console script declared in
`pyproject.toml`:

```toml
[project.scripts]
reportkit = "reportkit.cli:main"
```

This example demonstrates two plugins and one path resolver:

- `ProjectScriptsPlugin` reads `[project.scripts]` and treats each target as
  an entrypoint. It keeps `reportkit.cli.main` -- and everything `main`
  reaches -- alive.
- `ModuleDundersPlugin` (always on) keeps the `__all__` variable in
  `reportkit/renderer.py` from being reported as a dead variable. It also
  preserves any other module-level dunder (e.g. `__version__`).
- `PyprojectResolver` notices the `src/` directory and feeds it to
  `build_symbol_graph` as the analysis base, so `reportkit.cli` resolves to
  `src/reportkit/cli.py`.

## Run the analysis

```bash
uv run dead-cst analyze examples/scripts-and-all \
    --resolver pyproject --plugin project_scripts
```

Expected output:

```
.../examples/scripts-and-all/src:
  function: 6 total, 3 dead
  import: 2 total
  module: 5 total, 1 dead
  synthetic: 1 total
  variable: 1 total

Dead symbols (4):
  reportkit.formatters.format_xml (function) ...
  reportkit.legacy_writer (module) ...
  reportkit.legacy_writer._legacy_header (function) ...
  reportkit.legacy_writer.write_legacy_report (function) ...
```

The `variable: 1 total` line is `reportkit.renderer.__all__`; note that it is
*not* in the dead list because `ModuleDundersPlugin` flagged it as an
entrypoint.

## Equivalent invocation without the resolver

If you'd rather pass the search path explicitly instead of relying on
`PyprojectResolver`:

```bash
uv run dead-cst analyze examples/scripts-and-all -p src --plugin project_scripts
```

## Note on `__all__` re-exports

`dead-cst` does *not* follow string names inside `__all__`. If
`reportkit/__init__.py` had

```python
from reportkit.renderer import render
__all__ = ["render"]
```

the `from ... import render` line would be reported as a dead import, even
though `__all__` mentions it -- string lookup is out of scope for static
analysis. This is documented in the project README under "Limitations".
