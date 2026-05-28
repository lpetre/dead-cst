"""Prebuilt dead-cst runtime bundle for building native plugins.

Data-only package: it ships the dead-cst runtime dylib + its dependency
closure (rlib + proc-macro dylibs) + libstd, all relocatable. `dead-cst
build-plugin` locates this package (``import dead_cst_plugin_host``) and
compiles plugins against it via ``rustc --extern`` — no source checkout, no
ruff recompile. Installed on demand via ``pip install dead-cst[build-plugin]``;
populated by ``dead-cst bundle-plugin-host``.
"""
