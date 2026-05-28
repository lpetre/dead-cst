# dead-cst-plugin-host

Prebuilt [dead-cst](https://github.com/lpetre/dead-cst) runtime + dependency
closure, used by `dead-cst build-plugin` to compile native plugins via
`rustc --extern` without a source checkout or a ruff recompile.

This is a **data-only, platform-specific** package — it ships the
`libdead_cst_runtime` dylib, its `.rlib` / proc-macro-dylib dependency closure,
and `libstd`, all relocatable. It's large (~130 MB compressed), so it is **not**
a dependency of `dead-cst`; install it on demand:

```
pip install dead-cst[build-plugin]
```

Its version must match the installed `dead-cst` exactly — the runtime ABI
fingerprint (rustc commit + version) is checked when a plugin loads.
