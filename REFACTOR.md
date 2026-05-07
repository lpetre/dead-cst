# Refactor: replace `PathMap` with `Package` + two-phase parse

This is a self-contained spec for redoing the resolver/analyzer refactor on
top of current `main`. It captures the design, the API shape, the two-phase
pipeline mechanics, and the simplifications that landed on the
`claude/refactor-resolver-package-dirs-hqHxQ` branch so you don't have to
rediscover them.

## Where `main` is today

The starting point:

```python
PathMap = dict[Path, list[Path]]   # base path -> list of search paths

class PathResolver(Cacheable, Protocol):
    def resolve(self, project_root: Path) -> PathMap: ...
    def resolve_import(self, name, search_paths) -> str | Path | None: ...

class Analysis:
    def __init__(self, paths: PathMap, *, resolvers=(), plugins=(), ...): ...
    @property
    def paths(self) -> PathMap: ...
```

Key supporting pieces on `main`:

- `dead_cst.resolvers.merge_paths(*pathmaps)` — combines multiple resolvers'
  `PathMap`s with deduplication.
- `_infer_project_root(paths)` heuristic — needed because `Analysis`'s
  positional arg is `paths`, not `project_root`.
- Per-base machinery: `_base_specs: dict[Path, _BaseSpec]`,
  `_contributions: dict[Path, _BaseContribution]`,
  `_closure_graphs: dict[Path, nx.MultiDiGraph]`, `_order_paths(paths)` for
  topo over the search-path DAG.
- Resolvers: `PyprojectResolver`, `ManualResolver`, `VenvResolver`,
  `UvWorkspaceResolver`. `MissingVenvError` for the venv path.
- `dead_cst.resolvers.exported_roots(project_dir)` already exists.

## Goal

Replace the `PathMap`-based resolver model with a flat `list[Package]` keyed
on `(path, name, exported, deps)`, and rewire the analyzer to parse each
package in two phases.

The motivating use case is **non-exported code with apparent cross-package
cycles** — `A.tests` importing `B.lib` while `B.tests` imports `A.lib` —
which `PathMap` can't express because it conflates "what depends on what"
(the build DAG) with "what can see what" (visibility). The
search-path-as-DAG model rejects all cycles in `paths`, including ones that
only exist between non-shipped code.

The split is: production code (exported) participates in a strict deps DAG;
non-exported code (tests, scripts, app entrypoints) reads against finalized
export tries in a second pass and therefore can have cycles freely.

## The `Package` dataclass

```python
@dataclass(frozen=True, slots=True)
class Package:
    path: Path                       # the package directory
    name: str                        # unique within an Analysis
    exported: tuple[Path, ...] = ()  # subdirs of `path` whose .py files ship
    deps: tuple[str, ...] = ()       # other package names; production-only DAG
```

Files under `path` but not under any `exported` subdir are *internal*.
Examples:

- **src layout**: `path = pkg_dir`, `exported = (pkg_dir/"src",)`. Tests live
  in `pkg_dir/tests/` (internal).
- **flat layout**: `path = pkg_dir`, `exported = (pkg_dir/"mypkg",)`. Tests in
  `pkg_dir/tests/` (internal).
- **virtual / app**: `exported = ()`. Everything is internal.

### Mapping from today's `PathMap`

A `PathMap` like `{Path("a"): [], Path("b"): [Path("a")]}` becomes:

```python
[
    Package(path=Path("a"), name="a", exported=(Path("a"),)),
    Package(path=Path("b"), name="b", exported=(Path("b"),), deps=("a",)),
]
```

Two semantic shifts:

1. **Deps are package names, not paths.** That's what makes the test-cycle
   case expressible: `deps` is the production DAG; phase-2 visibility is
   separately the union of every package's exports, with no DAG check.
2. **Files under `path` outside `exported` are internal.** The `tests/`
   sibling that today is folded into the package's single PathMap entry
   becomes phase-2 work (parsed last, against every package's export trie).

### Validation (`validate_packages`)

- Non-empty unique names.
- Unique paths.
- Every `exported` entry is under (or equal to) `path`.
- Every `deps` name refers to another package in the list.
- No self-deps.
- `deps` is acyclic.

Returns a `_ValidatedPackages` with `by_name`, `by_path`, and `topo_order`
indices.

### Helpers shipped from `dead_cst.resolvers`

- `assign_file_to_package(file, packages)` — longest-prefix-match routing.
- `is_exported_file(file, package)` — `True` iff under any `package.exported`.
- `export_search_root(package)` — `path/"src"` if all exported subdirs are
  under it, else `path`. The `sys.path` entry that resolves cross-package
  imports of this package's exports.
- `exported_tree_root(project_dir)` — derives a single exported subdir from
  the build backend's metadata via `exported_roots(project_dir)`. Use this
  in `UvResolver` (and any custom resolver) so build-backend introspection
  stays consistent with `pyproject.toml` aware tooling.

## The two-phase parse

This is the core of the refactor. Each package is walked in two phases:

### Phase 1 (exports), in topological order over `deps`

For each package in topo order:

1. Walk `.py` files under any `pkg.exported` subdir.
2. `sys.path` during visit = `(export_search_root(pkg), *deps' export_search_roots)`.
3. Build the package's **export trie** from the resulting payloads.
4. Stitch cross-package imports against (own export trie under construction +
   each dep's already-finalized export trie).

The DAG requirement is what makes this sound — by the time we stitch
package X's exports, every dep's export trie is finalized.

### Phase 2 (internals), any order

For each package (after every package's phase 1 is done):

1. Walk `.py` files under `pkg.path` *not* under any `pkg.exported` subdir.
2. `sys.path` during visit = `(pkg.path, export_search_root(pkg), *every other
   package's export_search_root)` — the all-to-all phase-2 visibility.
3. Stitch imports against (own combined trie + the union of every package's
   export trie).

Phase 2 reads from already-finalized tries, so apparent cross-package cycles
between non-exported files (`A.tests → B.lib`, `B.tests → A.lib`) resolve
cleanly. There's no ordering requirement on phase 2 because no phase-2 trie
ever needs to read from another phase-2 trie.

### Why this is strictly more permissive than `PathMap`

`PathMap` has a single relation (paths + their search paths) that has to
satisfy both roles: it's both the "build order" graph (the analyzer topo-
sorts it) *and* the visibility graph (a base's search paths are what its
files can import from). Cycles in the merged thing are forbidden, which is
why test fixtures crossing package boundaries can't be modeled today.

Splitting into `deps` (production, DAG-checked) + a phase-2 read-only union
gets you cycle tolerance for free, without weakening the production
guarantees.

## `Analysis` API

```python
class Analysis:
    def __init__(
        self,
        project_root: Path,                       # ← positional, replaces `paths`
        *,
        resolvers: Sequence[PathResolver] = (),
        plugins: Sequence[EdgePlugin] = (),
        cache: GraphCache | None = None,
        unreachable_detector: UnreachableRegionDetector | None = None,
        workers: int | None = None,
    ): ...

    @property
    def packages(self) -> list[Package]: ...   # replaces `.paths`
    @property
    def package_names(self) -> list[str]: ...

    def reverse_closure(self, package: str) -> frozenset[str]: ...   # walks deps DAG
    def refresh(self, packages: Iterable[str] | None = None) -> Self: ...
    def package(self, name: str) -> PackageView: ...
    def package_views(self) -> Iterator[PackageView]: ...

    def materialize_all(self) -> nx.MultiDiGraph: ...
    def materialize_closure(self, package: str) -> nx.MultiDiGraph: ...
    # ↑ now collapses to materialize_all() — phase 2's all-to-all visibility
    # makes a closure-scoped graph unsound. Kept as a stable API surface but
    # no longer narrows work. _closure_graphs goes away.

    def reachable(self) -> set[SymbolNode]: ...
    def dead(self) -> Iterator[SymbolNode]: ...
    def kept_alive_by_dead_branches(self) -> set[SymbolNode]: ...
    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]: ...
```

Drops vs. main:

- `paths: PathMap` positional arg → `project_root: Path` positional arg.
- `Analysis.paths` property → `Analysis.packages` (returns `list[Package]`).
- `_infer_project_root` heuristic — gone, root is explicit.
- `_closure_graphs`, `materialize_closure`'s narrowing — collapsed.
- `_base_specs` / `_contributions` keyed by `Path` → keyed by
  `tuple[str, Phase]` (see Phase enum below).

`PackageView` is filtered by `package.path` rather than by base paths. Its
methods (`modules`, `declarations`, `dead`, `reachable`, `graph`,
`remove_dead_code`, etc.) all key off the package's name; the implementation
iterates the package's per-phase contributions via a private accessor
`_phase_contributions(name)` so it doesn't reach into `_contributions`
directly.

## `Phase` enum

The pipeline has a `Phase` enum with `EXPORTS` and `INTERNALS`. All phase
keys in the analyzer are `tuple[str, Phase]`. Don't use raw strings —
typo'd keys silently miss in dicts.

```python
class Phase(enum.Enum):
    EXPORTS = "exports"
    INTERNALS = "internals"
```

## Resolver changes

### `PathResolver` protocol

```python
class PathResolver(Cacheable, Protocol):
    def resolve(self, project_root: Path) -> list[Package]: ...   # was PathMap
    def resolve_import(self, name, search_paths) -> str | Path | None: ...
```

### `ManualResolver`

Spec format unchanged: `path[:dep1,dep2]`. **Deps are now package names**, not
paths. Each spec produces one `Package` with `exported=(path,)` (entire dir
exported) and `deps=tuple(dep names)`. The package's `name` is the path's
final component.

The CLI default stays `ManualResolver(specs=["."])` when neither `-p` nor
`--resolver` is passed.

### `UvResolver` (renamed from `UvWorkspaceResolver`)

- Module: `dead_cst.contrib.uv_workspace` → `dead_cst.contrib.uv_resolver`.
- Builtin name: `uv_workspace` → `uv`.
- Returns one `Package` per workspace member from `uv.lock`:
  - `path = member_dir`
  - `exported = (exported_tree_root(member_dir),)` if non-`None`,
    else `(member_dir/"src",)` if `src/` exists, else `()`
  - `deps = tuple(d for d in lockfile_deps if d in workspace_members)`
- Workspace root (`virtual = "."`) is skipped.
- Both `editable` and `virtual` members are emitted — virtual members are
  apps/services that don't ship as wheels but are first-party code to
  analyze. Treated identically except `virtual = "."` is the workspace-root
  marker that gets dropped.

### Drop `PyprojectResolver`

The conventional-layout fallback offered no value over `-p .` (or a custom
resolver), and `[[tool.dead-cst.trees]]` (or any explicit-config shape)
duplicated what a small custom `PathResolver` can express more directly.
`UvResolver` and custom resolvers still call `exported_tree_root` internally
so build-backend introspection stays available.

### Drop `VenvResolver` + `MissingVenvError`

The new contract is "run `dead-cst` with the project's venv active"
(typically `uv run dead-cst …`). `default_resolve_import` then resolves
third-party dists via the running Python's `sys.path`. The previous
machinery (threading `site-packages` paths through the resolver protocol,
raising `MissingVenvError` when no venv was found) is replaced by a
construction-time warning when the project root has a sibling `.venv` /
`venv` that isn't the running interpreter:

```python
def _warn_if_project_venv_inactive(project_root: Path) -> None: ...
```

Called from `Analysis.__init__`. More actionable than the downstream
`[unresolved]` synthetic / `UnresolvedDependencyError` symptoms users hit
without it.

### Drop `merge_paths`

Multiple resolvers' `Package` lists concatenate. `validate_packages`
catches duplicates.

## Cache fingerprint (recommendation, not landed on the source branch)

The source branch shipped per-(package, phase) fingerprints, derived from
`(base=pkg.path, search_paths=phase-specific tuple, ...chain)`. The cleaner
model is **per-package**, derived from the full `Package` list:

```python
def compute_package_fingerprint(
    *,
    package: Package,
    packages: Sequence[Package],
    resolvers, plugins, detector,
) -> str: ...
```

Why "all packages" and not just the one package's fields:

- Phase 1's `sys.path` includes deps' `export_search_root`s. So this
  package's fingerprint correctness depends on its deps' configs.
- Phase 2's `sys.path` includes every other package's `export_search_root`.
  So this package's phase-2 fingerprint correctness depends on the whole
  workspace.

Hashing the full `Package` list captures both. One fingerprint per package
covers both phases (which walk disjoint files), the cache layer drops
phase-awareness entirely, and invalidation becomes "any change to any
`Package` or the chain invalidates this package." Strictly more conservative
than today's `(base, search_paths)` formulation but typically ≤2× cache
misses on workspace structural changes, which are rare relative to file
edits.

If you keep `compute_fingerprint(base=, search_paths=)` as a low-level
primitive for power users, add a thin `compute_package_fingerprint(pkg,
packages, ...)` that derives the inputs and have the analyzer call that.

## Pipeline-level simplifications worth replicating

These all landed on the source branch and are worth carrying forward:

1. **One composition loop, not two.** Iterate `for phase in (EXPORTS,
   INTERNALS): for pkg in topo_order: ...` with a single
   `_build_phase_lookup(pkg, phase, all_exports)` helper. Don't write
   parallel phase-1 and phase-2 loops with copy-paste.

2. **Pre-build the all-packages export trie once** before phase 2 starts,
   reuse for every phase-2 lookup. Was O(N²K) trie merges across all
   phase-2 compositions; now O(NK) build + O(K) merge per lookup.

3. **Don't emit empty phase specs.** A package with no exported files
   (virtual app) shouldn't have a phase-1 contribution; a package with no
   internal files shouldn't have a phase-2 contribution. Otherwise
   `_compose_contribution` runs on empty data and plugin `finalize` runs
   redundantly per (package, phase).

4. **Don't cache file partitions.** Walking `pkg.path.rglob("*.py")` is
   cheap relative to per-file visiting, and a `_partitions` cache only
   saves repeat `refresh()` calls on the same packages, which already
   short-circuit on `key in self._contributions`.

5. **Don't reach through `PackageView` into private analyzer state.** Add an
   `Analysis._phase_contributions(name) -> Iterator[_PhaseContribution]`
   accessor and use it from `PackageView`. Same for `Analysis._validated`
   access via a `Analysis._package(name)` helper.

## A subtle bug the agents caught

The first stab at `_build_phase_lookup` for phase 2 looked like:

```python
lookup.merge(contrib.current_trie)             # X's internals
lookup.merge(own_exports.current_trie)         # X's exports
lookup.merge(all_exports)                      # everyone's exports — including X
```

This double-merges X's exports (`own_exports.current_trie` + the X portion of
`all_exports`) and emits `SymbolTrie collision at p.foo: keeping ...,
dropping ...` warnings. The fix: drop the `own_exports.current_trie` line.
For phase 1 contributions, `current_trie == export_trie` (because
`_apply_payload` adds the same nodes to both when `export_trie is not
None`), so `all_exports` already covers it.

```python
lookup.merge(contrib.current_trie)             # X's internals
lookup.merge(all_exports)                      # everyone's exports (incl X)
```

## Public surface delta

### Added

- `dead_cst.Package` (re-exported at top level).
- `dead_cst.resolvers.{Package, validate_packages, assign_file_to_package,
  is_exported_file, export_search_root}`.
- `dead_cst.resolvers.exported_tree_root` (if not already on main; the
  existing `exported_roots` is its lower-level helper).

### Removed

- `dead_cst.resolvers.{PathMap, merge_paths}`.
- `dead_cst.resolvers.PyprojectResolver`.
- `dead_cst.resolvers.{VenvResolver, MissingVenvError,
  find_venv_site_packages}`.
- `dead_cst.contrib.UvWorkspaceResolver` (renamed to `UvResolver`).

The public-API snapshot in `tests/test_public_api.py` needs to match.

## Files that need touching

```
dead_cst/__init__.py                     # Package re-export, drop PathMap, docstring
dead_cst/analyze.py                      # full rewrite of orchestration
dead_cst/cli.py                          # source `paths` -> packages,
                                         # build_resolvers default to ManualResolver(["."])
dead_cst/contrib/__init__.py             # UvWorkspaceResolver -> UvResolver
dead_cst/contrib/uv_workspace.py         # DELETE (replaced by uv_resolver.py)
dead_cst/contrib/uv_resolver.py          # NEW: emit Package per member
dead_cst/resolvers/__init__.py           # __all__, BUILTIN_RESOLVERS, drop
                                         # PathMap/merge_paths/PyprojectResolver/VenvResolver
dead_cst/resolvers/_core.py              # new Package model + helpers + validation
dead_cst/resolvers/_exports.py           # docstrings cleanup
dead_cst/resolvers/manual.py             # deps as names, exported=(path,)
dead_cst/resolvers/pyproject.py          # DELETE
dead_cst/resolvers/venv.py               # DELETE
dead_cst/cache.py                        # signature: per-package fingerprint
                                         # (or keep as-is for power users + add helper)
tests/conftest.py                        # `manual()` helper that wraps
                                         # ManualResolver(specs=...)
tests/test_analysis.py                   # _contributions key shape, packages property
tests/test_cache.py                      # use UvResolver as the second-resolver
                                         # for fingerprint variation tests
tests/test_cli.py                        # ManualResolver spec parser test
tests/test_public_api.py                 # snapshot
tests/test_resolvers/test_core.py        # rewrite for Package
tests/test_resolvers/test_pyproject.py   # DELETE
tests/test_resolvers/test_venv.py        # DELETE
tests/test_resolvers/test_uv_resolver.py # NEW (was test_uv_workspace.py)
README.md
CHANGELOG.md                             # [Unreleased] block — Added/Changed/Removed
CLAUDE.md                                # path resolution + two-phase composition sections
CONTRIBUTING.md
examples/README.md
examples/scripts-and-all/README.md       # switch to -p src
examples/uv-workspace/README.md          # `--resolver uv` (was uv_workspace)
```

## Migration strategy

Recommended order to keep the diff reviewable:

1. **Land the new `_core.py`** — `Package` dataclass, `validate_packages`,
   `assign_file_to_package`, `is_exported_file`, `export_search_root`. Keep
   `PathMap` / `merge_paths` exported for one commit so nothing else
   breaks yet.
2. **Migrate the resolvers** — `ManualResolver` and the renamed
   `UvResolver` emit `Package` lists. Delete `pyproject.py` and `venv.py`.
   Update `BUILTIN_RESOLVERS`.
3. **Rewrite `analyze.py`** — this is where the two-phase pipeline lives.
   Keep the existing per-file visitor / observe / cache machinery intact;
   only the orchestration changes (per-(package, phase) specs and
   contributions, the dual-loop composition with the pre-built
   `all_exports` trie). Drop `_infer_project_root` and `_closure_graphs`.
4. **Update `cli.py`** — `Analysis(root, resolvers=...)` shape, drop the
   `paths` plumbing, default to `ManualResolver(specs=["."])`.
5. **Drop `PathMap` / `merge_paths`** from `__init__.py` and `_core.py`.
6. **Migrate tests** — easiest path is the `manual()` conftest helper that
   wraps `ManualResolver(specs=...)`; most existing test sites collapse to
   `Analysis(tmp_path, resolvers=manual())`.
7. **Update docs + CHANGELOG.**

## Validation gates

1. `uv run pytest` — the source branch ran 768 unit tests cleanly.
2. `uv run pytest -m e2e` — 10 e2e in ~29s.
3. `uv run prek run --all-files` — ruff + format + ty all clean.
4. CLI smoke: `uv run dead-cst analyze examples/uv-workspace --resolver uv`.

## Reference

The source branch is `claude/refactor-resolver-package-dirs-hqHxQ`. Key
commits:

- `cdc5612` — initial Package + two-phase rewrite (atomic SourceTree-stage
  → Package-stage move).
- `d98a754` — drop `PyprojectResolver`.
- `a447d94` — `simplify` pass: `Phase` enum, dedupe phase loops, pre-built
  `all_exports`, drop `_partitions`, `_phase_contributions` accessor.

If the diffs are useful, they're at HEAD of that branch. The bulk of new
code is `dead_cst/analyze.py` (~700 lines diff) and
`dead_cst/resolvers/_core.py` (~270 lines diff).
