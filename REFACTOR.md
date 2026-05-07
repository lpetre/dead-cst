# Refactor: replace `SourceTree` with `Package` + two-phase parse

This is a self-contained spec for redoing the resolver/analyzer refactor on
top of current `main`. It captures the design, the API shape, the two-phase
pipeline mechanics, and the simplifications that landed on
`claude/refactor-resolver-package-dirs-hqHxQ` so you don't have to rediscover
them.

## Goal

Replace the `SourceTree` / `SourceTreeFlags` / `search_trees` resolver model
with a flat `list[Package]` keyed on `(path, name, exported, deps)`, and rewire
the analyzer to parse each package in two phases. The motivating use case is
**non-exported code with apparent cross-package cycles** — `A.tests` importing
`B.lib` while `B.tests` imports `A.lib` — which the existing
`search_trees`-based DAG can't express because it conflates "what depends on
what" (the build DAG) with "what can see what" (visibility).

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

### Why this is strictly more permissive than `search_trees`

The old model has a single relation that has to satisfy both roles at once
("what depends on what" *and* "what can see what"), and `validate_source_trees`
enforces acyclicity on the merged thing. That's what blocks the
test-fixture-cycle case. Splitting into `deps` (production, DAG-checked) +
phase-2 read-only union gets you the cycle tolerance for free.

## `Analysis` API

```python
class Analysis:
    def __init__(
        self,
        project_root: Path,
        *,
        resolvers: Sequence[PathResolver] = (),
        plugins: Sequence[EdgePlugin] = (),
        cache: GraphCache | None = None,
        unreachable_detector: UnreachableRegionDetector | None = None,
        workers: int | None = None,
    ): ...

    @property
    def packages(self) -> list[Package]: ...
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
    # no longer narrows work.

    def reachable(self) -> set[SymbolNode]: ...
    def dead(self) -> Iterator[SymbolNode]: ...
    def kept_alive_by_dead_branches(self) -> set[SymbolNode]: ...
    def count_nodes(self, prefix: Path | None = None) -> dict[str, int]: ...
```

`PackageView` is filtered by `package.path` rather than by tree paths. Its
methods (`modules`, `declarations`, `dead`, `reachable`, `graph`,
`remove_dead_code`, etc.) all key off the package's name; the implementation
iterates the package's per-phase contributions via a private accessor
`_phase_contributions(name)` so it doesn't reach into `_contributions`
directly.

## Phase enum, not strings

The pipeline has a `Phase` enum with `EXPORTS` and `INTERNALS`. All phase keys
in the analyzer are `tuple[str, Phase]`. Don't use raw strings.

```python
class Phase(enum.Enum):
    EXPORTS = "exports"
    INTERNALS = "internals"
```

## Resolver changes

### `PathResolver` protocol

```python
class PathResolver(Cacheable, Protocol):
    def resolve(self, project_root: Path) -> list[Package]: ...
    def resolve_import(self, name, search_paths) -> str | Path | None: ...
```

### `ManualResolver`

Spec format unchanged: `path[:dep1,dep2]`. **Deps are now package names**, not
paths. Each spec produces one `Package` with `exported=(path,)` (entire dir
exported) and `deps=tuple(dep names)`. The package's `name` is the path's
final component.

### `UvResolver`

One `Package` per workspace member from `uv.lock`:

- `path = member_dir`
- `exported = (exported_tree_root(member_dir),)` if it returns a path,
  else `(member_dir/"src",)` if `src/` exists, else `()`
- `deps = tuple(d for d in lockfile_deps if d in workspace_members)`

The workspace root (`virtual = "."`) is skipped. Both `editable` and
`virtual` members are emitted.

### Drop `PyprojectResolver`

The conventional-layout fallback offered no value over `-p .` (or a custom
resolver), and `[[tool.dead-cst.packages]]` / `[[tool.dead-cst.trees]]`
duplicated what a small custom `PathResolver` can express more directly.
`UvResolver` still uses `exported_tree_root` internally so build-backend
introspection stays available to other resolvers.

The CLI default is unchanged: `ManualResolver(specs=["."])` when no `-p` /
`--resolver` is passed.

## Cache fingerprint (recommendation, not landed on the branch)

The branch ships per-(package, phase) fingerprints, derived from
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
phase-awareness, and invalidation becomes "any change to any `Package` or the
chain invalidates" — strictly more conservative than today (today, a non-dep
sibling change only invalidates *phase 2* of consumers, not phase 1) but
typically ≤2× cache misses on workspace structural changes, which are rare
relative to file edits.

The branch did not land this; it kept per-(package, phase) for fidelity with
the prior cache contract.

## Pipeline-level simplifications worth replicating

These all landed on the branch and are worth carrying forward:

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

4. **Drop the `_partitions` cache.** Walking `pkg.path.rglob("*.py")` is
   cheap relative to per-file visiting, and the cache only saved repeat
   `refresh()` calls on the same packages, which already short-circuit on
   `key in self._contributions`.

5. **Don't reach through `PackageView` into private analyzer state.** Add an
   `Analysis._phase_contributions(name) -> Iterator[_PhaseContribution]`
   accessor and use it from `PackageView`. Same for `Analysis._validated`
   access.

## A subtle bug the agents caught

The first stab at `_build_phase_lookup` for phase 2 looked like:

```python
lookup.merge(contrib.current_trie)             # X's internals
lookup.merge(own_exports.current_trie)         # X's exports
lookup.merge(all_exports)                      # everyone's exports — including X
```

This double-merges X's exports (`own_exports.current_trie` + the X portion of
`all_exports`) and emits `SymbolTrie collision at p.foo: keeping ..., dropping
...` warnings. The fix: drop the `own_exports.current_trie` line. For phase 1
contributions, `current_trie == export_trie` (because `_apply_payload` adds
the same nodes to both when `export_trie is not None`), so `all_exports`
already covers it.

```python
lookup.merge(contrib.current_trie)             # X's internals
lookup.merge(all_exports)                      # everyone's exports (incl X)
```

## Public surface delta

### Added

- `dead_cst.Package` (re-exported at top level).
- `dead_cst.resolvers.{Package, validate_packages, assign_file_to_package,
  is_exported_file, export_search_root}`.

### Removed

- `dead_cst.SourceTree`, `dead_cst.SourceTreeFlags`.
- `dead_cst.resolvers.{validate_source_trees, assign_file_to_tree}`.
- `dead_cst.resolvers.PyprojectResolver`.
- `[[tool.dead-cst.trees]]` / `[[tool.dead-cst.packages]]` config keys.
- `VenvResolver`, `MissingVenvError`, `find_venv_site_packages`.
- `dead_cst.resolvers.{PathMap, merge_paths}`.

The public-API snapshot in `tests/test_public_api.py` needs to match.

## Tests + docs touchpoints

Files that needed updating in the original branch (use as a checklist):

```
dead_cst/__init__.py                     # Package re-export, docstring
dead_cst/analyze.py                      # full rewrite of orchestration
dead_cst/cli.py                          # source_trees -> packages, _dead_suite_locations
dead_cst/contrib/uv_resolver.py          # emit Package per member
dead_cst/resolvers/__init__.py           # __all__, BUILTIN_RESOLVERS, drop PyprojectResolver
dead_cst/resolvers/_core.py              # new Package model + helpers + validation
dead_cst/resolvers/_exports.py           # docstrings (drop PyprojectResolver mention)
dead_cst/resolvers/manual.py             # deps as names
dead_cst/resolvers/pyproject.py          # DELETE
tests/test_analysis.py                   # _contributions key shape
tests/test_cache.py                      # use UvResolver as the second-resolver fingerprint
tests/test_cli.py                        # ManualResolver spec parser test
tests/test_public_api.py                 # snapshot
tests/test_resolvers/test_core.py        # rewrite for Package
tests/test_resolvers/test_pyproject.py   # DELETE
tests/test_resolvers/test_uv_resolver.py # rewrite for Package
README.md
CHANGELOG.md                             # [Unreleased] block
CLAUDE.md                                # path resolution + two-phase composition sections
CONTRIBUTING.md
examples/README.md
examples/scripts-and-all/README.md       # switch to -p src
examples/uv-workspace/README.md          # `--resolver uv` (was uv_workspace)
```

## What `main` likely changed under you

Without seeing the competing refactor, expect conflicts in:

- `dead_cst/analyze.py` — the orchestration layer is the most rewritten file
  on this branch; any structural change on main collides hard.
- `dead_cst/resolvers/_core.py` — wholly replaced.
- `dead_cst/resolvers/__init__.py` — public surface re-export list.
- `tests/test_public_api.py` — snapshot.
- Any test that builds resolvers manually.

Strategy: re-implement the data model + helpers first (`_core.py`),
land the new resolvers (`manual.py`, `uv_resolver.py`, drop `pyproject.py`),
*then* rewire `analyze.py` against whatever main looks like now. The
two-phase pipeline has an obvious shape:

```python
# in Analysis._materialize_full:
all_exports = SymbolTrie()
for c in self._contributions.values():
    if c.phase is Phase.EXPORTS:
        all_exports.merge(c.export_trie)

for phase in (Phase.EXPORTS, Phase.INTERNALS):
    for pkg in self._validated.topo_order:
        contrib = self._contributions.get((pkg.name, phase))
        if contrib is None:
            continue
        _compose_contribution(
            contrib,
            target_graph=g,
            symbol_lookup=self._build_phase_lookup(pkg, phase, all_exports),
            plugins=self._plugins,
            project_root=self._project_root,
        )
```

## Validation gates

1. `uv run pytest` — the original branch ran 768 unit tests cleanly.
2. `uv run pytest -m e2e` — 10 e2e in ~29s.
3. `uv run prek run --all-files` — ruff + format + ty all clean.
4. CLI smoke: `uv run dead-cst analyze examples/uv-workspace --resolver uv`.
