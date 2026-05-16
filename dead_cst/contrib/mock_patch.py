"""Plugin: keep symbols referenced by string-fqname patch calls alive.

``unittest.mock.patch``, pytest-mock's ``mocker.patch``, and pytest's
``monkeypatch.setattr`` / ``monkeypatch.delattr`` reference their
target by fully-qualified string name. The static analyzer can't see
those references, so a symbol whose only consumers are tests patching
it looks unused even though removing it would break the test.

For each file the plugin scans every call expression with a
string-literal first argument matching one of these forms::

    from unittest.mock import patch; patch("pkg.mod.func")
    from mock import patch;          patch("pkg.mod.func")
    from unittest import mock;       mock.patch("pkg.mod.func")
    import mock;                     mock.patch("pkg.mod.func")
    import unittest.mock;            unittest.mock.patch("pkg.mod.func")
    import mock as m;                m.patch("pkg.mod.func")
    @patch("pkg.mod.func")                   # decorator form
    mocker.patch("pkg.mod.func")             # pytest-mock fixture
    monkeypatch.setattr("pkg.mod.func", v)   # pytest monkeypatch fixture
    monkeypatch.delattr("pkg.mod.func")      # pytest monkeypatch fixture

Two phases:

* :meth:`observe` walks the parsed CST. For each top-level decl it
  collects every recognized patch call inside that decl's subtree
  (including its decorators) and emits a
  ``<patch-target>:<fqname>`` synthetic plus an edge from the
  enclosing top-level decl to the synthetic. Module-level patches
  (rare) hang off the file's module node.
* :meth:`finalize` walks every ``<patch-target>:`` synthetic and
  resolves the fqname through ``ctx.find_declarations`` (which walks
  back through dotted segments to find the enclosing first-party
  decl). Unresolved fqnames -- patches against third-party code --
  stay as dangling synthetics, which is harmless.

The ``mocker`` and ``monkeypatch`` parameter names are recognized
unconditionally because they're the pytest-mock and pytest
conventions; an unrelated ``mocker.patch(...)`` /
``monkeypatch.setattr(...)`` call whose string doesn't resolve to a
first-party decl produces a dangling synthetic with no observable
effect.

For ``monkeypatch.setattr`` / ``monkeypatch.delattr`` the plugin
distinguishes the fqname-string form from the object form by
positional-argument count -- ``setattr("X.Y", value)`` is two
positional args (fqname + value), while ``setattr(obj, "name", value)``
is three positional args (object + name + value). Only the
two-positional / one-positional form (``delattr("X.Y")``) is treated
as a string fqname; the object form is already a real reference the
analyzer sees.

Limitations:

* Only string-literal first args are recognized. A patch target
  computed at runtime (``patch(target_var)``) is invisible.
* ``patch.object(Cls, "attr")`` is not handled here -- ``Cls`` is
  already a real reference the analyzer sees, and the ``"attr"``
  string only names a method/attribute that doesn't have its own
  graph node anyway (methods are attributed to their enclosing class).
* ``patch.dict`` / ``patch.multiple`` are not recognized: their first
  arg can be either a string or a live object, and the kwargs name
  attributes that the class node already keeps alive.
* ``monkeypatch.setitem`` / ``setenv`` / ``syspath_prepend`` / etc.
  are not recognized -- their string args name dict keys / env vars
  / paths, not first-party symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst
from libcst.metadata import CodeRange

from ..graph import SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    GraphOp,
    ObserveContext,
    PluginContext,
    _asname_value,
    decls_by_simple_name,
    dotted_name,
    is_name,
    make_payload,
    module_node,
    string_value,
    synthetic_node,
)

if TYPE_CHECKING:
    import dead_cst_ty_native as native

    from ..graph import VisitorPayload

PATCH_TARGET_PREFIX = "<patch-target>:"

_MOCK_MODULES = frozenset({"unittest.mock", "mock"})
_MOCKER_NAME = "mocker"
_MONKEYPATCH_NAME = "monkeypatch"

# Methods of pytest's ``monkeypatch`` fixture whose first arg can be a
# fully-qualified string name. The value is the positional-argument
# count for the string-fqname form (the object form takes one extra
# positional arg, which is how we disambiguate).
_MONKEYPATCH_FQNAME_METHODS: dict[str, int] = {
    "setattr": 2,  # setattr("X.Y", value)              [vs setattr(obj, "name", value)]
    "delattr": 1,  # delattr("X.Y")                     [vs delattr(obj, "name")]
}


@dataclass
class MockPatchPlugin:
    """Resolve string-fqname ``patch(...)`` calls to their target decl.

    See module docstring for the recognized forms and limitations.
    """

    name: str = "mock_patch"
    version: int = 1778025601

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        patch_aliases, mock_module_aliases = _collect_mock_imports(ctx.module)
        # ``mocker.patch`` (pytest-mock) and ``monkeypatch.setattr`` /
        # ``delattr`` (pytest) are recognized regardless of imports
        # because they're the standard pytest-fixture call shape.
        # Folding ``mocker`` into ``mock_module_aliases`` here lets
        # ``_PatchCallFinder`` treat all ``<X>.patch(...)`` matches
        # uniformly.
        mock_module_aliases.add(_MOCKER_NAME)
        module = module_node(ctx.payload)
        if module is None:
            return None
        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        finder = _PatchCallFinder(patch_aliases, mock_module_aliases)

        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        for stmt in ctx.module.body:
            if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
                owners = decls_by_name.get(stmt.name.value, [])
            else:
                owners = []
            if not owners:
                owners = [module]

            for fqname in finder.find(stmt):
                synth = synthetic_node(f"{PATCH_TARGET_PREFIX}{fqname}", ctx.path)
                nodes.append(synth)
                for owner in owners:
                    edges.append((owner, synth, SYNTHETIC_POSITION))

        if not nodes:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Memoize trie lookups: the same fqname appears in one synthetic
        # per file that patches it, and trie walks are O(parts).
        decls_cache: dict[str, list[SymbolNode]] = {}
        modules_cache: dict[str, SymbolNode | None] = {}
        for node in ctx.contribution.nodes:
            if node.type != "synthetic" or not node.fqname.startswith(PATCH_TARGET_PREFIX):
                continue
            fqname = node.fqname[len(PATCH_TARGET_PREFIX) :]
            if fqname not in decls_cache:
                decls_cache[fqname] = ctx.find_declarations(fqname)
                modules_cache[fqname] = ctx.find_module(fqname)
            existing = {
                ctx.graph.raw[i] for i in ctx.graph.raw.successor_indices(ctx.graph.index(node))
            }
            for decl in decls_cache[fqname]:
                if decl not in existing:
                    yield AddEdge(node, decl)
            mod = modules_cache[fqname]
            if mod is not None and mod not in existing:
                yield AddEdge(node, mod)

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        import dead_cst_ty_native as native

        pairs: list[tuple[native.NativeNode, str]] = []
        for module in _MOCK_MODULES:
            pairs.extend(ctx.find_calls_to_imported(module, "patch", 0))
        pairs.extend(ctx.find_calls_on_var(_MOCKER_NAME, "patch", 0, required_positional=None))
        for attr, required in _MONKEYPATCH_FQNAME_METHODS.items():
            pairs.extend(
                ctx.find_calls_on_var(_MONKEYPATCH_NAME, attr, 0, required_positional=required)
            )

        owners_by_fqname: dict[str, list[native.NativeNode]] = {}
        for owner, fqname in pairs:
            owners_by_fqname.setdefault(fqname, []).append(owner)

        for fqname, owners in owners_by_fqname.items():
            targets = list(ctx.find_declarations(fqname))
            mod = ctx.find_module(fqname)
            if mod is not None:
                targets.append(mod)
            # The marker is per-fqname, not per-file: any owner's path is
            # an acceptable anchor for `why-alive` output.
            yield native.AddNode(
                fqname=f"{PATCH_TARGET_PREFIX}{fqname}",
                path=owners[0].path,
                edges_from=owners,
                edges_to=targets,
            )


def _collect_mock_imports(module: cst.Module) -> tuple[set[str], set[str]]:
    """Return ``(patch_aliases, mock_module_aliases)``.

    ``patch_aliases`` are local names bound directly to
    ``unittest.mock.patch`` / ``mock.patch`` (e.g. ``patch`` after
    ``from unittest.mock import patch``, or ``p`` after ``... as p``).
    ``mock_module_aliases`` are local names bound to the
    ``unittest.mock`` or ``mock`` module so ``<alias>.patch(...)``
    matches.

    ``import unittest.mock`` (no asname) binds the local name
    ``unittest`` rather than ``unittest.mock``; the dotted call form
    ``unittest.mock.patch(...)`` is matched syntactically in
    :class:`_PatchCallFinder` instead, so no alias entry is recorded
    for that case.
    """
    patch_aliases: set[str] = set()
    module_aliases: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                source = _import_from_source(small)
                if source is None:
                    continue
                if isinstance(small.names, cst.ImportStar):
                    if source in _MOCK_MODULES:
                        patch_aliases.add("patch")
                    continue
                if source in _MOCK_MODULES:
                    for alias in small.names:
                        if is_name(alias.name, "patch"):
                            patch_aliases.add(_asname_value(alias) or "patch")
                if source == "unittest":
                    for alias in small.names:
                        if is_name(alias.name, "mock"):
                            module_aliases.add(_asname_value(alias) or "mock")
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    name = dotted_name(alias.name)
                    if name not in _MOCK_MODULES:
                        continue
                    asname = _asname_value(alias)
                    if asname:
                        module_aliases.add(asname)
                    elif name == "mock":
                        module_aliases.add("mock")
    return patch_aliases, module_aliases


def _import_from_source(node: cst.ImportFrom) -> str | None:
    if node.relative:
        return None
    return dotted_name(node.module)


class _PatchCallFinder(cst.CSTVisitor):
    """Walk a CST subtree, collecting fqname strings from patch calls.

    A single instance is reused across statements via :meth:`find`,
    which clears the per-statement target list before visiting.
    """

    def __init__(self, patch_aliases: set[str], mock_module_aliases: set[str]) -> None:
        super().__init__()
        self._patch_aliases = patch_aliases
        self._mock_module_aliases = mock_module_aliases
        self._targets: list[str] = []

    def find(self, stmt: cst.CSTNode) -> list[str]:
        """Return the deduped fqname strings collected from ``stmt``'s subtree."""
        self._targets.clear()
        stmt.visit(self)
        # ``dict.fromkeys`` preserves first-seen order while deduping --
        # multiple ``patch("X")`` calls in the same statement collapse
        # to one synthetic per fqname.
        return list(dict.fromkeys(self._targets))

    def visit_Call(self, node: cst.Call) -> bool | None:
        if self._matches_patch(node.func):
            target = _first_string_arg(node)
            if target is not None:
                self._targets.append(target)
        else:
            target = _monkeypatch_target(node)
            if target is not None:
                self._targets.append(target)
        return True

    def _matches_patch(self, func: cst.BaseExpression) -> bool:
        if isinstance(func, cst.Name):
            return func.value in self._patch_aliases
        if not isinstance(func, cst.Attribute) or func.attr.value != "patch":
            return False
        if isinstance(func.value, cst.Name):
            return func.value.value in self._mock_module_aliases
        if isinstance(func.value, cst.Attribute):
            return dotted_name(func.value) in _MOCK_MODULES
        return False


def _first_string_arg(call: cst.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if first.keyword is not None:
        return None
    return string_value(first.value)


def _monkeypatch_target(call: cst.Call) -> str | None:
    """Return the fqname string from ``monkeypatch.setattr`` / ``delattr``.

    Disambiguates the string-fqname form from the object form by
    positional-argument count. Returns ``None`` if the call isn't a
    monkeypatch fqname-form call.
    """
    func = call.func
    if not isinstance(func, cst.Attribute):
        return None
    if not isinstance(func.value, cst.Name) or func.value.value != _MONKEYPATCH_NAME:
        return None
    expected_positional = _MONKEYPATCH_FQNAME_METHODS.get(func.attr.value)
    if expected_positional is None:
        return None
    positional = [a for a in call.args if a.keyword is None]
    if len(positional) != expected_positional:
        return None
    return string_value(positional[0].value)
