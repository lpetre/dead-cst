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
    make_payload,
    synthetic_node,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload

PATCH_TARGET_PREFIX = "<patch-target>:"

_MOCK_MODULES = frozenset({"unittest.mock", "mock"})
_MOCKER_NAME = "mocker"
_MONKEYPATCH_NAME = "monkeypatch"


@dataclass
class MockPatchPlugin:
    """Resolve string-fqname ``patch(...)`` calls to their target decl.

    See module docstring for the recognized forms and limitations.
    """

    name: str = "mock_patch"
    version: int = 1778025601

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        patch_aliases, mock_module_aliases = _collect_mock_imports(ctx.module)
        module_node = next((n for n in ctx.payload.nodes if n.type == "module"), None)
        if module_node is None:
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
                owners = [module_node]

            finder.targets.clear()
            stmt.visit(finder)
            seen: set[str] = set()
            for fqname in finder.targets:
                if fqname in seen:
                    continue
                seen.add(fqname)
                synth = synthetic_node(f"{PATCH_TARGET_PREFIX}{fqname}", ctx.path)
                nodes.append(synth)
                for owner in owners:
                    edges.append((owner, synth, SYNTHETIC_POSITION))

        if not nodes:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for node in ctx.base_nodes():
            if node.type != "synthetic" or not node.fqname.startswith(PATCH_TARGET_PREFIX):
                continue
            fqname = node.fqname[len(PATCH_TARGET_PREFIX) :]
            existing = set(ctx.graph.successors(node))
            for decl in ctx.find_declarations(fqname):
                if decl not in existing:
                    yield AddEdge(node, decl)
            mod = ctx.find_module(fqname)
            if mod is not None and mod not in existing:
                yield AddEdge(node, mod)


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
                        if isinstance(alias.name, cst.Name) and alias.name.value == "patch":
                            patch_aliases.add(_asname_value(alias) or "patch")
                if source == "unittest":
                    for alias in small.names:
                        if isinstance(alias.name, cst.Name) and alias.name.value == "mock":
                            module_aliases.add(_asname_value(alias) or "mock")
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    name = _dotted_import_name(alias.name)
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
    return _dotted_import_name(node.module)


def _dotted_import_name(node: cst.CSTNode | None) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        head = _dotted_import_name(node.value)
        if head is None:
            return None
        return f"{head}.{node.attr.value}"
    return None


class _PatchCallFinder(cst.CSTVisitor):
    """Walk a CST subtree, collecting fqname strings from patch calls."""

    def __init__(self, patch_aliases: set[str], mock_module_aliases: set[str]) -> None:
        super().__init__()
        self._patch_aliases = patch_aliases
        self._mock_module_aliases = mock_module_aliases
        self.targets: list[str] = []

    def visit_Call(self, node: cst.Call) -> bool | None:
        if self._matches_patch(node.func):
            target = _first_string_arg(node)
            if target is not None:
                self.targets.append(target)
        else:
            target = _monkeypatch_target(node)
            if target is not None:
                self.targets.append(target)
        return True

    def _matches_patch(self, func: cst.BaseExpression) -> bool:
        if isinstance(func, cst.Name):
            return func.value in self._patch_aliases
        if not isinstance(func, cst.Attribute) or func.attr.value != "patch":
            return False
        if isinstance(func.value, cst.Name):
            return func.value.value in self._mock_module_aliases or func.value.value == _MOCKER_NAME
        if isinstance(func.value, cst.Attribute):
            return _dotted_attr_name(func.value) in _MOCK_MODULES
        return False


def _first_string_arg(call: cst.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if first.keyword is not None:
        return None
    return _string_value(first.value)


# Methods of pytest's ``monkeypatch`` fixture whose first arg can be a
# fully-qualified string name. The value is the positional-argument
# count for the *string-fqname* form (the object form takes one extra
# positional arg, which is how we disambiguate).
_MONKEYPATCH_FQNAME_METHODS: dict[str, int] = {
    "setattr": 2,  # setattr("X.Y", value)              [vs setattr(obj, "name", value)]
    "delattr": 1,  # delattr("X.Y")                     [vs delattr(obj, "name")]
}


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
    return _string_value(positional[0].value)


def _string_value(expr: cst.BaseExpression) -> str | None:
    if not isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
        return None
    try:
        value = expr.evaluated_value
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _dotted_attr_name(node: cst.BaseExpression) -> str | None:
    parts: list[str] = []
    current: cst.BaseExpression = node
    while isinstance(current, cst.Attribute):
        parts.append(current.attr.value)
        current = current.value
    if not isinstance(current, cst.Name):
        return None
    parts.append(current.value)
    parts.reverse()
    return ".".join(parts)
