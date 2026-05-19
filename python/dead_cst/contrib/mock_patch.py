"""Plugin: keep symbols referenced by string-fqname patch calls alive.

``unittest.mock.patch``, pytest-mock's ``mocker.patch``, and pytest's
``monkeypatch.setattr`` / ``monkeypatch.delattr`` reference their
target by fully-qualified string name. The plugin resolves those
strings to their target decls and emits keep-alive edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..plugins._base import Plugin, native

PATCH_TARGET_PREFIX = "<patch-target>:"

_MOCK_MODULES = frozenset({"unittest.mock", "mock"})
_MOCKER_NAME = "mocker"
_MONKEYPATCH_NAME = "monkeypatch"

# Methods of pytest's ``monkeypatch`` fixture whose first arg can be a
# fully-qualified string name. The value is the positional-argument
# count for the string-fqname form.
_MONKEYPATCH_FQNAME_METHODS: dict[str, int] = {
    "setattr": 2,  # setattr("X.Y", value)              [vs setattr(obj, "name", value)]
    "delattr": 1,  # delattr("X.Y")                     [vs delattr(obj, "name")]
}


@dataclass
class MockPatchPlugin(Plugin):
    """Resolve string-fqname ``patch(...)`` calls to their target decl."""

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        refs: list[native.CallRef] = []
        for module in _MOCK_MODULES:
            refs.extend(
                native.query(ctx).calls().where_module(module).where_name("patch").string_arg_at(0)
            )
        refs.extend(
            native.query(ctx).calls().where_owner(_MOCKER_NAME).where_attr("patch").string_arg_at(0)
        )
        for attr, required in _MONKEYPATCH_FQNAME_METHODS.items():
            refs.extend(
                native.query(ctx)
                .calls()
                .where_owner(_MONKEYPATCH_NAME)
                .where_attr(attr)
                .string_arg_at(0)
                .where_required_positional(required)
            )

        owners_by_fqname: dict[str, list[native.SymbolNode]] = {}
        for ref in refs:
            owners_by_fqname.setdefault(ref.string_arg, []).append(ref.owner)

        for fqname, owners in owners_by_fqname.items():
            targets = list(native.query(ctx).declarations(fqname))
            mod = native.query(ctx).module(fqname)
            if mod is not None:
                targets.append(mod)
            yield native.AddNode(
                fqname=f"{PATCH_TARGET_PREFIX}{fqname}",
                path=owners[0].path,
                edges_from=owners,
                edges_to=targets,
            )
