"""Builder API over the rust ProjectContext queries.

Phase 1 pure-Python wrapper — internally delegates to the rust
``find_*`` methods. Phase 2 ports the queries to rust to enable
predicate fusion (path/module filters that prune file iteration
before parsing). Plugin imports won't change between phases —
the public surface (``CallRef`` / ``DecoratorRef`` /
``ConstructionRef`` / ``query``) stays put.

Usage::

    from dead_cst_ty_native import query, AddEntrypoint

    for ref in query(ctx).decorators().where_module("celery").where_name("shared_task"):
        yield AddEntrypoint(ref.decorated, marker="<celery-task>")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:
    from . import NativeNode, ProjectContext


# ----- Result types -------------------------------------------------------


@dataclass(frozen=True)
class DecoratorRef:
    """One decorator application on a top-level function or class.

    Field nullability follows the underlying query shape:

    * ``where_module`` + ``where_name`` queries surface the decorated
      node but the legacy rust API doesn't tell us *which* of the
      names matched, so ``decorator_name`` is ``None`` in Phase 1.
      Phase 2 will populate it.
    * ``where_owner_attr`` / ``where_owner_attr_via`` surface the
      textual owner prefix as ``decorator_owner``.
    * ``where_owner_attr_via`` additionally fills ``decorator_via``
      with the middle attribute name.
    """

    decorated: "NativeNode"
    decorator_name: str | None
    decorator_owner: str | None
    decorator_via: str | None

    @property
    def path(self) -> str:
        return self.decorated.path


@dataclass(frozen=True)
class ConstructionRef:
    """One ``<var> = <Ctor>(...)`` construction at module scope."""

    var: "NativeNode"
    class_name: str

    @property
    def path(self) -> str:
        return self.var.path


@dataclass(frozen=True)
class CallRef:
    """One matched call site. ``string_arg`` is the positional
    string-literal at the index passed to
    :meth:`CallQuery.string_arg_at`."""

    owner: "NativeNode"
    string_arg: str

    @property
    def path(self) -> str:
        return self.owner.path


# ----- Query builder ------------------------------------------------------


def query(ctx: "ProjectContext") -> "QueryBuilder":
    """Open a chainable query builder against ``ctx``."""
    return QueryBuilder(ctx)


class QueryBuilder:
    """Entry point for the chainable query API.

    Use ``decorators()``, ``constructions()``, or ``calls()`` to
    pick the result-type stream. Each returned ``*Query`` accepts
    predicates and terminates with ``__iter__`` / ``collect()`` /
    ``first()`` / ``count()``.
    """

    def __init__(self, ctx: "ProjectContext") -> None:
        self._ctx = ctx

    def decorators(self) -> "DecoratorQuery":
        return DecoratorQuery(self._ctx)

    def constructions(self) -> "ConstructionQuery":
        return ConstructionQuery(self._ctx)

    def calls(self) -> "CallQuery":
        return CallQuery(self._ctx)


def _to_tuple(names: str | Sequence[str]) -> tuple[str, ...]:
    return (names,) if isinstance(names, str) else tuple(names)


def _apply_path_filter(refs, path_regex):
    if path_regex is None:
        return refs
    return [r for r in refs if path_regex.search(r.path)]


class DecoratorQuery:
    """Find decorated top-level functions / classes.

    Pick exactly one of the four decorator-shape predicates per
    chain; mixing them raises ``ValueError`` at ``collect()`` time.

    * ``where_module(m).where_name(n)`` — ``@m.x`` / ``@x`` where
      ``x`` is imported from ``m``.
    * ``where_callee(fqn)`` — fqn-form ``@<fqn>``.
    * ``where_owner_attr(attrs)`` — ``@<owner>.<attr>(...)`` shape;
      ``decorator_owner`` is the textual prefix.
    * ``where_owner_attr_via(via, attrs)`` —
      ``@<owner>.<via>.<attr>(...)`` two-level chain.
    * ``in_decl(node).where_name(names)`` — ``@<node>.<name>``
      same-file instance-method decorators.
    """

    def __init__(self, ctx: "ProjectContext") -> None:
        self._ctx = ctx
        self._module: str | None = None
        self._callee_fqn: str | None = None
        self._names: tuple[str, ...] | None = None
        self._owner_attrs: tuple[str, ...] | None = None
        self._via_attr: str | None = None
        self._in_decl: "NativeNode | None" = None
        self._path_regex: re.Pattern[str] | None = None

    def where_module(self, module: str) -> "DecoratorQuery":
        self._module = module
        return self

    def where_callee(self, fqn: str) -> "DecoratorQuery":
        self._callee_fqn = fqn
        return self

    def where_name(self, names: str | Sequence[str]) -> "DecoratorQuery":
        self._names = _to_tuple(names)
        return self

    def where_owner_attr(self, attrs: str | Sequence[str]) -> "DecoratorQuery":
        self._owner_attrs = _to_tuple(attrs)
        return self

    def where_owner_attr_via(self, via: str, attrs: str | Sequence[str]) -> "DecoratorQuery":
        self._via_attr = via
        self._owner_attrs = _to_tuple(attrs)
        return self

    def in_decl(self, node: "NativeNode") -> "DecoratorQuery":
        self._in_decl = node
        return self

    def where_path(self, regex: str) -> "DecoratorQuery":
        self._path_regex = re.compile(regex)
        return self

    def collect(self) -> list[DecoratorRef]:
        if self._owner_attrs is not None:
            if self._via_attr is not None:
                pairs = self._ctx.find_handler_decorators_via(
                    self._via_attr, list(self._owner_attrs)
                )
            else:
                pairs = self._ctx.find_handler_decorators(list(self._owner_attrs))
            refs = [
                DecoratorRef(
                    decorated=decorated,
                    decorator_name=None,
                    decorator_owner=owner_name,
                    decorator_via=self._via_attr,
                )
                for owner_name, decorated in pairs
            ]
        elif self._in_decl is not None:
            if self._names is None:
                raise ValueError("DecoratorQuery.in_decl(...) requires .where_name(...)")
            decls = self._ctx.find_decorations_on(self._in_decl, list(self._names))
            owner_simple = self._in_decl.fqname.rsplit(".", 1)[-1]
            refs = [
                DecoratorRef(
                    decorated=d,
                    decorator_name=None,
                    decorator_owner=owner_simple,
                    decorator_via=None,
                )
                for d in decls
            ]
        elif self._callee_fqn is not None:
            decls = self._ctx.find_decorated(self._callee_fqn)
            refs = [
                DecoratorRef(
                    decorated=d,
                    decorator_name=None,
                    decorator_owner=None,
                    decorator_via=None,
                )
                for d in decls
            ]
        elif self._module is not None and self._names is not None:
            decls = self._ctx.find_decorated_decls(self._module, list(self._names))
            refs = [
                DecoratorRef(
                    decorated=d,
                    decorator_name=None,
                    decorator_owner=None,
                    decorator_via=None,
                )
                for d in decls
            ]
        else:
            raise ValueError(
                "DecoratorQuery requires one of: where_callee(...); "
                "where_module(...) + where_name(...); "
                "where_owner_attr(...); where_owner_attr_via(via, attrs); "
                "or in_decl(node) + where_name(...)"
            )
        return _apply_path_filter(refs, self._path_regex)

    def first(self) -> DecoratorRef | None:
        items = self.collect()
        return items[0] if items else None

    def count(self) -> int:
        return len(self.collect())

    def __iter__(self) -> Iterator[DecoratorRef]:
        return iter(self.collect())


class ConstructionQuery:
    """Find module-scope ``<var> = <Ctor>(...)`` sites.

    Pick exactly one shape per chain:

    * ``where_module(m).where_name(n)`` — ctor imported from ``m``
      with bare name in ``n``.
    * ``where_class(fqn, include_subclasses=...)`` — every ctor for
      ``fqn`` (and optionally its subclass closure).
    """

    def __init__(self, ctx: "ProjectContext") -> None:
        self._ctx = ctx
        self._module: str | None = None
        self._names: tuple[str, ...] | None = None
        self._class_fqn: str | None = None
        self._include_subclasses = False
        self._path_regex: re.Pattern[str] | None = None

    def where_module(self, module: str) -> "ConstructionQuery":
        self._module = module
        return self

    def where_name(self, names: str | Sequence[str]) -> "ConstructionQuery":
        self._names = _to_tuple(names)
        return self

    def where_class(self, fqn: str, *, include_subclasses: bool = False) -> "ConstructionQuery":
        self._class_fqn = fqn
        self._include_subclasses = include_subclasses
        return self

    def where_path(self, regex: str) -> "ConstructionQuery":
        self._path_regex = re.compile(regex)
        return self

    def collect(self) -> list[ConstructionRef]:
        if self._class_fqn is not None:
            decls = self._ctx.find_constructions(
                self._class_fqn, include_subclasses=self._include_subclasses
            )
            cls_name = self._class_fqn.rsplit(".", 1)[-1]
            refs = [ConstructionRef(var=d, class_name=cls_name) for d in decls]
        elif self._module is not None and self._names is not None:
            pairs = self._ctx.find_instance_constructions(self._module, list(self._names))
            refs = [ConstructionRef(var=v, class_name=name) for v, name in pairs]
        else:
            raise ValueError(
                "ConstructionQuery requires either where_class(...) "
                "or where_module(...) + where_name(...)"
            )
        return _apply_path_filter(refs, self._path_regex)

    def first(self) -> ConstructionRef | None:
        items = self.collect()
        return items[0] if items else None

    def count(self) -> int:
        return len(self.collect())

    def __iter__(self) -> Iterator[ConstructionRef]:
        return iter(self.collect())


class CallQuery:
    """Find call sites whose positional string-literal at the configured
    index is captured.

    ``.string_arg_at(index)`` is required — it tells the query which
    positional argument to extract. Pick one shape:

    * ``where_module(m).where_name(n)`` — call to ``n`` imported from
      ``m`` (e.g. ``patch`` from ``unittest.mock``).
    * ``where_owner(o).where_attr(a)`` — ``<o>.<a>(...)`` literal
      receiver match.
    * ``where_attr(a)`` — ``<expr>.<a>(...)`` any receiver.
    """

    def __init__(self, ctx: "ProjectContext") -> None:
        self._ctx = ctx
        self._module: str | None = None
        self._name: str | None = None
        self._owner: str | None = None
        self._attr: str | None = None
        self._arg_index: int | None = None
        self._required_positional: int | None = None
        self._path_regex: re.Pattern[str] | None = None

    def where_module(self, module: str) -> "CallQuery":
        self._module = module
        return self

    def where_name(self, name: str) -> "CallQuery":
        self._name = name
        return self

    def where_owner(self, owner: str) -> "CallQuery":
        self._owner = owner
        return self

    def where_attr(self, attr: str) -> "CallQuery":
        self._attr = attr
        return self

    def string_arg_at(self, index: int) -> "CallQuery":
        self._arg_index = index
        return self

    def where_required_positional(self, n: int | None) -> "CallQuery":
        self._required_positional = n
        return self

    def where_path(self, regex: str) -> "CallQuery":
        self._path_regex = re.compile(regex)
        return self

    def collect(self) -> list[CallRef]:
        if self._arg_index is None:
            raise ValueError("CallQuery: .string_arg_at(index) is required")
        if self._module is not None and self._name is not None:
            pairs = self._ctx.find_calls_to_imported(self._module, self._name, self._arg_index)
        elif self._owner is not None and self._attr is not None:
            pairs = self._ctx.find_calls_on_var(
                self._owner,
                self._attr,
                self._arg_index,
                required_positional=self._required_positional,
            )
        elif self._attr is not None:
            pairs = self._ctx.find_calls_on_attr(self._attr, self._arg_index)
        else:
            raise ValueError(
                "CallQuery requires one of: where_module(...) + where_name(...); "
                "where_owner(...) + where_attr(...); or where_attr(...)"
            )
        refs = [CallRef(owner=o, string_arg=s) for o, s in pairs]
        return _apply_path_filter(refs, self._path_regex)

    def first(self) -> CallRef | None:
        items = self.collect()
        return items[0] if items else None

    def count(self) -> int:
        return len(self.collect())

    def __iter__(self) -> Iterator[CallRef]:
        return iter(self.collect())
