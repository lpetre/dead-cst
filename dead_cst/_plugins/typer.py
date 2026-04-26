"""Plugin: keep Typer command and callback handlers alive.

Strategy: find top-level ``Typer()`` instances in each module and emit
inverse edges (instance -> handler) for every ``@<instance>.command(...)``
and ``@<instance>.callback(...)`` decorator. Typer instances are *not*
seeded as entrypoints; reachability is expected to flow through
``[project.scripts]`` (handled by :class:`ProjectScriptsPlugin`) or an
``if __name__ == "__main__": app()`` block (handled by
:class:`MainBlockPlugin`). Sub-typers attached via ``app.add_typer(sub)``
are kept alive through the ordinary reference tracked on that call.

This routes ``why-alive`` chains through the Typer app variable users
recognize ("alive because it's a command on ``app``") and lets a
sub-typer that's never ``add_typer``'d surface as dead code, mirroring
the behavior of :class:`FastAPIPlugin` for ``APIRouter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
from libcst.metadata import FullRepoManager

from .._symbols import SymbolNode
from ._core import AddEdge, GraphOp, PluginContext

# Attribute names ``Typer`` uses to register a callable. Matched as the
# rightmost attribute of ``@<instance>.<name>(...)``.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset({"command", "callback"})

_TYPER_CLASS = "Typer"


@dataclass
class TyperPlugin:
    """Wire Typer command and callback handlers through their app instance.

    For each module the plugin:

    1. Inspects ``from typer import ...`` / ``import typer`` to learn
       which local names refer to ``Typer``.
    2. Finds top-level assignments ``X = Typer(...)`` (including
       ``AnnAssign`` and aliased / module-prefixed forms) and records
       ``X`` as a Typer instance.
    3. For every top-level function decorated ``@X.command(...)`` or
       ``@X.callback(...)``, emits an edge ``X -> handler`` so the
       handler is reachable whenever ``X`` is.

    Typer instances are not auto-marked as entrypoints. The expected
    wiring is ``[project.scripts]`` (``ProjectScriptsPlugin`` seeds
    ``module:app``) or an ``if __name__ == "__main__": app()`` block
    (``MainBlockPlugin``); both paths make the instance reachable, after
    which this plugin's edges keep its commands alive. Sub-typers added
    via ``app.add_typer(sub)`` are reached through the regular reference
    tracking on that call.

    Limitations: only top-level ``X = Typer(...)`` assignments with a
    single ``Name`` target are detected. Factory-style apps
    (``def make_app(): return Typer()``) and class-attribute apps
    (``self.app = Typer()``) are not handled; users can still keep those
    alive with explicit ``-e`` entrypoints.

    This is a :class:`CSTAwareEdgePlugin` because detection needs the
    original CST.
    """

    name: str = "typer"
    cst_aware: bool = True

    def contribute(
        self, ctx: PluginContext, managers: dict[Path, FullRepoManager]
    ) -> Iterable[GraphOp]:
        modules_by_path: dict[Path, SymbolNode] = {}
        for node in ctx.graph.nodes:
            if node.type == "module":
                modules_by_path[node.path] = node

        for path, module_node in modules_by_path.items():
            wrapper = _wrapper_for(path, managers)
            if wrapper is None:
                continue
            typer_imports = _collect_typer_imports(wrapper.module)
            if not typer_imports:
                continue
            instances = _find_instances(wrapper.module, typer_imports)
            if not instances:
                continue
            handlers = _find_handlers(wrapper.module, instances)

            module_fqname = module_node.fqname
            for var_name in instances:
                instance_decls = ctx.find_declarations(f"{module_fqname}.{var_name}")
                if not instance_decls:
                    continue
                for instance_decl in instance_decls:
                    for handler_name in handlers.get(var_name, ()):
                        for handler_decl in ctx.find_declarations(
                            f"{module_fqname}.{handler_name}"
                        ):
                            yield AddEdge(instance_decl, handler_decl)


def _wrapper_for(path: Path, managers: dict[Path, FullRepoManager]):
    for base, mgr in managers.items():
        if not path.is_relative_to(base):
            continue
        try:
            return mgr.get_metadata_wrapper_for_path(path)
        except Exception:
            return None
    return None


def _collect_typer_imports(module: cst.Module) -> dict[str, str]:
    """Return ``{local_name: target}`` for names imported from ``typer``.

    ``target`` is either ``"Typer"`` or ``"<module>"`` (the whole
    ``typer`` package, for ``import typer``).
    """
    bindings: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                if not _is_typer_module(small):
                    continue
                if isinstance(small.names, cst.ImportStar):
                    continue
                for alias in small.names:
                    target = alias.name.value if isinstance(alias.name, cst.Name) else None
                    if target != _TYPER_CLASS:
                        continue
                    local = alias.asname.name.value if alias.asname else target
                    if isinstance(local, str):
                        bindings[local] = _TYPER_CLASS
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    if not _is_name(alias.name, "typer"):
                        continue
                    local = alias.asname.name.value if alias.asname else "typer"
                    if isinstance(local, str):
                        bindings[local] = "<module>"
    return bindings


def _is_typer_module(node: cst.ImportFrom) -> bool:
    if node.relative:
        return False
    return _is_name(node.module, "typer")


def _is_name(node: cst.CSTNode | None, value: str) -> bool:
    return isinstance(node, cst.Name) and node.value == value


def _find_instances(module: cst.Module, typer_imports: dict[str, str]) -> set[str]:
    """Return the set of top-level names bound to a ``Typer(...)`` call."""
    instances: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target_name, value = _single_target_assignment(small)
            if target_name is None or not isinstance(value, cst.Call):
                continue
            if _is_typer_call(value.func, typer_imports):
                instances.add(target_name)
    return instances


def _single_target_assignment(
    stmt: cst.BaseSmallStatement,
) -> tuple[str | None, cst.BaseExpression | None]:
    """Extract ``(name, rhs)`` for ``X = ...`` / ``X: T = ...``; else ``(None, None)``."""
    if isinstance(stmt, cst.Assign):
        if len(stmt.targets) != 1:
            return None, None
        target = stmt.targets[0].target
        if isinstance(target, cst.Name):
            return target.value, stmt.value
    elif isinstance(stmt, cst.AnnAssign):
        if isinstance(stmt.target, cst.Name) and stmt.value is not None:
            return stmt.target.value, stmt.value
    return None, None


def _is_typer_call(func: cst.BaseExpression, typer_imports: dict[str, str]) -> bool:
    """Return ``True`` if ``func`` denotes a ``Typer`` constructor."""
    if isinstance(func, cst.Name):
        return typer_imports.get(func.value) == _TYPER_CLASS
    if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
        # ``typer.Typer(...)`` / ``t.Typer(...)``
        if typer_imports.get(func.value.value) == "<module>":
            return func.attr.value == _TYPER_CLASS
    return False


def _find_handlers(module: cst.Module, instance_vars: set[str]) -> dict[str, list[str]]:
    """Return ``{instance_var: [handler_func_name, ...]}`` for decorated handlers."""
    handlers: dict[str, list[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            owner = _registration_decorator_owner(dec.decorator)
            if owner is None or owner not in instance_vars:
                continue
            handlers.setdefault(owner, []).append(stmt.name.value)
            break
    return handlers


def _registration_decorator_owner(expr: cst.BaseExpression) -> str | None:
    """For ``@X.command(...)`` / ``@X.command`` return ``"X"`` (only when the
    rightmost attribute is a known Typer registration name and ``X`` is a
    bare ``Name``). Returns ``None`` otherwise."""
    if isinstance(expr, cst.Call):
        expr = expr.func
    if not isinstance(expr, cst.Attribute):
        return None
    if expr.attr.value not in _REGISTRATION_DECORATORS:
        return None
    if not isinstance(expr.value, cst.Name):
        return None
    return expr.value.value
