"""Plugin: keep discord.py bot/client handlers, Cogs, and extension hooks alive.

Strategy:

1. Find top-level ``X = commands.Bot(...)`` / ``X = discord.Client(...)``
   assignments (and the ``AutoSharded*`` variants). Each one is seeded
   as a synthetic entrypoint -- discord.py bots are loaded by
   ``bot.run(token)`` in a ``__main__`` block or script entry-point, so
   the bot itself is the framework-visible entrypoint, mirroring how
   :class:`FastAPIPlugin` treats a top-level ``FastAPI()`` instance.

2. Wire ``X -> handler`` edges for every ``@<bot>.<verb>(...)`` decorator
   on a top-level function. Single-attr decorators (``command``,
   ``event``, ``listen``, ``group``, ``hybrid_command``,
   ``hybrid_group``, ``check``, ``check_once``, ``before_invoke``,
   ``after_invoke``) plus the two-level ``@<bot>.tree.<verb>(...)`` slash-
   command decorators (``command``, ``context_menu``) are both
   recognized.

3. Detect Cog subclasses (``class Foo(commands.Cog): ...``). Each file
   that defines a Cog gets a ``<discordpy-cog>:<filename>`` synthetic
   entrypoint with edges to every Cog class and to the file's module-
   level ``setup`` / ``teardown`` functions, mirroring how discord.py
   dynamically loads extension modules at runtime.

4. Detect ``<expr>.load_extension("dotted.path")`` /
   ``load_extensions([...])`` call sites in files that import discord.
   Each captured string-literal target produces a
   ``<discordpy-extension>:<dotted.path>`` synthetic entrypoint;
   :meth:`finalize` stitches edges to the target module's whole
   surface, matching ``importlib.import_module`` semantics for dynamic
   extension loading.

Limitations:

* Bot / Client / Cog detection only handles top-level assignments and
  class definitions with bare ``Name`` targets. Factory-style apps
  (``def make_bot(): return commands.Bot()``) and class-attribute apps
  (``self.bot = commands.Bot()``) are not handled; users can keep
  those alive with explicit ``-e`` entrypoints.
* Slash-command decorators on Cog methods (``@app_commands.command()``,
  ``@commands.Cog.listener()``, ``@tasks.loop(...)``) need no special
  handling: methods are not graph nodes, so a live Cog class keeps
  every decorated method alive transitively.
* ``bot.add_command(...)`` / ``bot.add_cog(...)`` calls flow through
  the analyzer's ordinary reference edges -- no plugin synthetic
  required.
* ``load_extension`` arguments that are non-literal (variables,
  f-strings, dynamic lists) are skipped silently. Most extension
  registries use literal ``["cogs.greet", "cogs.admin"]`` constants
  and round-trip cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Container, Iterable

import libcst as cst
from libcst.metadata import CodeRange

from ..graph import NodeFlags, SymbolNode
from ..plugins._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    GraphOp,
    ObserveContext,
    PluginContext,
    decls_by_simple_name,
    decorator_owner,
    dotted_name,
    dotted_parts,
    is_name,
    make_payload,
    payload_imports_module,
    single_target_assignment,
    string_value,
    synthetic_node,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload


# Sentinel values used as the "target" entry in the unified discord
# import map. ``"<discord.ext.commands>"`` marks a local name bound to
# the ``commands`` submodule (``from discord.ext import commands``);
# ``"<discord>"`` marks a local name bound to the top-level ``discord``
# package (``import discord``). Concrete class names ("Bot", "Cog", ...)
# appear as themselves when imported directly.
_COMMANDS_MODULE = "<discord.ext.commands>"
_DISCORD_MODULE = "<discord>"

# Classes from ``discord.ext.commands`` that produce a bot instance.
_COMMANDS_BOT_KINDS: frozenset[str] = frozenset({"Bot", "AutoShardedBot"})

# Classes from ``discord`` that produce a client instance.
_DISCORD_CLIENT_KINDS: frozenset[str] = frozenset({"Client", "AutoShardedClient"})

# Cog base classes (from ``discord.ext.commands``). ``GroupCog`` is the
# slash-command analogue introduced in 2.0.
_COG_BASES: frozenset[str] = frozenset({"Cog", "GroupCog"})

# Single-attribute decorators on a bot/client variable.
_BOT_DECORATORS: frozenset[str] = frozenset(
    {
        "command",
        "event",
        "listen",
        "group",
        "hybrid_command",
        "hybrid_group",
        "check",
        "check_once",
        "before_invoke",
        "after_invoke",
    }
)

# Two-level decorators ``@<bot>.tree.<attr>(...)`` for slash commands.
_TREE_DECORATORS: frozenset[str] = frozenset({"command", "context_menu"})

# Extension-loader method names whose first positional argument is the
# dotted module path. ``load_extension`` takes a single string;
# ``load_extensions`` takes an iterable of strings.
_LOAD_EXTENSION_METHODS: frozenset[str] = frozenset({"load_extension", "load_extensions"})

DISCORDPY_APP_PREFIX = "<discordpy-app>:"
DISCORDPY_COG_PREFIX = "<discordpy-cog>:"
DISCORDPY_EXTENSION_PREFIX = "<discordpy-extension>:"


@dataclass
class DiscordPyPlugin:
    """discord.py-aware reachability.

    See module docstring for the four-pillar strategy (bot instances,
    decorator handlers, Cog subclasses, ``load_extension`` targets).
    """

    name: str = "discordpy"
    version: int = 1778566342

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        # Cheap gate: bail out for files that don't touch discord at all.
        if not payload_imports_module(ctx.payload, "discord"):
            return None

        imports = _collect_discord_imports(ctx.module)
        if not imports:
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        # 1. Bot/Client instances ----------------------------------------
        bot_vars = _find_bot_instances(ctx.module, imports)
        for var_name in bot_vars:
            for var_decl in decls_by_name.get(var_name, []):
                seed = synthetic_node(
                    f"{DISCORDPY_APP_PREFIX}{var_decl.fqname}",
                    ctx.path,
                    flags=NodeFlags.ENTRYPOINT,
                )
                nodes.append(seed)
                edges.append((seed, var_decl, SYNTHETIC_POSITION))

        # 2. Decorator handlers wired to a bot var -----------------------
        handlers = _find_discordpy_handlers(ctx.module, bot_vars)
        for var_name, handler_names in handlers.items():
            for var_decl in decls_by_name.get(var_name, []):
                for handler_name in handler_names:
                    for handler_decl in decls_by_name.get(handler_name, []):
                        edges.append((var_decl, handler_decl, SYNTHETIC_POSITION))

        # 3. Cog classes + the module's setup/teardown hooks -------------
        cog_class_decls: list[SymbolNode] = []
        for cog_name in _find_cog_subclass_names(ctx.module, imports):
            for decl in decls_by_name.get(cog_name, []):
                if decl.type == "class":
                    cog_class_decls.append(decl)
        if cog_class_decls:
            cog_seed = synthetic_node(
                f"{DISCORDPY_COG_PREFIX}{ctx.path.name}",
                ctx.path,
                flags=NodeFlags.ENTRYPOINT,
            )
            nodes.append(cog_seed)
            for cog_decl in cog_class_decls:
                edges.append((cog_seed, cog_decl, SYNTHETIC_POSITION))
            for hook_name in ("setup", "teardown"):
                for hook_decl in decls_by_name.get(hook_name, []):
                    if hook_decl.type == "function":
                        edges.append((cog_seed, hook_decl, SYNTHETIC_POSITION))

        # 4. load_extension(...) string-literal targets ------------------
        for fqname in _find_load_extension_targets(ctx.module):
            ext_seed = synthetic_node(
                f"{DISCORDPY_EXTENSION_PREFIX}{fqname}",
                ctx.path,
                flags=NodeFlags.ENTRYPOINT,
            )
            nodes.append(ext_seed)

        if not nodes and not edges:
            return None
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Stitch each ``<discordpy-extension>:<fqname>`` synthetic to its
        # target module's surface so the module + its top-level decls
        # (including the ``setup`` / ``teardown`` hooks discord.py calls
        # dynamically) stay alive. Mirrors how the visitor folds
        # ``importlib.import_module("m")`` into a star import on ``m``.
        for synth in list(ctx.package_nodes()):
            if synth.type != "synthetic" or not synth.fqname.startswith(DISCORDPY_EXTENSION_PREFIX):
                continue
            captured = synth.fqname[len(DISCORDPY_EXTENSION_PREFIX) :]
            for target in ctx.module_surface(captured):
                yield AddEdge(synth, target)


# --- helpers ----------------------------------------------------------------


def _collect_discord_imports(module: cst.Module) -> dict[str, str]:
    """Return a unified ``{local_name: target}`` map for discord.py imports.

    Five legal shapes are recognized:

    * ``from discord.ext.commands import Bot``       -> ``{"Bot": "Bot"}``
    * ``from discord.ext.commands import Bot as B``  -> ``{"B": "Bot"}``
    * ``from discord.ext import commands``           -> ``{"commands": "<discord.ext.commands>"}``
    * ``from discord import Client``                 -> ``{"Client": "Client"}``
    * ``import discord``                             -> ``{"discord": "<discord>"}``

    Class names outside the recognized discord.py whitelist
    (``Bot``/``AutoShardedBot``/``Client``/``AutoShardedClient``/``Cog``/
    ``GroupCog``) are dropped to keep the map small. Aliased forms
    (``as ...``) flow through naturally.
    """
    bindings: dict[str, str] = {}
    known_targets = _COMMANDS_BOT_KINDS | _DISCORD_CLIENT_KINDS | _COG_BASES
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom):
                if isinstance(small.names, cst.ImportStar):
                    continue
                module_name = _import_from_module_name(small)
                if module_name is None:
                    continue
                if module_name == "discord.ext.commands":
                    for alias in small.names:
                        if not isinstance(alias.name, cst.Name):
                            continue
                        target = alias.name.value
                        if target not in known_targets:
                            continue
                        local = _alias_local(alias) or target
                        bindings[local] = target
                elif module_name == "discord.ext":
                    for alias in small.names:
                        if not isinstance(alias.name, cst.Name):
                            continue
                        if alias.name.value != "commands":
                            continue
                        local = _alias_local(alias) or "commands"
                        bindings[local] = _COMMANDS_MODULE
                elif module_name == "discord":
                    for alias in small.names:
                        if not isinstance(alias.name, cst.Name):
                            continue
                        target = alias.name.value
                        if target not in known_targets:
                            continue
                        local = _alias_local(alias) or target
                        bindings[local] = target
            elif isinstance(small, cst.Import):
                for alias in small.names:
                    if is_name(alias.name, "discord"):
                        local = _alias_local(alias) or "discord"
                        bindings[local] = _DISCORD_MODULE
    return bindings


def _import_from_module_name(node: cst.ImportFrom) -> str | None:
    """Return the dotted module name for ``from <name> import ...``.

    ``None`` for relative imports (``from .x import y``) or unparseable
    module references. Supports dotted module names
    (``from discord.ext.commands import Bot``) that ``is_from_module``
    rejects because its underlying ``is_name`` check only matches a
    bare ``cst.Name``.
    """
    if node.relative:
        return None
    return dotted_name(node.module)


def _alias_local(alias: cst.ImportAlias) -> str | None:
    if alias.asname is None:
        return None
    name = alias.asname.name
    return name.value if isinstance(name, cst.Name) else None


def _matched_constructor(
    expr: cst.BaseExpression,
    imports: dict[str, str],
    valid_targets: Container[str],
) -> str | None:
    """Return the matched class name for ``Bot(...)``, ``commands.Bot(...)``,
    or ``discord.Client(...)``; ``None`` otherwise.

    Caller is expected to pass the ``Call.func`` directly (i.e. the
    expression has been unwrapped from the surrounding ``Call``).
    """
    if isinstance(expr, cst.Name):
        target = imports.get(expr.value)
        if target is not None and target in valid_targets:
            return target
    elif isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
        sentinel = imports.get(expr.value.value)
        attr = expr.attr.value
        if sentinel in (_COMMANDS_MODULE, _DISCORD_MODULE) and attr in valid_targets:
            return attr
    return None


def _find_bot_instances(module: cst.Module, imports: dict[str, str]) -> set[str]:
    """Top-level var names bound to a Bot/Client constructor call."""
    valid = _COMMANDS_BOT_KINDS | _DISCORD_CLIENT_KINDS
    found: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target, value = single_target_assignment(small)
            if target is None or not isinstance(value, cst.Call):
                continue
            kind = _matched_constructor(value.func, imports, valid)
            if kind is not None:
                found.add(target)
    return found


def _tree_decorator_owner(expr: cst.BaseExpression) -> str | None:
    """For ``@X.tree.<attr>(...)`` return ``"X"`` when ``attr`` is in
    :data:`_TREE_DECORATORS`. Returns ``None`` otherwise.

    The ``.tree`` attribute is the ``CommandTree`` instance discord.py
    auto-creates on every ``Bot`` -- it isn't a separate variable, so
    the owner is whatever bare ``Name`` heads the attribute chain.
    """
    if isinstance(expr, cst.Call):
        expr = expr.func
    parts = dotted_parts(expr)
    if parts is None or len(parts) != 3:
        return None
    if parts[1] != "tree" or parts[2] not in _TREE_DECORATORS:
        return None
    return parts[0]


def _find_discordpy_handlers(
    module: cst.Module, instance_vars: Container[str]
) -> dict[str, list[str]]:
    """Map ``bot_var -> [handler_func_name, ...]`` for decorated top-level functions.

    Recognizes both single-attribute decorators (``@bot.command()``,
    ``@bot.event``) and the two-level slash-command form
    (``@bot.tree.command()``, ``@bot.tree.context_menu()``).
    """
    handlers: dict[str, list[str]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            owner = decorator_owner(dec.decorator, _BOT_DECORATORS)
            if owner is None:
                owner = _tree_decorator_owner(dec.decorator)
            if owner is None or owner not in instance_vars:
                continue
            handlers.setdefault(owner, []).append(stmt.name.value)
            break
    return handlers


def _find_cog_subclass_names(module: cst.Module, imports: dict[str, str]) -> list[str]:
    """Top-level class names inheriting from a recognized Cog base."""
    out: list[str] = []
    for stmt in module.body:
        if not isinstance(stmt, cst.ClassDef):
            continue
        if _is_cog_subclass(stmt, imports):
            out.append(stmt.name.value)
    return out


def _is_cog_subclass(class_def: cst.ClassDef, imports: dict[str, str]) -> bool:
    for arg in class_def.bases:
        if arg.keyword is not None:
            continue
        if _matched_base(arg.value, imports):
            return True
    return False


def _matched_base(expr: cst.BaseExpression, imports: dict[str, str]) -> bool:
    """Return True if ``expr`` resolves to a Cog base class."""
    # ``class Foo(commands.Cog[Generic[T]])`` is unusual but safe to unwrap.
    if isinstance(expr, cst.Subscript):
        expr = expr.value
    if isinstance(expr, cst.Name):
        return imports.get(expr.value) in _COG_BASES
    if isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
        sentinel = imports.get(expr.value.value)
        return sentinel == _COMMANDS_MODULE and expr.attr.value in _COG_BASES
    return False


def _find_load_extension_targets(module: cst.Module) -> list[str]:
    """Return ``[fqname, ...]`` for every ``X.load_extension("...")`` or
    ``X.load_extensions([...])`` call in ``module``.

    Walks the entire CST (these calls live in module bodies, async
    ``setup_hook`` overrides, ``main()`` functions, etc.). Non-literal
    arguments are dropped silently -- the analyzer can't resolve a
    dynamic ``f"cogs.{name}"`` and emitting an entrypoint for the
    unresolvable string would just produce noise.
    """
    finder = _LoadExtensionFinder()
    module.visit(finder)
    return finder.captured


class _LoadExtensionFinder(cst.CSTVisitor):
    """Collect string-literal ``load_extension`` / ``load_extensions`` args.

    Match shape: ``<expr>.load_extension("dotted.path")`` or
    ``<expr>.load_extensions(["a", "b"])``. The ``<expr>`` is not
    constrained to be a bot var -- in well-formed discord.py code
    nothing else uses these method names, and constraining to known
    bot vars would miss calls inside async ``main()`` functions where
    the bot is a local variable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[str] = []

    def visit_Call(self, node: cst.Call) -> bool | None:
        if not isinstance(node.func, cst.Attribute):
            return None
        attr = node.func.attr.value
        if attr not in _LOAD_EXTENSION_METHODS:
            return None
        if not node.args:
            return None
        first = node.args[0]
        if first.keyword is not None:
            return None
        value = first.value
        if attr == "load_extension":
            captured = string_value(value)
            if captured:
                self.captured.append(captured)
            return None
        # load_extensions: iterable of string literals.
        if isinstance(value, (cst.List, cst.Tuple)):
            for elt in value.elements:
                if not isinstance(elt, cst.Element):
                    continue
                captured = string_value(elt.value)
                if captured:
                    self.captured.append(captured)
        return None
