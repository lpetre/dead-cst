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
    collect_module_imports,
    decls_by_simple_name,
    decorator_owner,
    dotted_parts,
    find_call_assignments,
    make_payload,
    matched_attr_call,
    payload_imports_module,
    string_value,
    synthetic_node,
)

if TYPE_CHECKING:
    import dead_cst_ty_native as native

    from ..graph import VisitorPayload


# Classes from ``discord.ext.commands`` that produce a bot instance.
_COMMANDS_BOT_KINDS: frozenset[str] = frozenset({"Bot", "AutoShardedBot"})

# Classes from ``discord`` that produce a client instance.
_DISCORD_CLIENT_KINDS: frozenset[str] = frozenset({"Client", "AutoShardedClient"})

# Cog base classes (from ``discord.ext.commands``). ``GroupCog`` is the
# slash-command analogue introduced in 2.0.
_COG_BASES: frozenset[str] = frozenset({"Cog", "GroupCog"})

# Combined whitelist of class names the plugin recognizes when imported
# from ``discord.ext.commands``.
_COMMANDS_DIRECT_TARGETS: frozenset[str] = _COMMANDS_BOT_KINDS | _COG_BASES

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
    """Wire discord.py bots, Cogs, and extension hooks into reachability.

    Two phases:

    * :meth:`observe` (per-file) classifies top-level ``commands.Bot`` /
      ``discord.Client`` constructions, emits ``bot -> handler`` edges
      for decorated handlers (including ``@bot.tree.command()`` slash
      commands), anchors Cog subclasses + their module-level ``setup``
      / ``teardown`` hooks to a per-file entrypoint synthetic, and
      captures string-literal ``load_extension`` targets.
    * :meth:`finalize` (per-package) stitches each captured extension
      target to its module's surface so the dynamically-loaded module
      and its hooks survive.
    """

    name: str = "discordpy"
    version: int = 1778566342

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        # Cheap gate: bail out for files that don't touch discord at all.
        if not payload_imports_module(ctx.payload, "discord"):
            return None

        commands_imports, discord_imports = _collect_discord_imports(ctx.module)
        if not (commands_imports or discord_imports):
            return None

        decls_by_name = decls_by_simple_name(ctx.payload.nodes)
        nodes: list[SymbolNode] = []
        edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []

        # 1. Bot/Client instances ----------------------------------------
        bot_vars: set[str] = set(
            find_call_assignments(ctx.module, commands_imports, _COMMANDS_BOT_KINDS)
        )
        bot_vars |= set(find_call_assignments(ctx.module, discord_imports, _DISCORD_CLIENT_KINDS))
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
        if bot_vars:
            for var_name, handler_names in _find_handlers(ctx.module, bot_vars).items():
                for var_decl in decls_by_name.get(var_name, []):
                    for handler_name in handler_names:
                        for handler_decl in decls_by_name.get(handler_name, []):
                            edges.append((var_decl, handler_decl, SYNTHETIC_POSITION))

        # 3. Cog classes + the module's setup/teardown hooks -------------
        cog_class_decls: list[SymbolNode] = []
        for cog_name in _find_cog_subclass_names(ctx.module, commands_imports):
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
        for synth in ctx.contribution.nodes:
            if synth.type != "synthetic" or not synth.fqname.startswith(DISCORDPY_EXTENSION_PREFIX):
                continue
            captured = synth.fqname[len(DISCORDPY_EXTENSION_PREFIX) :]
            for target in ctx.module_surface(captured):
                yield AddEdge(synth, target)

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        import dead_cst_ty_native as native

        # Per-file gate: only fire on files that import discord. Mirrors
        # the libcst-side ``payload_imports_module("discord")`` check.
        discord_paths: set[str] = set()
        for module in ("discord", "discord.ext", "discord.ext.commands"):
            for imp in ctx.find_imports_of(module):
                discord_paths.add(imp.path)
        if not discord_paths:
            return

        # 1. Bot / Client constructions. find_instance_constructions
        # handles `from X import Y`, `from X import Y as Z`, and
        # `import X.Y; X.Y.Z(...)` through syntactic import resolution.
        bot_constructions: list[tuple[native.NativeNode, str]] = []
        bot_constructions.extend(
            ctx.find_instance_constructions("discord.ext.commands", sorted(_COMMANDS_BOT_KINDS))
        )
        bot_constructions.extend(
            ctx.find_instance_constructions("discord", sorted(_DISCORD_CLIENT_KINDS))
        )

        bot_vars_by_file: dict[str, dict[str, native.NativeNode]] = {}
        for var_node, _kind in bot_constructions:
            if var_node.path not in discord_paths:
                continue
            simple = var_node.fqname.rsplit(".", 1)[-1]
            bot_vars_by_file.setdefault(var_node.path, {})[simple] = var_node
            yield native.AddEntrypoint(var_node, marker="<discordpy-app>")

        # 2. Single-attr decorators @<bot>.<verb>(...).
        for owner_name, handler in ctx.find_handler_decorators(sorted(_BOT_DECORATORS)):
            owner_node = bot_vars_by_file.get(handler.path, {}).get(owner_name)
            if owner_node is not None:
                yield native.AddEdge(owner_node, handler)

        # 3. Two-level slash-command decorators @<bot>.tree.<verb>(...).
        for owner_name, handler in ctx.find_handler_decorators_via(
            "tree", sorted(_TREE_DECORATORS)
        ):
            owner_node = bot_vars_by_file.get(handler.path, {}).get(owner_name)
            if owner_node is not None:
                yield native.AddEdge(owner_node, handler)

        # 4. Cog subclasses + module-level setup / teardown hooks.
        # Use the syntactic ``find_classes_subclassing`` query rather
        # than ty's ``find_subclasses``: discord usually isn't in the
        # analyzer's venv, so ty can't resolve ``discord.ext.commands.Cog``
        # as a seed type. The syntactic walk matches the libcst plugin's
        # behavior (direct ``commands.Cog`` / ``Cog`` references through
        # known imports).
        cogs_by_path: dict[str, list[native.NativeNode]] = {}
        for cog, _base in ctx.find_classes_subclassing("discord.ext.commands", sorted(_COG_BASES)):
            cogs_by_path.setdefault(cog.path, []).append(cog)

        if cogs_by_path:
            hook_funcs_by_path: dict[str, list[native.NativeNode]] = {}
            for n in ctx.nodes():
                if n.kind != "function" or n.path not in cogs_by_path:
                    continue
                if n.fqname.rsplit(".", 1)[-1] in ("setup", "teardown"):
                    hook_funcs_by_path.setdefault(n.path, []).append(n)

            for path, cogs in cogs_by_path.items():
                targets = list(cogs) + hook_funcs_by_path.get(path, [])
                filename = path.rsplit("/", 1)[-1]
                yield native.AddNode(
                    fqname=f"{DISCORDPY_COG_PREFIX}{filename}",
                    path=path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to=targets,
                )

        # 5. load_extension / load_extensions string-literal targets.
        # The receiver doesn't matter (could be bot / client / self.bot
        # / a local in async def main()); the per-file discord-import
        # gate above is what keeps unrelated `loader.load_extension`
        # calls from firing.
        seen_extensions: set[str] = set()
        for attr in ("load_extension", "load_extensions"):
            for owner, fqname in ctx.find_calls_on_attr(attr, 0):
                if owner.path not in discord_paths:
                    continue
                if fqname in seen_extensions:
                    continue
                targets = list(ctx.module_surface(fqname))
                if not targets:
                    continue
                seen_extensions.add(fqname)
                yield native.AddNode(
                    fqname=f"{DISCORDPY_EXTENSION_PREFIX}{fqname}",
                    path=owner.path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to=targets,
                )


# --- helpers ----------------------------------------------------------------


def _collect_discord_imports(
    module: cst.Module,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(commands_imports, discord_imports)`` for discord.py imports.

    Each dict has the shape ``collect_module_imports`` already produces:
    ``{local_name: target}``, with the literal ``"<module>"`` sentinel
    marking a bare module binding (``import discord``,
    ``from discord.ext import commands``). Keeping the two source
    modules in separate dicts lets the call-site / base-class matchers
    validate each name against the right whitelist
    (``_COMMANDS_DIRECT_TARGETS`` for ``commands_imports``,
    ``_DISCORD_CLIENT_KINDS`` for ``discord_imports``) without an
    extra discord-specific sentinel layer.

    The ``commands`` alias produced by ``from discord.ext import commands``
    is spliced into ``commands_imports`` as a ``"<module>"`` entry so
    ``commands.Bot(...)`` resolves through the standard
    :func:`matched_attr_call` attribute-form check.
    """
    commands_imports = collect_module_imports(
        module, "discord.ext.commands", _COMMANDS_DIRECT_TARGETS
    )
    for local in collect_module_imports(module, "discord.ext", {"commands"}):
        commands_imports[local] = "<module>"
    discord_imports = collect_module_imports(module, "discord", _DISCORD_CLIENT_KINDS)
    return commands_imports, discord_imports


def _find_handlers(module: cst.Module, instance_vars: Container[str]) -> dict[str, list[str]]:
    """Map ``bot_var -> [handler_func_name, ...]`` for decorated top-level functions.

    Single-attribute decorators (``@bot.command()``, ``@bot.event``) and
    the two-level slash-command form (``@bot.tree.command()``,
    ``@bot.tree.context_menu()``) are both recognized in one pass over
    ``module.body``.
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


def _tree_decorator_owner(expr: cst.BaseExpression) -> str | None:
    """For ``@X.tree.<attr>(...)`` return ``"X"`` when ``attr`` is a
    recognized tree decorator; ``None`` otherwise.

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


def _find_cog_subclass_names(module: cst.Module, commands_imports: dict[str, str]) -> list[str]:
    """Top-level class names inheriting from a recognized Cog base."""
    out: list[str] = []
    for stmt in module.body:
        if not isinstance(stmt, cst.ClassDef):
            continue
        for arg in stmt.bases:
            if arg.keyword is not None:
                continue
            base_expr = arg.value
            # ``class Foo(commands.Cog[Generic[T]])`` is unusual but safe to unwrap.
            if isinstance(base_expr, cst.Subscript):
                base_expr = base_expr.value
            if matched_attr_call(base_expr, commands_imports, _COG_BASES, unwrap_call=False):
                out.append(stmt.name.value)
                break
    return out


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
        if isinstance(value, (cst.List, cst.Tuple)):
            for elt in value.elements:
                if not isinstance(elt, cst.Element):
                    continue
                captured = string_value(elt.value)
                if captured:
                    self.captured.append(captured)
        return None
