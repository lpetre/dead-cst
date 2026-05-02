"""Project-specific plugin prototypes for the flux0 e2e tests.

These plugins close flux0's two ``importlib``-driven blind spots
using only the public ``dead_cst`` API. They live in the test tree
so we can iterate on the shape; anything here that ends up shared
across multiple targets is a candidate for promotion into
``dead_cst._plugins`` proper.

Design note: the plugins do not re-run any ``importlib`` /
``pkgutil`` / ``importlib.resources`` code. Everything they need is
already in the symbol graph -- dead-cst's visitor pass parsed every
``*.py`` under the analyzed base, so the modules and decls the
runtime loader would import are already present as graph nodes. The
plugins' job is to pattern-match those nodes by fqname prefix and
mark the right ones alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dead_cst import AddEdge, AddNode, GraphOp, PluginContext
from dead_cst._plugins import synthetic_node
from dead_cst._symbols import SymbolNode


@dataclass
class SubpackageDiscoveryPlugin:
    """Mark every top-level decl under a package as alive.

    Encodes the ``pkgutil.iter_modules + importlib.import_module``
    auto-discovery idiom (Click sub-command loaders, Flask blueprint
    discovery, pytest plugin loading, Django app auto-discovery, ...)
    *without* re-running the discovery code. dead-cst's visitor has
    already parsed every ``*.py`` under the analyzed base, so the
    modules the runtime loader would import are already in the
    symbol graph; the plugin just pattern-matches them by fqname.

    Importing a module runs every one of its top-level statements,
    so the right semantics is "every node whose fqname starts with
    ``<package>.`` is reachable" -- decls, imports, top-level
    assignments. This is the most faithful mirror of
    ``importlib.import_module``.

    Subclass and set ``package`` to use it. Project subclasses are
    typically three lines: docstring, ``package = "..."``, and a
    fresh ``name`` so it can be addressed individually if you compose
    it with siblings.
    """

    package: str = ""
    name: str = "subpackage_discovery"
    version: str = "1"

    def observe(self, ctx) -> None:
        # Finalize-only plugin -- nothing to contribute per file.
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        if not self.package:
            return
        prefix = self.package + "."
        targets: list[SymbolNode] = [
            n for n in ctx.base_nodes() if n.type != "module" and n.fqname.startswith(prefix)
        ]
        if not targets:
            return

        # Inlined ``mark_entrypoints`` -- private in ``_plugins._core``
        # today, so user plugins spell the four lines by hand. Promoting
        # it would shave them off every finalize-only plugin.
        synth = synthetic_node(f"<discovered>:{self.package}", ctx.base)
        yield AddNode(synth, entrypoint=True)
        for target in targets:
            yield AddEdge(synth, target)


@dataclass
class Flux0CliCommandsPlugin(SubpackageDiscoveryPlugin):
    """Close the ``flux0_cli/main.py:register_commands`` blind spot.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/cli/src/flux0_cli/main.py#L61

    The runtime loop iterates every ``.py`` under ``flux0_cli.cmds``
    via ``pkgutil`` and ``importlib``-loads each one. The whole shape
    is delegated to :class:`SubpackageDiscoveryPlugin`; we just spell
    the package name.
    """

    package: str = "flux0_cli.cmds"
    name: str = "flux0_cli_cmds"


@dataclass
class FqnRegistryPlugin:
    """Mark a fixed list of fqnames -- and everything inside them -- alive.

    Encodes the "string-FQN registry" idiom: a module-level constant
    list of dotted names that some other component feeds to
    ``importlib.import_module``. flux0_server uses this for its
    ``INTERNAL_MODULES`` list; similar shapes show up in plugin
    registries, dependency-injection wiring, and feature-flag
    switchboards. The plugin doesn't read the registry at runtime --
    you list the fqnames directly when constructing it.

    Each fqname can name a module (everything under it goes alive,
    matching ``importlib.import_module`` semantics) or a single decl.
    """

    fqnames: tuple[str, ...] = ()
    name: str = "fqn_registry"
    version: str = "1"

    def observe(self, ctx) -> None:
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        targets: list[SymbolNode] = []
        for fqname in self.fqnames:
            mod_node = ctx.find_module(fqname)
            if mod_node is not None:
                # Whole module: every top-level node under it.
                prefix = fqname + "."
                targets.extend(
                    n for n in ctx.base_nodes() if n.fqname == fqname or n.fqname.startswith(prefix)
                )
                continue
            # Single decl: ``find_declarations`` handles ``pkg.mod.func``.
            targets.extend(ctx.find_declarations(fqname))

        if not targets:
            return
        synth = synthetic_node(f"<{self.name}>", ctx.base)
        yield AddNode(synth, entrypoint=True)
        for target in targets:
            yield AddEdge(synth, target)


@dataclass
class Flux0InternalModulesPlugin(FqnRegistryPlugin):
    """Close the ``flux0_server.main.INTERNAL_MODULES`` blind spot.

    The list literal is at the top of ``flux0_server/main.py``; we
    re-state it here so the plugin doesn't have to evaluate any
    runtime code. If the upstream list grows, this tuple grows too.
    """

    fqnames: tuple[str, ...] = ("flux0_server.replay_agent",)
    name: str = "flux0_internal_modules"
