"""Tests for :class:`SymbolTrie.merge` and :meth:`merge_exported`.

The trie's two merge methods are how :meth:`Analysis._build_symbol_lookup`
composes a per-package lookup view: ``merge`` for the package's own
trie (sees every decl), ``merge_exported`` for each dep's trie
(filters to nodes flagged :data:`NodeFlags.EXPORTED`).
"""

from __future__ import annotations

from pathlib import Path

from libcst.metadata import CodePosition, CodeRange

from dead_cst.graph import NodeFlags, SymbolNode, SymbolTrie

POS = CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0))


def _module(fqname: str, *, exported: bool = False) -> SymbolNode:
    flags = NodeFlags.EXPORTED if exported else NodeFlags.NONE
    return SymbolNode(fqname, "module", Path(f"/tmp/{fqname}.py"), POS, flags=flags)


def _decl(fqname: str, *, exported: bool = False) -> SymbolNode:
    flags = NodeFlags.EXPORTED if exported else NodeFlags.NONE
    return SymbolNode(fqname, "function", Path(f"/tmp/{fqname.split('.')[0]}.py"), POS, flags=flags)


def test_merge_exported_drops_non_exported_decls():
    """``merge_exported`` filters decls by ``NodeFlags.EXPORTED``."""
    src = SymbolTrie()
    src.add_declaration(_module("pkg", exported=True))
    src.add_declaration(_decl("pkg.public", exported=True))
    src.add_declaration(_decl("pkg.private", exported=False))

    dst = SymbolTrie()
    dst.merge_exported(src)

    pkg_node = dst.children["pkg"]
    assert pkg_node.module is not None
    assert "public" in pkg_node.declarations
    assert "private" not in pkg_node.declarations


def test_merge_exported_drops_non_exported_module():
    """An unexported module is not visible to consumers."""
    src = SymbolTrie()
    src.add_declaration(_module("pkg", exported=False))
    src.add_declaration(_decl("pkg.foo", exported=False))

    dst = SymbolTrie()
    dst.merge_exported(src)

    # Non-exported module is dropped, but the trie node still exists
    # because the recursion walks every child unconditionally.
    pkg_node = dst.children.get("pkg")
    if pkg_node is not None:
        assert pkg_node.module is None


def test_merge_exported_walks_through_unexported_intermediate():
    """An unexported intermediate module doesn't block exported descendants.

    ``pkg/__init__.py`` may not be exported while ``pkg/api/__init__.py``
    is -- the consumer needs ``pkg`` -> ``api`` to traverse so the
    descendant resolves.
    """
    src = SymbolTrie()
    src.add_declaration(_module("pkg", exported=False))
    src.add_declaration(_module("pkg.api", exported=True))
    src.add_declaration(_decl("pkg.api.foo", exported=True))

    dst = SymbolTrie()
    dst.merge_exported(src)

    assert dst.children["pkg"].module is None
    api_node = dst.children["pkg"].children["api"]
    assert api_node.module is not None
    assert api_node.module.fqname == "pkg.api"
    assert "foo" in api_node.declarations


def test_merge_exported_collision_keeps_self():
    """When both sides have an exported module at the same FQN, self wins."""
    self_mod = _module("pkg", exported=True)
    other_mod = _module("pkg", exported=True)

    src = SymbolTrie()
    src.add_declaration(other_mod)

    dst = SymbolTrie()
    dst.add_declaration(self_mod)
    dst.merge_exported(src)

    assert dst.children["pkg"].module is self_mod


def test_merge_exported_skips_non_exported_decls_only():
    """Mixed exported / non-exported decls under the same name keep only exported."""
    src = SymbolTrie()
    src.add_declaration(_module("pkg", exported=True))
    src.add_declaration(_decl("pkg.foo", exported=True))
    src.add_declaration(_decl("pkg.foo", exported=False))

    dst = SymbolTrie()
    dst.merge_exported(src)

    bucket = dst.children["pkg"].declarations["foo"]
    assert all(d.flags & NodeFlags.EXPORTED for d in bucket)
