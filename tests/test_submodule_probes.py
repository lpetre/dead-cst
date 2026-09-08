"""Speculative submodule probes on package aliases.

Every ``alias.attr`` chain rooted at an imported *package* makes the
reference walk ask "is ``pkg.attr`` a submodule?" before it falls back
to a global-scope decl. Those probes are answered from the resolved
parent package's own directory listing when that is exact (a regular
``__init__.py`` package keeps its submodules beside it) instead of
making ty's resolver scan every search path per distinct name; the
resolver is still consulted for namespace-style parents whose
submodules can live on other search paths. These tests pin both halves.
"""

from __future__ import annotations


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_attribute_on_package_alias_lands_on_decl(tmp_path, make_analysis, assert_edges):
    """``sub.f()`` where ``sub`` is a *package*: the ``A.sub.f`` probe misses
    (no such submodule) and the use lands on the decl in ``A/sub/__init__.py``."""
    _write(tmp_path / "pkg_a" / "A" / "__init__.py")
    _write(tmp_path / "pkg_a" / "A" / "sub" / "__init__.py", "def f(): ...\ndef g(): ...\n")
    _write(tmp_path / "pkg_b" / "B" / "__init__.py", "from A import sub\nsub.f()\n")

    graph = make_analysis(["pkg_b:pkg_a", "pkg_a"]).materialize_all()
    assert_edges(
        graph,
        {
            "A.sub -> A",
            "A.sub.f -> A.sub",
            "A.sub.g -> A.sub",
            "B -> A.sub",
            "B -> A.sub.f",
            "B -> B.sub",
            "B.sub -> A.sub",
            "B.sub -> B",
        },
    )


def test_attribute_on_package_alias_lands_on_submodule(tmp_path, make_analysis, assert_edges):
    """``sub.nested.h()`` where ``nested`` *is* a submodule of the package:
    the probe hits (``nested.py`` sits beside ``sub/__init__.py``) and the
    chain walks into it."""
    _write(tmp_path / "pkg_a" / "A" / "__init__.py")
    _write(tmp_path / "pkg_a" / "A" / "sub" / "__init__.py")
    _write(tmp_path / "pkg_a" / "A" / "sub" / "nested.py", "def h(): ...\n")
    _write(tmp_path / "pkg_b" / "B" / "__init__.py", "from A import sub\nsub.nested.h()\n")

    graph = make_analysis(["pkg_b:pkg_a", "pkg_a"]).materialize_all()
    assert_edges(
        graph,
        {
            "A.sub -> A",
            "A.sub.nested -> A.sub",
            "A.sub.nested.h -> A.sub.nested",
            "B -> A.sub.nested",
            "B -> A.sub.nested.h",
            "B -> B.sub",
            "B.sub -> A.sub",
            "B.sub -> B",
        },
    )


def test_probe_on_legacy_namespace_package_spans_members(tmp_path, make_analysis, assert_edges):
    """A ``pkgutil.extend_path`` package has an ``__init__.py`` but its
    submodules may live on *other* search paths: ``NS/b.py`` ships in a
    second member with no ``__init__``. The ``NS.b`` probe must still
    reach the resolver (which merges the two directories) rather than be
    answered from ``NS/__init__.py``'s own directory."""
    _write(
        tmp_path / "pkg_a" / "NS" / "__init__.py",
        "__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n",
    )
    _write(tmp_path / "pkg_a" / "NS" / "a.py", "def fa(): ...\n")
    _write(tmp_path / "pkg_b" / "NS" / "b.py", "def fb(): ...\n")
    _write(tmp_path / "pkg_c" / "C" / "__init__.py", "import NS\nNS.a.fa()\nNS.b.fb()\n")

    graph = make_analysis(["pkg_c:pkg_a,pkg_b", "pkg_a", "pkg_b"]).materialize_all()
    assert_edges(
        graph,
        {
            "NS -> NS.__path__",
            "NS.__path__ -> NS",
            "NS.a -> NS",
            "NS.a.fa -> NS.a",
            "NS.b -> NS",
            "NS.b.fb -> NS.b",
            "C -> C.NS",
            "C -> NS.a",
            "C -> NS.a.fa",
            "C -> NS.b",
            "C -> NS.b.fb",
            "C.NS -> C",
            "C.NS -> NS",
        },
    )
