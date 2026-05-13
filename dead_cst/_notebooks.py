"""Jupyter ``.ipynb`` ingestion: turn a notebook into parseable Python source.

Code cells are concatenated in document order; IPython magics, shell
escapes, and trailing-help forms are rewritten to ``pass  # <orig>`` so
``libcst.parse_module`` accepts the result.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from libcst.helpers.module import ModuleNameAndPackage

logger = logging.getLogger(__name__)

# ``!`` is only a shell-escape when not followed by ``=`` (so ``!= 0`` stays
# Python). ``?`` prefix is only help-syntax when followed by an identifier.
_LINE_MAGIC_RE = re.compile(r"^\s*(%[%A-Za-z_]|!(?!=)|\?[A-Za-z_])")
_CELL_MAGIC_RE = re.compile(r"^\s*%%[A-Za-z_]")
_HELP_SUFFIX_RE = re.compile(r"^\s*[A-Za-z_]\w*(\.[A-Za-z_]\w*)*\?{1,2}\s*$")
# Cheap pre-check: if a line's stripped form starts with none of these and
# doesn't end in ``?``, the regex pass can't fire. Avoids three regex hits
# per pure-Python line in a magics-free notebook.
_MAGIC_TRIGGERS = ("%", "!", "?")


def is_notebook(path: Path) -> bool:
    return path.suffix == ".ipynb"


def notebook_to_module(path: Path) -> str | None:
    """Concatenate ``path``'s code cells into one parseable Python source string.

    Returns ``None`` if the file is not valid notebook JSON, has no
    ``cells`` array, or contains no usable code cells.
    """
    try:
        with path.open() as f:
            nb = json.load(f)
    except OSError as exc:
        logger.warning("Skipping notebook %s: could not read file: %s", path, exc)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Skipping notebook %s: invalid JSON: %s", path, exc)
        return None
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        return None

    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        text = _cell_source_to_text(cell.get("source"))
        if text is None:
            continue
        scrubbed = _strip_ipython_magics(text)
        # Without a trailing newline a cell's last expression glues onto the
        # next cell's first line.
        if not scrubbed.endswith("\n"):
            scrubbed += "\n"
        parts.append(scrubbed)

    if not parts:
        return None
    return "".join(parts)


def _cell_source_to_text(source: object) -> str | None:
    """nbformat allows either a single string or a list of strings per cell."""
    if isinstance(source, list):
        parts: list[str] = []
        for s in source:
            if not isinstance(s, str):
                return None
            parts.append(s)
        return "".join(parts)
    if isinstance(source, str):
        return source
    return None


def _strip_ipython_magics(src: str) -> str:
    """A ``%%cell`` magic swallows the rest of its cell. Line count is
    preserved so libcst positions still map back to the original cell.
    """
    out: list[str] = []
    in_cell_magic = False
    for line in src.splitlines(keepends=True):
        if in_cell_magic:
            out.append(_neutralize(line))
            continue
        stripped = line.lstrip()
        if not stripped or not (
            stripped.startswith(_MAGIC_TRIGGERS) or stripped.rstrip().endswith("?")
        ):
            out.append(line)
            continue
        if _CELL_MAGIC_RE.match(line):
            in_cell_magic = True
            out.append(_neutralize(line))
        elif _LINE_MAGIC_RE.match(line) or _HELP_SUFFIX_RE.match(line):
            out.append(_neutralize(line))
        else:
            out.append(line)
    return "".join(out)


def _neutralize(line: str) -> str:
    body = line.rstrip("\r\n")
    return f"pass  # {body}\n"


_FQN_SANITIZE = re.compile(r"[^0-9A-Za-z_]")


def notebook_fqn_entry(path: Path) -> ModuleNameAndPackage:
    """Synthesize a ``ModuleNameAndPackage`` from a notebook path's stem.

    Notebooks never enter the cross-module lookup trie, so name
    collisions with a real module are harmless.
    """
    stem = path.stem or "notebook"
    sanitized = _FQN_SANITIZE.sub("_", stem)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return ModuleNameAndPackage(name=sanitized, package="")
