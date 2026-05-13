"""Jupyter ``.ipynb`` ingestion: turn a notebook into parseable Python source.

A notebook is opened, its ``code`` cells concatenated in document order, and
IPython magics / shell escapes are line-rewritten to a no-op statement so
``libcst.parse_module`` accepts the result. Non-code cells (markdown, raw) and
malformed inputs cause :func:`notebook_to_module` to return ``None``; the
caller falls through to the same ``[unparseable]`` placeholder used for
``.py`` files that fail to parse.

Notebooks are not importable modules, so the synthetic FQN is derived from
the path's stem (sanitized to a Python identifier); nothing imports a
notebook -- every node from a notebook is flagged ``NOTEBOOK | ENTRYPOINT``
so reachability seeds the whole file, and notebooks are deliberately kept
out of the cross-module lookup trie via ``add_to_trie=False``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from libcst.helpers.module import ModuleNameAndPackage

logger = logging.getLogger(__name__)


# Line-magic forms IPython recognizes; ``%%`` cell magics are matched
# separately because they swallow the rest of the cell. ``!`` is only a
# shell-escape when not followed by ``=`` (so ``!= 0`` stays Python).
# ``?`` prefix is only help-syntax when followed by an identifier.
_LINE_MAGIC_RE = re.compile(r"^\s*(%[%A-Za-z_]|!(?!=)|\?[A-Za-z_])")
_CELL_MAGIC_RE = re.compile(r"^\s*%%[A-Za-z_]")
# A line that is exactly ``obj?`` or ``obj??`` (possibly dotted) is IPython
# trailing-help, not Python.
_HELP_SUFFIX_RE = re.compile(r"^\s*[A-Za-z_]\w*(\.[A-Za-z_]\w*)*\?{1,2}\s*$")


@dataclass(frozen=True, slots=True)
class NotebookSource:
    """Concatenated code-cell source plus enough metadata to map back.

    ``text`` is the joined, magic-neutralized source ready for
    ``libcst.parse_module``. ``cell_line_starts`` records the 1-indexed line
    in ``text`` where each preserved code cell begins, in document order;
    ``cell_indices`` is the parallel list of original cell indices from
    ``nb["cells"]`` so callers can map a libcst position back to
    ``(cell_index, line_in_cell)``.
    """

    text: str
    cell_line_starts: tuple[int, ...]
    cell_indices: tuple[int, ...]

    def locate(self, line: int) -> tuple[int, int] | None:
        """Return ``(cell_index, line_in_cell)`` for a 1-indexed ``line`` in ``text``.

        ``line_in_cell`` is also 1-indexed. Returns ``None`` if ``line`` is out
        of range or falls before the first cell.
        """
        if line < 1 or not self.cell_line_starts:
            return None
        # Find the last cell whose start is <= line.
        idx = -1
        for i, start in enumerate(self.cell_line_starts):
            if start <= line:
                idx = i
            else:
                break
        if idx < 0:
            return None
        return self.cell_indices[idx], line - self.cell_line_starts[idx] + 1


def is_notebook(path: Path) -> bool:
    """Cheap suffix check used by ingestion and codemod gates."""
    return path.suffix == ".ipynb"


def notebook_to_module(path: Path) -> NotebookSource | None:
    """Load ``path`` and return its code cells joined into parseable Python.

    Returns ``None`` if the file is not valid notebook JSON, has no
    ``cells`` array, or contains no code cells. Magics and shell escapes
    are replaced with a same-line ``pass`` so the line-to-cell map stays
    byte-faithful.
    """
    try:
        raw = path.read_text()
    except OSError as exc:
        logger.warning("Skipping notebook %s: could not read file: %s", path, exc)
        return None
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Skipping notebook %s: invalid JSON: %s", path, exc)
        return None
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        return None

    parts: list[str] = []
    cell_line_starts: list[int] = []
    cell_indices: list[int] = []
    current_line = 1
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        text = _cell_source_to_text(source)
        if text is None:
            continue
        scrubbed = _strip_ipython_magics(text)
        # Ensure each cell ends with a newline so the next cell's
        # statements start fresh -- otherwise a trailing-expression cell
        # would glue onto the next cell's first line.
        if not scrubbed.endswith("\n"):
            scrubbed += "\n"
        cell_indices.append(idx)
        cell_line_starts.append(current_line)
        parts.append(scrubbed)
        current_line += scrubbed.count("\n")

    if not parts:
        return None
    return NotebookSource(
        text="".join(parts),
        cell_line_starts=tuple(cell_line_starts),
        cell_indices=tuple(cell_indices),
    )


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
    """Replace lines starting with an IPython magic / shell escape with ``pass``.

    A ``%%cell`` magic swallows the remainder of the cell, so we replace
    every subsequent line with ``pass`` too. The output preserves line
    count so libcst positions still map back to the original notebook.
    """
    out: list[str] = []
    in_cell_magic = False
    for line in src.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        terminator = line[len(stripped) :]
        if in_cell_magic:
            out.append(f"pass  # {stripped}" + (terminator or "\n"))
            continue
        if _CELL_MAGIC_RE.match(stripped):
            in_cell_magic = True
            out.append(f"pass  # {stripped}" + (terminator or "\n"))
            continue
        if _LINE_MAGIC_RE.match(stripped) or _HELP_SUFFIX_RE.match(stripped):
            out.append(f"pass  # {stripped}" + (terminator or "\n"))
            continue
        out.append(line)
    return "".join(out)


_FQN_SANITIZE = re.compile(r"[^0-9A-Za-z_]")


def notebook_fqn_entry(path: Path) -> ModuleNameAndPackage:
    """Build a synthetic FQN entry for a notebook path.

    The notebook's stem is sanitized to a valid Python identifier; if the
    stem starts with a digit it's prefixed with ``_``. ``package`` is the
    empty string because notebooks aren't packaged. Notebooks never enter
    the cross-module lookup trie, so collisions with a real module of the
    same name are harmless -- the trie skip is enforced upstream.
    """
    stem = path.stem or "notebook"
    sanitized = _FQN_SANITIZE.sub("_", stem)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return ModuleNameAndPackage(name=sanitized, package="")
