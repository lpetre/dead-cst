"""Progress reporting that survives non-TTY consumers.

``tqdm`` is great for humans on a terminal but its ``\\r``-overwriting
output renders to mush when stderr is captured (CI logs, pytest, agent
harnesses). :func:`progress` keeps the live bar on TTYs and falls back
to newline-terminated decile checkpoints otherwise -- one line at 0%,
each ~10% boundary, and 100% -- so log consumers can still track a
long-running pass.
"""

from __future__ import annotations

import sys
from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    total: int,
    desc: str,
    unit: str,
) -> Iterator[T]:
    """Yield from ``iterable`` while reporting progress to stderr.

    On a TTY: hands off to :func:`tqdm.tqdm`. Off a TTY: emits one
    ``"<desc>: i/total <unit>"`` line at 0%, every decile boundary,
    and ``total`` itself. Counts that fall on the same milestone (small
    ``total``s) coalesce so the same line is never printed twice.
    """
    if sys.stderr.isatty():
        yield from tqdm(iterable, total=total, desc=desc, unit=unit, disable=False)
        return

    milestones = sorted({total * k // 10 for k in range(11)})
    next_idx = 0

    def _emit_through(count: int) -> None:
        nonlocal next_idx
        while next_idx < len(milestones) and milestones[next_idx] <= count:
            print(
                f"{desc}: {milestones[next_idx]}/{total} {unit}",
                file=sys.stderr,
                flush=True,
            )
            next_idx += 1

    _emit_through(0)
    count = 0
    for item in iterable:
        yield item
        count += 1
        _emit_through(count)
