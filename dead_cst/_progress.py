"""TTY-aware progress reporting; emits decile log lines off-TTY."""

from __future__ import annotations

import sys
from typing import Iterable, Iterator, TypeVar

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
    if total == 0:
        return

    if sys.stderr.isatty():
        # Lazy import: tqdm is only needed for the live-bar path,
        # and skipping it on non-TTY keeps CLI startup snappy.
        from tqdm import tqdm

        yield from tqdm(iterable, total=total, desc=desc, unit=unit)
        return

    milestones = iter(sorted({total * k // 10 for k in range(11)}))
    next_threshold: int | None = next(milestones, None)

    while next_threshold is not None and next_threshold <= 0:
        print(f"{desc}: {next_threshold}/{total} {unit}", file=sys.stderr, flush=True)
        next_threshold = next(milestones, None)

    count = 0
    for item in iterable:
        yield item
        count += 1
        while next_threshold is not None and count >= next_threshold:
            print(f"{desc}: {next_threshold}/{total} {unit}", file=sys.stderr, flush=True)
            next_threshold = next(milestones, None)
