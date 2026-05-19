"""Progress reporting: tqdm bar on a TTY, logger-driven off it.

Verbosity is controlled by the root logger level. On a TTY tqdm
renders the live bar regardless, and ``logging_redirect_tqdm``
re-routes concurrent ``logger.*`` calls so DEBUG / INFO lines from
callers print above the bar without shattering it. Off a TTY tqdm
is suppressed and decile checkpoints surface via ``logger.info`` on
the ``dead_cst._progress`` logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

T = TypeVar("T")

_logger = logging.getLogger(__name__)


def progress(
    iterable: Iterable[T],
    *,
    total: int,
    desc: str,
    unit: str,
) -> Iterator[T]:
    if total == 0:
        return

    if sys.stderr.isatty():
        with logging_redirect_tqdm():
            yield from tqdm(iterable, total=total, desc=desc, unit=unit)
        return

    milestones = iter(sorted({total * k // 10 for k in range(11)}))
    next_threshold: int | None = next(milestones, None)

    while next_threshold is not None and next_threshold <= 0:
        _logger.info("%s: %d/%d %s", desc, next_threshold, total, unit)
        next_threshold = next(milestones, None)

    count = 0
    for item in iterable:
        yield item
        count += 1
        while next_threshold is not None and count >= next_threshold:
            _logger.info("%s: %d/%d %s", desc, next_threshold, total, unit)
            next_threshold = next(milestones, None)
