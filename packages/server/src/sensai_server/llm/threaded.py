"""Bridge a blocking iterator into async code, for the sync calls of the provider (model stage, ReAct, later RAG)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator

_DONE = object()


async def iterate_in_thread[T](make_iterator: Callable[[], Iterator[T]]) -> AsyncGenerator[T]:
    """Stream the items of the iterator `make_iterator` opens, every blocking step run in a thread.

    The iterator is opened, advanced and closed on one dedicated worker, so its `close()` queues behind a `next()`
    still running (a generator cannot be closed while it executes) and the event loop never blocks. When this
    generator is closed early, by a cancellation or an `aclose()`, the sync iterator is closed too, which releases
    an HTTP stream. Errors raised by the iterator propagate.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    iterator: Iterator[T] | None = None
    try:
        iterator = opened = await loop.run_in_executor(executor, make_iterator)
        while (item := await loop.run_in_executor(executor, _next_or_done, opened)) is not _DONE:
            yield cast("T", item)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            executor.submit(close)
        executor.shutdown(wait=False)


def _next_or_done[T](iterator: Iterator[T]) -> T | object:
    """Return the next item of `iterator`, or `_DONE` once it is exhausted.

    The sentinel keeps `StopIteration` from crossing the future, which cannot carry it.
    """
    return next(iterator, _DONE)
