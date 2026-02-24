"""Exponential backoff with jitter for retrying transient failures."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (),
) -> T:
    """Retry a coroutine factory with exponential backoff and jitter.

    Calls ``coro_factory()`` to obtain a fresh coroutine on each attempt.
    Retries only on exceptions matching ``retryable_exceptions``. Non-retryable
    exceptions propagate immediately.

    Args:
        coro_factory: Callable returning a new awaitable on each invocation.
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Base delay in seconds before exponential scaling.
        max_delay: Maximum delay cap in seconds.
        jitter: Upper bound for uniform random jitter added to delay.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The result of a successful ``coro_factory()`` call.

    Raises:
        Exception: The last retryable exception after exhausting retries,
            or any non-retryable exception immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable_exceptions:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            delay += random.uniform(0, jitter)
            await asyncio.sleep(delay)
    msg = "Unreachable: loop must execute at least once"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover
