"""Exponential backoff with jitter for retrying transient failures."""

from __future__ import annotations

import asyncio
import math
import random
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

T = TypeVar("T")

_RETRYABLE_STATUSES = frozenset({408, 409, 429})


def is_retryable_status(status: object) -> bool:
    """Return whether an HTTP status code marks a transient failure.

    408 (timeout), 409 (conflict), 429 (rate limit) and every 5xx status
    are retryable. Other 4xx statuses (bad request, auth, not found) are
    not, because repeating the request cannot fix them.

    Args:
        status: The status code, or any other value (never retryable).

    Returns:
        ``True`` if the request should be retried.
    """
    return isinstance(status, int) and (status in _RETRYABLE_STATUSES or status >= 500)


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Read the server-requested retry delay from HTTP response headers.

    ``retry-after-ms`` (milliseconds) takes precedence over
    ``retry-after`` (seconds). The HTTP-date form of ``retry-after`` is
    not supported and yields ``None``.

    Args:
        headers: Response headers, or ``None``.

    Returns:
        The delay in seconds, or ``None`` if no usable header is present.
    """
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) / scale
        except ValueError:
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (),
    is_retryable: Callable[[Exception], bool] | None = None,
    retry_after: Callable[[Exception], float | None] | None = None,
) -> T:
    """Retry a coroutine factory with exponential backoff and jitter.

    Calls ``coro_factory()`` to obtain a fresh coroutine on each attempt.
    Retries only on exceptions matching ``retryable_exceptions`` or
    accepted by ``is_retryable``. Non-retryable exceptions propagate
    immediately.

    Args:
        coro_factory: Callable returning a new awaitable on each invocation.
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Base delay in seconds before exponential scaling.
        max_delay: Maximum delay cap in seconds.
        jitter: Upper bound for uniform random jitter added to delay.
        retryable_exceptions: Exception types that trigger a retry.
        is_retryable: Predicate that returns ``True`` for exceptions that
            trigger a retry, checked in addition to
            ``retryable_exceptions``.
        retry_after: Returns the server-requested delay in seconds for an
            exception, or ``None``. When it returns a value, that value
            (capped at ``max_delay``) replaces the backoff delay.

    Returns:
        The result of a successful ``coro_factory()`` call.

    Raises:
        Exception: The last retryable exception after exhausting retries,
            or any non-retryable exception immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            retryable = isinstance(exc, retryable_exceptions) or (
                is_retryable is not None and is_retryable(exc)
            )
            if not retryable or attempt == max_retries:
                raise
            requested = retry_after(exc) if retry_after is not None else None
            if requested is not None:
                delay = min(requested, max_delay)
            else:
                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, jitter)
            await asyncio.sleep(delay)
    msg = "Unreachable: loop must execute at least once"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover
