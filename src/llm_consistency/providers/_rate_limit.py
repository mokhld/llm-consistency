"""Async token-bucket rate limiter for LLM provider requests."""

from __future__ import annotations

import asyncio
import time


class AsyncTokenBucket:
    """Token bucket rate limiter for asyncio.

    Limits request rate over time using a token bucket algorithm.
    Each ``acquire()`` call consumes one token; tokens refill at the
    configured rate up to the bucket capacity.

    Args:
        rate: Tokens added per second (e.g., ``requests_per_minute / 60``).
        capacity: Maximum tokens the bucket can hold (burst size).
    """

    def __init__(self, rate: float, capacity: int) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate,
            )
            self._last_refill = now
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0
