"""Tests for llm_consistency.providers._rate_limit module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_consistency.providers._rate_limit import AsyncTokenBucket

_RATE_LIMIT_MOD = "llm_consistency.providers._rate_limit"


class TestAsyncTokenBucketInit:
    """Tests for AsyncTokenBucket construction."""

    def test_stores_rate_and_capacity(self) -> None:
        bucket = AsyncTokenBucket(rate=10.0, capacity=100)
        assert bucket._rate == 10.0
        assert bucket._capacity == 100

    def test_initial_tokens_equal_capacity(self) -> None:
        bucket = AsyncTokenBucket(rate=5.0, capacity=50)
        assert bucket._tokens == 50.0

    def test_rate_from_requests_per_minute(self) -> None:
        rpm = 120
        rate = rpm / 60.0
        bucket = AsyncTokenBucket(rate=rate, capacity=rpm)
        assert bucket._rate == pytest.approx(2.0)
        assert bucket._capacity == 120


class TestAsyncTokenBucketAcquire:
    """Tests for AsyncTokenBucket.acquire()."""

    async def test_acquire_succeeds_when_tokens_available(self) -> None:
        bucket = AsyncTokenBucket(rate=10.0, capacity=10)
        # Should not block or raise
        await bucket.acquire()

    async def test_acquire_blocks_when_no_tokens_then_succeeds(
        self,
    ) -> None:
        bucket = AsyncTokenBucket(rate=100.0, capacity=1)
        # Drain the single token
        await bucket.acquire()

        # Second acquire should sleep (mock asyncio.sleep to verify)
        with patch(
            f"{_RATE_LIMIT_MOD}.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            await bucket.acquire()
            mock_sleep.assert_called_once()
            # The sleep time should be positive
            sleep_time = mock_sleep.call_args[0][0]
            assert sleep_time > 0

    async def test_tokens_refill_over_time_up_to_capacity(self) -> None:
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)
        # Drain all tokens
        for _ in range(5):
            await bucket.acquire()

        # Simulate time passing (enough for full refill)
        with patch(
            f"{_RATE_LIMIT_MOD}.time.monotonic",
        ) as mock_time:
            # Advance 1s (refill 10 tokens, capped at 5) minus 1 = 4
            mock_time.return_value = bucket._last_refill + 1.0
            await bucket.acquire()
            assert bucket._tokens == pytest.approx(4.0)

    async def test_tokens_never_exceed_capacity(self) -> None:
        bucket = AsyncTokenBucket(rate=100.0, capacity=5)
        # Simulate a large time gap
        with patch(
            f"{_RATE_LIMIT_MOD}.time.monotonic",
        ) as mock_time:
            mock_time.return_value = bucket._last_refill + 1000.0
            await bucket.acquire()
            # Capped at capacity (5) minus 1 consumed = 4
            assert bucket._tokens <= 5.0

    async def test_multiple_sequential_acquires_drain_bucket(
        self,
    ) -> None:
        bucket = AsyncTokenBucket(rate=0.001, capacity=3)
        # Freeze time so no refill happens
        frozen_time = bucket._last_refill
        with patch(
            f"{_RATE_LIMIT_MOD}.time.monotonic",
            return_value=frozen_time,
        ):
            await bucket.acquire()  # 3 -> 2
            assert bucket._tokens == pytest.approx(2.0)
            await bucket.acquire()  # 2 -> 1
            assert bucket._tokens == pytest.approx(1.0)
            await bucket.acquire()  # 1 -> 0
            assert bucket._tokens == pytest.approx(0.0)
