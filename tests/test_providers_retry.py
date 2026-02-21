"""Tests for llm_consistency.providers._retry module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_consistency.providers._retry import retry_with_backoff

_RETRY_MOD = "llm_consistency.providers._retry"


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    async def test_succeeds_on_first_attempt(self) -> None:
        factory = AsyncMock(return_value="ok")
        result = await retry_with_backoff(
            factory,
            retryable_exceptions=(ValueError,),
        )
        assert result == "ok"
        factory.assert_called_once()

    async def test_retries_on_retryable_exception_succeeds_second(
        self,
    ) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient")
            return "ok"

        with patch(
            f"{_RETRY_MOD}.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await retry_with_backoff(
                factory,
                retryable_exceptions=(ValueError,),
            )
        assert result == "ok"
        assert call_count == 2

    async def test_raises_after_max_retries_exhausted(self) -> None:
        async def factory() -> str:
            raise ValueError("always fails")

        with (
            patch(
                f"{_RETRY_MOD}.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(ValueError, match="always fails"),
        ):
            await retry_with_backoff(
                factory,
                max_retries=2,
                retryable_exceptions=(ValueError,),
            )

    async def test_non_retryable_exception_raised_immediately(
        self,
    ) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            await retry_with_backoff(
                factory,
                retryable_exceptions=(ValueError,),
            )
        assert call_count == 1

    async def test_delay_increases_exponentially(self) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ValueError("fail")
            return "ok"

        with (
            patch(
                f"{_RETRY_MOD}.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
            patch(
                f"{_RETRY_MOD}.random.uniform",
                return_value=0.0,
            ),
        ):
            result = await retry_with_backoff(
                factory,
                max_retries=3,
                base_delay=1.0,
                max_delay=60.0,
                jitter=1.0,
                retryable_exceptions=(ValueError,),
            )
        assert result == "ok"
        # Delays: 1*2^0=1, 1*2^1=2, 1*2^2=4 (jitter=0)
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == pytest.approx([1.0, 2.0, 4.0])

    async def test_delay_capped_at_max_delay(self) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ValueError("fail")
            return "ok"

        with (
            patch(
                f"{_RETRY_MOD}.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
            patch(
                f"{_RETRY_MOD}.random.uniform",
                return_value=0.0,
            ),
        ):
            await retry_with_backoff(
                factory,
                max_retries=3,
                base_delay=10.0,
                max_delay=15.0,
                jitter=1.0,
                retryable_exceptions=(ValueError,),
            )
        # Delays: min(10*1,15)=10, min(10*2,15)=15, min(10*4,15)=15
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == pytest.approx([10.0, 15.0, 15.0])

    async def test_jitter_adds_randomness(self) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("fail")
            return "ok"

        with (
            patch(
                f"{_RETRY_MOD}.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
            patch(
                f"{_RETRY_MOD}.random.uniform",
                return_value=0.5,
            ) as mock_rand,
        ):
            await retry_with_backoff(
                factory,
                max_retries=1,
                base_delay=1.0,
                jitter=1.0,
                retryable_exceptions=(ValueError,),
            )
        mock_rand.assert_called_once_with(0, 1.0)
        # Delay: 1.0 + 0.5 = 1.5
        assert mock_sleep.call_args[0][0] == pytest.approx(1.5)

    async def test_coro_factory_called_fresh_each_attempt(
        self,
    ) -> None:
        """Verify factory is called (not reusing a spent coroutine)."""
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry")
            return "done"

        with patch(
            f"{_RETRY_MOD}.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await retry_with_backoff(
                factory,
                max_retries=3,
                retryable_exceptions=(ValueError,),
            )
        assert result == "done"
        assert call_count == 3
