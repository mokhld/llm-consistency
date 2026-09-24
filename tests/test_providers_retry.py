"""Tests for llm_consistency.providers._retry module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_consistency.providers._retry import (
    is_retryable_status,
    parse_retry_after,
    retry_with_backoff,
)

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

    async def test_is_retryable_predicate_triggers_retry(self) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyError("transient")
            return "ok"

        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(
                factory,
                is_retryable=lambda exc: isinstance(exc, KeyError),
            )
        assert result == "ok"
        assert call_count == 2

    async def test_is_retryable_false_raises_immediately(self) -> None:
        factory = AsyncMock(side_effect=KeyError("permanent"))
        with pytest.raises(KeyError):
            await retry_with_backoff(factory, is_retryable=lambda exc: False)
        factory.assert_called_once()

    async def test_retry_after_replaces_backoff_delay(self) -> None:
        factory = AsyncMock(side_effect=[ValueError("429"), "ok"])
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await retry_with_backoff(
                factory,
                base_delay=10.0,
                jitter=5.0,
                retryable_exceptions=(ValueError,),
                retry_after=lambda exc: 0.5,
            )
        sleep.assert_awaited_once_with(0.5)

    async def test_retry_after_capped_at_max_delay(self) -> None:
        factory = AsyncMock(side_effect=[ValueError("429"), "ok"])
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await retry_with_backoff(
                factory,
                max_delay=3.0,
                retryable_exceptions=(ValueError,),
                retry_after=lambda exc: 90.0,
            )
        sleep.assert_awaited_once_with(3.0)

    async def test_retry_after_none_falls_back_to_backoff(self) -> None:
        factory = AsyncMock(side_effect=[ValueError("503"), "ok"])
        with (
            patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep,
            patch(f"{_RETRY_MOD}.random.uniform", return_value=0.0),
        ):
            await retry_with_backoff(
                factory,
                base_delay=2.0,
                retryable_exceptions=(ValueError,),
                retry_after=lambda exc: None,
            )
        sleep.assert_awaited_once_with(2.0)


class TestIsRetryableStatus:
    @pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
    def test_transient_statuses_retry(self, status: int) -> None:
        assert is_retryable_status(status)

    @pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 413, 422, -1])
    def test_permanent_statuses_do_not_retry(self, status: int) -> None:
        assert not is_retryable_status(status)

    @pytest.mark.parametrize("status", [None, "429"])
    def test_non_int_does_not_retry(self, status: object) -> None:
        assert not is_retryable_status(status)


class TestParseRetryAfter:
    def test_none_headers(self) -> None:
        assert parse_retry_after(None) is None

    def test_missing_header(self) -> None:
        assert parse_retry_after({"content-type": "application/json"}) is None

    def test_seconds(self) -> None:
        assert parse_retry_after({"retry-after": "2"}) == 2.0

    def test_fractional_seconds(self) -> None:
        assert parse_retry_after({"retry-after": "0.5"}) == 0.5

    def test_milliseconds_take_precedence(self) -> None:
        headers = {"retry-after-ms": "250", "retry-after": "9"}
        assert parse_retry_after(headers) == pytest.approx(0.25)

    def test_bad_milliseconds_fall_back_to_seconds(self) -> None:
        headers = {"retry-after-ms": "soon", "retry-after": "3"}
        assert parse_retry_after(headers) == 3.0

    @pytest.mark.parametrize(
        "value", ["Wed, 21 Oct 2026 07:28:00 GMT", "-1", "nan", "inf"]
    )
    def test_unusable_values(self, value: str) -> None:
        assert parse_retry_after({"retry-after": value}) is None
