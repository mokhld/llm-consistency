"""Tests for BaseLLMProvider ABC with Template Method pattern."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.providers._base import (
    BaseLLMProvider,
    EmptyResponseError,
    _RawResponse,
)
from llm_consistency.providers._batch_result import BatchResult
from llm_consistency.providers._budget import BudgetExceededError, CostPerToken
from llm_consistency.providers._rate_limit import AsyncTokenBucket
from llm_consistency.types import LLMResponse

_RETRY_MOD = "llm_consistency.providers._retry"


# ---------------------------------------------------------------------------
# _MockProvider -- concrete subclass for testing the base class
# ---------------------------------------------------------------------------
class _MockProvider(BaseLLMProvider):
    """Concrete test provider that returns canned _RawResponse values."""

    def __init__(
        self,
        responses: list[_RawResponse] | None = None,
        *,
        fail_on: dict[int, Exception] | None = None,
        delay_s: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(model="mock-model", **kwargs)  # type: ignore[arg-type]
        self._responses = list(responses or [])
        self._fail_on: dict[int, Exception] = fail_on or {}
        self._delay_s = delay_s
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        attempt = self._call_count
        self._call_count += 1
        if attempt in self._fail_on:
            raise self._fail_on[attempt]
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._responses:
            return self._responses[attempt % len(self._responses)]
        return _RawResponse(
            content="mock answer",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )


_DEFAULT_RAW = _RawResponse(
    content="Answer: B",
    prompt_tokens=20,
    completion_tokens=8,
    latency_ms=42.5,
)

# $1 per million tokens both ways: _DEFAULT_RAW costs 28e-6, and the
# 200 + 50 token pre-request estimate is 250e-6.
_PRICING = CostPerToken(input_per_token=1e-6, output_per_token=1e-6)


class _HTTPStatusError(ConnectionError):
    """Retryable error carrying an HTTP response, like SDK status errors."""

    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__("503 Service Unavailable")
        self.response = SimpleNamespace(headers=headers)


# ---------------------------------------------------------------------------
# Tests: _RawResponse dataclass
# ---------------------------------------------------------------------------
class TestRawResponse:
    def test_frozen(self) -> None:
        raw = _RawResponse(
            content="x",
            prompt_tokens=1,
            completion_tokens=2,
            latency_ms=0.5,
        )
        with pytest.raises(AttributeError):
            raw.content = "y"  # type: ignore[misc]

    def test_fields(self) -> None:
        raw = _RawResponse(
            content="hello",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=3.2,
        )
        assert raw.content == "hello"
        assert raw.prompt_tokens == 10
        assert raw.completion_tokens == 5
        assert raw.latency_ms == 3.2

    def test_none_tokens(self) -> None:
        raw = _RawResponse(
            content="x",
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=0.1,
        )
        assert raw.prompt_tokens is None
        assert raw.completion_tokens is None


# ---------------------------------------------------------------------------
# Tests: Abstract base class cannot be instantiated
# ---------------------------------------------------------------------------
class TestAbstract:
    def test_cannot_instantiate_base(self) -> None:
        """BaseLLMProvider is ABC; cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLMProvider(model="x")  # type: ignore[abstract]

    def test_provider_name_is_abstract(self) -> None:
        """provider_name must be overridden."""

        class _Incomplete(BaseLLMProvider):
            async def _send_request(
                self,
                prompt: str,
                *,
                system: str | None = None,
            ) -> _RawResponse:
                return _DEFAULT_RAW  # pragma: no cover

        with pytest.raises(TypeError):
            _Incomplete(model="x")  # type: ignore[abstract]

    def test_send_request_is_abstract(self) -> None:
        """_send_request must be overridden."""

        class _Incomplete(BaseLLMProvider):
            @property
            def provider_name(self) -> str:
                return "incomplete"  # pragma: no cover

        with pytest.raises(TypeError):
            _Incomplete(model="x")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Tests: query() returns LLMResponse with correct fields
# ---------------------------------------------------------------------------
class TestQuery:
    @pytest.mark.asyncio
    async def test_query_returns_llm_response(self) -> None:
        provider = _MockProvider(responses=[_DEFAULT_RAW])
        resp = await provider.query("What is 2+2?", question_id="q1")
        assert isinstance(resp, LLMResponse)

    @pytest.mark.asyncio
    async def test_query_maps_fields_correctly(self) -> None:
        provider = _MockProvider(responses=[_DEFAULT_RAW])
        resp = await provider.query("prompt", question_id="q42")
        assert resp.question_id == "q42"
        assert resp.raw_output == "Answer: B"
        assert resp.extracted_answer == ""  # placeholder
        assert resp.model == "mock-model"
        assert resp.provider == "mock"
        assert resp.prompt_tokens == 20
        assert resp.completion_tokens == 8
        assert resp.latency_ms == 42.5

    @pytest.mark.asyncio
    async def test_query_none_tokens(self) -> None:
        raw = _RawResponse(
            content="x",
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=0.1,
        )
        provider = _MockProvider(responses=[raw])
        resp = await provider.query("p", question_id="q1")
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None


# ---------------------------------------------------------------------------
# Tests: query() rate limiting
# ---------------------------------------------------------------------------
class TestQueryRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limiter_causes_delay(self) -> None:
        """Two quick queries with low-capacity bucket cause delay."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            requests_per_minute=60,
        )
        # Replace with a bucket: capacity=1, rate=2 tokens/s
        # First query drains the single token; second waits ~0.5s
        provider._rate_limiter = AsyncTokenBucket(rate=2.0, capacity=1)
        await provider.query("p1", question_id="q1")
        t0 = time.monotonic()
        await provider.query("p2", question_id="q2")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.3, f"Expected delay >= 0.3s, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Tests: query() retry on retryable exceptions
# ---------------------------------------------------------------------------
class TestQueryRetry:
    @pytest.mark.asyncio
    async def test_retries_on_retryable_exception(self) -> None:
        """query() retries on TimeoutError from _send_request."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: TimeoutError("timed out")},
            max_retries=2,
            base_delay=0.01,
            max_delay=0.02,
            jitter=0.0,
        )
        resp = await provider.query("p", question_id="q1")
        assert isinstance(resp, LLMResponse)
        # Call 0 fails, call 1 succeeds
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_permission_error_not_retried(self) -> None:
        """PermissionError is an OSError, but retrying cannot fix it."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: PermissionError("401 Unauthorized")},
            max_retries=3,
            base_delay=0.0,
            jitter=0.0,
        )
        with pytest.raises(PermissionError):
            await provider.query("p", question_id="q1")
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_header_sets_delay(self) -> None:
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: _HTTPStatusError({"retry-after": "0.25"})},
            max_retries=1,
            base_delay=5.0,
            jitter=1.0,
        )
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await provider.query("p", question_id="q1")
        sleep.assert_awaited_once_with(0.25)

    @pytest.mark.asyncio
    async def test_retry_after_capped_at_max_delay(self) -> None:
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: _HTTPStatusError({"retry-after-ms": "120000"})},
            max_retries=1,
            max_delay=2.0,
        )
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await provider.query("p", question_id="q1")
        sleep.assert_awaited_once_with(2.0)

    @pytest.mark.asyncio
    async def test_each_attempt_takes_a_rate_limit_token(self) -> None:
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: TimeoutError("t"), 1: ConnectionError("c")},
            max_retries=2,
            base_delay=0.0,
            jitter=0.0,
        )
        with patch.object(
            provider._rate_limiter, "acquire", new_callable=AsyncMock
        ) as acquire:
            await provider.query("p", question_id="q1")
        assert provider._call_count == 3
        assert acquire.await_count == 3

    @pytest.mark.asyncio
    async def test_empty_response_not_retried(self) -> None:
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: EmptyResponseError("no text")},
            max_retries=3,
            base_delay=0.0,
            jitter=0.0,
        )
        with pytest.raises(EmptyResponseError):
            await provider.query("p", question_id="q1")
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self) -> None:
        """query() raises after exhausting retries."""
        provider = _MockProvider(
            fail_on={
                0: ConnectionError("fail"),
                1: ConnectionError("fail"),
                2: ConnectionError("fail"),
                3: ConnectionError("fail"),
            },
            max_retries=2,
            base_delay=0.01,
            max_delay=0.02,
            jitter=0.0,
        )
        with pytest.raises(ConnectionError, match="fail"):
            await provider.query("p", question_id="q1")


# ---------------------------------------------------------------------------
# Tests: query() budget enforcement
# ---------------------------------------------------------------------------
class TestQueryBudget:
    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self) -> None:
        """query() raises BudgetExceededError when exceeded."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            max_budget_usd=0.001,
            pricing=_PRICING,
        )
        # Manually push budget tracker past limit
        await provider._budget.settle(reserved=0.0, actual_cost=0.002)
        with pytest.raises(BudgetExceededError):
            await provider.query("p", question_id="q1")
        assert provider._call_count == 0

    @pytest.mark.asyncio
    async def test_budget_reserved_before_request(self) -> None:
        """budget.reserve() gets the 200 + 50 token estimate."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        with patch.object(
            provider._budget,
            "reserve",
            new_callable=AsyncMock,
        ) as mock_reserve:
            await provider.query("p", question_id="q1")
            mock_reserve.assert_called_once_with(
                estimated_cost=pytest.approx(250e-6),
            )

    @pytest.mark.asyncio
    async def test_budget_settled_after_request(self) -> None:
        """budget.settle() swaps the reservation for the actual cost."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        with patch.object(
            provider._budget,
            "settle",
            new_callable=AsyncMock,
        ) as mock_settle:
            await provider.query("p", question_id="q1")
            mock_settle.assert_called_once_with(
                reserved=pytest.approx(250e-6),
                actual_cost=pytest.approx(28e-6),
            )

    @pytest.mark.asyncio
    async def test_failed_request_releases_reservation(self) -> None:
        provider = _MockProvider(
            fail_on={0: ValueError("bad request")},
            max_retries=0,
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        with pytest.raises(ValueError, match="bad request"):
            await provider.query("p", question_id="q1")
        assert provider._budget._reserved == 0.0
        assert provider._budget.spent == 0.0

    @pytest.mark.asyncio
    async def test_cancelled_request_keeps_its_reservation(self) -> None:
        """A cancelled request may already have been billed, so it counts."""
        provider = _MockProvider(
            delay_s=10.0,
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        task = asyncio.create_task(provider.query("p", question_id="q1"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider._budget._reserved == 0.0
        assert provider._budget.spent == pytest.approx(250e-6)

    @pytest.mark.asyncio
    async def test_unreported_usage_is_charged_the_estimate(self) -> None:
        raw = _RawResponse(
            content="x", prompt_tokens=None, completion_tokens=None, latency_ms=1.0
        )
        provider = _MockProvider(
            responses=[raw],
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        await provider.query("p", question_id="q1")
        assert provider._budget.spent == pytest.approx(250e-6)

    @pytest.mark.asyncio
    async def test_empty_response_is_charged_its_usage(self) -> None:
        """An empty response was billed, so its tokens count."""
        provider = _MockProvider(
            fail_on={
                0: EmptyResponseError(
                    "no text", prompt_tokens=100, completion_tokens=900
                ),
            },
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        with pytest.raises(EmptyResponseError):
            await provider.query("p", question_id="q1")
        assert provider._budget.spent == pytest.approx(1000e-6)
        assert provider._budget._reserved == 0.0

    @pytest.mark.asyncio
    async def test_reservation_grows_to_largest_cost_seen(self) -> None:
        big = _RawResponse(
            content="x", prompt_tokens=1000, completion_tokens=1000, latency_ms=1.0
        )
        provider = _MockProvider(
            responses=[big, _DEFAULT_RAW],
            max_budget_usd=10.0,
            pricing=_PRICING,
        )
        assert provider._estimate_cost("p", None) == pytest.approx(250e-6)
        await provider.query("p", question_id="q1")
        assert provider._estimate_cost("p", None) == pytest.approx(2000e-6)
        await provider.query("p", question_id="q2")
        # A cheaper response does not shrink the reservation
        assert provider._estimate_cost("p", None) == pytest.approx(2000e-6)

    @pytest.mark.asyncio
    async def test_concurrent_queries_stop_at_the_cap(self) -> None:
        """Requests in flight hold their reservation, so spend stays capped.

        Each call costs 0.001 (200 input + 50 output tokens at gpt-4o
        prices). Before reservations, all 50 concurrent calls passed the
        check and spent 0.05 against a 0.0205 cap.
        """
        raw = _RawResponse(
            content="x", prompt_tokens=200, completion_tokens=50, latency_ms=1.0
        )
        provider = _MockProvider(
            responses=[raw],
            delay_s=0.01,
            max_budget_usd=0.0205,
            pricing=CostPerToken(input_per_token=2.5e-6, output_per_token=10e-6),
            requests_per_minute=60_000,
        )
        results = await asyncio.gather(
            *(provider.query("p", question_id=f"q{i}") for i in range(50)),
            return_exceptions=True,
        )
        succeeded = [r for r in results if isinstance(r, LLMResponse)]
        refused = [r for r in results if isinstance(r, BudgetExceededError)]
        assert len(succeeded) == 20
        assert len(refused) == 30
        assert provider._call_count == 20
        assert provider._budget.spent == pytest.approx(0.020)


# ---------------------------------------------------------------------------
# Tests: constructor pricing and rate-limit settings
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_budget_without_price_raises(self) -> None:
        with pytest.raises(ValidationError, match="no price is known"):
            _MockProvider(max_budget_usd=1.0)

    def test_pricing_override_allows_budget(self) -> None:
        provider = _MockProvider(max_budget_usd=1.0, pricing=_PRICING)
        assert provider._pricing is _PRICING

    def test_no_budget_needs_no_price(self) -> None:
        provider = _MockProvider()
        assert provider._pricing is None
        assert provider._estimate_cost("p", None) == 0.0

    @pytest.mark.parametrize(
        ("rpm", "capacity"),
        [(600, 60), (60, 6), (5, 1)],
    )
    def test_burst_is_a_tenth_of_a_minute(self, rpm: int, capacity: int) -> None:
        provider = _MockProvider(requests_per_minute=rpm)
        assert provider._rate_limiter._capacity == capacity
        assert provider._rate_limiter._rate == pytest.approx(rpm / 60)


class TestEmptyResponseError:
    def test_is_llm_consistency_error(self) -> None:
        assert issubclass(EmptyResponseError, LLMConsistencyError)

    def test_carries_usage(self) -> None:
        err = EmptyResponseError("no text", prompt_tokens=3, completion_tokens=4)
        assert str(err) == "no text"
        assert err.prompt_tokens == 3
        assert err.completion_tokens == 4


# ---------------------------------------------------------------------------
# Tests: query() timeout enforcement
# ---------------------------------------------------------------------------
class TestQueryTimeout:
    @pytest.mark.asyncio
    async def test_request_timeout_raises(self) -> None:
        """query() raises TimeoutError when exceeding timeout."""
        provider = _MockProvider(
            delay_s=2.0,
            request_timeout_s=0.1,
            max_retries=0,
        )
        with pytest.raises(TimeoutError):
            await provider.query("p", question_id="q1")


# ---------------------------------------------------------------------------
# Tests: query_batch()
# ---------------------------------------------------------------------------
class TestQueryBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_batch_result(self) -> None:
        provider = _MockProvider(responses=[_DEFAULT_RAW])
        prompts = [("prompt1", "q1"), ("prompt2", "q2")]
        result = await provider.query_batch(prompts)
        assert isinstance(result, BatchResult)
        assert result.attempted == 2
        assert result.completed == 2
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_batch_captures_individual_failures(self) -> None:
        """One failing query appears in errors, not crash."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            fail_on={0: ValueError("bad prompt")},
            max_retries=0,
        )
        prompts = [("prompt1", "q1"), ("prompt2", "q2")]
        result = await provider.query_batch(prompts)
        assert result.completed == 1
        assert result.failed == 1
        assert len(result.errors) == 1
        qid, msg = result.errors[0]
        assert qid == "q1"
        assert "bad prompt" in msg

    @pytest.mark.asyncio
    async def test_batch_timeout_raises(self) -> None:
        """query_batch() raises TimeoutError on batch timeout."""
        provider = _MockProvider(
            delay_s=2.0,
            batch_timeout_s=0.1,
            max_retries=0,
            request_timeout_s=10.0,
        )
        prompts = [("p1", "q1"), ("p2", "q2")]
        with pytest.raises(TimeoutError):
            await provider.query_batch(prompts)

    @pytest.mark.asyncio
    async def test_batch_runs_concurrently(self) -> None:
        """Two 0.2s queries complete in < 0.35s (concurrent)."""
        raw = _RawResponse(
            content="x",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=200.0,
        )
        provider = _MockProvider(
            responses=[raw],
            delay_s=0.2,
            requests_per_minute=600,
        )
        prompts = [("p1", "q1"), ("p2", "q2")]
        t0 = time.monotonic()
        result = await provider.query_batch(prompts)
        elapsed = time.monotonic() - t0
        assert result.completed == 2
        assert elapsed < 0.35, f"Expected < 0.35s (concurrent), got {elapsed:.3f}s"


class TestBudgetUnderConcurrency:
    @pytest.mark.asyncio
    async def test_first_request_runs_alone_so_the_cap_holds(self) -> None:
        """Ten concurrent requests costing $0.00775 each against a $0.02 cap.

        The first request runs alone, so later requests reserve the observed
        cost: two fit under the cap and the rest are refused, instead of all
        ten passing the small default estimate at once.
        """
        raw = _RawResponse(
            content="x", prompt_tokens=1500, completion_tokens=400, latency_ms=1.0
        )
        pricing = CostPerToken(input_per_token=2.5e-6, output_per_token=10e-6)
        provider = _MockProvider(
            responses=[raw], delay_s=0.01, max_budget_usd=0.02, pricing=pricing
        )
        results = await asyncio.gather(
            *(provider.query("p", question_id=f"q{i}") for i in range(10)),
            return_exceptions=True,
        )
        assert sum(not isinstance(r, BaseException) for r in results) == 2
        assert all(
            isinstance(r, BudgetExceededError)
            for r in results
            if isinstance(r, BaseException)
        )
        assert provider._budget.spent == pytest.approx(0.0155)

    def test_estimate_scales_with_prompt_length(self) -> None:
        provider = _MockProvider(max_budget_usd=1.0, pricing=_PRICING)
        short = provider._estimate_cost("p", None)
        long = provider._estimate_cost("x" * 30_000, "system")
        assert long > short
