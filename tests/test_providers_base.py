"""Tests for BaseLLMProvider ABC with Template Method pattern."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse
from llm_consistency.providers._batch_result import BatchResult
from llm_consistency.providers._budget import BudgetExceededError
from llm_consistency.types import LLMResponse


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
        """Two quick queries with rate=1/s cause measurable delay."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            requests_per_minute=60,  # 1/s
        )
        # Drain the bucket with first query
        await provider.query("p1", question_id="q1")
        # Second query should be delayed by ~1s
        t0 = time.monotonic()
        await provider.query("p2", question_id="q2")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.5, f"Expected delay >= 0.5s, got {elapsed:.3f}s"


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
        )
        # Manually push budget tracker past limit
        await provider._budget.record(actual_cost=0.002)
        with pytest.raises(BudgetExceededError):
            await provider.query("p", question_id="q1")

    @pytest.mark.asyncio
    async def test_budget_check_called_before_request(self) -> None:
        """budget.check() called with estimated_cost=0.0."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            max_budget_usd=10.0,
        )
        with patch.object(
            provider._budget,
            "check",
            new_callable=AsyncMock,
        ) as mock_check:
            await provider.query("p", question_id="q1")
            mock_check.assert_called_once_with(
                estimated_cost=0.0,
            )

    @pytest.mark.asyncio
    async def test_budget_record_called_after_request(self) -> None:
        """budget.record() called with actual_cost=0.0."""
        provider = _MockProvider(
            responses=[_DEFAULT_RAW],
            max_budget_usd=10.0,
        )
        with patch.object(
            provider._budget,
            "record",
            new_callable=AsyncMock,
        ) as mock_record:
            await provider.query("p", question_id="q1")
            mock_record.assert_called_once_with(
                actual_cost=0.0,
            )


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
