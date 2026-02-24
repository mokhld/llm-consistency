"""BaseLLMProvider ABC with Template Method pattern.

Composes rate limiting, retry with backoff, budget enforcement, and
per-request/batch timeout into a single abstract base class.  Concrete
providers override only ``_send_request()`` and ``provider_name``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_consistency.providers._batch_result import BatchResult
from llm_consistency.providers._budget import BudgetTracker
from llm_consistency.providers._rate_limit import AsyncTokenBucket
from llm_consistency.providers._retry import retry_with_backoff
from llm_consistency.types import LLMResponse


@dataclass(frozen=True)
class _RawResponse:
    """Internal contract between ``_send_request()`` and the base class.

    Concrete providers return this from ``_send_request()``.  The base
    class maps it to the public ``LLMResponse`` frozen dataclass.

    Attributes:
        content: Raw text content from the LLM response.
        prompt_tokens: Number of input tokens, or ``None``.
        completion_tokens: Number of output tokens, or ``None``.
        latency_ms: Response latency in milliseconds.
    """

    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class BaseLLMProvider(ABC):
    """Abstract async LLM provider with rate limiting, retry, and budget.

    Uses the Template Method pattern: ``query()`` and ``query_batch()``
    implement all cross-cutting concerns.  Subclasses override only
    ``_send_request()`` and ``provider_name``.

    Args:
        model: LLM model identifier (e.g., ``"gpt-4o"``).
        requests_per_minute: Rate limit for the token bucket.
        max_retries: Maximum retry attempts on transient failures.
        request_timeout_s: Per-request timeout in seconds.
        batch_timeout_s: Per-batch timeout in seconds.
        max_budget_usd: Budget ceiling in USD, or ``None`` for
            unlimited.
        base_delay: Backoff base delay in seconds.
        max_delay: Backoff maximum delay in seconds.
        jitter: Backoff jitter upper bound in seconds.
    """

    def __init__(
        self,
        *,
        model: str,
        requests_per_minute: int = 60,
        max_retries: int = 3,
        request_timeout_s: float = 60.0,
        batch_timeout_s: float = 300.0,
        max_budget_usd: float | None = None,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
    ) -> None:
        self._model = model
        self._rate_limiter = AsyncTokenBucket(
            rate=requests_per_minute / 60.0,
            capacity=requests_per_minute,
        )
        self._budget = BudgetTracker(max_budget_usd=max_budget_usd)
        self._max_retries = max_retries
        self._request_timeout_s = request_timeout_s
        self._batch_timeout_s = batch_timeout_s
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g., ``'openai'``, ``'anthropic'``)."""
        ...  # pragma: no cover

    @abstractmethod
    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        """Make one API call.  No retry/rate-limit logic here.

        Args:
            prompt: The user prompt to send.
            system: Optional system message.

        Returns:
            A ``_RawResponse`` with content and token metadata.
        """
        ...  # pragma: no cover

    @property
    def _retryable_exceptions(
        self,
    ) -> tuple[type[Exception], ...]:
        """Exception types that trigger a retry.

        Subclasses can override to add provider-specific transient
        errors.
        """
        return (TimeoutError, ConnectionError, OSError)

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
    ) -> LLMResponse:
        """Rate-limited, retried, budgeted single query.

        Orchestration order:
        1. Check budget
        2. Acquire rate-limit token
        3. Retry loop with backoff + per-request timeout
        4. Map ``_RawResponse`` to ``LLMResponse``
        5. Record actual cost

        Args:
            prompt: The user prompt to send.
            question_id: Back-reference to the originating question.
            system: Optional system message.

        Returns:
            An ``LLMResponse`` frozen dataclass.

        Raises:
            BudgetExceededError: If the budget ceiling is exceeded.
            TimeoutError: If the request exceeds ``request_timeout_s``.
        """
        # 1. Check budget
        await self._budget.check(estimated_cost=0.0)

        # 2. Acquire rate-limit token
        await self._rate_limiter.acquire()

        # 3. Retry with backoff + per-request timeout
        async def _attempt() -> _RawResponse:
            async with asyncio.timeout(self._request_timeout_s):
                return await self._send_request(prompt, system=system)

        raw = await retry_with_backoff(
            _attempt,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            jitter=self._jitter,
            retryable_exceptions=self._retryable_exceptions,
        )

        # 4. Map _RawResponse to LLMResponse
        response = LLMResponse(
            question_id=question_id,
            raw_output=raw.content,
            extracted_answer="",
            model=self._model,
            provider=self.provider_name,
            latency_ms=raw.latency_ms,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
        )

        # 5. Record actual cost (placeholder 0.0)
        await self._budget.record(actual_cost=0.0)

        return response

    async def query_batch(
        self,
        prompts: list[tuple[str, str]],
        *,
        system: str | None = None,
    ) -> BatchResult:
        """Concurrent batch with partial failure tracking.

        Wraps the entire batch in ``batch_timeout_s``.  Individual
        failures are captured in ``BatchResult.errors``; the method
        never raises on individual query failures.

        Args:
            prompts: List of ``(prompt_text, question_id)`` pairs.
            system: Optional system message for all queries.

        Returns:
            A ``BatchResult`` with successful responses and errors.

        Raises:
            TimeoutError: If the batch exceeds ``batch_timeout_s``.
        """
        async with asyncio.timeout(self._batch_timeout_s):
            tasks = [
                self.query(prompt, question_id, system=system)
                for prompt, question_id in prompts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        successes: list[LLMResponse] = []
        errors: list[tuple[str, str]] = []

        for (_, question_id), result in zip(prompts, results, strict=True):
            if isinstance(result, BaseException):
                errors.append((question_id, str(result)))
            else:
                successes.append(result)

        return BatchResult(
            responses=tuple(successes),
            errors=tuple(errors),
        )
