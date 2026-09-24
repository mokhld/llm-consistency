"""BaseLLMProvider ABC with Template Method pattern.

Composes rate limiting, retry with backoff, budget enforcement, and
per-request/batch timeout into a single abstract base class.  Concrete
providers override only ``_send_request()`` and ``provider_name``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.providers._batch_result import BatchResult
from llm_consistency.providers._budget import BudgetTracker, CostPerToken
from llm_consistency.providers._cost import get_model_pricing
from llm_consistency.providers._rate_limit import AsyncTokenBucket
from llm_consistency.providers._retry import parse_retry_after, retry_with_backoff
from llm_consistency.types import GenerationParams, LLMResponse


class EmptyResponseError(LLMConsistencyError):
    """Raised when a provider returns no answer text.

    Covers responses cut off by the output token limit before any text,
    and reasoning-only responses. It is not retried, so the runner
    records the variant as an error instead of scoring an empty answer
    as wrong.

    Args:
        message: Description of the empty response.
        prompt_tokens: Input tokens billed for the response, if known.
        completion_tokens: Output tokens billed for the response, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


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
        model: LLM model identifier (e.g., ``"gpt-5-mini"``).
        requests_per_minute: Sustained rate limit for the token bucket.
            Every attempt, including retries, takes a token. Up to a
            tenth of a minute's requests may go out as an initial burst.
        max_retries: Maximum retry attempts on transient failures.
        request_timeout_s: Per-request timeout in seconds.
        batch_timeout_s: Per-batch timeout in seconds.
        max_budget_usd: Budget ceiling in USD, or ``None`` for
            unlimited.
        base_delay: Backoff base delay in seconds.
        max_delay: Backoff maximum delay in seconds. Also caps a
            server-requested ``retry-after`` delay.
        jitter: Backoff jitter upper bound in seconds.
        pricing: Per-token prices for ``model``. Overrides the built-in
            pricing table; needed for models it does not list.

    Raises:
        ValidationError: If ``max_budget_usd`` is set but no price is
            known for ``model``.
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
        pricing: CostPerToken | None = None,
    ) -> None:
        self._model = model
        self._pricing = pricing if pricing is not None else get_model_pricing(model)
        if max_budget_usd is not None and self._pricing is None:
            msg = (
                f"max_budget_usd is set, but no price is known for model "
                f"{model!r}, so the budget cannot be enforced. Remove the "
                f"budget (--max-budget-usd on the CLI), or in Python pass "
                f"pricing=CostPerToken(input_per_token=..., "
                f"output_per_token=...) to the provider."
            )
            raise ValidationError(msg)
        self._rate_limiter = AsyncTokenBucket(
            rate=requests_per_minute / 60.0,
            capacity=max(1, requests_per_minute // 10),
        )
        self._budget = BudgetTracker(max_budget_usd=max_budget_usd)
        self._largest_cost = 0.0
        # With a budget, the first request runs alone so later reservations
        # are sized from a real cost rather than a guess.
        self._first_request_lock = asyncio.Lock()
        self._cost_observed = False
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

    @property
    def max_budget_usd(self) -> float | None:
        """Budget ceiling in USD enforced by this provider, or ``None``."""
        return self._budget.max_budget_usd

    @abstractmethod
    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> _RawResponse:
        """Make one API call.  No retry/rate-limit logic here.

        Args:
            prompt: The user prompt to send.
            system: Optional system message.
            generation: Decoding settings. Send each field that is not
                ``None``. :meth:`query` passes this argument only when it
                is set, so an override written without it keeps working
                until generation settings are used.

        Returns:
            A ``_RawResponse`` with content and token metadata.
        """
        ...  # pragma: no cover

    @property
    def _retryable_exceptions(
        self,
    ) -> tuple[type[Exception], ...]:
        """Exception types that trigger a retry.

        Network-level failures only. ``OSError`` as a whole is not
        included, because it also covers errors such as
        ``PermissionError`` that retrying cannot fix. Subclasses can
        override to add provider-specific transient errors.
        """
        return (TimeoutError, ConnectionError)

    def _is_retryable(self, exc: Exception) -> bool:
        """Return whether ``exc`` is a transient failure worth retrying.

        The default accepts ``_retryable_exceptions``. Concrete providers
        extend it with their SDK's connection errors and retryable HTTP
        statuses (408, 409, 429 and 5xx).
        """
        return isinstance(exc, self._retryable_exceptions)

    def _retry_after(self, exc: Exception) -> float | None:
        """Return the server-requested retry delay for ``exc``, if any.

        Reads the ``retry-after-ms`` or ``retry-after`` header of the
        HTTP response that SDK status errors carry as ``exc.response``.
        """
        response = getattr(exc, "response", None)
        return parse_retry_after(getattr(response, "headers", None))

    def _estimate_cost(self, prompt: str, system: str | None) -> float:
        """Return the cost to reserve before sending a request.

        Input tokens are estimated as one per three characters of prompt
        and system text (at least 200), output as 50 tokens. The largest
        cost of any request so far is used instead when it is higher.
        """
        if self._pricing is None:
            return 0.0
        input_tokens = max(200, (len(prompt) + len(system or "")) // 3)
        return max(self._pricing.estimate(input_tokens, 50), self._largest_cost)

    def _cost(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        estimated_cost: float,
    ) -> float:
        """Return the cost of a response in USD.

        Falls back to ``estimated_cost`` when token usage was not
        reported, so the request still counts against the budget.
        """
        if self._pricing is None or prompt_tokens is None or completion_tokens is None:
            return estimated_cost
        return self._pricing.estimate(prompt_tokens, completion_tokens)

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> LLMResponse:
        """Rate-limited, retried, budgeted single query.

        Orchestration order:
        1. Reserve the estimated cost against the budget
        2. Retry loop with backoff; each attempt acquires a rate-limit
           token and runs under the per-request timeout
        3. Settle the reservation to the actual cost
        4. Map ``_RawResponse`` to ``LLMResponse``

        Args:
            prompt: The user prompt to send.
            question_id: Back-reference to the originating question.
            system: Optional system message.
            generation: Decoding settings, or ``None`` for the provider
                defaults.

        Returns:
            An ``LLMResponse`` frozen dataclass.

        Raises:
            BudgetExceededError: If the budget ceiling is exceeded.
            EmptyResponseError: If the provider returned no answer text.
            TimeoutError: If the request exceeds ``request_timeout_s``.
        """
        if self.max_budget_usd is not None and not self._cost_observed:
            async with self._first_request_lock:
                if not self._cost_observed:
                    try:
                        return await self._query(
                            prompt, question_id, system, generation
                        )
                    finally:
                        self._cost_observed = True
        return await self._query(prompt, question_id, system, generation)

    async def _query(
        self,
        prompt: str,
        question_id: str,
        system: str | None,
        generation: GenerationParams | None,
    ) -> LLMResponse:
        """Run one budgeted, rate-limited, retried query (see :meth:`query`)."""
        # 1. Reserve the estimated cost; raises if it does not fit
        estimated_cost = self._estimate_cost(prompt, system)
        await self._budget.reserve(estimated_cost=estimated_cost)

        # Passed only when set, so _send_request overrides written before
        # the argument existed keep working.
        extra = {"generation": generation} if generation is not None else {}

        # 2. Retry with backoff; every attempt takes a rate-limit token
        async def _attempt() -> _RawResponse:
            await self._rate_limiter.acquire()
            async with asyncio.timeout(self._request_timeout_s):
                return await self._send_request(prompt, system=system, **extra)

        # 3. Settle the reservation. A request that timed out or was
        #    cancelled keeps its reservation, because it may still be
        #    billed. A request the provider rejected costs nothing. An empty
        #    response is billed for the tokens it used.
        actual_cost = estimated_cost
        try:
            raw = await retry_with_backoff(
                _attempt,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                max_delay=self._max_delay,
                jitter=self._jitter,
                is_retryable=self._is_retryable,
                retry_after=self._retry_after,
            )
            actual_cost = self._cost(
                raw.prompt_tokens, raw.completion_tokens, estimated_cost
            )
        except EmptyResponseError as exc:
            actual_cost = self._cost(
                exc.prompt_tokens, exc.completion_tokens, estimated_cost
            )
            raise
        except TimeoutError:
            raise
        except Exception:
            actual_cost = 0.0
            raise
        finally:
            await self._budget.settle(reserved=estimated_cost, actual_cost=actual_cost)
            self._largest_cost = max(self._largest_cost, actual_cost)

        # 4. Map _RawResponse to LLMResponse
        return LLMResponse(
            question_id=question_id,
            raw_output=raw.content,
            extracted_answer="",
            model=self._model,
            provider=self.provider_name,
            latency_ms=raw.latency_ms,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
        )

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
