"""LiteLLMProvider thin adapter for universal LLM routing.

LiteLLM routes to 100+ providers via prefixed model strings
(e.g. ``"anthropic/claude-3-5-sonnet"``, ``"ollama/llama3"``).
The ``litellm`` SDK is an optional dependency; import fails
gracefully with a clear installation hint.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from llm_consistency.providers._base import (
    BaseLLMProvider,
    EmptyResponseError,
    _RawResponse,
)
from llm_consistency.providers._retry import is_retryable_status, parse_retry_after

if TYPE_CHECKING:
    from llm_consistency.types import GenerationParams


class LiteLLMProvider(BaseLLMProvider):  # pragma: no cover
    """Async LiteLLM universal proxy provider.

    Uses ``litellm.acompletion()`` for any LLM via prefixed model
    strings.  Response format is OpenAI-compatible.

    Args:
        model: LiteLLM model string with provider prefix
            (e.g. ``"anthropic/claude-3-5-sonnet"``).
        **kwargs: Forwarded to :class:`BaseLLMProvider`.
    """

    def __init__(
        self,
        *,
        model: str,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, **kwargs)  # type: ignore[arg-type]
        try:
            import litellm  # noqa: PLC0415

            # litellm depends on openai. Every LiteLLM exception subclasses
            # openai.APIError and carries the HTTP status it maps to.
            from openai import APIConnectionError, APIError  # noqa: PLC0415
        except ImportError:
            msg = "Install llm-consistency[litellm] to use the LiteLLM provider"
            raise ImportError(msg) from None
        self._litellm: Any = litellm
        self._connection_error = APIConnectionError
        self._api_error = APIError

    @property
    def provider_name(self) -> str:
        """Return ``'litellm'``."""
        return "litellm"

    def _is_retryable(self, exc: Exception) -> bool:
        """Retry connection errors, timeouts and 408/409/429/5xx statuses."""
        if isinstance(exc, self._connection_error):  # includes litellm.Timeout
            return True
        if isinstance(exc, self._api_error):
            return is_retryable_status(getattr(exc, "status_code", None))
        return super()._is_retryable(exc)

    def _retry_after(self, exc: Exception) -> float | None:
        """Read ``retry-after`` from the upstream headers LiteLLM keeps."""
        requested = parse_retry_after(getattr(exc, "litellm_response_headers", None))
        return requested if requested is not None else super()._retry_after(exc)

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> _RawResponse:
        """Send a single request via ``litellm.acompletion()``.

        LiteLLM returns OpenAI-compatible responses with
        ``response.choices[0].message.content`` and
        ``response.usage`` attributes.  A refusal is returned as the
        content.  Generation settings that are set are sent as
        ``temperature``, ``max_tokens`` and ``seed``.

        Raises:
            EmptyResponseError: If the output token limit was reached
                before any text was produced.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {}
        if generation is not None:
            if generation.temperature is not None:
                options["temperature"] = generation.temperature
            if generation.max_tokens is not None:
                options["max_tokens"] = generation.max_tokens
            if generation.seed is not None:
                options["seed"] = generation.seed

        t0 = time.monotonic()
        response = await self._litellm.acompletion(
            model=self._model,
            messages=messages,
            **options,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        usage = response.usage
        prompt_tokens: int | None = usage.prompt_tokens if usage else None
        completion_tokens: int | None = usage.completion_tokens if usage else None

        # LiteLLM passes an OpenAI refusal through provider_specific_fields.
        extra = choice.message.provider_specific_fields or {}
        content: str = choice.message.content or extra.get("refusal") or ""
        if not content and choice.finish_reason == "length":
            msg = (
                f"LiteLLM model {self._model!r} reached the output token limit "
                f"before producing any text"
            )
            raise EmptyResponseError(
                msg,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return _RawResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
