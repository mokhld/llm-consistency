"""Anthropic Claude provider adapter.

Thin subclass of :class:`BaseLLMProvider` that maps
``AsyncAnthropic`` message responses to :class:`_RawResponse`.
Handles Anthropic-specific conventions: system prompt as a
top-level parameter (not in messages), ``input_tokens``/
``output_tokens`` naming, and required ``max_tokens``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from llm_consistency.providers._base import (
    BaseLLMProvider,
    EmptyResponseError,
    _RawResponse,
)
from llm_consistency.providers._retry import is_retryable_status

if TYPE_CHECKING:
    from llm_consistency.types import GenerationParams

_logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):  # pragma: no cover
    """Anthropic Claude provider.

    Args:
        model: Model identifier (e.g., ``"claude-sonnet-4-20250514"``).
        api_key: Anthropic API key, or ``None`` to use env default.
        max_tokens: Maximum tokens per response when the request's
            generation settings do not set one.  Anthropic requires this
            on every request.  Defaults to 1024 (sufficient for MC
            question answers).
        **kwargs: Forwarded to :class:`BaseLLMProvider`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        try:
            from anthropic import (  # noqa: PLC0415
                APIConnectionError,
                APIStatusError,
                AsyncAnthropic,
            )
        except ImportError:
            msg = "Install llm-consistency[anthropic] to use the Anthropic provider"
            raise ImportError(msg) from None
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self._max_tokens = max_tokens
        self._connection_error = APIConnectionError
        self._status_error = APIStatusError

    @property
    def provider_name(self) -> str:
        """Return ``'anthropic'``."""
        return "anthropic"

    def _is_retryable(self, exc: Exception) -> bool:
        """Retry connection errors, timeouts and 408/409/429/5xx statuses.

        5xx includes 529 (``OverloadedError``).
        """
        if isinstance(exc, self._connection_error):  # includes APITimeoutError
            return True
        if isinstance(exc, self._status_error):
            return is_retryable_status(exc.status_code)
        return super()._is_retryable(exc)

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> _RawResponse:
        """Send a single messages API request.

        Builds kwargs with system as a top-level parameter (not in
        messages), calls the Anthropic Messages API, and maps
        ``input_tokens``/``output_tokens`` to the standard
        ``prompt_tokens``/``completion_tokens`` in :class:`_RawResponse`.
        The content is all text blocks joined; thinking blocks are
        skipped.  Generation settings that are set are sent as
        ``temperature`` (in ``extra_body``, because anthropic 1.x has no
        temperature argument) and ``max_tokens``.  Anthropic has no seed
        parameter, so a seed is ignored.

        Raises:
            EmptyResponseError: If ``max_tokens`` was reached before any
                text was produced.
        """
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            create_kwargs["system"] = system
        if generation is not None:
            if generation.temperature is not None:
                # anthropic 1.x dropped the temperature argument, but the API
                # still accepts it for models before Opus 4.7.
                create_kwargs["extra_body"] = {"temperature": generation.temperature}
            if generation.max_tokens is not None:
                create_kwargs["max_tokens"] = generation.max_tokens
            if generation.seed is not None:
                _logger.debug(
                    "Anthropic has no seed parameter; ignoring generation seed %d",
                    generation.seed,
                )

        t0 = time.monotonic()
        response = await self._client.messages.create(**create_kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        prompt_tokens = getattr(response.usage, "input_tokens", None)
        completion_tokens = getattr(response.usage, "output_tokens", None)
        if not content and response.stop_reason == "max_tokens":
            msg = (
                f"Anthropic model {self._model!r} reached max_tokens "
                f"({create_kwargs['max_tokens']}) before producing any text"
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
