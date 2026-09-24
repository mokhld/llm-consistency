"""OpenAI and OpenAI-compatible provider adapter.

Thin subclass of :class:`BaseLLMProvider` that maps
``AsyncOpenAI`` responses to :class:`_RawResponse`.
Supports OpenAI-compatible servers (vLLM, Together, etc.)
via the ``base_url`` parameter.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from llm_consistency.providers._base import (
    BaseLLMProvider,
    EmptyResponseError,
    _RawResponse,
)
from llm_consistency.providers._retry import is_retryable_status

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class OpenAIProvider(BaseLLMProvider):  # pragma: no cover
    """OpenAI and OpenAI-compatible provider.

    Args:
        model: Model identifier (e.g., ``"gpt-5-mini"``).
        api_key: OpenAI API key, or ``None`` to use env default.
        base_url: Custom API endpoint for OpenAI-compatible servers.
        **kwargs: Forwarded to :class:`BaseLLMProvider`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        try:
            from openai import (  # noqa: PLC0415
                APIConnectionError,
                APIStatusError,
                AsyncOpenAI,
            )
        except ImportError:
            msg = "Install llm-consistency[openai] to use the OpenAI provider"
            raise ImportError(msg) from None
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._connection_error = APIConnectionError
        self._status_error = APIStatusError

    @property
    def provider_name(self) -> str:
        """Return ``'openai'``."""
        return "openai"

    def _is_retryable(self, exc: Exception) -> bool:
        """Retry connection errors, timeouts and 408/409/429/5xx statuses."""
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
    ) -> _RawResponse:
        """Send a single chat completion request.

        Builds a messages list with optional system message,
        calls the OpenAI Chat Completions API, and maps the
        response to a :class:`_RawResponse`.  A refusal is returned as
        the content.

        Raises:
            EmptyResponseError: If the output token limit was reached
                before any text was produced.
        """
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        if not response.choices:
            msg = (
                f"OpenAI returned no choices for model {self._model!r}; "
                f"id={getattr(response, 'id', None)!r}"
            )
            raise RuntimeError(msg)

        choice = response.choices[0]
        prompt_tokens = response.usage.prompt_tokens if response.usage else None
        completion_tokens = response.usage.completion_tokens if response.usage else None
        content = choice.message.content or choice.message.refusal or ""
        if not content and choice.finish_reason == "length":
            msg = (
                f"OpenAI model {self._model!r} reached the output token limit "
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
