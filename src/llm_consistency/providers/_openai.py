"""OpenAI and OpenAI-compatible provider adapter.

Thin subclass of :class:`BaseLLMProvider` that maps
``AsyncOpenAI`` responses to :class:`_RawResponse`.
Supports OpenAI-compatible servers (vLLM, Together, etc.)
via the ``base_url`` parameter.
"""

from __future__ import annotations

import time
from typing import Any

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse


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
            from openai import AsyncOpenAI  # noqa: PLC0415
        except ImportError:
            msg = "Install llm-consistency[openai] to use the OpenAI provider"
            raise ImportError(msg) from None
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    @property
    def provider_name(self) -> str:
        """Return ``'openai'``."""
        return "openai"

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        """Send a single chat completion request.

        Builds a messages list with optional system message,
        calls the OpenAI Chat Completions API, and maps the
        response to a :class:`_RawResponse`.
        """
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        latency_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        return _RawResponse(
            content=choice.message.content or "",
            prompt_tokens=(response.usage.prompt_tokens if response.usage else None),
            completion_tokens=(
                response.usage.completion_tokens if response.usage else None
            ),
            latency_ms=latency_ms,
        )
