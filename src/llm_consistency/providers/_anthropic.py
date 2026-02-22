"""Anthropic Claude provider adapter.

Thin subclass of :class:`BaseLLMProvider` that maps
``AsyncAnthropic`` message responses to :class:`_RawResponse`.
Handles Anthropic-specific conventions: system prompt as a
top-level parameter (not in messages), ``input_tokens``/
``output_tokens`` naming, and required ``max_tokens``.
"""

from __future__ import annotations

import time
from typing import Any

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse


class AnthropicProvider(BaseLLMProvider):  # pragma: no cover
    """Anthropic Claude provider.

    Args:
        model: Model identifier (e.g., ``"claude-sonnet-4-20250514"``).
        api_key: Anthropic API key, or ``None`` to use env default.
        max_tokens: Maximum tokens per response.  Anthropic requires
            this on every request.  Defaults to 1024 (sufficient for
            MC question answers).
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
            from anthropic import AsyncAnthropic  # noqa: PLC0415
        except ImportError:
            msg = "Install llm-consistency[anthropic] to use the Anthropic provider"
            raise ImportError(msg) from None
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        """Return ``'anthropic'``."""
        return "anthropic"

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        """Send a single messages API request.

        Builds kwargs with system as a top-level parameter (not in
        messages), calls the Anthropic Messages API, and maps
        ``input_tokens``/``output_tokens`` to the standard
        ``prompt_tokens``/``completion_tokens`` in :class:`_RawResponse`.
        """
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            create_kwargs["system"] = system

        t0 = time.monotonic()
        response = await self._client.messages.create(**create_kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        content = response.content[0].text if response.content else ""
        return _RawResponse(
            content=content,
            prompt_tokens=getattr(response.usage, "input_tokens", None),
            completion_tokens=getattr(response.usage, "output_tokens", None),
            latency_ms=latency_ms,
        )
