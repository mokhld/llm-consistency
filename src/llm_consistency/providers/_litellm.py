"""LiteLLMProvider thin adapter for universal LLM routing.

LiteLLM routes to 100+ providers via prefixed model strings
(e.g. ``"anthropic/claude-3-5-sonnet"``, ``"ollama/llama3"``).
The ``litellm`` SDK is an optional dependency; import fails
gracefully with a clear installation hint.
"""

from __future__ import annotations

import time
from typing import Any

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse


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
        except ImportError:
            msg = "Install llm-consistency[litellm] to use the LiteLLM provider"
            raise ImportError(msg) from None
        self._litellm: Any = litellm

    @property
    def provider_name(self) -> str:
        """Return ``'litellm'``."""
        return "litellm"

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        """Send a single request via ``litellm.acompletion()``.

        LiteLLM returns OpenAI-compatible responses with
        ``response.choices[0].message.content`` and
        ``response.usage`` attributes.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await self._litellm.acompletion(
            model=self._model,
            messages=messages,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        content: str = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens: int | None = usage.prompt_tokens if usage else None
        completion_tokens: int | None = usage.completion_tokens if usage else None

        return _RawResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
