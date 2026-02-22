"""OllamaProvider thin adapter for local Ollama models.

Ollama serves local LLMs via its HTTP API.  No API key is required.
The ``ollama`` SDK is an optional dependency; import fails gracefully
with a clear installation hint.
"""

from __future__ import annotations

import time

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse


class OllamaProvider(BaseLLMProvider):  # pragma: no cover
    """Async Ollama provider using ``ollama.AsyncClient``.

    Args:
        model: Ollama model tag (e.g., ``"llama3"``).
        host: Custom Ollama server URL, or ``None`` for the
            default ``http://localhost:11434``.
        **kwargs: Forwarded to :class:`BaseLLMProvider`.
    """

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, **kwargs)  # type: ignore[arg-type]
        try:
            from ollama import AsyncClient  # noqa: PLC0415
        except ImportError:
            msg = "Install llm-consistency[ollama] to use the Ollama provider"
            raise ImportError(msg) from None

        if host is not None:
            self._client = AsyncClient(host=host)
        else:
            self._client = AsyncClient()

    @property
    def provider_name(self) -> str:
        """Return ``'ollama'``."""
        return "ollama"

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        """Send a single chat request to the local Ollama server.

        Ollama-specific mapping:
        - ``prompt_eval_count`` -> ``prompt_tokens``
        - ``eval_count`` -> ``completion_tokens``
        - Response uses dict-style access for max compatibility.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await self._client.chat(
            model=self._model,
            messages=messages,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        return _RawResponse(
            content=response["message"]["content"],
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            latency_ms=latency_ms,
        )
