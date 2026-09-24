"""OllamaProvider thin adapter for local Ollama models.

Ollama serves local LLMs via its HTTP API.  No API key is required.
The ``ollama`` SDK is an optional dependency; import fails gracefully
with a clear installation hint.
"""

from __future__ import annotations

import time

import httpx

from llm_consistency.providers._base import (
    BaseLLMProvider,
    EmptyResponseError,
    _RawResponse,
)
from llm_consistency.providers._retry import is_retryable_status


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
            from ollama import AsyncClient, ResponseError  # noqa: PLC0415
        except ImportError:
            msg = "Install llm-consistency[ollama] to use the Ollama provider"
            raise ImportError(msg) from None

        if host is not None:
            self._client = AsyncClient(host=host)
        else:
            self._client = AsyncClient()
        self._response_error = ResponseError

    @property
    def provider_name(self) -> str:
        """Return ``'ollama'``."""
        return "ollama"

    def _is_retryable(self, exc: Exception) -> bool:
        """Retry network failures and 408/409/429/5xx responses.

        The ollama client turns a refused connection into the built-in
        ``ConnectionError`` but lets other httpx transport errors, such
        as read timeouts, through unchanged.
        """
        if isinstance(
            exc,
            (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError),
        ):
            return True
        if isinstance(exc, self._response_error):
            return is_retryable_status(exc.status_code)
        return super()._is_retryable(exc)

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

        Raises:
            EmptyResponseError: If the model produced reasoning in
                ``message.thinking`` but no answer content.
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

        message = response.get("message") if hasattr(response, "get") else None
        if not message or "content" not in message:
            shape = (
                list(response.keys())
                if hasattr(response, "keys")
                else type(response).__name__
            )
            msg = f"Ollama response missing 'message.content' field; got: {shape}"
            raise RuntimeError(msg)

        content = message["content"] or ""
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        if not content and message.get("thinking"):
            msg = (
                f"Ollama model {self._model!r} returned reasoning in "
                f"message.thinking but no answer content"
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
