"""Real SDK errors and responses raised through ``query()``.

The other provider tests replace each SDK with a ``MagicMock``, so they
never see the exception classes the SDKs really raise. These tests drive
the installed SDKs through an in-memory HTTP transport, so each SDK's own
status-to-exception mapping runs. CI installs ``.[all]``; each test skips
when its SDK is missing.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm_consistency.providers._base import EmptyResponseError

_RETRY_MOD = "llm_consistency.providers._retry"
_FAST_RETRY: dict[str, Any] = {
    "max_retries": 3,
    "base_delay": 0.0,
    "max_delay": 1.0,
    "jitter": 0.0,
}


def _replay(http: Any, items: list[Any]) -> tuple[Any, list[Any]]:
    """Return a mock transport that replays ``items`` in order, and its calls.

    Each item is a response to return or an exception to raise.
    """
    calls: list[Any] = []

    def handler(request: Any) -> Any:
        calls.append(request)
        item = items[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    return http.MockTransport(handler), calls


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
def _openai_completion(
    content: str | None = "B",
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": refusal,
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 400,
                "total_tokens": 500,
            },
        },
    )


def _openai_error(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": f"status {status}", "type": "error"}},
        headers=headers,
    )


def _openai_provider(items: list[Any], **kwargs: Any) -> tuple[Any, list[Any], Any]:
    openai = pytest.importorskip("openai")
    module = importlib.import_module("llm_consistency.providers._openai")
    provider = module.OpenAIProvider(
        model="gpt-4o", api_key="test-key", **{**_FAST_RETRY, **kwargs}
    )
    transport, calls = _replay(httpx, items)
    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    )
    return provider, calls, openai


class TestOpenAIErrors:
    @pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
    async def test_transient_status_is_retried(self, status: int) -> None:
        provider, calls, _ = _openai_provider(
            [_openai_error(status), _openai_completion()]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (400, "BadRequestError"),
            (401, "AuthenticationError"),
            (403, "PermissionDeniedError"),
            (404, "NotFoundError"),
        ],
    )
    async def test_client_error_is_not_retried(self, status: int, error: str) -> None:
        provider, calls, openai = _openai_provider(
            [_openai_error(status), _openai_completion()]
        )
        with pytest.raises(getattr(openai, error)):
            await provider.query("prompt", "q1")
        assert len(calls) == 1

    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
    async def test_network_failure_is_retried(self, error: type[Exception]) -> None:
        provider, calls, _ = _openai_provider([error("down"), _openai_completion()])
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    async def test_rate_limit_raises_after_retries(self) -> None:
        provider, calls, openai = _openai_provider([_openai_error(429)] * 4)
        with pytest.raises(openai.RateLimitError):
            await provider.query("prompt", "q1")
        assert len(calls) == 4

    async def test_retry_after_is_honoured(self) -> None:
        provider, calls, _ = _openai_provider(
            [_openai_error(429, {"retry-after": "0.25"}), _openai_completion()],
            base_delay=30.0,
        )
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await provider.query("prompt", "q1")
        sleep.assert_awaited_once_with(0.25)
        assert len(calls) == 2

    async def test_refusal_is_the_answer_text(self) -> None:
        provider, _, _ = _openai_provider(
            [_openai_completion(None, refusal="I can't help with that.")]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "I can't help with that."

    async def test_truncated_empty_answer_is_an_error(self) -> None:
        provider, calls, _ = _openai_provider(
            [_openai_completion("", finish_reason="length")],
            max_budget_usd=1.0,
        )
        with pytest.raises(EmptyResponseError):
            await provider.query("prompt", "q1")
        assert len(calls) == 1
        # Billed: 100 input and 400 output tokens at gpt-4o prices
        assert provider._budget.spent == pytest.approx(100 * 2.5e-6 + 400 * 10e-6)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
def _anthropic_http() -> Any:
    """Return the HTTP package the Anthropic SDK is built on.

    anthropic 1.x uses ``httpx2`` and rejects ``httpx`` objects.
    """
    try:
        return importlib.import_module("httpx2")
    except ImportError:
        return httpx


def _anthropic_message(
    http: Any,
    blocks: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
) -> Any:
    return http.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 400},
        },
    )


def _anthropic_text(http: Any) -> Any:
    return _anthropic_message(http, [{"type": "text", "text": "B"}])


def _anthropic_error(
    http: Any, status: int, headers: dict[str, str] | None = None
) -> Any:
    return http.Response(
        status,
        json={"type": "error", "error": {"type": "api_error", "message": "failed"}},
        headers=headers,
    )


def _anthropic_provider(make_items: Any, **kwargs: Any) -> tuple[Any, list[Any], Any]:
    anthropic = pytest.importorskip("anthropic")
    http = _anthropic_http()
    module = importlib.import_module("llm_consistency.providers._anthropic")
    provider = module.AnthropicProvider(
        model="claude-haiku-4-5-20251001",
        api_key="test-key",
        **{**_FAST_RETRY, **kwargs},
    )
    transport, calls = _replay(http, make_items(http))
    provider._client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="http://test",
        max_retries=0,
        http_client=http.AsyncClient(transport=transport),
    )
    return provider, calls, anthropic


class TestAnthropicErrors:
    @pytest.mark.parametrize("status", [408, 409, 429, 500, 503, 529])
    async def test_transient_status_is_retried(self, status: int) -> None:
        provider, calls, _ = _anthropic_provider(
            lambda http: [_anthropic_error(http, status), _anthropic_text(http)]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (400, "BadRequestError"),
            (401, "AuthenticationError"),
            (403, "PermissionDeniedError"),
            (404, "NotFoundError"),
        ],
    )
    async def test_client_error_is_not_retried(self, status: int, error: str) -> None:
        provider, calls, anthropic = _anthropic_provider(
            lambda http: [_anthropic_error(http, status), _anthropic_text(http)]
        )
        with pytest.raises(getattr(anthropic, error)):
            await provider.query("prompt", "q1")
        assert len(calls) == 1

    @pytest.mark.parametrize("error", ["ConnectError", "ReadTimeout"])
    async def test_network_failure_is_retried(self, error: str) -> None:
        provider, calls, _ = _anthropic_provider(
            lambda http: [getattr(http, error)("down"), _anthropic_text(http)]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    async def test_overloaded_raises_after_retries(self) -> None:
        provider, calls, anthropic = _anthropic_provider(
            lambda http: [_anthropic_error(http, 529)] * 4
        )
        with pytest.raises(anthropic.OverloadedError):
            await provider.query("prompt", "q1")
        assert len(calls) == 4

    async def test_retry_after_is_honoured(self) -> None:
        provider, _, _ = _anthropic_provider(
            lambda http: [
                _anthropic_error(http, 529, {"retry-after": "0.25"}),
                _anthropic_text(http),
            ],
            base_delay=30.0,
        )
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await provider.query("prompt", "q1")
        sleep.assert_awaited_once_with(0.25)

    async def test_thinking_block_is_skipped(self) -> None:
        provider, _, _ = _anthropic_provider(
            lambda http: [
                _anthropic_message(
                    http,
                    [
                        {"type": "thinking", "thinking": "B fits.", "signature": "s"},
                        {"type": "text", "text": "Answer: "},
                        {"type": "text", "text": "B"},
                    ],
                ),
            ]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "Answer: B"

    async def test_max_tokens_before_text_is_an_error(self) -> None:
        provider, calls, _ = _anthropic_provider(
            lambda http: [
                _anthropic_message(
                    http,
                    [{"type": "thinking", "thinking": "Hmm", "signature": "s"}],
                    stop_reason="max_tokens",
                ),
            ]
        )
        with pytest.raises(EmptyResponseError):
            await provider.query("prompt", "q1")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------
@pytest.fixture
def litellm_via(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Route ``litellm.acompletion`` through an OpenAI client on a mock transport.

    LiteLLM then maps the OpenAI SDK's errors to its own, as in production.
    """
    litellm = pytest.importorskip("litellm")
    openai = pytest.importorskip("openai")
    monkeypatch.setattr(litellm, "suppress_debug_info", True)

    def make(items: list[Any], **kwargs: Any) -> tuple[Any, list[Any]]:
        transport, calls = _replay(httpx, items)
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )
        monkeypatch.setattr(
            litellm,
            "acompletion",
            functools.partial(litellm.acompletion, client=client, num_retries=0),
        )
        module = importlib.import_module("llm_consistency.providers._litellm")
        provider = module.LiteLLMProvider(
            model="openai/gpt-4o", **{**_FAST_RETRY, **kwargs}
        )
        return provider, calls

    return make


class TestLiteLLMErrors:
    @pytest.mark.parametrize("status", [408, 429, 500, 503])
    async def test_transient_status_is_retried(
        self, litellm_via: Any, status: int
    ) -> None:
        provider, calls = litellm_via([_openai_error(status), _openai_completion()])
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    async def test_client_error_is_not_retried(
        self, litellm_via: Any, status: int
    ) -> None:
        provider, calls = litellm_via([_openai_error(status), _openai_completion()])
        with pytest.raises(Exception) as exc:
            await provider.query("prompt", "q1")
        assert exc.value.status_code == status  # type: ignore[attr-defined]
        assert len(calls) == 1

    async def test_network_failure_is_retried(self, litellm_via: Any) -> None:
        provider, calls = litellm_via(
            [httpx.ConnectError("down"), _openai_completion()]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    async def test_retry_after_is_honoured(self, litellm_via: Any) -> None:
        """LiteLLM keeps the upstream headers only on litellm_response_headers."""
        provider, _ = litellm_via(
            [_openai_error(429, {"retry-after": "0.25"}), _openai_completion()],
            base_delay=30.0,
        )
        with patch(f"{_RETRY_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await provider.query("prompt", "q1")
        sleep.assert_awaited_once_with(0.25)

    async def test_refusal_is_the_answer_text(self, litellm_via: Any) -> None:
        provider, _ = litellm_via(
            [_openai_completion(None, refusal="I can't help with that.")]
        )
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "I can't help with that."

    async def test_truncated_empty_answer_is_an_error(self, litellm_via: Any) -> None:
        provider, calls = litellm_via([_openai_completion("", finish_reason="length")])
        with pytest.raises(EmptyResponseError):
            await provider.query("prompt", "q1")
        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("name", "kwargs", "retryable"),
        [
            ("RateLimitError", {}, True),
            ("Timeout", {}, True),
            ("APIConnectionError", {}, True),
            ("ServiceUnavailableError", {}, True),
            ("InternalServerError", {}, True),
            ("APIError", {"status_code": 502}, True),
            ("AuthenticationError", {}, False),
            ("BadRequestError", {}, False),
            ("NotFoundError", {}, False),
            ("ContextWindowExceededError", {}, False),
            ("APIError", {"status_code": 400}, False),
        ],
    )
    def test_exception_classes(
        self, litellm_via: Any, name: str, kwargs: dict[str, Any], retryable: bool
    ) -> None:
        litellm = pytest.importorskip("litellm")
        provider, _ = litellm_via([])
        exc = getattr(litellm, name)(
            message="failed", llm_provider="openai", model="gpt-4o", **kwargs
        )
        assert provider._is_retryable(exc) is retryable


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _ollama_chat(content: str = "B", thinking: str | None = None) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return httpx.Response(
        200,
        json={
            "model": "llama3",
            "created_at": "2026-01-01T00:00:00Z",
            "message": message,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 2,
        },
    )


def _ollama_error(status: int) -> httpx.Response:
    return httpx.Response(status, json={"error": f"status {status}"})


def _ollama_provider(items: list[Any]) -> tuple[Any, list[Any], Any]:
    ollama = pytest.importorskip("ollama")
    module = importlib.import_module("llm_consistency.providers._ollama")
    provider = module.OllamaProvider(model="llama3", **_FAST_RETRY)
    transport, calls = _replay(httpx, items)
    provider._client = ollama.AsyncClient(host="http://test", transport=transport)
    return provider, calls, ollama


class TestOllamaErrors:
    @pytest.mark.parametrize("status", [429, 500, 503])
    async def test_transient_status_is_retried(self, status: int) -> None:
        provider, calls, _ = _ollama_provider([_ollama_error(status), _ollama_chat()])
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    @pytest.mark.parametrize("status", [400, 404])
    async def test_client_error_is_not_retried(self, status: int) -> None:
        provider, calls, ollama = _ollama_provider(
            [_ollama_error(status), _ollama_chat()]
        )
        with pytest.raises(ollama.ResponseError):
            await provider.query("prompt", "q1")
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "error", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError]
    )
    async def test_network_failure_is_retried(self, error: type[Exception]) -> None:
        provider, calls, _ = _ollama_provider([error("down"), _ollama_chat()])
        response = await provider.query("prompt", "q1")
        assert response.raw_output == "B"
        assert len(calls) == 2

    async def test_thinking_without_answer_is_an_error(self) -> None:
        provider, calls, _ = _ollama_provider([_ollama_chat("", thinking="B fits.")])
        with pytest.raises(EmptyResponseError):
            await provider.query("prompt", "q1")
        assert len(calls) == 1
