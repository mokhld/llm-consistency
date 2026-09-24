"""Tests for LiteLLMProvider thin adapter.

All tests use mocked SDK -- no real LiteLLM or API required.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_consistency.providers._base import EmptyResponseError, _RawResponse


@pytest.fixture(autouse=True)
def _stub_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for openai, whose exception bases LiteLLMProvider imports."""
    stub = types.ModuleType("openai")
    stub.APIError = type("APIError", (Exception,), {})  # type: ignore[attr-defined]
    stub.APIConnectionError = type(  # type: ignore[attr-defined]
        "APIConnectionError", (stub.APIError,), {}
    )
    monkeypatch.setitem(sys.modules, "openai", stub)


def _import_litellm_provider() -> type:
    """Import LiteLLMProvider via importlib to satisfy PLC0415."""
    mod = importlib.import_module("llm_consistency.providers._litellm")
    return mod.LiteLLMProvider  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Helper: mock litellm module with acompletion()
# ---------------------------------------------------------------------------
def _make_mock_litellm_module(
    *,
    content: str | None = "The answer is A",
    prompt_tokens: int | None = 20,
    completion_tokens: int | None = 10,
    has_usage: bool = True,
    provider_specific_fields: dict[str, str] | None = None,
    finish_reason: str = "stop",
) -> tuple[types.ModuleType, AsyncMock]:
    """Return (mock_module, mock_acompletion)."""
    mock_module = types.ModuleType("litellm")

    # Build mock response in OpenAI-compatible format
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.provider_specific_fields = provider_specific_fields

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    if has_usage:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_response.usage = mock_usage
    else:
        mock_response.usage = None

    mock_acompletion = AsyncMock(return_value=mock_response)
    mock_module.acompletion = mock_acompletion  # type: ignore[attr-defined]

    return mock_module, mock_acompletion


# ---------------------------------------------------------------------------
# Tests: ImportError when litellm is not installed
# ---------------------------------------------------------------------------
class TestLiteLLMImportError:
    def test_import_error_when_sdk_missing(self) -> None:
        """LiteLLMProvider raises ImportError with clear message."""
        with patch.dict(sys.modules, {"litellm": None}):
            cls = _import_litellm_provider()
            with pytest.raises(
                ImportError,
                match="llm-consistency\\[litellm\\]",
            ):
                cls(model="anthropic/claude-3-5-sonnet")


# ---------------------------------------------------------------------------
# Tests: Constructor behavior
# ---------------------------------------------------------------------------
class TestLiteLLMConstructor:
    def test_constructor_succeeds(self) -> None:
        """Constructor succeeds when litellm is importable."""
        mock_module, _ = _make_mock_litellm_module()
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="anthropic/claude-3-5-sonnet")
            assert provider._model == "anthropic/claude-3-5-sonnet"


# ---------------------------------------------------------------------------
# Tests: provider_name property
# ---------------------------------------------------------------------------
class TestLiteLLMProviderName:
    def test_provider_name_returns_litellm(self) -> None:
        mock_module, _ = _make_mock_litellm_module()
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="anthropic/claude-3-5-sonnet")
            assert provider.provider_name == "litellm"


# ---------------------------------------------------------------------------
# Tests: _send_request behavior
# ---------------------------------------------------------------------------
class TestLiteLLMSendRequest:
    @pytest.mark.asyncio
    async def test_calls_acompletion_with_model(self) -> None:
        """acompletion() called with correct model string."""
        mock_module, mock_acompletion = _make_mock_litellm_module()
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="anthropic/claude-3-5-sonnet")
            await provider._send_request("prompt")
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args
            assert call_kwargs.kwargs["model"] == "anthropic/claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_builds_messages_without_system(self) -> None:
        """Only user message when system is None."""
        mock_module, mock_acompletion = _make_mock_litellm_module()
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="ollama/llama3")
            await provider._send_request("What is 2+2?")
            call_kwargs = mock_acompletion.call_args
            messages = call_kwargs.kwargs["messages"]
            assert len(messages) == 1
            assert messages[0] == {
                "role": "user",
                "content": "What is 2+2?",
            }

    @pytest.mark.asyncio
    async def test_builds_messages_with_system(self) -> None:
        """System + user messages when system is provided."""
        mock_module, mock_acompletion = _make_mock_litellm_module()
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="ollama/llama3")
            await provider._send_request(
                "What is 2+2?",
                system="Be concise",
            )
            call_kwargs = mock_acompletion.call_args
            messages = call_kwargs.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0] == {
                "role": "system",
                "content": "Be concise",
            }
            assert messages[1] == {
                "role": "user",
                "content": "What is 2+2?",
            }

    @pytest.mark.asyncio
    async def test_maps_openai_compatible_response(self) -> None:
        """OpenAI-compatible response mapped to _RawResponse."""
        mock_module, _ = _make_mock_litellm_module(
            content="Answer: B",
            prompt_tokens=30,
            completion_tokens=15,
        )
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="openai/gpt-4o")
            raw = await provider._send_request("prompt")
            assert isinstance(raw, _RawResponse)
            assert raw.content == "Answer: B"
            assert raw.prompt_tokens == 30
            assert raw.completion_tokens == 15

    @pytest.mark.asyncio
    async def test_handles_none_usage(self) -> None:
        """None usage returns None for token counts."""
        mock_module, _ = _make_mock_litellm_module(has_usage=False)
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="openai/gpt-4o")
            raw = await provider._send_request("prompt")
            assert raw.prompt_tokens is None
            assert raw.completion_tokens is None

    @pytest.mark.asyncio
    async def test_handles_empty_content(self) -> None:
        """None content becomes empty string."""
        mock_module, _ = _make_mock_litellm_module()
        # Override content to None after creating mock
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="openai/gpt-4o")
            # Manually set content to None on the mock
            mock_acompletion = mock_module.acompletion  # type: ignore[attr-defined]
            resp = await mock_acompletion()
            resp.choices[0].message.content = None
            mock_acompletion.reset_mock()
            mock_acompletion.return_value = resp
            raw = await provider._send_request("prompt")
            assert raw.content == ""

    @pytest.mark.asyncio
    async def test_records_latency(self) -> None:
        """latency_ms is a positive float."""
        mock_module, _ = _make_mock_litellm_module()
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="openai/gpt-4o")
            raw = await provider._send_request("prompt")
            assert isinstance(raw.latency_ms, float)
            assert raw.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_refusal_returned_as_content(self) -> None:
        """LiteLLM carries an OpenAI refusal in provider_specific_fields."""
        mock_module, _ = _make_mock_litellm_module(
            content=None,
            provider_specific_fields={"refusal": "I can't help with that."},
        )
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="openai/gpt-4o")
            raw = await provider._send_request("prompt")
            assert raw.content == "I can't help with that."

    @pytest.mark.asyncio
    async def test_truncated_before_text_raises(self) -> None:
        mock_module, _ = _make_mock_litellm_module(
            content="",
            finish_reason="length",
            prompt_tokens=12,
            completion_tokens=256,
        )
        with patch.dict(sys.modules, {"litellm": mock_module}):
            cls = _import_litellm_provider()
            provider = cls(model="openai/gpt-4o")
            with pytest.raises(EmptyResponseError) as exc:
                await provider._send_request("prompt")
            assert exc.value.prompt_tokens == 12
            assert exc.value.completion_tokens == 256
