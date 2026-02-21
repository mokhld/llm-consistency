"""Tests for OllamaProvider thin adapter.

All tests use mocked SDK -- no real Ollama server or SDK required.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_consistency.providers._base import _RawResponse


def _import_ollama_provider() -> type:
    """Import OllamaProvider via importlib to satisfy PLC0415."""
    mod = importlib.import_module("llm_consistency.providers._ollama")
    return mod.OllamaProvider  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Helper: create a mock ollama SDK module for injection
# ---------------------------------------------------------------------------
def _make_mock_ollama_module(
    *,
    chat_response: dict[str, Any] | None = None,
) -> tuple[types.ModuleType, AsyncMock]:
    """Return (mock_module, mock_async_client_instance)."""
    mock_module = types.ModuleType("ollama")

    # Default chat response mimicking Ollama dict-style format
    if chat_response is None:
        chat_response = {
            "message": {"content": "The answer is B"},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=chat_response)

    mock_async_client_cls = MagicMock(return_value=mock_client)
    mock_module.AsyncClient = mock_async_client_cls  # type: ignore[attr-defined]

    return mock_module, mock_client


# ---------------------------------------------------------------------------
# Tests: ImportError when ollama is not installed
# ---------------------------------------------------------------------------
class TestOllamaImportError:
    def test_import_error_when_sdk_missing(self) -> None:
        """OllamaProvider raises ImportError with clear message."""
        with patch.dict(sys.modules, {"ollama": None}):
            cls = _import_ollama_provider()
            with pytest.raises(
                ImportError,
                match="llm-consistency\\[ollama\\]",
            ):
                cls(model="llama3")


# ---------------------------------------------------------------------------
# Tests: Constructor behavior
# ---------------------------------------------------------------------------
class TestOllamaConstructor:
    def test_creates_async_client_no_host(self) -> None:
        """AsyncClient() called with no args when host is None."""
        mock_module, _ = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            mock_module.AsyncClient.assert_called_once_with()  # type: ignore[attr-defined]
            assert provider._model == "llama3"

    def test_creates_async_client_with_host(self) -> None:
        """AsyncClient(host=...) called when host provided."""
        mock_module, _ = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            cls(model="llama3", host="http://localhost:11434")
            mock_module.AsyncClient.assert_called_once_with(  # type: ignore[attr-defined]
                host="http://localhost:11434",
            )


# ---------------------------------------------------------------------------
# Tests: provider_name property
# ---------------------------------------------------------------------------
class TestOllamaProviderName:
    def test_provider_name_returns_ollama(self) -> None:
        mock_module, _ = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            assert provider.provider_name == "ollama"


# ---------------------------------------------------------------------------
# Tests: _send_request behavior
# ---------------------------------------------------------------------------
class TestOllamaSendRequest:
    @pytest.mark.asyncio
    async def test_builds_messages_without_system(self) -> None:
        """Only user message when system is None."""
        mock_module, mock_client = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            await provider._send_request("What is 2+2?")
            mock_client.chat.assert_called_once()
            call_kwargs = mock_client.chat.call_args
            messages = call_kwargs.kwargs["messages"]
            assert len(messages) == 1
            assert messages[0] == {
                "role": "user",
                "content": "What is 2+2?",
            }

    @pytest.mark.asyncio
    async def test_builds_messages_with_system(self) -> None:
        """System + user messages when system is provided."""
        mock_module, mock_client = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            await provider._send_request(
                "What is 2+2?",
                system="Be concise",
            )
            call_kwargs = mock_client.chat.call_args
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
    async def test_maps_token_fields(self) -> None:
        """prompt_eval_count/eval_count map correctly."""
        chat_response = {
            "message": {"content": "B"},
            "prompt_eval_count": 25,
            "eval_count": 12,
        }
        mock_module, _ = _make_mock_ollama_module(
            chat_response=chat_response,
        )
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            raw = await provider._send_request("prompt")
            assert isinstance(raw, _RawResponse)
            assert raw.prompt_tokens == 25
            assert raw.completion_tokens == 12

    @pytest.mark.asyncio
    async def test_handles_missing_token_fields(self) -> None:
        """Missing token fields return None."""
        chat_response: dict[str, Any] = {
            "message": {"content": "answer"},
        }
        mock_module, _ = _make_mock_ollama_module(
            chat_response=chat_response,
        )
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            raw = await provider._send_request("prompt")
            assert raw.prompt_tokens is None
            assert raw.completion_tokens is None

    @pytest.mark.asyncio
    async def test_extracts_content_dict_style(self) -> None:
        """Content via response['message']['content']."""
        chat_response = {
            "message": {"content": "The answer is C"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_module, _ = _make_mock_ollama_module(
            chat_response=chat_response,
        )
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            raw = await provider._send_request("prompt")
            assert raw.content == "The answer is C"

    @pytest.mark.asyncio
    async def test_records_latency(self) -> None:
        """latency_ms is a positive float."""
        mock_module, _ = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="llama3")
            raw = await provider._send_request("prompt")
            assert isinstance(raw.latency_ms, float)
            assert raw.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_passes_model_to_chat(self) -> None:
        """Model string forwarded to client.chat()."""
        mock_module, mock_client = _make_mock_ollama_module()
        with patch.dict(sys.modules, {"ollama": mock_module}):
            cls = _import_ollama_provider()
            provider = cls(model="codellama:7b")
            await provider._send_request("prompt")
            call_kwargs = mock_client.chat.call_args
            assert call_kwargs.kwargs["model"] == "codellama:7b"
