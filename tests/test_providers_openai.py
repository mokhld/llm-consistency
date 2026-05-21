"""Tests for OpenAIProvider -- fully mocked, no real SDK required."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_consistency.providers._base import _RawResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_module() -> MagicMock:
    """Return a mock ``openai`` module with ``AsyncOpenAI``."""
    mod = MagicMock()
    mod.AsyncOpenAI = MagicMock()
    return mod


def _mock_response(
    *,
    content: str | None = "Answer: B",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
) -> SimpleNamespace:
    """Build a mock response matching ``client.chat.completions.create()``."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    if prompt_tokens is not None or completion_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    else:
        usage = None
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_provider(
    mock_mod: MagicMock,
    **kwargs: object,
) -> object:
    """Create an OpenAIProvider with a mocked openai module."""
    with patch.dict(sys.modules, {"openai": mock_mod}):
        mod = importlib.import_module("llm_consistency.providers._openai")
        importlib.reload(mod)
        return mod.OpenAIProvider(**{"model": "gpt-4o", **kwargs})


# ---------------------------------------------------------------------------
# ImportError guard
# ---------------------------------------------------------------------------


class TestImportError:
    """Verify clear error when openai SDK is missing."""

    def test_import_error_message(self) -> None:
        with (
            patch.dict(sys.modules, {"openai": None}),
            pytest.raises(ImportError, match="llm-consistency\\[openai\\]"),
        ):
            mod = importlib.import_module("llm_consistency.providers._openai")
            importlib.reload(mod)
            mod.OpenAIProvider(model="gpt-4o")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Verify constructor wires AsyncOpenAI correctly."""

    def test_creates_client_with_max_retries_zero(self) -> None:
        mock_mod = _mock_openai_module()
        _make_provider(mock_mod)

        call_kwargs = mock_mod.AsyncOpenAI.call_args[1]
        assert call_kwargs["max_retries"] == 0

    def test_passes_api_key_and_base_url(self) -> None:
        mock_mod = _mock_openai_module()
        _make_provider(
            mock_mod,
            api_key="sk-test",
            base_url="https://custom.api/v1",
        )

        call_kwargs = mock_mod.AsyncOpenAI.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["base_url"] == "https://custom.api/v1"


# ---------------------------------------------------------------------------
# provider_name
# ---------------------------------------------------------------------------


class TestProviderName:
    """Verify provider_name property."""

    def test_returns_openai(self) -> None:
        mock_mod = _mock_openai_module()
        provider = _make_provider(mock_mod)

        assert provider.provider_name == "openai"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    """Verify _send_request maps SDK responses to _RawResponse."""

    def _provider_with_mock_client(
        self,
        response: SimpleNamespace | None = None,
    ) -> tuple[object, MagicMock]:
        """Create a provider whose client.chat.completions.create is mocked."""
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=response or _mock_response(),
        )
        mock_mod.AsyncOpenAI.return_value = mock_client
        provider = _make_provider(mock_mod)
        return provider, mock_client

    @pytest.mark.asyncio
    async def test_maps_response_correctly(self) -> None:
        provider, _ = self._provider_with_mock_client(
            _mock_response(content="B", prompt_tokens=15, completion_tokens=8),
        )

        raw = await provider._send_request("What is 2+2?")  # type: ignore[union-attr]

        assert isinstance(raw, _RawResponse)
        assert raw.content == "B"
        assert raw.prompt_tokens == 15
        assert raw.completion_tokens == 8
        # time.monotonic resolution is coarse on Windows (often 16 ms);
        # an instant mock call can yield exactly 0.0. Allow >= 0.
        assert raw.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_builds_messages_with_system(self) -> None:
        provider, mock_client = self._provider_with_mock_client()

        await provider._send_request("prompt", system="Be helpful")  # type: ignore[union-attr]

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "prompt"}

    @pytest.mark.asyncio
    async def test_builds_messages_without_system(self) -> None:
        provider, mock_client = self._provider_with_mock_client()

        await provider._send_request("prompt")  # type: ignore[union-attr]

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "prompt"}

    @pytest.mark.asyncio
    async def test_handles_none_usage(self) -> None:
        provider, _ = self._provider_with_mock_client(
            _mock_response(prompt_tokens=None, completion_tokens=None),
        )

        raw = await provider._send_request("prompt")  # type: ignore[union-attr]

        assert raw.prompt_tokens is None
        assert raw.completion_tokens is None

    @pytest.mark.asyncio
    async def test_handles_none_content(self) -> None:
        provider, _ = self._provider_with_mock_client(
            _mock_response(content=None),
        )

        raw = await provider._send_request("prompt")  # type: ignore[union-attr]

        assert raw.content == ""
