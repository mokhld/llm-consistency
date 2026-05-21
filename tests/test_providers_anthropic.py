"""Tests for AnthropicProvider -- fully mocked, no real SDK required."""

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


def _mock_anthropic_module() -> MagicMock:
    """Return a mock ``anthropic`` module with ``AsyncAnthropic``."""
    mod = MagicMock()
    mod.AsyncAnthropic = MagicMock()
    return mod


def _mock_response(
    *,
    text: str | None = "Answer: B",
    input_tokens: int = 10,
    output_tokens: int = 5,
    empty_content: bool = False,
) -> SimpleNamespace:
    """Build a mock response matching ``client.messages.create()``.

    Anthropic responses differ from OpenAI:
    - Content is ``response.content[0].text`` (not choices)
    - Tokens are ``input_tokens``/``output_tokens`` (not prompt_tokens)
    """
    if empty_content:
        content: list[object] = []
    else:
        content = [SimpleNamespace(text=text)]
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return SimpleNamespace(content=content, usage=usage)


def _make_provider(
    mock_mod: MagicMock,
    **kwargs: object,
) -> object:
    """Create an AnthropicProvider with a mocked anthropic module."""
    with patch.dict(sys.modules, {"anthropic": mock_mod}):
        mod = importlib.import_module("llm_consistency.providers._anthropic")
        importlib.reload(mod)
        return mod.AnthropicProvider(**{"model": "claude-sonnet-4-20250514", **kwargs})


# ---------------------------------------------------------------------------
# ImportError guard
# ---------------------------------------------------------------------------


class TestImportError:
    """Verify clear error when anthropic SDK is missing."""

    def test_import_error_message(self) -> None:
        with (
            patch.dict(sys.modules, {"anthropic": None}),
            pytest.raises(ImportError, match="llm-consistency\\[anthropic\\]"),
        ):
            mod = importlib.import_module(
                "llm_consistency.providers._anthropic",
            )
            importlib.reload(mod)
            mod.AnthropicProvider(model="claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Verify constructor wires AsyncAnthropic correctly."""

    def test_creates_client_with_max_retries_zero(self) -> None:
        mock_mod = _mock_anthropic_module()
        _make_provider(mock_mod)

        call_kwargs = mock_mod.AsyncAnthropic.call_args[1]
        assert call_kwargs["max_retries"] == 0

    def test_passes_api_key(self) -> None:
        mock_mod = _mock_anthropic_module()
        _make_provider(mock_mod, api_key="sk-ant-test")

        call_kwargs = mock_mod.AsyncAnthropic.call_args[1]
        assert call_kwargs["api_key"] == "sk-ant-test"

    @pytest.mark.asyncio
    async def test_max_tokens_defaults_to_1024(self) -> None:
        mock_mod = _mock_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_response(),
        )
        mock_mod.AsyncAnthropic.return_value = mock_client
        provider = _make_provider(mock_mod)

        await provider._send_request("prompt")  # type: ignore[union-attr]

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_max_tokens_can_be_overridden(self) -> None:
        mock_mod = _mock_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_response(),
        )
        mock_mod.AsyncAnthropic.return_value = mock_client
        provider = _make_provider(mock_mod, max_tokens=2048)

        await provider._send_request("prompt")  # type: ignore[union-attr]

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# provider_name
# ---------------------------------------------------------------------------


class TestProviderName:
    """Verify provider_name property."""

    def test_returns_anthropic(self) -> None:
        mock_mod = _mock_anthropic_module()
        provider = _make_provider(mock_mod)

        assert provider.provider_name == "anthropic"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    """Verify _send_request maps SDK responses to _RawResponse."""

    def _provider_with_mock_client(
        self,
        response: SimpleNamespace | None = None,
        **kwargs: object,
    ) -> tuple[object, MagicMock]:
        """Create a provider whose client.messages.create is mocked."""
        mock_mod = _mock_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=response or _mock_response(),
        )
        mock_mod.AsyncAnthropic.return_value = mock_client
        provider = _make_provider(mock_mod, **kwargs)
        return provider, mock_client

    @pytest.mark.asyncio
    async def test_maps_response_correctly(self) -> None:
        provider, _ = self._provider_with_mock_client(
            _mock_response(text="B", input_tokens=15, output_tokens=8),
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
    async def test_system_as_top_level_param(self) -> None:
        """Anthropic uses system as a top-level kwarg, NOT in messages."""
        provider, mock_client = self._provider_with_mock_client()

        await provider._send_request("prompt", system="Be helpful")  # type: ignore[union-attr]

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"
        # System should NOT be in messages
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "prompt"}

    @pytest.mark.asyncio
    async def test_no_system_param_when_none(self) -> None:
        """When system is None, it should not be in kwargs at all."""
        provider, mock_client = self._provider_with_mock_client()

        await provider._send_request("prompt")  # type: ignore[union-attr]

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "prompt"}

    @pytest.mark.asyncio
    async def test_maps_input_output_tokens(self) -> None:
        """Anthropic uses input_tokens/output_tokens, not prompt_tokens."""
        provider, _ = self._provider_with_mock_client(
            _mock_response(input_tokens=42, output_tokens=17),
        )

        raw = await provider._send_request("prompt")  # type: ignore[union-attr]

        assert raw.prompt_tokens == 42
        assert raw.completion_tokens == 17

    @pytest.mark.asyncio
    async def test_handles_empty_content(self) -> None:
        provider, _ = self._provider_with_mock_client(
            _mock_response(empty_content=True),
        )

        raw = await provider._send_request("prompt")  # type: ignore[union-attr]

        assert raw.content == ""
