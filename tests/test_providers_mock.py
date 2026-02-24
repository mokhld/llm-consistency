"""Tests for MockLLMProvider -- deterministic testing provider."""

from __future__ import annotations

import pytest

from llm_consistency.providers import get_provider
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.types import LLMResponse


class TestMockProviderConstruction:
    """Construction and provider_name."""

    def test_default_construction(self) -> None:
        provider = MockLLMProvider(model="mock")
        assert isinstance(provider, MockLLMProvider)

    def test_provider_name(self) -> None:
        provider = MockLLMProvider(model="mock")
        assert provider.provider_name == "mock"


class TestDefaultResponse:
    """Default response mode."""

    @pytest.mark.asyncio
    async def test_default_returns_a(self) -> None:
        provider = MockLLMProvider(model="mock")
        response = await provider.query("any prompt", "q1")
        assert isinstance(response, LLMResponse)
        assert response.raw_output == "A"

    @pytest.mark.asyncio
    async def test_custom_default(self) -> None:
        provider = MockLLMProvider(model="mock", default_response="B")
        response = await provider.query("any prompt", "q1")
        assert response.raw_output == "B"


class TestResponseMapMode:
    """Response map mode -- question_id to answer mapping."""

    @pytest.mark.asyncio
    async def test_map_returns_correct_answer(self) -> None:
        provider = MockLLMProvider(
            model="mock",
            responses={"q1": "A", "q2": "B"},
        )
        r1 = await provider.query("prompt", "q1")
        assert r1.raw_output == "A"
        r2 = await provider.query("prompt", "q2")
        assert r2.raw_output == "B"

    @pytest.mark.asyncio
    async def test_map_unknown_falls_back_to_default(self) -> None:
        provider = MockLLMProvider(
            model="mock",
            responses={"q1": "A"},
            default_response="X",
        )
        response = await provider.query("prompt", "unknown")
        assert response.raw_output == "X"


class TestCyclingListMode:
    """Cycling list mode -- successive queries cycle through list."""

    @pytest.mark.asyncio
    async def test_cycling_through_list(self) -> None:
        provider = MockLLMProvider(
            model="mock",
            responses=["A", "B", "C"],
        )
        r1 = await provider.query("p1", "q1")
        assert r1.raw_output == "A"
        r2 = await provider.query("p2", "q2")
        assert r2.raw_output == "B"
        r3 = await provider.query("p3", "q3")
        assert r3.raw_output == "C"
        r4 = await provider.query("p4", "q4")
        assert r4.raw_output == "A"  # wraps around


class TestCostAndLatency:
    """Cost and latency reporting."""

    @pytest.mark.asyncio
    async def test_fixed_token_counts(self) -> None:
        provider = MockLLMProvider(model="mock")
        response = await provider.query("prompt", "q1")
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_near_zero_latency(self) -> None:
        provider = MockLLMProvider(model="mock")
        response = await provider.query("prompt", "q1")
        assert response.latency_ms is not None
        assert response.latency_ms < 100


class TestFactoryRegistration:
    """Factory registration via get_provider."""

    def test_get_provider_mock(self) -> None:
        provider = get_provider("mock", model="mock")
        assert isinstance(provider, MockLLMProvider)


class TestQueryBatch:
    """Batch query support."""

    @pytest.mark.asyncio
    async def test_batch_returns_correct_results(self) -> None:
        provider = MockLLMProvider(model="mock")
        result = await provider.query_batch([("p1", "q1"), ("p2", "q2")])
        assert len(result.responses) == 2
        assert len(result.errors) == 0
