"""Tests for static pricing table and estimate_cost function."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse
from llm_consistency.providers._budget import BudgetExceededError, CostPerToken
from llm_consistency.providers._cost import (
    MODEL_PRICING,
    estimate_cost,
    get_model_pricing,
)


# ---------------------------------------------------------------------------
# Tests: MODEL_PRICING table
# ---------------------------------------------------------------------------
class TestModelPricing:
    def test_contains_at_least_nine_models(self) -> None:
        """MODEL_PRICING has entries for all listed models."""
        assert len(MODEL_PRICING) >= 10

    def test_all_entries_are_cost_per_token(self) -> None:
        """Every value in MODEL_PRICING is a CostPerToken."""
        for model, pricing in MODEL_PRICING.items():
            assert isinstance(pricing, CostPerToken), f"{model} not CostPerToken"

    def test_all_entries_have_positive_costs(self) -> None:
        """Every CostPerToken has positive input and output costs."""
        for model, pricing in MODEL_PRICING.items():
            assert pricing.input_per_token > 0, f"{model} input <= 0"
            assert pricing.output_per_token > 0, f"{model} output <= 0"

    def test_known_models_present(self) -> None:
        """All specified models are in MODEL_PRICING."""
        expected = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-opus-4-20250514",
        ]
        for model in expected:
            assert model in MODEL_PRICING, f"{model} missing from MODEL_PRICING"


# ---------------------------------------------------------------------------
# Tests: estimate_cost()
# ---------------------------------------------------------------------------
class TestEstimateCost:
    def test_known_model_returns_expected_usd(self) -> None:
        """estimate_cost for gpt-4o returns expected USD value."""
        # gpt-4o: $2.50/1M input, $10.00/1M output
        # 10 prompts * (200 * 2.50/1M + 50 * 10.00/1M)
        # = 10 * (0.0005 + 0.0005) = 10 * 0.001 = 0.01
        result = estimate_cost("gpt-4o", num_prompts=10)
        assert result == pytest.approx(0.01)

    def test_unknown_model_returns_zero(self) -> None:
        """estimate_cost for unknown model returns 0.0."""
        result = estimate_cost("nonexistent-model-xyz", num_prompts=100)
        assert result == 0.0

    def test_custom_token_counts(self) -> None:
        """estimate_cost with custom token counts."""
        # gpt-4o: $2.50/1M input, $10.00/1M output
        # 1 prompt * (500 * 2.50/1M + 100 * 10.00/1M)
        # = 1 * (0.00125 + 0.001) = 0.00225
        result = estimate_cost(
            "gpt-4o",
            num_prompts=1,
            avg_prompt_tokens=500,
            avg_completion_tokens=100,
        )
        assert result == pytest.approx(0.00225)

    def test_zero_prompts_returns_zero(self) -> None:
        """estimate_cost with num_prompts=0 returns 0.0."""
        result = estimate_cost("gpt-4o", num_prompts=0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests: get_model_pricing()
# ---------------------------------------------------------------------------
class TestGetModelPricing:
    def test_known_model_returns_cost_per_token(self) -> None:
        """get_model_pricing for known model returns CostPerToken."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert isinstance(pricing, CostPerToken)

    def test_unknown_model_returns_none(self) -> None:
        """get_model_pricing for unknown model returns None."""
        assert get_model_pricing("nonexistent-model") is None


# ---------------------------------------------------------------------------
# Helper: concrete mock provider for cost integration tests
# ---------------------------------------------------------------------------
class _CostTestProvider(BaseLLMProvider):
    """Concrete test provider for cost integration tests."""

    def __init__(
        self,
        *,
        content: str = "Answer: A",
        prompt_tokens: int | None = 200,
        completion_tokens: int | None = 50,
        latency_ms: float = 10.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._raw = _RawResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        return self._raw


# ---------------------------------------------------------------------------
# Tests: BaseLLMProvider.query() cost integration
# ---------------------------------------------------------------------------
class TestBaseLLMProviderCostIntegration:
    """BaseLLMProvider.query() wires actual_cost via get_model_pricing."""

    @pytest.mark.asyncio
    async def test_query_records_nonzero_cost_for_known_model(
        self,
    ) -> None:
        """query() records positive actual_cost for known model."""
        provider = _CostTestProvider(
            model="gpt-4o",
            max_budget_usd=10.0,
        )
        with patch.object(
            provider._budget,
            "record",
            new_callable=AsyncMock,
        ) as mock_record:
            await provider.query("test prompt", question_id="q1")
            mock_record.assert_called_once()
            actual_cost = mock_record.call_args.kwargs["actual_cost"]
            assert actual_cost > 0.0, f"Expected positive cost, got {actual_cost}"

    @pytest.mark.asyncio
    async def test_query_records_zero_cost_for_unknown_model(
        self,
    ) -> None:
        """query() records actual_cost=0.0 for unknown model."""
        provider = _CostTestProvider(
            model="unknown-model-xyz",
            max_budget_usd=10.0,
        )
        with patch.object(
            provider._budget,
            "record",
            new_callable=AsyncMock,
        ) as mock_record:
            await provider.query("test prompt", question_id="q1")
            mock_record.assert_called_once()
            actual_cost = mock_record.call_args.kwargs["actual_cost"]
            assert actual_cost == 0.0, f"Expected 0.0, got {actual_cost}"

    @pytest.mark.asyncio
    async def test_query_records_zero_cost_when_tokens_none(
        self,
    ) -> None:
        """query() records actual_cost=0.0 when tokens are None."""
        provider = _CostTestProvider(
            model="gpt-4o",
            prompt_tokens=None,
            completion_tokens=None,
            max_budget_usd=10.0,
        )
        with patch.object(
            provider._budget,
            "record",
            new_callable=AsyncMock,
        ) as mock_record:
            await provider.query("test prompt", question_id="q1")
            mock_record.assert_called_once()
            actual_cost = mock_record.call_args.kwargs["actual_cost"]
            assert actual_cost == 0.0, f"Expected 0.0 (None tokens), got {actual_cost}"

    @pytest.mark.asyncio
    async def test_budget_exceeded_with_known_model_pricing(
        self,
    ) -> None:
        """BudgetExceededError when costs accumulate past budget."""
        # gpt-4o: 200*2.50/1M + 50*10.00/1M = 0.001 per query
        # Budget = 0.0015, first succeeds, second should fail
        provider = _CostTestProvider(
            model="gpt-4o",
            max_budget_usd=0.0015,
        )

        # First query should succeed
        await provider.query("test prompt", question_id="q1")

        # Second query should fail -- budget exceeded
        with pytest.raises(BudgetExceededError):
            await provider.query("test prompt", question_id="q2")
