"""Tests for llm_consistency.providers._budget module."""

from __future__ import annotations

import asyncio

import pytest

from llm_consistency._exceptions import LLMConsistencyError
from llm_consistency.providers._budget import (
    BudgetExceededError,
    BudgetTracker,
    CostPerToken,
)


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_is_subclass_of_llm_consistency_error(self) -> None:
        assert issubclass(BudgetExceededError, LLMConsistencyError)

    def test_stores_spent_estimated_and_limit(self) -> None:
        err = BudgetExceededError(spent=1.5, estimated=0.8, limit=2.0)
        assert err.spent == 1.5
        assert err.estimated == 0.8
        assert err.limit == 2.0

    def test_str_representation_is_clear(self) -> None:
        err = BudgetExceededError(spent=1.5, estimated=0.8, limit=2.0)
        msg = str(err)
        assert "1.5" in msg
        assert "0.8" in msg
        assert "2.0" in msg


class TestCostPerToken:
    """Tests for CostPerToken frozen dataclass."""

    def test_stores_input_and_output_per_token(self) -> None:
        cost = CostPerToken(
            input_per_token=0.00001,
            output_per_token=0.00003,
        )
        assert cost.input_per_token == 0.00001
        assert cost.output_per_token == 0.00003

    def test_is_frozen(self) -> None:
        cost = CostPerToken(
            input_per_token=0.00001,
            output_per_token=0.00003,
        )
        with pytest.raises(AttributeError):
            cost.input_per_token = 0.0  # type: ignore[misc]

    def test_estimate_computes_correct_cost(self) -> None:
        cost = CostPerToken(
            input_per_token=0.01,
            output_per_token=0.03,
        )
        result = cost.estimate(
            prompt_tokens=100,
            completion_tokens=50,
        )
        # 100 * 0.01 + 50 * 0.03 = 1.0 + 1.5 = 2.5
        assert result == pytest.approx(2.5)


class TestBudgetTracker:
    """Tests for BudgetTracker with ceiling enforcement."""

    async def test_unlimited_budget_never_raises(self) -> None:
        tracker = BudgetTracker(max_budget_usd=None)
        await tracker.reserve(estimated_cost=1000.0)
        await tracker.settle(reserved=1000.0, actual_cost=1000.0)
        await tracker.reserve(estimated_cost=1000.0)
        # No exception raised

    async def test_reserve_raises_when_exceeds_budget(self) -> None:
        tracker = BudgetTracker(max_budget_usd=2.0)
        await tracker.settle(reserved=0.0, actual_cost=1.5)
        with pytest.raises(BudgetExceededError) as exc_info:
            await tracker.reserve(estimated_cost=0.8)
        assert exc_info.value.spent == pytest.approx(1.5)
        assert exc_info.value.estimated == pytest.approx(0.8)
        assert exc_info.value.limit == pytest.approx(2.0)

    async def test_reservations_count_against_budget(self) -> None:
        """An unsettled reservation blocks a request that would overshoot."""
        tracker = BudgetTracker(max_budget_usd=1.0)
        await tracker.reserve(estimated_cost=0.6)
        with pytest.raises(BudgetExceededError) as exc_info:
            await tracker.reserve(estimated_cost=0.6)
        assert exc_info.value.spent == pytest.approx(0.6)
        assert tracker.spent == 0.0

    async def test_rejected_reservation_is_not_held(self) -> None:
        tracker = BudgetTracker(max_budget_usd=1.0)
        with pytest.raises(BudgetExceededError):
            await tracker.reserve(estimated_cost=2.0)
        await tracker.reserve(estimated_cost=1.0)

    async def test_settle_replaces_reservation_with_actual_cost(self) -> None:
        tracker = BudgetTracker(max_budget_usd=1.0)
        await tracker.reserve(estimated_cost=0.6)
        await tracker.settle(reserved=0.6, actual_cost=0.1)
        assert tracker.spent == pytest.approx(0.1)
        # 0.1 spent + 0.9 fits exactly
        await tracker.reserve(estimated_cost=0.9)

    async def test_settle_with_zero_cost_releases_reservation(self) -> None:
        tracker = BudgetTracker(max_budget_usd=1.0)
        await tracker.reserve(estimated_cost=1.0)
        await tracker.settle(reserved=1.0, actual_cost=0.0)
        assert tracker.spent == 0.0
        await tracker.reserve(estimated_cost=1.0)

    async def test_settle_accumulates_cost(self) -> None:
        tracker = BudgetTracker(max_budget_usd=10.0)
        await tracker.settle(reserved=0.0, actual_cost=1.0)
        await tracker.settle(reserved=0.0, actual_cost=2.5)
        assert tracker.spent == pytest.approx(3.5)

    async def test_spent_property_returns_accumulated(self) -> None:
        tracker = BudgetTracker(max_budget_usd=None)
        assert tracker.spent == 0.0
        await tracker.settle(reserved=0.0, actual_cost=5.0)
        assert tracker.spent == pytest.approx(5.0)

    async def test_reserve_passes_when_within_budget(self) -> None:
        tracker = BudgetTracker(max_budget_usd=10.0)
        await tracker.settle(reserved=0.0, actual_cost=3.0)
        await tracker.reserve(estimated_cost=5.0)
        # No exception -- 3.0 + 5.0 = 8.0 <= 10.0

    async def test_concurrent_reservations_never_overshoot(self) -> None:
        """Concurrent reserve-then-settle cycles stop exactly at the cap."""
        tracker = BudgetTracker(max_budget_usd=1.0)

        async def reserve_and_settle(cost: float) -> bool:
            try:
                await tracker.reserve(estimated_cost=cost)
            except BudgetExceededError:
                return False
            await asyncio.sleep(0)  # let the other tasks interleave
            await tracker.settle(reserved=cost, actual_cost=cost)
            return True

        results = await asyncio.gather(*(reserve_and_settle(0.1) for _ in range(50)))

        assert sum(results) == 10
        assert tracker.spent == pytest.approx(1.0)
