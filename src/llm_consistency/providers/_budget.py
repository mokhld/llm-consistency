"""Budget tracking and cost estimation for LLM provider requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llm_consistency._exceptions import LLMConsistencyError


class BudgetExceededError(LLMConsistencyError):
    """Raised when a request would exceed the configured budget ceiling.

    Attributes:
        spent: Total USD already spent, plus the amount reserved by
            requests still in flight.
        estimated: Estimated cost of the next request in USD.
        limit: Maximum budget ceiling in USD.
    """

    def __init__(
        self,
        spent: float,
        estimated: float,
        limit: float,
    ) -> None:
        self.spent = spent
        self.estimated = estimated
        self.limit = limit
        super().__init__(
            f"Budget exceeded: spent=${spent:.4f} + "
            f"estimated=${estimated:.4f} > limit=${limit:.4f}"
        )


@dataclass(frozen=True)
class CostPerToken:
    """Per-token cost rates for a model.

    Attributes:
        input_per_token: USD cost per input/prompt token.
        output_per_token: USD cost per output/completion token.
    """

    input_per_token: float
    output_per_token: float

    def estimate(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate total cost for the given token counts.

        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Estimated total cost in USD.
        """
        return (
            self.input_per_token * prompt_tokens
            + self.output_per_token * completion_tokens
        )


class BudgetTracker:
    """Tracks cumulative cost and enforces a budget ceiling.

    Each request reserves its estimated cost before it is sent and
    settles the reservation against the actual cost when it finishes.
    Because the check and the reservation happen under one lock,
    concurrent requests cannot all pass the check and then overshoot
    the ceiling together. When ``max_budget_usd`` is ``None``, no
    ceiling is enforced.

    Args:
        max_budget_usd: Maximum allowed spend in USD, or ``None``
            for unlimited budget.
    """

    def __init__(self, max_budget_usd: float | None) -> None:
        self._max = max_budget_usd
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = asyncio.Lock()

    @property
    def max_budget_usd(self) -> float | None:
        """The budget ceiling in USD, or ``None`` when unlimited."""
        return self._max

    async def reserve(self, estimated_cost: float) -> None:
        """Reserve ``estimated_cost`` if it fits within the budget.

        Args:
            estimated_cost: Estimated cost of the next request in USD.

        Raises:
            BudgetExceededError: If ``spent + reserved + estimated_cost``
                would exceed ``max_budget_usd``.
        """
        async with self._lock:
            committed = self._spent + self._reserved
            if self._max is not None and committed + estimated_cost > self._max:
                raise BudgetExceededError(committed, estimated_cost, self._max)
            self._reserved += estimated_cost

    async def settle(self, reserved: float, actual_cost: float) -> None:
        """Release a reservation and record what the request really cost.

        Pass ``actual_cost=0.0`` to release a reservation for a request
        that failed without being billed.

        Args:
            reserved: The amount previously passed to :meth:`reserve`.
            actual_cost: Actual cost of the request in USD.
        """
        async with self._lock:
            self._reserved -= reserved
            self._spent += actual_cost

    @property
    def spent(self) -> float:
        """Total USD spent so far."""
        return self._spent
