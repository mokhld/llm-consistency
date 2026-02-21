"""Budget tracking and cost estimation for LLM provider requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llm_consistency._exceptions import LLMConsistencyError


class BudgetExceededError(LLMConsistencyError):
    """Raised when a request would exceed the configured budget ceiling.

    Attributes:
        spent: Total USD already spent.
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

    Uses ``asyncio.Lock`` for safe concurrent access during batch
    queries. When ``max_budget_usd`` is ``None``, no ceiling is enforced.

    Args:
        max_budget_usd: Maximum allowed spend in USD, or ``None``
            for unlimited budget.
    """

    def __init__(self, max_budget_usd: float | None) -> None:
        self._max = max_budget_usd
        self._spent = 0.0
        self._lock = asyncio.Lock()

    async def check(self, estimated_cost: float) -> None:
        """Check whether adding estimated_cost would exceed budget.

        Args:
            estimated_cost: Estimated cost of the next request in USD.

        Raises:
            BudgetExceededError: If ``spent + estimated_cost > max_budget_usd``.
        """
        async with self._lock:
            if self._max is not None and self._spent + estimated_cost > self._max:
                raise BudgetExceededError(
                    self._spent,
                    estimated_cost,
                    self._max,
                )

    async def record(self, actual_cost: float) -> None:
        """Record actual cost after a completed request.

        Args:
            actual_cost: Actual cost of the completed request in USD.
        """
        async with self._lock:
            self._spent += actual_cost

    @property
    def spent(self) -> float:
        """Total USD spent so far."""
        return self._spent
