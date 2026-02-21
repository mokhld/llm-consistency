"""Static pricing table and cost estimation for LLM models.

Provides per-model pricing data and helper functions for estimating
and computing costs before and after API calls.
"""

from __future__ import annotations

from llm_consistency.providers._budget import CostPerToken

MODEL_PRICING: dict[str, CostPerToken] = {
    # OpenAI models -- prices as of February 2026 (USD per token)
    "gpt-4o": CostPerToken(
        input_per_token=2.50 / 1_000_000,
        output_per_token=10.00 / 1_000_000,
    ),
    "gpt-4o-mini": CostPerToken(
        input_per_token=0.15 / 1_000_000,
        output_per_token=0.60 / 1_000_000,
    ),
    "gpt-4.1": CostPerToken(
        input_per_token=2.00 / 1_000_000,
        output_per_token=8.00 / 1_000_000,
    ),
    "gpt-4.1-mini": CostPerToken(
        input_per_token=0.40 / 1_000_000,
        output_per_token=1.60 / 1_000_000,
    ),
    "gpt-4.1-nano": CostPerToken(
        input_per_token=0.10 / 1_000_000,
        output_per_token=0.40 / 1_000_000,
    ),
    # Anthropic models -- prices as of February 2026 (USD per token)
    "claude-sonnet-4-20250514": CostPerToken(
        input_per_token=3.00 / 1_000_000,
        output_per_token=15.00 / 1_000_000,
    ),
    "claude-3-5-sonnet-20241022": CostPerToken(
        input_per_token=3.00 / 1_000_000,
        output_per_token=15.00 / 1_000_000,
    ),
    "claude-3-5-haiku-20241022": CostPerToken(
        input_per_token=0.80 / 1_000_000,
        output_per_token=4.00 / 1_000_000,
    ),
    "claude-opus-4-20250514": CostPerToken(
        input_per_token=15.00 / 1_000_000,
        output_per_token=75.00 / 1_000_000,
    ),
}
"""Static pricing table mapping model identifiers to per-token costs.

All prices are in USD per token, derived from per-million-token
published pricing.
"""


def estimate_cost(
    model: str,
    num_prompts: int,
    avg_prompt_tokens: int = 200,
    avg_completion_tokens: int = 50,
) -> float:
    """Estimate total USD cost for a batch of prompts.

    Looks up the model in :data:`MODEL_PRICING` and computes the
    estimated cost based on average token counts.  Returns ``0.0``
    for unknown models (no error).

    Args:
        model: LLM model identifier (e.g., ``"gpt-4o"``).
        num_prompts: Number of prompts to estimate for.
        avg_prompt_tokens: Average input tokens per prompt.
        avg_completion_tokens: Average output tokens per prompt.

    Returns:
        Estimated total cost in USD, or ``0.0`` for unknown models.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    return num_prompts * (
        pricing.input_per_token * avg_prompt_tokens
        + pricing.output_per_token * avg_completion_tokens
    )


def get_model_pricing(model: str) -> CostPerToken | None:
    """Look up per-token pricing for a model.

    Args:
        model: LLM model identifier.

    Returns:
        A :class:`CostPerToken` if the model is known, otherwise
        ``None``.
    """
    return MODEL_PRICING.get(model)
