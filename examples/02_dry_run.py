"""Programmatic dry-run: validate the pipeline without spending tokens.

The `--dry-run` CLI flag generates one variant of the first MC question,
renders its prompt, and prints the estimated cost — all without calling
the provider. This example reproduces the same behaviour from Python,
which is useful when you want to wire dry-run validation into your own
tooling rather than shell out to the CLI.

We use the mock provider but a real *priced* model identifier so the
cost estimate prints a real number. Swap ``provider="openai"`` and the
arithmetic doesn't change.

Run from the repo root::

    uv run python examples/02_dry_run.py
"""

from __future__ import annotations

from pathlib import Path

from llm_consistency import (
    EvaluationConfig,
    MCDataset,
    MCQuestion,
    PerturbationType,
    estimate_cost,
    get_provider,
    get_scorer,
)
from llm_consistency.runners._pipeline import (
    generate_variants_for_question,
    render_prompt,
)


def dry_run(config: EvaluationConfig, dataset: MCDataset, seed: int = 42) -> None:
    mc_questions = [q for q in dataset if isinstance(q, MCQuestion)]
    if not mc_questions:
        msg = "Dataset has no MCQuestion items."
        raise ValueError(msg)

    sample = mc_questions[0]
    variants = generate_variants_for_question(sample, config, seed)
    sample_prompt = render_prompt(variants[0])

    num_calls = len(mc_questions) * config.num_variants
    estimated = estimate_cost(config.model, num_calls)
    cost_str = (
        f"~${estimated:.4f}"
        if estimated > 0
        else "unknown (model not in pricing table)"
    )

    print("Dry run — no provider calls made.")
    print(f"  model:                {config.model}")
    print(f"  provider:             {config.provider}")
    print(f"  scorer:               {config.scorer}")
    print(
        "  perturbations:        "
        + ", ".join(pt.value for pt in config.perturbation_types)
    )
    print(f"  questions (MC):       {len(mc_questions)}")
    print(f"  variants per Q:       {config.num_variants}")
    print(f"  total provider calls: {num_calls}")
    print(f"  estimated cost:       {cost_str}")
    print()
    print("Sample prompt (variant 0 of first question):")
    print("  " + sample_prompt.replace("\n", "\n  "))


def main() -> None:
    dataset = MCDataset.load(Path(__file__).parent / "datasets" / "sample.jsonl")
    config = EvaluationConfig(
        # gpt-5-mini is in the static pricing table, so estimate_cost
        # returns a non-zero number. Any priced model works here.
        model="gpt-5-mini",
        provider="mock",
        scorer="exact_match",
        perturbation_types=(
            PerturbationType.OPTION_REORDER,
            PerturbationType.SEPARATOR_CHANGE,
        ),
        num_variants=3,
    )

    # Validate the provider + scorer construct cleanly. Errors here mean
    # your config would fail on the first call, not after burning budget.
    _ = get_provider(config.provider, model=config.model)
    _ = get_scorer(config.scorer)

    dry_run(config, dataset)


if __name__ == "__main__":
    main()
