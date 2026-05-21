"""A/B two MockLLMProvider instances on the same dataset.

The CLI's ``llm-consistency compare`` subcommand runs N models on one
dataset and writes per-model report files. This example does the same
from Python: configure two providers, run them through BatchRunner,
and print a side-by-side comparison of the headline metrics.

The two providers below differ in *consistency*: provider A returns
the same answer for every variant of each question (high RC_agree,
high RC_correct), while provider B cycles through A/B/C/D
(low RC_agree, low RC_correct). The CORE index reflects this gap.

Run from the repo root::

    uv run python examples/06_compare_models.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _helpers import BaseIdMockProvider

from llm_consistency import (
    BatchRunner,
    EvaluationConfig,
    EvaluationReport,
    ExactMatchScorer,
    MCDataset,
    PerturbationType,
    agreement_gated_accuracy,
    core_index,
    mca,
)


async def evaluate(
    model: str,
    responses: dict[str, str] | list[str],
) -> EvaluationReport:
    dataset = MCDataset.load(Path(__file__).parent / "datasets" / "sample.jsonl")
    config = EvaluationConfig(
        model=model,
        provider="mock",
        scorer="exact_match",
        perturbation_types=(PerturbationType.FORMAT_CHANGE,),
        num_variants=4,
        concurrency=4,
    )
    provider = BaseIdMockProvider(model=model, responses=responses)
    runner = BatchRunner()
    return await runner.run(dataset, config, provider, ExactMatchScorer(), seed=42)


async def main() -> None:
    # Model A: stable answers, knows the correct label for each question.
    report_a = await evaluate(
        "stable-model",
        responses={"q1": "B", "q2": "C", "q3": "B", "q4": "C", "q5": "C"},
    )

    # Model B: cycles A/B/C/D regardless of question — high variance,
    # so it'll get some right by coincidence but RC_agree collapses.
    report_b = await evaluate(
        "unstable-model",
        responses=["A", "B", "C", "D"],
    )

    print(f"{'Metric':<22} {'stable':>10} {'unstable':>10}  Δ")
    print("-" * 56)
    rows: list[tuple[str, float, float]] = [
        ("mean RC_correct", report_a.mean_rc_correct, report_b.mean_rc_correct),
        ("mean RC_agree", report_a.mean_rc_agree, report_b.mean_rc_agree),
        ("CORE index", core_index(report_a.results), core_index(report_b.results)),
        ("MCA(0.8)", mca(report_a.results, 0.8), mca(report_b.results, 0.8)),
        (
            "AGA(0.8)",
            agreement_gated_accuracy(report_a.results, tau_agree=0.8),
            agreement_gated_accuracy(report_b.results, tau_agree=0.8),
        ),
    ]
    for label, a, b in rows:
        delta = a - b
        print(f"{label:<22} {a:>10.4f} {b:>10.4f}  {delta:+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
