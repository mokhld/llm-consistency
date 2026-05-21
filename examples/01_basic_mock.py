"""Minimal end-to-end evaluation against MockLLMProvider.

Loads the bundled sample dataset, builds an EvaluationConfig, runs the
full pipeline through the mock provider, and displays the resulting
metrics. No API keys or network calls required.

Run from the repo root::

    uv run python examples/01_basic_mock.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _helpers import BaseIdMockProvider

from llm_consistency import (
    BatchRunner,
    ConsoleReporter,
    EvaluationConfig,
    ExactMatchScorer,
    MCDataset,
    PerturbationType,
    core_index,
    mca,
)


async def main() -> None:
    dataset_path = Path(__file__).parent / "datasets" / "sample.jsonl"
    dataset = MCDataset.load(dataset_path)

    config = EvaluationConfig(
        model="mock-model",
        provider="mock",
        scorer="exact_match",
        perturbation_types=(PerturbationType.FORMAT_CHANGE,),
        num_variants=3,
        concurrency=4,
    )

    # The model "knows" q1/q2/q3 and guesses wrong on q4/q5. Mean
    # RC_correct should land around 0.6 — a useful non-trivial signal.
    # FORMAT_CHANGE preserves label positions, so a mock keyed by
    # correct labels per question gives meaningful scores.
    responses = {"q1": "B", "q2": "C", "q3": "B", "q4": "A", "q5": "A"}
    provider = BaseIdMockProvider(model="mock-model", responses=responses)

    runner = BatchRunner()
    report = await runner.run(dataset, config, provider, ExactMatchScorer(), seed=42)

    ConsoleReporter().display(report, threshold=0.8)
    print(f"\nCORE index: {core_index(report.results):.4f}")
    print(f"MCA(0.8):   {mca(report.results, 0.8):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
