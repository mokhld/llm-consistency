"""Custom scoring via CustomScorerAdapter.

The built-in ExactMatchScorer is a cascading regex extractor that
handles formats like ``"Answer: B"``, ``"The answer is (C)"``, and
single-character outputs. When a model emits something more exotic
— e.g. a verbose chain-of-thought ending with ``"...therefore the
answer is option **B (Paris)**"`` — you can plug in your own scoring
logic via CustomScorerAdapter.

This example registers a regex scorer that extracts the label from a
``"\\\\boxed{X}"`` LaTeX-style answer marker (a common output style for
math/reasoning models) and wraps it as a `BaseScorer` via the adapter.

Run from the repo root::

    uv run python examples/04_custom_scorer.py
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from _helpers import BaseIdMockProvider

from llm_consistency import (
    BatchRunner,
    CustomScorerAdapter,
    EvaluationConfig,
    LLMResponse,
    MCDataset,
    MCQuestion,
    PerturbationType,
    ScoredResponse,
)

# A model emitting LaTeX-style boxed answers might write something like
# "The reasoning leads us to \\boxed{B}." We extract the label between
# braces and compare it to the correct option.
_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Z])\s*\}")


def boxed_scorer(response: LLMResponse, question: MCQuestion) -> ScoredResponse:
    """Extract a ``\\boxed{X}`` label and compare against the correct option."""
    match = _BOXED_RE.search(response.raw_output)
    extracted = match.group(1) if match else ""
    correct_label = next(o.label for o in question.options if o.is_correct)
    is_correct = extracted == correct_label
    return ScoredResponse(
        question_id=response.question_id,
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        scoring_method="boxed_regex",
    )


async def main() -> None:
    dataset = MCDataset.load(Path(__file__).parent / "datasets" / "sample.jsonl")
    config = EvaluationConfig(
        model="mock-model",
        provider="mock",
        scorer="exact_match",  # Identifier only; runner uses scorer arg below.
        perturbation_types=(PerturbationType.FORMAT_CHANGE,),
        num_variants=2,
        concurrency=2,
    )

    # Mock provider returns chain-of-thought wrappers around the answer.
    # ExactMatchScorer's "first standalone label" rule would still match
    # them — but the boxed scorer demonstrates a custom extraction path.
    responses = {
        "q1": "Working through this: Paris is the capital. \\boxed{B}",
        "q2": "Mercury is the innermost planet. \\boxed{C}",
        "q3": "Pride and Prejudice is by Jane Austen. \\boxed{B}",
        "q4": "Gold's symbol comes from Latin aurum. \\boxed{C}",
        "q5": "The seven continents are well-established. \\boxed{C}",
    }
    provider = BaseIdMockProvider(model="mock-model", responses=responses)
    scorer = CustomScorerAdapter(boxed_scorer, name="boxed_regex")

    runner = BatchRunner()
    report = await runner.run(dataset, config, provider, scorer, seed=42)

    print(f"Custom scorer: {scorer.name}")
    print(f"Total questions: {report.total_questions}")
    print(f"Mean RC_correct: {report.mean_rc_correct:.3f}")
    print(f"Mean RC_agree:   {report.mean_rc_agree:.3f}")
    print()
    print("Per-question:")
    for qcr in report.results:
        print(
            f"  {qcr.question_id}: rc_correct={qcr.rc_correct:.2f} "
            f"answers={dict(qcr.answer_distribution)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
