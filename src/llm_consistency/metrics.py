"""CAT framework metrics engine (MCA, CAR, CORE).

Pure functions for computing consistency and accuracy metrics from
``QuestionConsistencyResult`` instances.  No I/O, no side effects.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from llm_consistency.types import QuestionConsistencyResult


def build_question_consistency_result(
    question_id: str,
    variant_answers: Sequence[tuple[str, bool]],
) -> QuestionConsistencyResult:
    """Build a QuestionConsistencyResult from per-variant scored data.

    Computes ``rc_correct``, ``rc_agree``, and ``answer_distribution``
    from a sequence of ``(extracted_answer, is_correct)`` pairs.

    Args:
        question_id: The question identifier.
        variant_answers: Sequence of ``(extracted_answer, is_correct)``
            tuples, one per variant.

    Returns:
        A fully computed ``QuestionConsistencyResult``.

    Raises:
        ValueError: If *variant_answers* is empty.
    """
    if not variant_answers:
        msg = "variant_answers must be non-empty"
        raise ValueError(msg)

    total = len(variant_answers)
    correct_count = sum(1 for _, is_correct in variant_answers if is_correct)
    rc_correct = correct_count / total

    answer_counts = Counter(answer for answer, _ in variant_answers)
    rc_agree = max(answer_counts.values()) / total
    answer_distribution = dict(answer_counts)

    return QuestionConsistencyResult(
        question_id=question_id,
        rc_correct=rc_correct,
        rc_agree=rc_agree,
        total_variants=total,
        correct_count=correct_count,
        answer_distribution=answer_distribution,
        scored_responses=(),
    )
