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


def mca(
    results: Sequence[QuestionConsistencyResult],
    threshold: float,
) -> float:
    """Compute MCA_cat(c) -- fraction of questions with RC_correct >= c.

    CAT paper Equation 4:
        ``MCA(c) = (1/N) * sum(indicator(RC_i >= c) for i=1..N)``

    Args:
        results: Per-question consistency results.
        threshold: Consistency threshold *c* in [0.0, 1.0].

    Returns:
        Fraction of questions meeting the threshold, or 0.0 if
        *results* is empty.
    """
    if not results:
        return 0.0
    count = sum(1 for r in results if r.rc_correct >= threshold)
    return count / len(results)


def car_curve(
    results: Sequence[QuestionConsistencyResult],
    thresholds: Sequence[float] | None = None,
) -> list[tuple[float, float]]:
    """Build the CAR curve from MCA values across thresholds.

    CAT paper Equation 5:
        ``CAR = {(c_k, MCA(c_k)) | c_k in C}``

    Default thresholds are 11 evenly-spaced points from 0.0 to 1.0
    (matching IBM/cat ``consistency_resolution=10``).

    Args:
        results: Per-question consistency results.
        thresholds: Consistency thresholds.  Defaults to
            ``[0.0, 0.1, 0.2, ..., 1.0]``.

    Returns:
        List of ``(threshold, mca_value)`` pairs sorted by threshold.
    """
    if thresholds is None:
        thresholds = [i / 10 for i in range(11)]
    return [(c, mca(results, c)) for c in sorted(thresholds)]
