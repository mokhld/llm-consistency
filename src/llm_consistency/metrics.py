"""CAT framework metrics engine (MCA, CAR, CORE, AGA, Bootstrap CI).

Pure functions for computing consistency and accuracy metrics from
``QuestionConsistencyResult`` instances.  No I/O, no side effects.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


def trapezoidal_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute area under curve using the trapezoidal rule.

    CAT paper Equation 6-7:
        ``sum((MCA(c_k) + MCA(c_{k+1})) / 2 * (c_{k+1} - c_k))``

    Handles non-uniform x spacing correctly.

    Args:
        xs: X-coordinates (thresholds), must be sorted ascending.
        ys: Y-coordinates (MCA values at each threshold).

    Returns:
        Approximate area under the curve, or 0.0 if fewer than 2 points.

    Raises:
        ValueError: If *xs* and *ys* have different lengths.
    """
    if len(xs) != len(ys):
        msg = "xs and ys must have the same length"
        raise ValueError(msg)
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(len(xs) - 1):
        area += (ys[i] + ys[i + 1]) / 2.0 * (xs[i + 1] - xs[i])
    return area


def dtw_distance(s: Sequence[float], t: Sequence[float]) -> float:
    """Compute DTW distance between two 1-D sequences using L1 norm.

    Standard O(NM) dynamic programming algorithm with an ``(n+1) x (m+1)``
    cost matrix initialized to ``inf``, except ``cost[0][0] = 0``.

    Args:
        s: First sequence.
        t: Second sequence.

    Returns:
        The DTW distance (accumulated minimum-cost alignment distance),
        or 0.0 if either sequence is empty.
    """
    n = len(s)
    m = len(t)
    if n == 0 or m == 0:
        return 0.0

    # Cost matrix with infinity borders
    cost = [[math.inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(s[i - 1] - t[j - 1])
            cost[i][j] = d + min(
                cost[i - 1][j],      # insertion
                cost[i][j - 1],      # deletion
                cost[i - 1][j - 1],  # match
            )

    return cost[n][m]


def normalized_dtw(mca_values: Sequence[float]) -> float:
    """Compute normalized DTW for the CAT framework.

    CAT paper Equation 8:
        ``norm_DTW = 1 - (DTW_model / DTW_worst)``

    Where:
      - ``DTW_model = dtw_distance(mca_values, ideal)``
      - ``DTW_worst = dtw_distance(ideal, worst)``
      - ``ideal = [1.0, 1.0, ..., 1.0]``
      - ``worst = [1.0, 0.0, 0.0, ..., 0.0]``

    Args:
        mca_values: MCA values at each threshold (the model's CAR curve
            y-values).

    Returns:
        Normalized DTW in [0.0, 1.0].  1.0 for ideal curve, 0.0 for worst.
        Returns 1.0 for the degenerate single-point case.
    """
    n = len(mca_values)
    if n <= 1:
        return 1.0

    ideal = [1.0] * n
    worst = [1.0] + [0.0] * (n - 1)

    dtw_worst = dtw_distance(ideal, worst)
    if dtw_worst == 0.0:
        return 1.0  # pragma: no cover

    dtw_model = dtw_distance(list(mca_values), ideal)
    return 1.0 - (dtw_model / dtw_worst)


def core_index(
    results: Sequence[QuestionConsistencyResult],
    thresholds: Sequence[float] | None = None,
) -> float:
    """Compute the CORE index (Consistency-Oriented Robustness Estimate).

    CAT paper Equation 9:
        ``CORE = AUCAR * norm_DTW``

    Composes: ``car_curve`` -> extract xs/ys -> ``trapezoidal_auc`` +
    ``normalized_dtw`` -> multiply.

    Args:
        results: Per-question consistency results.
        thresholds: Consistency thresholds.  Defaults to
            ``[0.0, 0.1, 0.2, ..., 1.0]``.

    Returns:
        CORE index in [0.0, 1.0].
    """
    curve = car_curve(results, thresholds)
    xs = [c for c, _ in curve]
    ys = [m for _, m in curve]

    aucar = trapezoidal_auc(xs, ys)
    norm_dtw = normalized_dtw(ys)

    return aucar * norm_dtw


def agreement_gated_accuracy(
    results: Sequence[QuestionConsistencyResult],
    tau_agree: float,
) -> float:
    """Compute accuracy gated by answer agreement.

    Extension metric (not in CAT paper): mean of ``rc_correct`` among
    questions where ``rc_agree >= tau_agree``.  Reveals stable-but-wrong
    patterns by filtering for questions where the model consistently
    gives the same answer.

    Args:
        results: Per-question consistency results.
        tau_agree: Agreement threshold -- only questions with
            ``rc_agree >= tau_agree`` are included.

    Returns:
        Mean ``rc_correct`` of qualifying questions, or 0.0 if no
        questions pass the filter or *results* is empty.
    """
    if not results:
        return 0.0
    passing = [r for r in results if r.rc_agree >= tau_agree]
    if not passing:
        return 0.0
    return sum(r.rc_correct for r in passing) / len(passing)


def bootstrap_ci(
    results: Sequence[QuestionConsistencyResult],
    statistic: Callable[[Sequence[QuestionConsistencyResult]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute a bootstrap percentile confidence interval.

    Resamples *results* with replacement ``n_bootstrap`` times, computes
    *statistic* on each resample, and returns the ``(lower, upper)``
    percentile bounds.

    Uses ``random.Random(seed)`` for an isolated PRNG that does not
    affect global state.  Reproducible: same seed + same input = same
    output.

    Args:
        results: Per-question consistency results.
        statistic: A callable that computes a scalar metric from results.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility, or ``None``.

    Returns:
        ``(lower, upper)`` confidence interval bounds.
    """
    rng = random.Random(seed)
    result_list = list(results)
    n = len(result_list)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        resample = rng.choices(result_list, k=n)
        estimates.append(statistic(resample))
    estimates.sort()
    alpha = 1.0 - confidence
    lower_idx = math.floor((alpha / 2) * n_bootstrap)
    upper_idx = math.ceil((1.0 - alpha / 2) * n_bootstrap) - 1
    # Clamp indices
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))
    return (estimates[lower_idx], estimates[upper_idx])
