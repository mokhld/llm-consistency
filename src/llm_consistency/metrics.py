"""CAT framework metrics engine (MCA, CAR, CORE, AGA, Bootstrap CI).

Pure functions for computing consistency and accuracy metrics from
``QuestionConsistencyResult`` instances.  No I/O, no side effects.
"""

from __future__ import annotations

import math
import random
import statistics
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal

from llm_consistency._exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from llm_consistency.types import MetricResult, QuestionConsistencyResult

_SAMPLE_SIZE_WARNING_THRESHOLD = 200


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
                cost[i - 1][j],  # insertion
                cost[i][j - 1],  # deletion
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


def bootstrap_ci_bca(
    results: Sequence[QuestionConsistencyResult],
    statistic: Callable[[Sequence[QuestionConsistencyResult]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute a bias-corrected accelerated (BCa) bootstrap CI.

    BCa adjusts the percentile bounds by two corrections:

    - ``z0`` (bias): the inverse normal CDF of the proportion of bootstrap
      estimates strictly less than the observed estimate.
    - ``a`` (acceleration): a jackknife-derived skewness correction.

    The adjusted lower/upper percentiles are then:

    ``alpha_low  = Phi(z0 + (z0 + z_{alpha/2})   / (1 - a*(z0 + z_{alpha/2})))``
    ``alpha_high = Phi(z0 + (z0 + z_{1-alpha/2}) / (1 - a*(z0 + z_{1-alpha/2})))``

    BCa is preferred over the plain percentile method when the bootstrap
    distribution is biased or skewed, which is common for bounded
    statistics like MCA, CORE, and AGA.

    Degenerate cases (empty input, zero jackknife variance, or all
    bootstrap estimates equal) fall back to the percentile bounds.

    Args:
        results: Per-question consistency results.
        statistic: A callable that computes a scalar metric from results.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility, or ``None``.

    Returns:
        ``(lower, upper)`` BCa confidence interval bounds.
    """
    result_list = list(results)
    n = len(result_list)
    if n == 0:
        return (0.0, 0.0)

    observed = statistic(result_list)

    # Draw bootstrap estimates.
    rng = random.Random(seed)
    estimates: list[float] = [
        statistic(rng.choices(result_list, k=n)) for _ in range(n_bootstrap)
    ]
    estimates.sort()

    norm = statistics.NormalDist()
    alpha = 1.0 - confidence

    # Bias correction z0.
    below = sum(1 for e in estimates if e < observed)
    if below in (0, n_bootstrap):
        # All bootstrap estimates lie on one side of the observed value:
        # BCa is ill-defined here. Fall back to percentile.
        lower_idx = max(0, min(math.floor((alpha / 2) * n_bootstrap), n_bootstrap - 1))
        upper_idx = max(
            0, min(math.ceil((1.0 - alpha / 2) * n_bootstrap) - 1, n_bootstrap - 1)
        )
        return (estimates[lower_idx], estimates[upper_idx])
    z0 = norm.inv_cdf(below / n_bootstrap)

    # Acceleration via leave-one-out jackknife.
    jackknife = [statistic(result_list[:i] + result_list[i + 1 :]) for i in range(n)]
    jack_mean = sum(jackknife) / n
    num = sum((jack_mean - j) ** 3 for j in jackknife)
    den = 6.0 * (sum((jack_mean - j) ** 2 for j in jackknife) ** 1.5)
    accel = num / den if den > 0 else 0.0

    z_lo = norm.inv_cdf(alpha / 2)
    z_hi = norm.inv_cdf(1.0 - alpha / 2)

    def _adjust(z: float) -> float:
        denom = 1.0 - accel * (z0 + z)
        if denom == 0:
            return alpha / 2 if z < 0 else 1.0 - alpha / 2
        return norm.cdf(z0 + (z0 + z) / denom)

    alpha_low = _adjust(z_lo)
    alpha_high = _adjust(z_hi)

    lower_idx = max(0, min(math.floor(alpha_low * n_bootstrap), n_bootstrap - 1))
    upper_idx = max(0, min(math.ceil(alpha_high * n_bootstrap) - 1, n_bootstrap - 1))
    return (estimates[lower_idx], estimates[upper_idx])


_BootstrapMethod = Literal["bca", "percentile"]


def _bootstrap_for(
    method: _BootstrapMethod,
) -> Callable[
    [
        Sequence[QuestionConsistencyResult],
        Callable[[Sequence[QuestionConsistencyResult]], float],
        int,
        float,
        int | None,
    ],
    tuple[float, float],
]:
    """Dispatch to the bootstrap implementation named by *method*."""
    if method == "bca":
        return bootstrap_ci_bca
    if method == "percentile":
        return bootstrap_ci
    msg = f"Unknown bootstrap method {method!r}. Use 'bca' or 'percentile'."
    raise ValueError(msg)


def _wrap_result(
    value: float,
    ci: tuple[float, float],
    n: int,
    confidence: float,
    method: str,
) -> MetricResult:
    """Build a MetricResult ensuring ci_lower <= value <= ci_upper.

    Bootstrap CIs can occasionally fall slightly above or below the
    observed value due to sampling noise; we widen the interval to the
    observed value rather than constructing an invalid MetricResult.
    """
    ci_lower = min(ci[0], value)
    ci_upper = max(ci[1], value)
    return MetricResult(
        value=value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n,
        confidence=confidence,
        method=method,
    )


def mca_with_ci(
    results: Sequence[QuestionConsistencyResult],
    threshold: float,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    method: _BootstrapMethod = "bca",
) -> MetricResult:
    """Compute MCA at *threshold* with a bootstrap confidence interval.

    See :func:`mca` for the point estimate. The CI is over questions
    (each question is a bootstrap sample).

    Args:
        results: Per-question consistency results.
        threshold: Consistency threshold *c* in [0.0, 1.0].
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        A :class:`MetricResult` with point estimate, CI, and metadata.
    """

    def _stat(sample: Sequence[QuestionConsistencyResult]) -> float:
        return mca(sample, threshold)

    point = mca(results, threshold)
    ci = _bootstrap_for(method)(results, _stat, n_bootstrap, confidence, seed)
    return _wrap_result(point, ci, len(results), confidence, method)


def core_index_with_ci(
    results: Sequence[QuestionConsistencyResult],
    thresholds: Sequence[float] | None = None,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    method: _BootstrapMethod = "bca",
) -> MetricResult:
    """Compute CORE with a bootstrap confidence interval.

    See :func:`core_index` for the point estimate.

    Args:
        results: Per-question consistency results.
        thresholds: CAR curve thresholds.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        A :class:`MetricResult` for the CORE index.
    """

    def _stat(sample: Sequence[QuestionConsistencyResult]) -> float:
        return core_index(sample, thresholds)

    point = core_index(results, thresholds)
    ci = _bootstrap_for(method)(results, _stat, n_bootstrap, confidence, seed)
    return _wrap_result(point, ci, len(results), confidence, method)


def agreement_gated_accuracy_with_ci(
    results: Sequence[QuestionConsistencyResult],
    tau_agree: float,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    method: _BootstrapMethod = "bca",
) -> MetricResult:
    """Compute AGA with a bootstrap confidence interval.

    See :func:`agreement_gated_accuracy` for the point estimate.

    Args:
        results: Per-question consistency results.
        tau_agree: Agreement threshold.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        A :class:`MetricResult` for AGA.
    """

    def _stat(sample: Sequence[QuestionConsistencyResult]) -> float:
        return agreement_gated_accuracy(sample, tau_agree)

    point = agreement_gated_accuracy(results, tau_agree)
    ci = _bootstrap_for(method)(results, _stat, n_bootstrap, confidence, seed)
    return _wrap_result(point, ci, len(results), confidence, method)


def car_curve_with_ci(
    results: Sequence[QuestionConsistencyResult],
    thresholds: Sequence[float] | None = None,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    method: _BootstrapMethod = "bca",
) -> list[tuple[float, MetricResult]]:
    """Build the CAR curve with a CI per threshold.

    Each (threshold, MetricResult) pair carries the point estimate and
    bootstrap CI for MCA at that threshold. Thresholds are sorted
    ascending.

    Args:
        results: Per-question consistency results.
        thresholds: CAR curve thresholds.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        List of ``(threshold, MetricResult)`` pairs sorted by threshold.
    """
    if thresholds is None:
        thresholds = [i / 10 for i in range(11)]
    return [
        (
            c,
            mca_with_ci(
                results,
                c,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
                method=method,
            ),
        )
        for c in sorted(thresholds)
    ]


def validate_sample_size(
    n: int,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    """Power analysis for a single-proportion perturbation study.

    Uses the standard one-sample two-sided z-test on a proportion with
    Cohen's h as the effect size — appropriate for binary-outcome
    studies like MCA-based perturbation evaluation, where each question
    contributes a Bernoulli trial.

    The returned dict makes two computations explicit:

    - ``observed_power``: the power achieved at the supplied ``n``.
    - ``recommended_n``: the minimum ``n`` needed to reach
      ``target_power`` at the supplied ``effect_size`` and ``alpha``.

    Cohen's conventional anchors for ``effect_size``:

    - ``0.2`` — small
    - ``0.5`` — medium
    - ``0.8`` — large

    Args:
        n: Actual sample size (number of questions in the study).
        effect_size: Effect size as Cohen's h.  Must be > 0.
        alpha: Two-sided significance level.  Defaults to ``0.05``.
        power: Target power for ``recommended_n``.  Defaults to ``0.80``.

    Returns:
        Dict with keys ``n``, ``effect_size``, ``alpha``, ``target_power``,
        ``observed_power``, and ``recommended_n``.  All values are
        ``float``.

    Raises:
        ValidationError: If ``n < 1``, ``effect_size <= 0``, ``alpha``
            not in ``(0, 1)``, or ``power`` not in ``(0, 1)``.

    Warnings:
        Emits ``UserWarning`` when ``n < 200`` — the typical guideline
        for perturbation studies (Cavalin et al., 2025).
    """
    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValidationError(msg)
    if effect_size <= 0:
        msg = f"effect_size must be > 0, got {effect_size}"
        raise ValidationError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValidationError(msg)
    if not 0.0 < power < 1.0:
        msg = f"power must be in (0, 1), got {power}"
        raise ValidationError(msg)

    if n < _SAMPLE_SIZE_WARNING_THRESHOLD:
        warnings.warn(
            f"n={n} is below the typical guidance of n >= "
            f"{_SAMPLE_SIZE_WARNING_THRESHOLD} for perturbation studies; "
            f"results may be underpowered",
            UserWarning,
            stacklevel=2,
        )

    norm = statistics.NormalDist()
    z_alpha = norm.inv_cdf(1.0 - alpha / 2.0)
    z_beta = norm.inv_cdf(power)

    recommended_n = math.ceil(((z_alpha + z_beta) / effect_size) ** 2)
    observed_power = norm.cdf(effect_size * math.sqrt(n) - z_alpha)

    return {
        "n": float(n),
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "target_power": float(power),
        "observed_power": float(observed_power),
        "recommended_n": float(recommended_n),
    }
