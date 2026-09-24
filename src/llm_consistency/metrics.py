"""CAT framework metrics engine (MCA, CAR, CORE, AGA, Bootstrap CI).

Pure functions for computing consistency and accuracy metrics from
``QuestionConsistencyResult`` instances.  No I/O, no side effects.
"""

from __future__ import annotations

import functools
import math
import random
import statistics
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, TypeVar

from llm_consistency._exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

from llm_consistency.types import (
    EvaluationReport,
    MetricResult,
    PairedTestResult,
    PerturbationType,
    QuestionConsistencyResult,
)

_SAMPLE_SIZE_WARNING_THRESHOLD = 200

_K = TypeVar("_K")
_R = TypeVar("_R")


def _check_unit_interval(name: str, value: float) -> None:
    """Raise ``ValidationError`` unless ``0.0 <= value <= 1.0``."""
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0.0, 1.0], got {value}"
        raise ValidationError(msg)


def _sorted_thresholds(
    thresholds: Sequence[float] | None,
    *,
    require_endpoints: bool = False,
) -> list[float]:
    """Validate CAR thresholds and return them sorted ascending.

    ``None`` gives the default grid ``[0.0, 0.1, ..., 1.0]``. With
    *require_endpoints*, the thresholds must include both 0.0 and 1.0.
    """
    if thresholds is None:
        return [i / 10 for i in range(11)]
    for c in thresholds:
        _check_unit_interval("threshold", c)
    xs = sorted(thresholds)
    if require_endpoints and (not xs or xs[0] != 0.0 or xs[-1] != 1.0):
        msg = f"CORE thresholds must include 0.0 and 1.0, got {list(thresholds)}"
        raise ValidationError(msg)
    return xs


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

    Raises:
        ValidationError: If *threshold* is not in [0.0, 1.0].
    """
    _check_unit_interval("threshold", threshold)
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
        thresholds: Consistency thresholds, each in [0.0, 1.0].  Any
            subset of that range is allowed here; :func:`core_index`
            is stricter.  Defaults to ``[0.0, 0.1, 0.2, ..., 1.0]``.

    Returns:
        List of ``(threshold, mca_value)`` pairs sorted by threshold.

    Raises:
        ValidationError: If a threshold is not in [0.0, 1.0].
    """
    return [(c, mca(results, c)) for c in _sorted_thresholds(thresholds)]


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

    Custom thresholds must lie in [0.0, 1.0] and include both 0.0 and
    1.0, as the paper's grid does.  With 0.0 in the grid, MCA(0.0) is
    1.0 for any non-empty results, which keeps the normalized DTW in
    [0.0, 1.0]; with 1.0 in the grid, AUCAR covers the whole range, so
    CORE values from different grids stay comparable.

    Args:
        results: Per-question consistency results.
        thresholds: Consistency thresholds.  Defaults to
            ``[0.0, 0.1, 0.2, ..., 1.0]``.

    Returns:
        CORE index in [0.0, 1.0], or 0.0 if *results* is empty.

    Raises:
        ValidationError: If a threshold is not in [0.0, 1.0], or the
            thresholds do not include 0.0 and 1.0.
    """
    grid = _sorted_thresholds(thresholds, require_endpoints=True)
    if not results:
        return 0.0
    curve = car_curve(results, grid)
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

    Raises:
        ValidationError: If *tau_agree* is not in [0.0, 1.0].
    """
    _check_unit_interval("tau_agree", tau_agree)
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
        n_bootstrap: Number of bootstrap resamples.  Must be >= 1.
        confidence: Confidence level (e.g. 0.95 for 95% CI), in (0, 1).
        seed: Random seed for reproducibility, or ``None``.

    Returns:
        ``(lower, upper)`` confidence interval bounds.

    Raises:
        ValidationError: If ``n_bootstrap < 1`` or *confidence* is not
            in (0, 1).
    """
    return _interval(
        "percentile", list(results), statistic, n_bootstrap, confidence, seed
    )


def bootstrap_ci_bca(
    results: Sequence[QuestionConsistencyResult],
    statistic: Callable[[Sequence[QuestionConsistencyResult]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute a bias-corrected accelerated (BCa) bootstrap CI.

    BCa adjusts the percentile bounds by two corrections:

    - ``z0`` (bias): the inverse normal CDF of the share of bootstrap
      estimates below the observed estimate, counting estimates equal to
      it as half (the mid-rank rule scipy uses).  Counting only strictly
      smaller estimates would bias intervals downward for discrete
      statistics such as MCA.
    - ``a`` (acceleration): a jackknife-derived skewness correction.

    The adjusted lower/upper percentiles are then:

    ``alpha_low  = Phi(z0 + (z0 + z_{alpha/2})   / (1 - a*(z0 + z_{alpha/2})))``
    ``alpha_high = Phi(z0 + (z0 + z_{1-alpha/2}) / (1 - a*(z0 + z_{1-alpha/2})))``

    BCa is preferred over the plain percentile method when the bootstrap
    distribution is biased or skewed, which is common for bounded
    statistics like MCA, CORE, and AGA.

    Degenerate cases: empty input gives ``(0.0, 0.0)``; if every
    bootstrap estimate is equal, both bounds are that value; if every
    estimate lies strictly on one side of the observed value, the
    percentile bounds are returned; zero jackknife variance gives zero
    acceleration.

    The jackknife evaluates *statistic* on each of the n leave-one-out
    samples, so this generic version costs O(n^2) for statistics that
    are linear in n.  The built-in ``*_with_ci`` functions compute the
    leave-one-out values in closed form instead.

    Args:
        results: Per-question consistency results.
        statistic: A callable that computes a scalar metric from results.
        n_bootstrap: Number of bootstrap resamples.  Must be >= 1.
        confidence: Confidence level (e.g. 0.95 for 95% CI), in (0, 1).
        seed: Random seed for reproducibility, or ``None``.

    Returns:
        ``(lower, upper)`` BCa confidence interval bounds.

    Raises:
        ValidationError: If ``n_bootstrap < 1`` or *confidence* is not
            in (0, 1).
    """
    return _interval("bca", list(results), statistic, n_bootstrap, confidence, seed)


_BootstrapMethod = Literal["bca", "percentile"]


def _check_method(method: str) -> None:
    if method not in ("bca", "percentile"):
        msg = f"Unknown bootstrap method {method!r}. Use 'bca' or 'percentile'."
        raise ValueError(msg)


def _check_ci_args(n_bootstrap: int, confidence: float) -> None:
    if n_bootstrap < 1:
        msg = f"n_bootstrap must be >= 1, got {n_bootstrap}"
        raise ValidationError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1), got {confidence}"
        raise ValidationError(msg)


def _resample(
    keys: Sequence[_K],
    statistic: Callable[[list[_K]], _R],
    n_bootstrap: int,
    seed: int | None,
) -> list[_R]:
    """Evaluate *statistic* on *n_bootstrap* resamples of *keys*.

    Which questions are drawn depends only on ``len(keys)`` and *seed*,
    so a compact per-question encoding of the results (a pass flag, a
    threshold level) draws exactly the same resamples as the results.
    """
    rng = random.Random(seed)
    n = len(keys)
    return [statistic(rng.choices(keys, k=n)) for _ in range(n_bootstrap)]


def _interval(
    method: str,
    keys: list[_K],
    statistic: Callable[[list[_K]], float],
    n_bootstrap: int,
    confidence: float,
    seed: int | None,
    jackknife: Callable[[], Sequence[float]] | None = None,
) -> tuple[float, float]:
    """Bootstrap interval for *statistic* over per-question *keys*.

    *jackknife* returns the leave-one-out values of *statistic*.  When
    omitted they are computed by re-evaluating *statistic* n times.
    """
    _check_method(method)
    _check_ci_args(n_bootstrap, confidence)
    if method == "bca" and not keys:
        return (0.0, 0.0)
    estimates = sorted(_resample(keys, statistic, n_bootstrap, seed))
    if method == "percentile":
        return _percentile_bounds(estimates, confidence)
    if jackknife is None:
        jackknife = functools.partial(_generic_jackknife, keys, statistic)
    return _bca_bounds(estimates, statistic(keys), jackknife, confidence)


def _generic_jackknife(
    keys: list[_K], statistic: Callable[[list[_K]], float]
) -> list[float]:
    """Leave-one-out values of *statistic*, one full evaluation each."""
    return [statistic(keys[:i] + keys[i + 1 :]) for i in range(len(keys))]


def _percentile_bounds(
    estimates: Sequence[float], confidence: float
) -> tuple[float, float]:
    """Percentile interval from sorted bootstrap *estimates*."""
    n_bootstrap = len(estimates)
    alpha = 1.0 - confidence
    lower_idx = max(0, min(math.floor((alpha / 2) * n_bootstrap), n_bootstrap - 1))
    upper_idx = max(
        0, min(math.ceil((1.0 - alpha / 2) * n_bootstrap) - 1, n_bootstrap - 1)
    )
    return (estimates[lower_idx], estimates[upper_idx])


def _bca_bounds(
    estimates: Sequence[float],
    observed: float,
    jackknife: Callable[[], Sequence[float]],
    confidence: float,
) -> tuple[float, float]:
    """BCa interval from sorted bootstrap *estimates*.

    *jackknife* is only called when the bootstrap distribution is not
    degenerate.
    """
    n_bootstrap = len(estimates)
    if estimates[0] == estimates[-1]:
        # Every resample gave the same value, so every percentile is it.
        return (estimates[0], estimates[-1])

    below = sum(1 for e in estimates if e < observed)
    equal = sum(1 for e in estimates if e == observed)
    rank = (below + 0.5 * equal) / n_bootstrap
    if rank in (0.0, 1.0):
        # All bootstrap estimates lie strictly on one side of the observed
        # value, so z0 is infinite. Fall back to percentile.
        return _percentile_bounds(estimates, confidence)

    norm = statistics.NormalDist()
    alpha = 1.0 - confidence
    z0 = norm.inv_cdf(rank)

    jack = jackknife()
    n = len(jack)
    jack_mean = sum(jack) / n
    num = sum((jack_mean - j) ** 3 for j in jack)
    den = 6.0 * (sum((jack_mean - j) ** 2 for j in jack) ** 1.5)
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


# The *_with_ci functions below bootstrap a compact encoding of each
# question (a pass flag, a threshold level, or a gated rc_correct) instead
# of the QuestionConsistencyResult, and compute the jackknife in closed
# form, in O(n * len(thresholds)) rather than O(n^2).  The resamples and
# estimates are the same as passing the scalar metric to bootstrap_ci or
# bootstrap_ci_bca.  AGA's leave-one-out means come from a running sum,
# so they can differ from direct evaluation in the last bits.


def _pass_rate(flags: Sequence[int]) -> float:
    """MCA of a resample encoded as 0/1 pass flags (same arithmetic as mca)."""
    return sum(flags) / len(flags) if flags else 0.0


def _pass_rate_jackknife(flags: Sequence[int]) -> list[float]:
    """Leave-one-out MCA values; there are only two possible values.

    Needs at least two flags.  With one question every resample is the
    same, so :func:`_bca_bounds` returns before asking for a jackknife.
    """
    n = len(flags)
    total = sum(flags)
    loo = (total / (n - 1), (total - 1) / (n - 1))
    return [loo[f] for f in flags]


def _levels(
    results: Sequence[QuestionConsistencyResult], xs: Sequence[float]
) -> list[int]:
    """Number of sorted thresholds *xs* each question meets.

    A question that meets threshold ``xs[j]`` meets every lower one, so
    it passes exactly the thresholds ``xs[:level]``.
    """
    return [sum(1 for c in xs if r.rc_correct >= c) for r in results]


def _car_ys(level_counts: Mapping[int, int], n: int, k: int) -> list[float]:
    """MCA at each of *k* sorted thresholds, from counts of levels."""
    ys: list[float] = []
    passing = n
    for j in range(k):
        passing -= level_counts.get(j, 0)
        ys.append(passing / n)
    return ys


def _core_from_counts(
    level_counts: Mapping[int, int], n: int, xs: Sequence[float]
) -> float:
    """CORE of *n* questions summarised by their level counts."""
    if n == 0:
        return 0.0
    ys = _car_ys(level_counts, n, len(xs))
    return trapezoidal_auc(xs, ys) * normalized_dtw(ys)


def _core_jackknife(levels: Sequence[int], xs: Sequence[float]) -> list[float]:
    """Leave-one-out CORE values, one per distinct level."""
    n = len(levels)
    counts = Counter(levels)
    loo: dict[int, float] = {}
    for level in counts:
        counts[level] -= 1
        loo[level] = _core_from_counts(counts, n - 1, xs)
        counts[level] += 1
    return [loo[level] for level in levels]


def _gated_mean(values: Sequence[float | None]) -> float:
    """AGA of a resample encoded as ``rc_correct`` or ``None`` if gated out."""
    passing = [v for v in values if v is not None]
    return sum(passing) / len(passing) if passing else 0.0


def _gated_mean_jackknife(values: Sequence[float | None]) -> list[float]:
    """Leave-one-out AGA values from the running sum and count."""
    passing = [v for v in values if v is not None]
    m = len(passing)
    total = sum(passing)
    full = total / m if m else 0.0
    return [
        full if v is None else ((total - v) / (m - 1) if m > 1 else 0.0) for v in values
    ]


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
        n_bootstrap: Number of bootstrap resamples.  Must be >= 1.
        confidence: Confidence level (e.g. 0.95 for 95% CI), in (0, 1).
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        A :class:`MetricResult` with point estimate, CI, and metadata.

    Raises:
        ValidationError: On an out-of-range *threshold*, *confidence*
            or *n_bootstrap*.
    """
    point = mca(results, threshold)
    flags = [1 if r.rc_correct >= threshold else 0 for r in results]
    ci = _interval(
        method,
        flags,
        _pass_rate,
        n_bootstrap,
        confidence,
        seed,
        functools.partial(_pass_rate_jackknife, flags),
    )
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

    See :func:`core_index` for the point estimate and the rules for
    custom thresholds.

    Args:
        results: Per-question consistency results.
        thresholds: CAR curve thresholds.
        n_bootstrap: Number of bootstrap resamples.  Must be >= 1.
        confidence: Confidence level, in (0, 1).
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        A :class:`MetricResult` for the CORE index.

    Raises:
        ValidationError: On invalid *thresholds*, *confidence* or
            *n_bootstrap*.
    """
    point = core_index(results, thresholds)
    xs = _sorted_thresholds(thresholds, require_endpoints=True)
    levels = _levels(results, xs)

    def _stat(sample: list[int]) -> float:
        return _core_from_counts(Counter(sample), len(sample), xs)

    ci = _interval(
        method,
        levels,
        _stat,
        n_bootstrap,
        confidence,
        seed,
        functools.partial(_core_jackknife, levels, xs),
    )
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
        tau_agree: Agreement threshold in [0.0, 1.0].
        n_bootstrap: Number of bootstrap resamples.  Must be >= 1.
        confidence: Confidence level, in (0, 1).
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        A :class:`MetricResult` for AGA.

    Raises:
        ValidationError: On an out-of-range *tau_agree*, *confidence*
            or *n_bootstrap*.
    """
    point = agreement_gated_accuracy(results, tau_agree)
    values = [r.rc_correct if r.rc_agree >= tau_agree else None for r in results]
    ci = _interval(
        method,
        values,
        _gated_mean,
        n_bootstrap,
        confidence,
        seed,
        functools.partial(_gated_mean_jackknife, values),
    )
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
    ascending.  All thresholds share the same bootstrap resamples, so
    with a fixed *seed* each interval equals the one :func:`mca_with_ci`
    gives for that threshold.

    Args:
        results: Per-question consistency results.
        thresholds: CAR curve thresholds, each in [0.0, 1.0].
        n_bootstrap: Number of bootstrap resamples.  Must be >= 1.
        confidence: Confidence level, in (0, 1).
        seed: Random seed for reproducibility.
        method: ``"bca"`` (default) or ``"percentile"``.

    Returns:
        List of ``(threshold, MetricResult)`` pairs sorted by threshold.

    Raises:
        ValidationError: On an out-of-range threshold, *confidence* or
            *n_bootstrap*.
    """
    _check_method(method)
    _check_ci_args(n_bootstrap, confidence)
    xs = _sorted_thresholds(thresholds)
    curve = car_curve(results, xs)
    n = len(results)
    if n == 0:
        return [
            (c, _wrap_result(p, (0.0, 0.0), 0, confidence, method)) for c, p in curve
        ]

    levels = _levels(results, xs)
    draws = _resample(
        levels, lambda s: _car_ys(Counter(s), n, len(xs)), n_bootstrap, seed
    )
    out: list[tuple[float, MetricResult]] = []
    for j, (c, point) in enumerate(curve):
        estimates = sorted(ys[j] for ys in draws)
        if method == "percentile":
            ci = _percentile_bounds(estimates, confidence)
        else:
            flags = [1 if level > j else 0 for level in levels]
            jackknife = functools.partial(_pass_rate_jackknife, flags)
            ci = _bca_bounds(estimates, point, jackknife, confidence)
        out.append((c, _wrap_result(point, ci, n, confidence, method)))
    return out


def validate_sample_size(
    n: int,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    """Sample size and power for a one-sample test of a proportion.

    Sizes a two-sided one-sample z-test of a single proportion (for
    example, one model's MCA at a fixed threshold against a reference
    value), with Cohen's h as the effect size and the normal
    approximation on the arcsine scale:

    - ``recommended_n = ceil(((z_{1-alpha/2} + z_{power}) / h) ** 2)``
    - ``power_at_n = Phi(h * sqrt(n) - z_{1-alpha/2})``

    It does not size a comparison of two models.  The power of
    :func:`compare_mca_paired` (McNemar's test) depends on the share of
    discordant questions, which this function does not model.

    The returned dict makes two computations explicit:

    - ``power_at_n``: the power at the supplied ``n`` if the true effect
      is exactly ``effect_size``.  It is computed from the assumed
      effect, not observed in any data.  ``observed_power`` holds the
      same value under its older name.
    - ``recommended_n``: the minimum ``n`` needed to reach
      ``target_power`` at the supplied ``effect_size`` and ``alpha``.

    Cohen's conventional anchors for ``effect_size``:

    - ``0.2``: small
    - ``0.5``: medium
    - ``0.8``: large

    Args:
        n: Actual sample size (number of questions in the study).
        effect_size: Effect size as Cohen's h.  Must be > 0.
        alpha: Two-sided significance level.  Defaults to ``0.05``.
        power: Target power for ``recommended_n``.  Defaults to ``0.80``.

    Returns:
        Dict with keys ``n``, ``effect_size``, ``alpha``, ``target_power``,
        ``power_at_n``, ``observed_power`` (equal to ``power_at_n``, kept
        for backward compatibility), and ``recommended_n``.  All values
        are ``float``.

    Raises:
        ValidationError: If ``n < 1``, ``effect_size <= 0``, ``alpha``
            not in ``(0, 1)``, or ``power`` not in ``(0, 1)``.

    Warnings:
        Emits ``UserWarning`` when ``n < 200``.  The cut-off is this
        package's rule of thumb for flagging small studies; it does not
        come from the CAT paper.
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
            f"n={n} is below {_SAMPLE_SIZE_WARNING_THRESHOLD} questions, this "
            f"package's rule of thumb for a perturbation study; check "
            f"power_at_n for the effect size you expect",
            UserWarning,
            stacklevel=2,
        )

    norm = statistics.NormalDist()
    z_alpha = norm.inv_cdf(1.0 - alpha / 2.0)
    z_beta = norm.inv_cdf(power)

    recommended_n = math.ceil(((z_alpha + z_beta) / effect_size) ** 2)
    power_at_n = norm.cdf(effect_size * math.sqrt(n) - z_alpha)

    return {
        "n": float(n),
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "target_power": float(power),
        "power_at_n": float(power_at_n),
        "observed_power": float(power_at_n),
        "recommended_n": float(recommended_n),
    }


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided p-value for McNemar's exact binomial test.

    Under the null ``b ~ Binomial(b + c, 0.5)``; the two-sided p-value
    is twice the lower-tail probability of the smaller discordant
    count, capped at 1.0.

    The tail ``sum_{i=0..k} C(n, i)`` is summed as an exact integer and
    divided by ``2**n`` in one correctly rounded step, so large ``n``
    neither overflows nor loses precision.
    """
    n = b + c
    k = min(b, c)
    if 2 * k >= n:
        # b == c: the lower tail is at least 1/2, so the p-value caps at 1.
        return 1.0
    term = 1
    tail = 1
    for i in range(k):
        term = term * (n - i) // (i + 1)  # C(n, i + 1), exact
        tail += term
    return min(1.0, 2 * tail / (1 << n))


def compare_mca_paired(
    results_a: Sequence[QuestionConsistencyResult],
    results_b: Sequence[QuestionConsistencyResult],
    threshold: float,
) -> PairedTestResult:
    """McNemar's exact binomial test on per-question MCA outcomes.

    For each question that appears in *both* result sets, mark it as
    "passing" if ``rc_correct >= threshold``.  The test compares the two
    discordant cells of the resulting 2x2 contingency table:

    - ``b`` = questions where A passes and B fails,
    - ``c`` = questions where A fails and B passes.

    Under the null hypothesis that the two models have equal MCA at the
    given threshold, ``b`` is distributed ``Binomial(b + c, 0.5)``.  The
    returned ``p_value`` is two-sided (lower-tail of ``min(b, c)``,
    doubled, capped at 1.0).

    Args:
        results_a: Per-question results for model A.
        results_b: Per-question results for model B.  Must include the
            same questions as ``results_a``; questions present in only
            one set are silently dropped.
        threshold: MCA threshold; a question passes when
            ``rc_correct >= threshold``.

    Returns:
        A :class:`PairedTestResult` with ``statistic = min(b, c)``,
        the two-sided exact p-value, ``n_discordant = b + c``, and
        ``method = "mcnemar_exact"``.

    Raises:
        ValidationError: If either input is empty, if ``threshold`` is
            not in ``[0.0, 1.0]``, or if the two sets share no
            question IDs.
    """
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be in [0.0, 1.0], got {threshold}"
        raise ValidationError(msg)
    if not results_a or not results_b:
        msg = "results_a and results_b must both be non-empty"
        raise ValidationError(msg)

    pass_a: dict[str, bool] = {
        r.question_id: r.rc_correct >= threshold for r in results_a
    }
    pass_b: dict[str, bool] = {
        r.question_id: r.rc_correct >= threshold for r in results_b
    }
    common = sorted(set(pass_a) & set(pass_b))
    if not common:
        msg = (
            "results_a and results_b share no question IDs; McNemar's test is undefined"
        )
        raise ValidationError(msg)

    b = sum(1 for qid in common if pass_a[qid] and not pass_b[qid])
    c = sum(1 for qid in common if not pass_a[qid] and pass_b[qid])

    return PairedTestResult(
        statistic=float(min(b, c)),
        p_value=_mcnemar_exact_p(b, c),
        n_discordant=b + c,
        method="mcnemar_exact",
    )


def perturbation_impact(
    report: EvaluationReport,
) -> dict[PerturbationType, float]:
    """Mean failure rate per perturbation type.

    Walks every ``ScoredResponse`` in ``report.results`` and groups by
    ``perturbation_type`` (set by the runner pipeline on each scored
    response).  Returns the mean failure rate (``1 - mean is_correct``)
    of the variants of each type.

    The failure rate includes the model's base error rate: a perfectly
    consistent model that is 70% accurate scores about 0.30 for every
    type.  Without an unperturbed baseline the value cannot tell how
    much of that error a perturbation caused, so it is not an
    attribution of consistency loss to perturbations and not a variance
    decomposition.  Compare types against each other with that in mind.

    Responses with ``perturbation_type=None`` (legacy reports, or
    responses constructed outside the runner pipeline) are skipped;
    perturbation type strings that don't match a known
    :class:`PerturbationType` enum value are also skipped (with no
    error, so future enum additions don't break old reports).

    Args:
        report: A completed :class:`EvaluationReport`.

    Returns:
        Mapping of :class:`PerturbationType` to mean failure rate in
        ``[0.0, 1.0]``.  Returns an empty dict if the report has no
        annotated scored responses.
    """
    # Map from PerturbationType.value -> (correct_count, total_count)
    counts: dict[str, list[int]] = {}
    for qcr in report.results:
        for sr in qcr.scored_responses:
            if sr.perturbation_type is None:
                continue
            slot = counts.setdefault(sr.perturbation_type, [0, 0])
            slot[0] += int(sr.is_correct)
            slot[1] += 1

    impact: dict[PerturbationType, float] = {}
    for pt_value, (correct, total) in counts.items():
        try:
            pt = PerturbationType(pt_value)
        except ValueError:
            continue
        impact[pt] = 1.0 - (correct / total) if total > 0 else 0.0
    return impact
