"""Tests for llm_consistency.metrics - builder, MCA, CAR, AUC, DTW, CORE, AGA, CI."""

from __future__ import annotations

import pytest

from llm_consistency.metrics import (
    agreement_gated_accuracy,
    bootstrap_ci,
    build_question_consistency_result,
    car_curve,
    core_index,
    dtw_distance,
    mca,
    normalized_dtw,
    trapezoidal_auc,
)
from llm_consistency.types import QuestionConsistencyResult


class TestBuildQuestionConsistencyResult:
    """Tests for the QCR builder function."""

    # ── Happy-path parametrized tests with hand-calculated values ──

    @pytest.mark.parametrize(
        "question_id, variant_answers, expected_rc_correct,"
        " expected_rc_agree, expected_distribution",
        [
            pytest.param(
                "q1",
                [("A", True), ("A", True), ("A", True), ("B", False), ("C", False)],
                0.6,
                0.6,
                {"A": 3, "B": 1, "C": 1},
                id="mixed-5-variants",
            ),
            pytest.param(
                "q2",
                [("B", False), ("B", False), ("B", False)],
                0.0,
                1.0,
                {"B": 3},
                id="stable-but-wrong",
            ),
            pytest.param(
                "q3",
                [("A", True), ("B", False), ("C", False), ("D", False)],
                0.25,
                0.25,
                {"A": 1, "B": 1, "C": 1, "D": 1},
                id="all-different-answers",
            ),
            pytest.param(
                "q4",
                [("A", True)],
                1.0,
                1.0,
                {"A": 1},
                id="single-variant-correct",
            ),
            pytest.param(
                "q5",
                [("X", False)],
                0.0,
                1.0,
                {"X": 1},
                id="single-variant-incorrect",
            ),
            pytest.param(
                "q6",
                [("A", True), ("A", True), ("A", True)],
                1.0,
                1.0,
                {"A": 3},
                id="all-same-all-correct",
            ),
        ],
    )
    def test_hand_calculated_cases(
        self,
        question_id: str,
        variant_answers: list[tuple[str, bool]],
        expected_rc_correct: float,
        expected_rc_agree: float,
        expected_distribution: dict[str, int],
    ) -> None:
        """Verify rc_correct, rc_agree, distribution for hand-calculated cases."""
        result = build_question_consistency_result(question_id, variant_answers)

        assert isinstance(result, QuestionConsistencyResult)
        assert result.question_id == question_id
        assert result.rc_correct == pytest.approx(expected_rc_correct)
        assert result.rc_agree == pytest.approx(expected_rc_agree)
        assert result.answer_distribution == expected_distribution
        assert result.total_variants == len(variant_answers)
        assert result.correct_count == sum(1 for _, c in variant_answers if c)

    # ── Edge cases ──

    def test_empty_input_raises_value_error(self) -> None:
        """Empty variant_answers must raise ValueError."""
        with pytest.raises(ValueError, match="variant_answers must be non-empty"):
            build_question_consistency_result("q0", [])

    def test_scored_responses_empty_tuple(self) -> None:
        """Builder should produce empty scored_responses tuple."""
        result = build_question_consistency_result("q1", [("A", True)])
        assert result.scored_responses == ()

    def test_tie_breaking_in_rc_agree(self) -> None:
        """When there's a tie in answer frequency, rc_agree is max / total."""
        # Two answers each appearing twice: max frequency = 2, total = 4
        result = build_question_consistency_result(
            "q_tie",
            [("A", True), ("A", True), ("B", False), ("B", False)],
        )
        assert result.rc_agree == pytest.approx(0.5)
        assert result.answer_distribution == {"A": 2, "B": 2}

    def test_return_type_is_frozen(self) -> None:
        """Result must be a frozen dataclass (QuestionConsistencyResult)."""
        result = build_question_consistency_result("q1", [("A", True)])
        with pytest.raises(AttributeError):
            result.rc_correct = 0.5  # type: ignore[misc]


# ── Helper to create QCR with a given rc_correct ──


def _qcr(question_id: str, rc_correct: float) -> QuestionConsistencyResult:
    """Create a minimal QCR with the given rc_correct for metric tests."""
    return QuestionConsistencyResult(
        question_id=question_id,
        rc_correct=rc_correct,
        rc_agree=1.0,
        total_variants=1,
        correct_count=round(rc_correct),
    )


# ── MCA tests ──


class TestMCA:
    """Tests for the mca() function -- CAT paper Equation 4."""

    # Case 1: Perfect model -- 3 questions, all rc_correct=1.0
    def test_perfect_model(self) -> None:
        """MCA at any threshold should be 1.0 for all-perfect results."""
        results = [_qcr("q1", 1.0), _qcr("q2", 1.0), _qcr("q3", 1.0)]
        assert mca(results, 0.0) == pytest.approx(1.0)
        assert mca(results, 0.5) == pytest.approx(1.0)
        assert mca(results, 1.0) == pytest.approx(1.0)

    # Case 2: Worst model -- 5 questions, all rc_correct=0.0
    def test_worst_model(self) -> None:
        """MCA(0.0)=1.0, MCA(any>0)=0.0 for all-zero results."""
        results = [_qcr(f"q{i}", 0.0) for i in range(5)]
        assert mca(results, 0.0) == pytest.approx(1.0)
        assert mca(results, 0.1) == pytest.approx(0.0)
        assert mca(results, 0.5) == pytest.approx(0.0)
        assert mca(results, 1.0) == pytest.approx(0.0)

    # Case 3: Mixed model -- 4 questions, rc_correct=[1.0, 0.8, 0.6, 0.0]
    @pytest.mark.parametrize(
        "threshold, expected",
        [
            pytest.param(0.0, 1.0, id="threshold-0.0"),
            pytest.param(0.5, 0.75, id="threshold-0.5"),
            pytest.param(0.6, 0.75, id="threshold-0.6-inclusive"),
            pytest.param(0.7, 0.5, id="threshold-0.7"),
            pytest.param(0.8, 0.5, id="threshold-0.8-inclusive"),
            pytest.param(0.9, 0.25, id="threshold-0.9"),
            pytest.param(1.0, 0.25, id="threshold-1.0"),
        ],
    )
    def test_mixed_model(self, threshold: float, expected: float) -> None:
        """Hand-calculated MCA values for mixed rc_correct values."""
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.8),
            _qcr("q3", 0.6),
            _qcr("q4", 0.0),
        ]
        assert mca(results, threshold) == pytest.approx(expected)

    # Case 4: MCA(0.0) always returns 1.0 for any non-empty input
    def test_threshold_zero_always_one(self) -> None:
        """MCA(0.0) = 1.0 for any non-empty input (rc_correct >= 0)."""
        results = [
            _qcr("q1", 0.0),
            _qcr("q2", 0.3),
            _qcr("q3", 0.7),
            _qcr("q4", 1.0),
        ]
        assert mca(results, 0.0) == pytest.approx(1.0)

    # Case 5: Empty results
    def test_empty_results(self) -> None:
        """mca([], threshold) returns 0.0."""
        assert mca([], 0.5) == pytest.approx(0.0)
        assert mca([], 0.0) == pytest.approx(0.0)

    # Monotonicity property
    def test_monotonicity(self) -> None:
        """MCA values must be non-increasing as threshold increases."""
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.8),
            _qcr("q3", 0.6),
            _qcr("q4", 0.4),
            _qcr("q5", 0.2),
        ]
        thresholds = [i / 10 for i in range(11)]
        values = [mca(results, t) for t in thresholds]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"MCA not monotonic at thresholds "
                f"{thresholds[i]}->{thresholds[i + 1]}: "
                f"{values[i]} < {values[i + 1]}"
            )

    def test_single_question(self) -> None:
        """MCA with a single question returns 0.0 or 1.0."""
        result = [_qcr("q1", 0.5)]
        assert mca(result, 0.3) == pytest.approx(1.0)
        assert mca(result, 0.5) == pytest.approx(1.0)
        assert mca(result, 0.6) == pytest.approx(0.0)


# ── CAR curve tests ──


class TestCARCurve:
    """Tests for the car_curve() function -- CAT paper Equation 5."""

    def test_default_thresholds_count(self) -> None:
        """Default CAR curve has 11 points (0.0, 0.1, ..., 1.0)."""
        results = [_qcr("q1", 1.0)]
        curve = car_curve(results)
        assert len(curve) == 11

    def test_default_thresholds_values(self) -> None:
        """Default thresholds are 0.0, 0.1, ..., 1.0."""
        results = [_qcr("q1", 1.0)]
        curve = car_curve(results)
        thresholds_out = [c for c, _ in curve]
        expected = [i / 10 for i in range(11)]
        for actual, exp in zip(thresholds_out, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_perfect_model_car(self) -> None:
        """Perfect model: all CAR points at 1.0."""
        results = [_qcr("q1", 1.0), _qcr("q2", 1.0), _qcr("q3", 1.0)]
        curve = car_curve(results)
        for _, mca_val in curve:
            assert mca_val == pytest.approx(1.0)

    def test_worst_model_car(self) -> None:
        """Worst model: MCA(0.0)=1.0, all others 0.0."""
        results = [_qcr(f"q{i}", 0.0) for i in range(5)]
        curve = car_curve(results)
        assert curve[0] == pytest.approx((0.0, 1.0))
        for c, m in curve[1:]:
            assert m == pytest.approx(0.0), f"MCA({c}) should be 0.0"

    def test_mixed_model_custom_thresholds(self) -> None:
        """Mixed model with custom thresholds [0.0, 0.5, 1.0]."""
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.8),
            _qcr("q3", 0.6),
            _qcr("q4", 0.0),
        ]
        curve = car_curve(results, thresholds=[0.0, 0.5, 1.0])
        assert len(curve) == 3
        assert curve[0] == pytest.approx((0.0, 1.0))
        assert curve[1] == pytest.approx((0.5, 0.75))
        assert curve[2] == pytest.approx((1.0, 0.25))

    def test_custom_thresholds_sorted(self) -> None:
        """Custom thresholds are sorted in output regardless of input order."""
        results = [_qcr("q1", 0.5)]
        curve = car_curve(results, thresholds=[1.0, 0.0, 0.5])
        thresholds_out = [c for c, _ in curve]
        assert thresholds_out == pytest.approx([0.0, 0.5, 1.0])

    def test_empty_results_car(self) -> None:
        """CAR curve with empty results returns all zeros."""
        curve = car_curve([])
        assert len(curve) == 11
        for c, m in curve:
            assert m == pytest.approx(0.0), f"MCA({c}) should be 0.0 for empty"

    def test_car_monotonicity(self) -> None:
        """CAR curve MCA values are monotonically non-increasing."""
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.7),
            _qcr("q3", 0.3),
            _qcr("q4", 0.0),
        ]
        curve = car_curve(results)
        values = [m for _, m in curve]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_single_custom_threshold(self) -> None:
        """Single custom threshold returns a one-element list."""
        results = [_qcr("q1", 0.8)]
        curve = car_curve(results, thresholds=[0.5])
        assert len(curve) == 1
        assert curve[0] == pytest.approx((0.5, 1.0))


# ── Trapezoidal AUC tests ──


class TestTrapezoidalAUC:
    """Tests for the trapezoidal_auc() function -- CAT paper Eq.6-7."""

    def test_rectangle(self) -> None:
        """Constant function y=1 over [0,1] has area 1.0."""
        assert trapezoidal_auc([0.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)

    def test_triangle(self) -> None:
        """Linear function y=1->0 over [0,1] has area 0.5."""
        assert trapezoidal_auc([0.0, 1.0], [1.0, 0.0]) == pytest.approx(0.5)

    def test_three_points_uniform(self) -> None:
        """Hand-calculated: xs=[0, 0.5, 1.0], ys=[1.0, 0.75, 0.25].

        area = 0.5*(1.0+0.75)/2 + 0.5*(0.75+0.25)/2 = 0.4375 + 0.25 = 0.6875
        """
        assert trapezoidal_auc([0.0, 0.5, 1.0], [1.0, 0.75, 0.25]) == pytest.approx(
            0.6875
        )

    def test_non_uniform_spacing(self) -> None:
        """Non-uniform x spacing: xs=[0, 0.3, 1.0], ys=[1.0, 1.0, 0.0].

        area = 0.3*(1.0+1.0)/2 + 0.7*(1.0+0.0)/2 = 0.3 + 0.35 = 0.65
        """
        assert trapezoidal_auc([0.0, 0.3, 1.0], [1.0, 1.0, 0.0]) == pytest.approx(0.65)

    def test_single_point_returns_zero(self) -> None:
        """Fewer than 2 points returns 0.0."""
        assert trapezoidal_auc([0.5], [1.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self) -> None:
        """Empty sequences return 0.0."""
        assert trapezoidal_auc([], []) == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched xs and ys lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            trapezoidal_auc([0.0, 1.0], [1.0])

    def test_all_zeros(self) -> None:
        """All-zero y values produce area 0.0."""
        assert trapezoidal_auc([0.0, 0.5, 1.0], [0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_worst_model_aucar(self) -> None:
        """Worst model CAR: y=[1,0,...,0] over default 11 thresholds.

        AUCAR = 0.5 * 0.1 * (1.0+0.0) = 0.05
        """
        xs = [i / 10 for i in range(11)]
        ys = [1.0] + [0.0] * 10
        assert trapezoidal_auc(xs, ys) == pytest.approx(0.05)


# ── DTW distance tests ──


class TestDTWDistance:
    """Tests for the dtw_distance() function -- O(NM) DP with L1 norm."""

    def test_identical_sequences(self) -> None:
        """DTW of identical sequences is 0.0."""
        assert dtw_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_single_elements(self) -> None:
        """DTW of [0] and [1] is 1.0."""
        assert dtw_distance([0.0], [1.0]) == pytest.approx(1.0)

    def test_empty_first_returns_zero(self) -> None:
        """DTW with empty first sequence returns 0.0."""
        assert dtw_distance([], [1.0, 2.0]) == pytest.approx(0.0)

    def test_empty_second_returns_zero(self) -> None:
        """DTW with empty second sequence returns 0.0."""
        assert dtw_distance([1.0, 2.0], []) == pytest.approx(0.0)

    def test_both_empty_returns_zero(self) -> None:
        """DTW with both empty sequences returns 0.0."""
        assert dtw_distance([], []) == pytest.approx(0.0)

    def test_hand_calculated_worst_vs_ideal_3pt(self) -> None:
        """Hand-calculated DTW for 3-point worst-vs-ideal case.

        ideal = [1.0, 1.0, 1.0], worst = [1.0, 0.0, 0.0]
        DTW cost matrix (from plan):
          cost[1][1] = |1-1| + 0 = 0.0
          cost[1][2] = |1-0| + 0.0 = 1.0
          cost[1][3] = |1-0| + 1.0 = 2.0
          cost[2][1] = |1-1| + 0.0 = 0.0
          cost[2][2] = |1-0| + min(0.0, 1.0, 0.0) = 1.0
          cost[2][3] = |1-0| + min(1.0, 2.0, 1.0) = 2.0
          cost[3][1] = |1-1| + 0.0 = 0.0
          cost[3][2] = |1-0| + min(0.0, 0.0, 1.0) = 1.0
          cost[3][3] = |1-0| + min(1.0, 2.0, 1.0) = 2.0
        DTW = 2.0
        """
        assert dtw_distance([1.0, 1.0, 1.0], [1.0, 0.0, 0.0]) == pytest.approx(2.0)

    def test_symmetric(self) -> None:
        """DTW(s, t) == DTW(t, s) for known sequences."""
        s = [1.0, 2.0, 3.0]
        t = [1.0, 1.0, 2.0]
        assert dtw_distance(s, t) == pytest.approx(dtw_distance(t, s))

    def test_different_lengths(self) -> None:
        """DTW handles sequences of different lengths.

        s = [1.0, 1.0], t = [1.0, 0.0, 0.0]
        cost[1][1] = 0.0
        cost[1][2] = |1-0| + 0.0 = 1.0
        cost[1][3] = |1-0| + 1.0 = 2.0
        cost[2][1] = |1-1| + 0.0 = 0.0
        cost[2][2] = |1-0| + min(0.0, 1.0, 0.0) = 1.0
        cost[2][3] = |1-0| + min(1.0, 2.0, 1.0) = 2.0
        DTW = 2.0
        """
        assert dtw_distance([1.0, 1.0], [1.0, 0.0, 0.0]) == pytest.approx(2.0)

    def test_all_same_values(self) -> None:
        """DTW of sequences with same constant value is 0.0."""
        assert dtw_distance([5.0, 5.0, 5.0], [5.0, 5.0]) == pytest.approx(0.0)


# ── Normalized DTW tests ──


class TestNormalizedDTW:
    """Tests for normalized_dtw() -- CAT paper Equation 8."""

    def test_ideal_curve_returns_one(self) -> None:
        """Ideal curve [1,1,...,1] has normalized DTW = 1.0."""
        assert normalized_dtw([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_worst_curve_returns_zero(self) -> None:
        """Worst curve [1,0,...,0] has normalized DTW = 0.0."""
        assert normalized_dtw([1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_single_point_returns_one(self) -> None:
        """Single-point degenerate case returns 1.0."""
        assert normalized_dtw([0.5]) == pytest.approx(1.0)

    def test_partial_curve(self) -> None:
        """Partial curve [1.0, 0.75, 0.25] -- verify range [0,1].

        DTW_worst = dtw([1,1,1], [1,0,0]) = 2.0 (hand-calculated above)
        DTW_model = dtw([1.0, 0.75, 0.25], [1,1,1])
        Need to compute:
          cost[1][1] = |1-1| = 0.0
          cost[1][2] = |0.75-1| + 0.0 = 0.25
          cost[1][3] = |0.25-1| + 0.25 = 1.0
          cost[2][1] = |1-1| + 0.0 = 0.0
          cost[2][2] = |0.75-1| + min(0.0, 0.25, 0.0) = 0.25
          cost[2][3] = |0.25-1| + min(0.25, 1.0, 0.25) = 1.0
          cost[3][1] = |1-1| + 0.0 = 0.0
          cost[3][2] = |0.75-1| + min(0.0, 0.0, 0.25) = 0.25
          cost[3][3] = |0.25-1| + min(0.25, 1.0, 0.25) = 1.0
        DTW_model = 1.0
        norm_DTW = 1 - (1.0 / 2.0) = 0.5
        """
        assert normalized_dtw([1.0, 0.75, 0.25]) == pytest.approx(0.5)

    def test_result_in_range(self) -> None:
        """Normalized DTW is always in [0.0, 1.0]."""
        values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        result = normalized_dtw(values)
        assert 0.0 <= result <= 1.0

    def test_11_point_ideal(self) -> None:
        """11-point ideal curve (default thresholds) returns 1.0."""
        assert normalized_dtw([1.0] * 11) == pytest.approx(1.0)

    def test_11_point_worst(self) -> None:
        """11-point worst curve returns 0.0."""
        assert normalized_dtw([1.0] + [0.0] * 10) == pytest.approx(0.0)


# ── CORE index tests ──


class TestCOREIndex:
    """Tests for the core_index() function -- CAT paper Equation 9."""

    def test_perfect_model(self) -> None:
        """Perfect model (all rc_correct=1.0) has CORE = 1.0."""
        results = [_qcr("q1", 1.0), _qcr("q2", 1.0), _qcr("q3", 1.0)]
        assert core_index(results) == pytest.approx(1.0)

    def test_worst_model(self) -> None:
        """Worst model (all rc_correct=0.0) has CORE = 0.0."""
        results = [_qcr(f"q{i}", 0.0) for i in range(5)]
        assert core_index(results) == pytest.approx(0.0)

    def test_mixed_model_custom_thresholds(self) -> None:
        """Mixed model with custom thresholds [0.0, 0.5, 1.0].

        Results: rc_correct = [1.0, 0.8, 0.6, 0.0]
        CAR: [(0.0, 1.0), (0.5, 0.75), (1.0, 0.25)]
        AUCAR = 0.6875
        MCA values = [1.0, 0.75, 0.25]
        norm_DTW = 0.5 (from TestNormalizedDTW.test_partial_curve)
        CORE = 0.6875 * 0.5 = 0.34375
        """
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.8),
            _qcr("q3", 0.6),
            _qcr("q4", 0.0),
        ]
        assert core_index(results, thresholds=[0.0, 0.5, 1.0]) == pytest.approx(0.34375)

    def test_core_in_range(self) -> None:
        """CORE values are always in [0.0, 1.0]."""
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.5),
            _qcr("q3", 0.3),
            _qcr("q4", 0.0),
        ]
        result = core_index(results)
        assert 0.0 <= result <= 1.0

    def test_core_with_default_thresholds(self) -> None:
        """CORE with default 11 thresholds for a perfect model is 1.0."""
        results = [_qcr("q1", 1.0), _qcr("q2", 1.0)]
        assert core_index(results) == pytest.approx(1.0)

    def test_single_question_perfect(self) -> None:
        """Single perfect question: CORE = 1.0."""
        results = [_qcr("q1", 1.0)]
        assert core_index(results) == pytest.approx(1.0)

    def test_composes_car_curve_auc_dtw(self) -> None:
        """Verify CORE = AUCAR * norm_DTW by computing components separately.

        Use custom thresholds [0.0, 0.5, 1.0] with perfect model.
        CAR: [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
        AUCAR = 1.0
        norm_DTW(MCA=[1.0, 1.0, 1.0]) = 1.0
        CORE = 1.0 * 1.0 = 1.0
        """
        results = [_qcr("q1", 1.0), _qcr("q2", 1.0)]
        curve = car_curve(results, thresholds=[0.0, 0.5, 1.0])
        xs = [c for c, _ in curve]
        ys = [m for _, m in curve]
        aucar = trapezoidal_auc(xs, ys)
        ndtw = normalized_dtw(ys)
        expected = aucar * ndtw
        assert core_index(results, thresholds=[0.0, 0.5, 1.0]) == pytest.approx(
            expected
        )


# ── Helper for AGA tests (QCR with both rc_correct and rc_agree) ──


def _qcr_two_axis(
    question_id: str, rc_correct: float, rc_agree: float
) -> QuestionConsistencyResult:
    """Create a QCR with explicit rc_correct and rc_agree for AGA tests."""
    return QuestionConsistencyResult(
        question_id=question_id,
        rc_correct=rc_correct,
        rc_agree=rc_agree,
        total_variants=1,
        correct_count=round(rc_correct),
    )


# ── AgreementGatedAccuracy tests ──


class TestAgreementGatedAccuracy:
    """Tests for agreement_gated_accuracy() -- extension metric."""

    def test_tau_filters_correctly(self) -> None:
        """tau_agree=0.5 keeps only questions with rc_agree >= 0.5.

        4 questions: rc_correct=[1.0, 0.8, 0.6, 0.0], rc_agree=[1.0, 0.8, 0.4, 0.2]
        Filter keeps q1(1.0) and q2(0.8) -> mean(1.0, 0.8) = 0.9
        """
        results = [
            _qcr_two_axis("q1", rc_correct=1.0, rc_agree=1.0),
            _qcr_two_axis("q2", rc_correct=0.8, rc_agree=0.8),
            _qcr_two_axis("q3", rc_correct=0.6, rc_agree=0.4),
            _qcr_two_axis("q4", rc_correct=0.0, rc_agree=0.2),
        ]
        assert agreement_gated_accuracy(results, tau_agree=0.5) == pytest.approx(0.9)

    def test_tau_zero_includes_all(self) -> None:
        """tau_agree=0.0 includes all questions -> mean of all rc_correct."""
        results = [
            _qcr_two_axis("q1", rc_correct=1.0, rc_agree=1.0),
            _qcr_two_axis("q2", rc_correct=0.8, rc_agree=0.8),
            _qcr_two_axis("q3", rc_correct=0.6, rc_agree=0.4),
            _qcr_two_axis("q4", rc_correct=0.0, rc_agree=0.2),
        ]
        assert agreement_gated_accuracy(results, tau_agree=0.0) == pytest.approx(0.6)

    def test_tau_one_keeps_only_perfect_agree(self) -> None:
        """tau_agree=1.0 keeps only q1 (rc_agree=1.0) -> mean(1.0) = 1.0."""
        results = [
            _qcr_two_axis("q1", rc_correct=1.0, rc_agree=1.0),
            _qcr_two_axis("q2", rc_correct=0.8, rc_agree=0.8),
            _qcr_two_axis("q3", rc_correct=0.6, rc_agree=0.4),
            _qcr_two_axis("q4", rc_correct=0.0, rc_agree=0.2),
        ]
        assert agreement_gated_accuracy(results, tau_agree=1.0) == pytest.approx(1.0)

    def test_tau_099_keeps_only_perfect(self) -> None:
        """tau_agree=0.99 keeps only q1 (rc_agree=1.0) -> 1.0."""
        results = [
            _qcr_two_axis("q1", rc_correct=1.0, rc_agree=1.0),
            _qcr_two_axis("q2", rc_correct=0.8, rc_agree=0.8),
        ]
        assert agreement_gated_accuracy(results, tau_agree=0.99) == pytest.approx(1.0)

    def test_all_below_threshold_returns_zero(self) -> None:
        """When no question passes the filter, return 0.0."""
        results = [
            _qcr_two_axis("q1", rc_correct=1.0, rc_agree=0.3),
            _qcr_two_axis("q2", rc_correct=0.8, rc_agree=0.4),
        ]
        assert agreement_gated_accuracy(results, tau_agree=0.5) == pytest.approx(0.0)

    def test_empty_results_returns_zero(self) -> None:
        """Empty results returns 0.0."""
        assert agreement_gated_accuracy([], tau_agree=0.5) == pytest.approx(0.0)


# ── Bootstrap CI tests ──


class TestBootstrapCI:
    """Tests for bootstrap_ci() -- confidence interval estimation."""

    def test_reproducibility_same_seed(self) -> None:
        """Same seed + same input produces identical CI bounds."""
        results = [
            _qcr_two_axis("q1", rc_correct=1.0, rc_agree=1.0),
            _qcr_two_axis("q2", rc_correct=0.5, rc_agree=0.5),
            _qcr_two_axis("q3", rc_correct=0.0, rc_agree=0.3),
        ]

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, threshold=0.5)

        ci1 = bootstrap_ci(results, statistic=stat, n_bootstrap=500, seed=42)
        ci2 = bootstrap_ci(results, statistic=stat, n_bootstrap=500, seed=42)
        assert ci1[0] == pytest.approx(ci2[0])
        assert ci1[1] == pytest.approx(ci2[1])

    def test_perfect_model_tight_ci(self) -> None:
        """Perfect model (all rc_correct=1.0): CI should be tight around 1.0."""
        results = [_qcr("q1", 1.0), _qcr("q2", 1.0), _qcr("q3", 1.0)]

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, threshold=1.0)

        lower, upper = bootstrap_ci(results, statistic=stat, n_bootstrap=1000, seed=42)
        assert lower == pytest.approx(1.0)
        assert upper == pytest.approx(1.0)

    def test_mixed_model_meaningful_ci(self) -> None:
        """Mixed model: lower < point_estimate < upper."""
        results = [
            _qcr("q1", 1.0),
            _qcr("q2", 0.8),
            _qcr("q3", 0.6),
            _qcr("q4", 0.0),
        ]

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, threshold=0.5)

        point_estimate = stat(list(results))
        lower, upper = bootstrap_ci(results, statistic=stat, n_bootstrap=2000, seed=42)
        assert lower <= point_estimate
        assert upper >= point_estimate
        # CI should have some width for mixed data
        assert upper > lower

    def test_different_seeds_different_results(self) -> None:
        """Different seeds produce different CI bounds."""
        results = [_qcr(f"q{i}", rc_correct=i / 20) for i in range(21)]

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return sum(q.rc_correct for q in r) / len(r) if r else 0.0

        ci_a = bootstrap_ci(results, statistic=stat, n_bootstrap=500, seed=42)
        ci_b = bootstrap_ci(results, statistic=stat, n_bootstrap=500, seed=99)
        # With different seeds, at least one bound should differ
        assert ci_a != ci_b

    def test_lambda_statistic_works(self) -> None:
        """Partial function pattern: lambda wrapping mca should work."""
        results = [_qcr("q1", 1.0), _qcr("q2", 0.5)]
        lower, upper = bootstrap_ci(
            results,
            statistic=lambda r: mca(r, threshold=0.5),
            n_bootstrap=100,
            seed=42,
        )
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    def test_returns_tuple_of_floats(self) -> None:
        """Return type is a tuple of two floats."""
        results = [_qcr("q1", 1.0)]

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, threshold=0.5)

        result = bootstrap_ci(results, statistic=stat, n_bootstrap=50, seed=1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ── Public API import tests ──


class TestPublicAPIImports:
    """Verify all metric functions are importable from llm_consistency."""

    def test_all_metrics_importable(self) -> None:
        """All metric functions are importable from the llm_consistency package."""
        from llm_consistency import (  # noqa: F401, PLC0415
            agreement_gated_accuracy,
            bootstrap_ci,
            build_question_consistency_result,
            car_curve,
            core_index,
            dtw_distance,
            mca,
            normalized_dtw,
            trapezoidal_auc,
        )

    def test_metrics_in_all(self) -> None:
        """All metric function names are in __all__."""
        import llm_consistency  # noqa: PLC0415

        all_names = llm_consistency.__all__
        metric_names = [
            "build_question_consistency_result",
            "mca",
            "car_curve",
            "trapezoidal_auc",
            "dtw_distance",
            "normalized_dtw",
            "core_index",
            "agreement_gated_accuracy",
            "bootstrap_ci",
            "bootstrap_ci_bca",
            "mca_with_ci",
            "core_index_with_ci",
            "agreement_gated_accuracy_with_ci",
            "car_curve_with_ci",
            "MetricResult",
        ]
        for name in metric_names:
            assert name in all_names, f"{name} not in __all__"


# ── MetricResult dataclass and BCa bootstrap ──


class TestMetricResult:
    """Frozen MetricResult dataclass invariants."""

    def test_construction_happy(self) -> None:
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        r = MetricResult(
            value=0.5,
            ci_lower=0.4,
            ci_upper=0.6,
            n_samples=100,
            confidence=0.95,
            method="bca",
        )
        assert r.value == 0.5
        assert r.ci_lower <= r.value <= r.ci_upper

    def test_round_trip(self) -> None:
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        r = MetricResult(
            value=0.5,
            ci_lower=0.4,
            ci_upper=0.6,
            n_samples=10,
            confidence=0.95,
            method="bca",
        )
        assert MetricResult.from_dict(r.to_dict()) == r

    def test_rejects_negative_n_samples(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        with pytest.raises(ValidationError, match="n_samples"):
            MetricResult(
                value=0.5,
                ci_lower=0.4,
                ci_upper=0.6,
                n_samples=-1,
                confidence=0.95,
                method="bca",
            )

    def test_rejects_inverted_ci(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        with pytest.raises(ValidationError, match="ci_lower"):
            MetricResult(
                value=0.5,
                ci_lower=0.7,
                ci_upper=0.6,
                n_samples=10,
                confidence=0.95,
                method="bca",
            )

    def test_rejects_invalid_confidence(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        with pytest.raises(ValidationError, match="confidence"):
            MetricResult(
                value=0.5,
                ci_lower=0.4,
                ci_upper=0.6,
                n_samples=10,
                confidence=1.5,
                method="bca",
            )


def _mixed_results(n: int = 30, seed: int = 0) -> list[QuestionConsistencyResult]:
    """A reproducible mix of QCRs with varied rc_correct/rc_agree."""
    import random as _r  # noqa: PLC0415

    rng = _r.Random(seed)
    return [
        QuestionConsistencyResult(
            question_id=f"q{i}",
            rc_correct=rng.random(),
            rc_agree=rng.random(),
            total_variants=5,
            correct_count=rng.randint(0, 5),
        )
        for i in range(n)
    ]


class TestBootstrapCIBCa:
    """BCa bootstrap returns a tuple of (lower, upper) sane bounds."""

    def test_returns_tuple_of_floats(self) -> None:
        from llm_consistency.metrics import bootstrap_ci_bca  # noqa: PLC0415

        results = _mixed_results()

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, 0.5)

        lo, hi = bootstrap_ci_bca(results, statistic=stat, n_bootstrap=200, seed=1)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi

    def test_reproducible_with_seed(self) -> None:
        from llm_consistency.metrics import bootstrap_ci_bca  # noqa: PLC0415

        results = _mixed_results()

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, 0.5)

        a = bootstrap_ci_bca(results, statistic=stat, n_bootstrap=200, seed=42)
        b = bootstrap_ci_bca(results, statistic=stat, n_bootstrap=200, seed=42)
        assert a == b

    def test_empty_returns_zeros(self) -> None:
        from llm_consistency.metrics import bootstrap_ci_bca  # noqa: PLC0415

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, 0.5)

        assert bootstrap_ci_bca([], statistic=stat, n_bootstrap=100, seed=1) == (
            0.0,
            0.0,
        )

    def test_degenerate_falls_back_to_percentile(self) -> None:
        """When every bootstrap estimate equals the observed value, falls back."""
        from llm_consistency.metrics import bootstrap_ci_bca  # noqa: PLC0415

        # All-equal rc_correct: every bootstrap resample yields the same MCA.
        results = [
            QuestionConsistencyResult(
                question_id=f"q{i}",
                rc_correct=1.0,
                rc_agree=1.0,
                total_variants=5,
                correct_count=5,
            )
            for i in range(10)
        ]

        def stat(r: list[QuestionConsistencyResult]) -> float:
            return mca(r, 0.5)

        lo, hi = bootstrap_ci_bca(results, statistic=stat, n_bootstrap=100, seed=1)
        assert lo == hi == 1.0


class TestMetricsWithCI:
    """*_with_ci wrappers return MetricResult with sane fields."""

    def test_mca_with_ci(self) -> None:
        from llm_consistency.metrics import mca_with_ci  # noqa: PLC0415
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        results = _mixed_results()
        r = mca_with_ci(results, 0.5, n_bootstrap=200, seed=1)
        assert isinstance(r, MetricResult)
        assert r.n_samples == len(results)
        assert r.method == "bca"
        assert r.ci_lower <= r.value <= r.ci_upper

    def test_mca_with_ci_percentile_method(self) -> None:
        from llm_consistency.metrics import mca_with_ci  # noqa: PLC0415

        results = _mixed_results()
        r = mca_with_ci(results, 0.5, n_bootstrap=200, seed=1, method="percentile")
        assert r.method == "percentile"

    def test_mca_with_ci_unknown_method_raises(self) -> None:
        from llm_consistency.metrics import mca_with_ci  # noqa: PLC0415

        results = _mixed_results()
        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            mca_with_ci(results, 0.5, n_bootstrap=10, seed=1, method="nope")  # type: ignore[arg-type]

    def test_core_index_with_ci(self) -> None:
        from llm_consistency.metrics import core_index_with_ci  # noqa: PLC0415
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        results = _mixed_results()
        r = core_index_with_ci(results, n_bootstrap=100, seed=1)
        assert isinstance(r, MetricResult)
        assert 0.0 <= r.value <= 1.0
        assert r.ci_lower <= r.value <= r.ci_upper

    def test_agreement_gated_accuracy_with_ci(self) -> None:
        from llm_consistency.metrics import (  # noqa: PLC0415
            agreement_gated_accuracy_with_ci,
        )
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        results = _mixed_results()
        r = agreement_gated_accuracy_with_ci(
            results, tau_agree=0.5, n_bootstrap=100, seed=1
        )
        assert isinstance(r, MetricResult)
        assert r.ci_lower <= r.value <= r.ci_upper

    def test_car_curve_with_ci_default_thresholds(self) -> None:
        from llm_consistency.metrics import car_curve_with_ci  # noqa: PLC0415
        from llm_consistency.types import MetricResult  # noqa: PLC0415

        results = _mixed_results()
        curve = car_curve_with_ci(results, n_bootstrap=100, seed=1)
        assert len(curve) == 11
        for c, mr in curve:
            assert isinstance(c, float)
            assert isinstance(mr, MetricResult)
            assert mr.ci_lower <= mr.value <= mr.ci_upper

    def test_car_curve_with_ci_sorts_thresholds(self) -> None:
        from llm_consistency.metrics import car_curve_with_ci  # noqa: PLC0415

        results = _mixed_results()
        curve = car_curve_with_ci(
            results, thresholds=[0.7, 0.1, 0.4], n_bootstrap=50, seed=1
        )
        assert [c for c, _ in curve] == [0.1, 0.4, 0.7]

    def test_point_matches_scalar_metric(self) -> None:
        """The point estimate in MetricResult equals the scalar metric."""
        from llm_consistency.metrics import (  # noqa: PLC0415
            core_index_with_ci,
            mca_with_ci,
        )

        results = _mixed_results()
        assert mca_with_ci(results, 0.5, n_bootstrap=50, seed=1).value == pytest.approx(
            mca(results, 0.5)
        )
        assert core_index_with_ci(
            results, n_bootstrap=50, seed=1
        ).value == pytest.approx(core_index(results))


class TestValidateSampleSize:
    """Power analysis utility for perturbation-study sample sizes."""

    def test_returns_expected_keys(self) -> None:
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        out = validate_sample_size(n=500, effect_size=0.5)
        assert set(out.keys()) == {
            "n",
            "effect_size",
            "alpha",
            "target_power",
            "observed_power",
            "recommended_n",
        }
        assert all(isinstance(v, float) for v in out.values())

    def test_recommended_n_matches_cohen_h(self) -> None:
        """Cohen's h=0.5, alpha=0.05 (two-sided), power=0.80 -> n=32."""
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        out = validate_sample_size(n=200, effect_size=0.5, alpha=0.05, power=0.80)
        assert out["recommended_n"] == pytest.approx(32.0)

    def test_observed_power_monotone_in_n(self) -> None:
        import warnings as _w  # noqa: PLC0415

        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            low = validate_sample_size(n=50, effect_size=0.2)["observed_power"]
            mid = validate_sample_size(n=200, effect_size=0.2)["observed_power"]
        high = validate_sample_size(n=2000, effect_size=0.2)["observed_power"]
        assert low < mid < high

    def test_recommended_n_decreases_with_larger_effect(self) -> None:
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        small = validate_sample_size(n=500, effect_size=0.2)["recommended_n"]
        medium = validate_sample_size(n=500, effect_size=0.5)["recommended_n"]
        large = validate_sample_size(n=500, effect_size=0.8)["recommended_n"]
        assert small > medium > large

    def test_warns_when_n_below_200(self) -> None:
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with pytest.warns(UserWarning, match="below the typical guidance"):
            validate_sample_size(n=100, effect_size=0.5)

    def test_no_warning_at_or_above_200(self) -> None:
        import warnings as _w  # noqa: PLC0415

        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning would raise here
            validate_sample_size(n=200, effect_size=0.5)
            validate_sample_size(n=1000, effect_size=0.5)

    def test_echoes_inputs(self) -> None:
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        out = validate_sample_size(n=500, effect_size=0.42, alpha=0.01, power=0.95)
        assert out["n"] == 500.0
        assert out["effect_size"] == pytest.approx(0.42)
        assert out["alpha"] == pytest.approx(0.01)
        assert out["target_power"] == pytest.approx(0.95)

    def test_invalid_n_raises(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with pytest.raises(ValidationError, match="n must be >= 1"):
            validate_sample_size(n=0, effect_size=0.5)

    def test_invalid_effect_size_raises(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with pytest.raises(ValidationError, match="effect_size must be > 0"):
            validate_sample_size(n=500, effect_size=0.0)
        with pytest.raises(ValidationError, match="effect_size must be > 0"):
            validate_sample_size(n=500, effect_size=-0.3)

    def test_invalid_alpha_raises(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with pytest.raises(ValidationError, match=r"alpha must be in \(0, 1\)"):
            validate_sample_size(n=500, effect_size=0.5, alpha=0.0)
        with pytest.raises(ValidationError, match=r"alpha must be in \(0, 1\)"):
            validate_sample_size(n=500, effect_size=0.5, alpha=1.0)

    def test_invalid_power_raises(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import validate_sample_size  # noqa: PLC0415

        with pytest.raises(ValidationError, match=r"power must be in \(0, 1\)"):
            validate_sample_size(n=500, effect_size=0.5, power=0.0)
        with pytest.raises(ValidationError, match=r"power must be in \(0, 1\)"):
            validate_sample_size(n=500, effect_size=0.5, power=1.0)

    def test_exposed_at_top_level(self) -> None:
        import llm_consistency  # noqa: PLC0415

        assert hasattr(llm_consistency, "validate_sample_size")
        assert "validate_sample_size" in llm_consistency.__all__


class TestCompareMcaPaired:
    """McNemar's exact test on paired MCA-pass/fail outcomes."""

    @staticmethod
    def _qcrs(scores: dict[str, float]) -> list[QuestionConsistencyResult]:
        return [_qcr(qid, score) for qid, score in scores.items()]

    def test_identical_results_p_value_one(self) -> None:
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        rs = self._qcrs({"q1": 1.0, "q2": 0.9, "q3": 0.3, "q4": 0.0})
        out = compare_mca_paired(rs, rs, threshold=0.8)
        assert out.n_discordant == 0
        assert out.statistic == 0.0
        assert out.p_value == pytest.approx(1.0)
        assert out.method == "mcnemar_exact"

    def test_a_strictly_better(self) -> None:
        """A passes all, B fails all -- maximally discordant in one direction."""
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        a = self._qcrs(dict.fromkeys((f"q{i}" for i in range(8)), 1.0))
        b = self._qcrs(dict.fromkeys((f"q{i}" for i in range(8)), 0.0))
        out = compare_mca_paired(a, b, threshold=0.8)
        assert out.n_discordant == 8
        # b=8 (A passes, B fails), c=0; min=0; p = 2 * 0.5^8 = 2/256
        assert out.statistic == 0.0
        assert out.p_value == pytest.approx(2.0 / 256.0)

    def test_b_strictly_better_symmetric(self) -> None:
        """Symmetry: swapping a and b doesn't change p_value or n_discordant."""
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        a = self._qcrs(dict.fromkeys((f"q{i}" for i in range(8)), 0.0))
        b = self._qcrs(dict.fromkeys((f"q{i}" for i in range(8)), 1.0))
        out = compare_mca_paired(a, b, threshold=0.8)
        assert out.n_discordant == 8
        assert out.statistic == 0.0
        assert out.p_value == pytest.approx(2.0 / 256.0)

    def test_known_two_by_two(self) -> None:
        """Hand-built 2x2: b=5, c=1 -- two-sided exact p via Binomial(6, 0.5)."""
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        # 6 questions where A passes & B fails, 1 where A fails & B passes,
        # rest concordant (1 both pass, 1 both fail).
        a_scores = {
            **{f"ab{i}": 1.0 for i in range(5)},  # A pass, B fail
            "ba1": 0.0,  # A fail, B pass
            "both_pass": 1.0,
            "both_fail": 0.0,
        }
        b_scores = {
            **{f"ab{i}": 0.0 for i in range(5)},
            "ba1": 1.0,
            "both_pass": 1.0,
            "both_fail": 0.0,
        }
        out = compare_mca_paired(
            self._qcrs(a_scores), self._qcrs(b_scores), threshold=0.8
        )
        assert out.n_discordant == 6
        assert out.statistic == 1.0
        # 2 * (C(6,0) + C(6,1)) * 0.5^6 = 2 * 7 / 64 = 0.21875
        assert out.p_value == pytest.approx(0.21875)

    def test_drops_unpaired_question_ids(self) -> None:
        """Questions present in only one set are silently ignored."""
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        a = self._qcrs({"shared": 1.0, "only_a": 0.0})
        b = self._qcrs({"shared": 1.0, "only_b": 0.0})
        out = compare_mca_paired(a, b, threshold=0.8)
        # Only "shared" counted; both pass -> 0 discordant.
        assert out.n_discordant == 0
        assert out.p_value == pytest.approx(1.0)

    def test_no_shared_ids_raises(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        a = self._qcrs({"q1": 1.0})
        b = self._qcrs({"q2": 1.0})
        with pytest.raises(ValidationError, match="share no question IDs"):
            compare_mca_paired(a, b, threshold=0.8)

    def test_empty_inputs_raise(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        with pytest.raises(ValidationError, match="must both be non-empty"):
            compare_mca_paired([], self._qcrs({"q1": 1.0}), threshold=0.8)
        with pytest.raises(ValidationError, match="must both be non-empty"):
            compare_mca_paired(self._qcrs({"q1": 1.0}), [], threshold=0.8)

    def test_invalid_threshold_raises(self) -> None:
        from llm_consistency._exceptions import ValidationError  # noqa: PLC0415
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        a = self._qcrs({"q1": 1.0})
        b = self._qcrs({"q1": 0.0})
        with pytest.raises(ValidationError, match=r"threshold must be in"):
            compare_mca_paired(a, b, threshold=-0.1)
        with pytest.raises(ValidationError, match=r"threshold must be in"):
            compare_mca_paired(a, b, threshold=1.1)

    def test_p_value_capped_at_one(self) -> None:
        """When min(b,c) = floor(n/2), the doubled tail saturates at 1.0."""
        from llm_consistency.metrics import compare_mca_paired  # noqa: PLC0415

        # 2 discordant pairs in opposite directions -> b=1, c=1
        a = self._qcrs({"q1": 1.0, "q2": 0.0})
        b = self._qcrs({"q1": 0.0, "q2": 1.0})
        out = compare_mca_paired(a, b, threshold=0.8)
        assert out.n_discordant == 2
        assert out.statistic == 1.0
        assert out.p_value == pytest.approx(1.0)

    def test_paired_test_result_exposed_at_top_level(self) -> None:
        import llm_consistency  # noqa: PLC0415

        assert hasattr(llm_consistency, "PairedTestResult")
        assert hasattr(llm_consistency, "compare_mca_paired")
        assert "PairedTestResult" in llm_consistency.__all__
        assert "compare_mca_paired" in llm_consistency.__all__
