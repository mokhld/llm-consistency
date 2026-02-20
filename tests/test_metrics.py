"""Tests for llm_consistency.metrics - builder, MCA, CAR curve, AUC, DTW, CORE."""

from __future__ import annotations

import pytest

from llm_consistency.metrics import (
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
        assert trapezoidal_auc(
            [0.0, 0.5, 1.0], [1.0, 0.75, 0.25]
        ) == pytest.approx(0.6875)

    def test_non_uniform_spacing(self) -> None:
        """Non-uniform x spacing: xs=[0, 0.3, 1.0], ys=[1.0, 1.0, 0.0].

        area = 0.3*(1.0+1.0)/2 + 0.7*(1.0+0.0)/2 = 0.3 + 0.35 = 0.65
        """
        assert trapezoidal_auc(
            [0.0, 0.3, 1.0], [1.0, 1.0, 0.0]
        ) == pytest.approx(0.65)

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
        assert trapezoidal_auc([0.0, 0.5, 1.0], [0.0, 0.0, 0.0]) == pytest.approx(
            0.0
        )

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
        assert dtw_distance(
            [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]
        ) == pytest.approx(2.0)

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
        assert core_index(results, thresholds=[0.0, 0.5, 1.0]) == pytest.approx(
            0.34375
        )

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
