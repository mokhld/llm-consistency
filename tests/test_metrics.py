"""Tests for llm_consistency.metrics - builder, MCA, and CAR curve."""

from __future__ import annotations

import pytest

from llm_consistency.metrics import (
    build_question_consistency_result,
    car_curve,
    mca,
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
        for actual, exp in zip(thresholds_out, expected):
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
