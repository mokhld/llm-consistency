"""Tests for llm_consistency.metrics - build_question_consistency_result."""

from __future__ import annotations

import pytest

from llm_consistency.metrics import build_question_consistency_result
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
