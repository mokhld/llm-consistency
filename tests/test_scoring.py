"""Tests for llm_consistency.scoring module."""

from __future__ import annotations

import pytest

from llm_consistency.scoring import BaseScorer, ExactMatchScorer
from llm_consistency.types import LLMResponse, MCOption, MCQuestion, ScoredResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_question_4() -> MCQuestion:
    """Standard 4-option question with A correct."""
    return MCQuestion(
        id="q1",
        stem="What is the capital of France?",
        options=(
            MCOption(label="A", text="Paris", is_correct=True),
            MCOption(label="B", text="London", is_correct=False),
            MCOption(label="C", text="Berlin", is_correct=False),
            MCOption(label="D", text="Madrid", is_correct=False),
        ),
    )


def _make_question_5() -> MCQuestion:
    """5-option question with E correct."""
    return MCQuestion(
        id="q2",
        stem="Which planet is largest?",
        options=(
            MCOption(label="A", text="Mars", is_correct=False),
            MCOption(label="B", text="Venus", is_correct=False),
            MCOption(label="C", text="Earth", is_correct=False),
            MCOption(label="D", text="Saturn", is_correct=False),
            MCOption(label="E", text="Jupiter", is_correct=True),
        ),
    )


def _make_response(raw_output: str, question_id: str = "q1") -> LLMResponse:
    """Helper to create an LLMResponse with a given raw_output."""
    return LLMResponse(
        question_id=question_id,
        raw_output=raw_output,
        extracted_answer="",
        model="test-model",
        provider="test-provider",
    )


# ---------------------------------------------------------------------------
# BaseScorer ABC tests (SCOR-01)
# ---------------------------------------------------------------------------


class TestBaseScorerABC:
    """Tests for BaseScorer abstract base class enforcement."""

    def test_cannot_instantiate_directly(self) -> None:
        """Attempting to instantiate BaseScorer directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseScorer()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_score(self) -> None:
        """Subclass missing score() raises TypeError on instantiation."""

        class NameOnlyScorer(BaseScorer):
            @property
            def name(self) -> str:
                return "name_only"

        with pytest.raises(TypeError):
            NameOnlyScorer()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_name(self) -> None:
        """Subclass missing name property raises TypeError on instantiation."""

        class ScoreOnlyScorer(BaseScorer):
            def score(
                self, response: LLMResponse, question: MCQuestion
            ) -> ScoredResponse:
                return ScoredResponse(
                    question_id=response.question_id,
                    is_correct=False,
                    score=0.0,
                    scoring_method="test",
                )

        with pytest.raises(TypeError):
            ScoreOnlyScorer()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """A subclass implementing both name and score() can be created."""

        class CompleteScorer(BaseScorer):
            @property
            def name(self) -> str:
                return "complete"

            def score(
                self, response: LLMResponse, question: MCQuestion
            ) -> ScoredResponse:
                return ScoredResponse(
                    question_id=response.question_id,
                    is_correct=False,
                    score=0.0,
                    scoring_method=self.name,
                )

        scorer = CompleteScorer()
        assert scorer.name == "complete"


# ---------------------------------------------------------------------------
# ExactMatchScorer tests (SCOR-02)
# ---------------------------------------------------------------------------


class TestExactMatchScorerName:
    """Tests for ExactMatchScorer.name property."""

    def test_name_returns_exact_match(self) -> None:
        """ExactMatchScorer.name returns 'exact_match'."""
        scorer = ExactMatchScorer()
        assert scorer.name == "exact_match"


class TestExactMatchScorerStrategy1:
    """Strategy 1: 'Answer: X' format extraction."""

    @pytest.mark.parametrize(
        ("raw_output", "expected_correct"),
        [
            ("Answer: A", True),
            ("answer: B", False),
            ("Answer: (C)", False),
            ("ANSWER: D", False),
            ("The reasoning is complex. Answer: A", True),
        ],
    )
    def test_answer_colon_format(
        self, raw_output: str, expected_correct: bool
    ) -> None:
        """'Answer: X' pattern correctly extracts the label."""
        question = _make_question_4()
        response = _make_response(raw_output)
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is expected_correct
        assert result.score == (1.0 if expected_correct else 0.0)


class TestExactMatchScorerStrategy2:
    """Strategy 2: 'The answer is X' format extraction."""

    @pytest.mark.parametrize(
        ("raw_output", "expected_correct"),
        [
            ("The answer is A", True),
            ("I think the answer is B because...", False),
            ("the answer is (C)", False),
            ("THE ANSWER IS A", True),
        ],
    )
    def test_the_answer_is_format(
        self, raw_output: str, expected_correct: bool
    ) -> None:
        """'The answer is X' pattern correctly extracts the label."""
        question = _make_question_4()
        response = _make_response(raw_output)
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is expected_correct
        assert result.score == (1.0 if expected_correct else 0.0)


class TestExactMatchScorerStrategy3:
    """Strategy 3: First standalone valid label."""

    def test_standalone_label_extraction(self) -> None:
        """Extracts first standalone valid label from prose."""
        question = _make_question_4()
        # "B" appears as a standalone word boundary match
        response = _make_response("After analysis, B is correct")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is False  # B is not correct, A is
        assert result.score == 0.0

    def test_first_valid_label_wins(self) -> None:
        """When multiple labels appear, the first one is extracted."""
        question = _make_question_4()
        response = _make_response("I considered A and B, but C is best")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        # First standalone label is A, which is correct
        assert result.is_correct is True
        assert result.score == 1.0


class TestExactMatchScorerStrategy4:
    """Strategy 4: Single character output."""

    def test_single_char(self) -> None:
        """Single character 'A' extracts correctly."""
        question = _make_question_4()
        response = _make_response("A")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is True
        assert result.score == 1.0

    def test_single_char_with_whitespace(self) -> None:
        """Single character with surrounding whitespace extracts correctly."""
        question = _make_question_4()
        response = _make_response(" B ")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is False
        assert result.score == 0.0


class TestExactMatchScorerUnextractable:
    """Unextractable output edge cases."""

    def test_empty_string(self) -> None:
        """Empty string returns is_correct=False, score=0.0."""
        question = _make_question_4()
        response = _make_response("")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is False
        assert result.score == 0.0

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns is_correct=False, score=0.0."""
        question = _make_question_4()
        response = _make_response("  ")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is False
        assert result.score == 0.0

    def test_no_valid_label(self) -> None:
        """Text with no valid label returns is_correct=False, score=0.0."""
        question = _make_question_4()
        response = _make_response("I don't know")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is False
        assert result.score == 0.0


class TestExactMatchScorerCaseInsensitivity:
    """Case-insensitive extraction tests."""

    def test_lowercase_extracted_as_uppercase(self) -> None:
        """Lowercase 'a' is extracted and compared as 'A'."""
        question = _make_question_4()
        response = _make_response("a")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is True
        assert result.score == 1.0


class TestExactMatchScorerDynamicLabels:
    """Dynamic valid label set tests."""

    def test_five_option_question(self) -> None:
        """5-option question where E is correct and extracted."""
        question = _make_question_5()
        response = _make_response("Answer: E", question_id="q2")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is True
        assert result.score == 1.0

    def test_five_option_wrong_answer(self) -> None:
        """5-option question where B is extracted but E is correct."""
        question = _make_question_5()
        response = _make_response("Answer: B", question_id="q2")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.is_correct is False
        assert result.score == 0.0


class TestExactMatchScorerScoringMethod:
    """ScoredResponse.scoring_method field tests."""

    def test_scoring_method_always_exact_match(self) -> None:
        """Every ScoredResponse has scoring_method='exact_match'."""
        question = _make_question_4()
        scorer = ExactMatchScorer()

        for raw in ["Answer: A", "B", "I don't know", ""]:
            response = _make_response(raw)
            result = scorer.score(response, question)
            assert result.scoring_method == "exact_match"


class TestExactMatchScorerQuestionId:
    """ScoredResponse.question_id propagation tests."""

    def test_question_id_matches_response(self) -> None:
        """ScoredResponse.question_id matches response.question_id."""
        question = _make_question_4()
        response = _make_response("Answer: A", question_id="q1")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.question_id == "q1"

    def test_question_id_propagates_custom_id(self) -> None:
        """ScoredResponse carries the exact question_id from response."""
        question = _make_question_5()
        response = _make_response("Answer: E", question_id="q2")
        scorer = ExactMatchScorer()
        result = scorer.score(response, question)
        assert result.question_id == "q2"
