"""Tests for llm_consistency.scoring module."""

from __future__ import annotations

import importlib

import pytest

from llm_consistency._exceptions import ValidationError
from llm_consistency.scoring import (
    BaseScorer,
    CustomScorerAdapter,
    ExactMatchScorer,
    get_scorer,
)
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
    def test_answer_colon_format(self, raw_output: str, expected_correct: bool) -> None:
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


# ---------------------------------------------------------------------------
# CustomScorerAdapter tests (SCOR-05)
# ---------------------------------------------------------------------------


class TestCustomScorerAdapterFullSignature:
    """Tests for CustomScorerAdapter wrapping a full-signature callable."""

    def test_full_callable_returns_scored_response_unchanged(self) -> None:
        """Full (LLMResponse, MCQuestion) -> ScoredResponse callable returns as-is."""
        expected = ScoredResponse(
            question_id="q1",
            is_correct=True,
            score=0.95,
            scoring_method="custom_full",
        )

        def full_scorer(response: LLMResponse, question: MCQuestion) -> ScoredResponse:
            return expected

        adapter = CustomScorerAdapter(fn=full_scorer, name="custom_full")
        question = _make_question_4()
        response = _make_response("Answer: A")
        result = adapter.score(response, question)
        assert result is expected
        assert result.score == 0.95
        assert result.scoring_method == "custom_full"

    def test_full_callable_receives_correct_args(self) -> None:
        """Full callable receives the original LLMResponse and MCQuestion."""
        received_args: list[object] = []

        def capturing_scorer(
            response: LLMResponse, question: MCQuestion
        ) -> ScoredResponse:
            received_args.extend([response, question])
            return ScoredResponse(
                question_id=response.question_id,
                is_correct=False,
                score=0.0,
                scoring_method="capture",
            )

        adapter = CustomScorerAdapter(fn=capturing_scorer, name="capture")
        question = _make_question_4()
        response = _make_response("Answer: A")
        adapter.score(response, question)
        assert received_args[0] is response
        assert received_args[1] is question


class TestCustomScorerAdapterSimpleSignature:
    """Tests for CustomScorerAdapter wrapping a simple (str, str) -> bool callable."""

    def test_simple_callable_correct_answer(self) -> None:
        """Simple callable with matching answer returns is_correct=True, score=1.0."""
        adapter = CustomScorerAdapter(
            fn=lambda extracted, correct: extracted == correct,
            name="my_scorer",
            simple=True,
        )
        question = _make_question_4()  # A is correct
        response = _make_response("Answer: A")
        result = adapter.score(response, question)
        assert result.is_correct is True
        assert result.score == 1.0
        assert result.scoring_method == "my_scorer"

    def test_simple_callable_incorrect_answer(self) -> None:
        """Non-matching answer returns is_correct=False, score=0.0."""
        adapter = CustomScorerAdapter(
            fn=lambda extracted, correct: extracted == correct,
            name="my_scorer",
            simple=True,
        )
        question = _make_question_4()  # A is correct
        response = _make_response("Answer: B")
        result = adapter.score(response, question)
        assert result.is_correct is False
        assert result.score == 0.0

    def test_simple_callable_unextractable_passes_empty_string(self) -> None:
        """When extraction fails, empty string is passed to simple callable."""
        received: list[str] = []

        def tracking_fn(extracted: str, correct: str) -> bool:
            received.append(extracted)
            return extracted == correct

        adapter = CustomScorerAdapter(fn=tracking_fn, name="tracker", simple=True)
        question = _make_question_4()
        response = _make_response("I don't know the answer")
        result = adapter.score(response, question)
        assert received[0] == ""
        assert result.is_correct is False

    def test_simple_callable_question_id_propagated(self) -> None:
        """ScoredResponse.question_id comes from the response."""
        adapter = CustomScorerAdapter(
            fn=lambda e, c: True,
            name="always_right",
            simple=True,
        )
        question = _make_question_4()
        response = _make_response("A", question_id="q42")
        result = adapter.score(response, question)
        assert result.question_id == "q42"


class TestCustomScorerAdapterName:
    """Tests for CustomScorerAdapter.name property."""

    def test_custom_name(self) -> None:
        """CustomScorerAdapter returns the user-supplied name."""
        adapter = CustomScorerAdapter(
            fn=lambda r, q: ScoredResponse(
                question_id="x", is_correct=False, score=0.0, scoring_method="x"
            ),
            name="custom_fuzzy",
        )
        assert adapter.name == "custom_fuzzy"

    def test_default_name(self) -> None:
        """CustomScorerAdapter defaults to 'custom' if no name given."""
        adapter = CustomScorerAdapter(
            fn=lambda r, q: ScoredResponse(
                question_id="x", is_correct=False, score=0.0, scoring_method="x"
            ),
        )
        assert adapter.name == "custom"


class TestCustomScorerAdapterInheritance:
    """Tests for CustomScorerAdapter inheritance from BaseScorer."""

    def test_isinstance_base_scorer(self) -> None:
        """CustomScorerAdapter is an instance of BaseScorer."""
        adapter = CustomScorerAdapter(
            fn=lambda r, q: ScoredResponse(
                question_id="x", is_correct=False, score=0.0, scoring_method="x"
            ),
        )
        assert isinstance(adapter, BaseScorer)


class TestCustomScorerAdapterFullCallableReturnValidation:
    """Tests for full callable return type validation."""

    def test_full_callable_non_scored_response_raises(self) -> None:
        """Full callable returning non-ScoredResponse raises TypeError."""

        def bad_scorer(response: LLMResponse, question: MCQuestion) -> bool:  # type: ignore[override]
            return True

        adapter = CustomScorerAdapter(fn=bad_scorer, name="bad")  # type: ignore[arg-type]
        question = _make_question_4()
        response = _make_response("Answer: A")
        with pytest.raises(TypeError, match="ScoredResponse"):
            adapter.score(response, question)


# ---------------------------------------------------------------------------
# Public API tests (scoring exports)
# ---------------------------------------------------------------------------


class TestScoringPublicAPI:
    """Tests for scoring class availability in llm_consistency public API."""

    def test_base_scorer_importable(self) -> None:
        """BaseScorer is importable from llm_consistency top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "BaseScorer")

    def test_exact_match_scorer_importable(self) -> None:
        """ExactMatchScorer is importable from llm_consistency top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "ExactMatchScorer")

    def test_custom_scorer_adapter_importable(self) -> None:
        """CustomScorerAdapter is importable from llm_consistency top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "CustomScorerAdapter")

    def test_scoring_classes_in_all(self) -> None:
        """BaseScorer, ExactMatchScorer, CustomScorerAdapter are in __all__."""
        mod = importlib.import_module("llm_consistency")
        all_names = mod.__all__
        assert "BaseScorer" in all_names
        assert "ExactMatchScorer" in all_names
        assert "CustomScorerAdapter" in all_names


class TestGetScorerRegistry:
    """get_scorer() looks up scorers by name."""

    def test_get_scorer_returns_exact_match(self) -> None:
        scorer = get_scorer("exact_match")
        assert isinstance(scorer, ExactMatchScorer)
        assert scorer.name == "exact_match"

    def test_get_scorer_unknown_raises(self) -> None:
        with pytest.raises(ValidationError, match="Unknown scorer"):
            get_scorer("not_a_real_scorer")

    def test_get_scorer_exposed_in_public_api(self) -> None:
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "get_scorer")
        assert "get_scorer" in mod.__all__
