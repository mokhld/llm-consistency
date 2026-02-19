"""Tests for llm_consistency.types module."""

from __future__ import annotations

import json

import pytest

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.types import (
    LLMResponse,
    MCOption,
    MCQuestion,
    OpenEndedQuestion,
    PerturbationType,
    PerturbedVariant,
    ScoredResponse,
)

# --- Exception hierarchy tests ---


class TestValidationError:
    """Tests for the custom exception hierarchy."""

    def test_validation_error_is_value_error(self) -> None:
        """ValidationError is catchable as both ValueError and LLMConsistencyError."""
        err = ValidationError("test message")
        assert isinstance(err, ValueError)
        assert isinstance(err, LLMConsistencyError)
        assert isinstance(err, Exception)

    def test_validation_error_message(self) -> None:
        """ValidationError carries the error message string."""
        err = ValidationError("field is invalid")
        assert str(err) == "field is invalid"


# --- PerturbationType enum tests ---


class TestPerturbationType:
    """Tests for the PerturbationType enum."""

    def test_perturbation_type_members(self) -> None:
        """Enum has exactly 5 members."""
        members = list(PerturbationType)
        assert len(members) == 5
        expected_names = {
            "OPTION_REORDER",
            "FORMAT_CHANGE",
            "SEPARATOR_CHANGE",
            "PARAPHRASE",
            "INSTRUCTION_REPHRASE",
        }
        assert {m.name for m in members} == expected_names

    def test_perturbation_type_name_serialization(self) -> None:
        """.name returns UPPER_CASE string; PerturbationType[name] round-trips."""
        for member in PerturbationType:
            name = member.name
            assert name == name.upper()
            assert PerturbationType[name] is member

    def test_perturbation_type_values(self) -> None:
        """Each member has a lowercase snake_case string value."""
        for member in PerturbationType:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()
            assert "_" in member.value or member.value.isalpha()


# --- MCOption tests ---


class TestMCOption:
    """Tests for the MCOption frozen dataclass."""

    def test_mc_option_construction(self) -> None:
        """Create MCOption and verify field access."""
        opt = MCOption(label="A", text="Paris", is_correct=True)
        assert opt.label == "A"
        assert opt.text == "Paris"
        assert opt.is_correct is True

    def test_mc_option_frozen(self) -> None:
        """Mutation raises FrozenInstanceError."""
        opt = MCOption(label="A", text="Paris", is_correct=True)
        with pytest.raises(AttributeError):
            opt.label = "B"  # type: ignore[misc]

    def test_mc_option_hashable(self) -> None:
        """hash(opt) works; two equal MCOptions have same hash."""
        opt1 = MCOption(label="A", text="Paris", is_correct=True)
        opt2 = MCOption(label="A", text="Paris", is_correct=True)
        assert hash(opt1) == hash(opt2)
        assert hash(opt1) is not None

    def test_mc_option_equality(self) -> None:
        """Two MCOptions with same fields are equal; different fields are not."""
        opt1 = MCOption(label="A", text="Paris", is_correct=True)
        opt2 = MCOption(label="A", text="Paris", is_correct=True)
        opt3 = MCOption(label="B", text="London", is_correct=False)
        assert opt1 == opt2
        assert opt1 != opt3

    def test_mc_option_round_trip(self) -> None:
        """MCOption.from_dict(json.loads(json.dumps(opt.to_dict()))) == opt."""
        opt = MCOption(label="A", text="Paris", is_correct=True)
        restored = MCOption.from_dict(json.loads(json.dumps(opt.to_dict())))
        assert opt == restored


# --- MCQuestion tests ---


class TestMCQuestion:
    """Tests for the MCQuestion frozen dataclass with validation."""

    @pytest.fixture()
    def valid_options(self) -> tuple[MCOption, ...]:
        """Return a valid set of MC options."""
        return (
            MCOption(label="A", text="Paris", is_correct=True),
            MCOption(label="B", text="London", is_correct=False),
            MCOption(label="C", text="Berlin", is_correct=False),
        )

    def test_mc_question_construction(self, valid_options) -> None:
        """Create MCQuestion with valid options, verify fields."""
        q = MCQuestion(id="q1", stem="Capital of France?", options=valid_options)
        assert q.id == "q1"
        assert q.stem == "Capital of France?"
        assert len(q.options) == 3

    def test_mc_question_exactly_one_correct_zero(self) -> None:
        """Zero correct options raises ValidationError with 'got 0'."""
        opts = (
            MCOption(label="A", text="X", is_correct=False),
            MCOption(label="B", text="Y", is_correct=False),
        )
        with pytest.raises(ValidationError, match="got 0"):
            MCQuestion(id="q1", stem="?", options=opts)

    def test_mc_question_exactly_one_correct_two(self) -> None:
        """Two correct options raises ValidationError with 'got 2'."""
        opts = (
            MCOption(label="A", text="X", is_correct=True),
            MCOption(label="B", text="Y", is_correct=True),
        )
        with pytest.raises(ValidationError, match="got 2"):
            MCQuestion(id="q1", stem="?", options=opts)

    def test_mc_question_empty_options(self) -> None:
        """Empty options tuple raises ValidationError."""
        with pytest.raises(ValidationError, match="at least one option"):
            MCQuestion(id="q1", stem="?", options=())

    def test_mc_question_empty_id(self, valid_options) -> None:
        """Empty string id raises ValidationError."""
        with pytest.raises(ValidationError, match="non-empty"):
            MCQuestion(id="", stem="?", options=valid_options)

    def test_mc_question_empty_stem(self, valid_options) -> None:
        """Empty string stem raises ValidationError."""
        with pytest.raises(ValidationError, match="non-empty"):
            MCQuestion(id="q1", stem="", options=valid_options)

    def test_mc_question_duplicate_labels(self) -> None:
        """Two options with same label raises ValidationError."""
        opts = (
            MCOption(label="A", text="X", is_correct=True),
            MCOption(label="A", text="Y", is_correct=False),
        )
        with pytest.raises(ValidationError, match=r"[Dd]uplicate"):
            MCQuestion(id="q1", stem="?", options=opts)

    def test_mc_question_frozen(self, valid_options) -> None:
        """Mutation raises FrozenInstanceError."""
        q = MCQuestion(id="q1", stem="?", options=valid_options)
        with pytest.raises(AttributeError):
            q.id = "q2"  # type: ignore[misc]

    def test_mc_question_hashable(self, valid_options) -> None:
        """hash(q) works; equal questions have same hash."""
        q1 = MCQuestion(id="q1", stem="?", options=valid_options)
        q2 = MCQuestion(id="q1", stem="?", options=valid_options)
        assert hash(q1) == hash(q2)

    def test_mc_question_round_trip(self, valid_options) -> None:
        """MCQuestion survives to_dict -> JSON -> from_dict -> equality."""
        q = MCQuestion(id="q1", stem="Capital of France?", options=valid_options)
        restored = MCQuestion.from_dict(json.loads(json.dumps(q.to_dict())))
        assert q == restored
        assert hash(q) == hash(restored)


# --- OpenEndedQuestion tests ---


class TestOpenEndedQuestion:
    """Tests for the OpenEndedQuestion frozen dataclass."""

    def test_open_ended_question_construction(self) -> None:
        """Create with id, stem, reference_answers tuple."""
        q = OpenEndedQuestion(
            id="oe1",
            stem="What is the capital of France?",
            reference_answers=("Paris", "paris"),
        )
        assert q.id == "oe1"
        assert q.stem == "What is the capital of France?"
        assert q.reference_answers == ("Paris", "paris")

    def test_open_ended_question_frozen(self) -> None:
        """Mutation raises FrozenInstanceError."""
        q = OpenEndedQuestion(id="oe1", stem="?", reference_answers=("Paris",))
        with pytest.raises(AttributeError):
            q.id = "oe2"  # type: ignore[misc]

    def test_open_ended_question_hashable(self) -> None:
        """hash works, equal instances have same hash."""
        q1 = OpenEndedQuestion(id="oe1", stem="?", reference_answers=("Paris",))
        q2 = OpenEndedQuestion(id="oe1", stem="?", reference_answers=("Paris",))
        assert hash(q1) == hash(q2)

    def test_open_ended_question_round_trip(self) -> None:
        """to_dict -> JSON -> from_dict -> equality."""
        q = OpenEndedQuestion(
            id="oe1",
            stem="What is the capital of France?",
            reference_answers=("Paris", "paris"),
        )
        restored = OpenEndedQuestion.from_dict(json.loads(json.dumps(q.to_dict())))
        assert q == restored
        assert hash(q) == hash(restored)

    def test_open_ended_question_empty_id_rejected(self) -> None:
        """Empty id raises ValidationError."""
        with pytest.raises(ValidationError, match="non-empty"):
            OpenEndedQuestion(id="", stem="?", reference_answers=("Paris",))

    def test_open_ended_question_empty_stem_rejected(self) -> None:
        """Empty stem raises ValidationError."""
        with pytest.raises(ValidationError, match="non-empty"):
            OpenEndedQuestion(id="oe1", stem="", reference_answers=("Paris",))


# --- PerturbedVariant tests ---


class TestPerturbedVariant:
    """Tests for the PerturbedVariant frozen dataclass."""

    def test_perturbed_variant_construction(self) -> None:
        """Create with required fields."""
        opts = (
            MCOption(label="A", text="Paris", is_correct=True),
            MCOption(label="B", text="London", is_correct=False),
        )
        v = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.OPTION_REORDER,
            seed=42,
            variant_index=0,
            stem="Capital of France?",
            options=opts,
        )
        assert v.original_question_id == "q1"
        assert v.perturbation_type is PerturbationType.OPTION_REORDER
        assert v.seed == 42
        assert v.variant_index == 0
        assert v.stem == "Capital of France?"
        assert v.options == opts

    def test_perturbed_variant_frozen(self) -> None:
        """Mutation raises FrozenInstanceError."""
        v = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.PARAPHRASE,
            seed=42,
            variant_index=0,
            stem="?",
        )
        with pytest.raises(AttributeError):
            v.seed = 99  # type: ignore[misc]

    def test_perturbed_variant_hashable(self) -> None:
        """hash works."""
        v = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.FORMAT_CHANGE,
            seed=42,
            variant_index=0,
            stem="?",
        )
        assert isinstance(hash(v), int)

    def test_perturbed_variant_provenance(self) -> None:
        """perturbation_type is PerturbationType enum, seed is int."""
        v = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.SEPARATOR_CHANGE,
            seed=123,
            variant_index=1,
            stem="?",
        )
        assert isinstance(v.perturbation_type, PerturbationType)
        assert isinstance(v.seed, int)

    def test_perturbed_variant_round_trip(self) -> None:
        """to_dict -> JSON -> from_dict -> equality; enum serializes as name."""
        v = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.INSTRUCTION_REPHRASE,
            seed=42,
            variant_index=2,
            stem="Rephased question?",
        )
        d = v.to_dict()
        # Verify enum serializes as UPPER_CASE name
        assert d["perturbation_type"] == "INSTRUCTION_REPHRASE"
        restored = PerturbedVariant.from_dict(json.loads(json.dumps(d)))
        assert v == restored

    def test_perturbed_variant_mc_with_options(self) -> None:
        """Variant with MC options round-trips."""
        opts = (
            MCOption(label="A", text="Paris", is_correct=True),
            MCOption(label="B", text="London", is_correct=False),
        )
        v = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.OPTION_REORDER,
            seed=42,
            variant_index=0,
            stem="Capital of France?",
            options=opts,
        )
        restored = PerturbedVariant.from_dict(json.loads(json.dumps(v.to_dict())))
        assert v == restored
        assert restored.options is not None
        assert len(restored.options) == 2

    def test_perturbed_variant_open_ended_no_options(self) -> None:
        """Variant without options (None) round-trips."""
        v = PerturbedVariant(
            original_question_id="oe1",
            perturbation_type=PerturbationType.PARAPHRASE,
            seed=42,
            variant_index=0,
            stem="Open ended paraphrase?",
            options=None,
        )
        restored = PerturbedVariant.from_dict(json.loads(json.dumps(v.to_dict())))
        assert v == restored
        assert restored.options is None


# --- LLMResponse tests ---


class TestLLMResponse:
    """Tests for the LLMResponse frozen dataclass."""

    def test_llm_response_construction(self) -> None:
        """Create with all required fields; verify field access."""
        resp = LLMResponse(
            question_id="q1",
            raw_output="The answer is A",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
        )
        assert resp.question_id == "q1"
        assert resp.raw_output == "The answer is A"
        assert resp.extracted_answer == "A"
        assert resp.model == "gpt-4o"
        assert resp.provider == "openai"

    def test_llm_response_optional_fields_default_none(self) -> None:
        """Omit latency_ms, prompt_tokens, completion_tokens; all default to None."""
        resp = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
        )
        assert resp.latency_ms is None
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None

    def test_llm_response_with_optional_fields(self) -> None:
        """Create with latency_ms=150.5, prompt_tokens=100, completion_tokens=50."""
        resp = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
            latency_ms=150.5,
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert resp.latency_ms == 150.5
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50

    def test_llm_response_frozen(self) -> None:
        """Mutation raises FrozenInstanceError."""
        resp = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
        )
        with pytest.raises(AttributeError):
            resp.question_id = "q2"  # type: ignore[misc]

    def test_llm_response_hashable(self) -> None:
        """hash works; equal instances have same hash."""
        resp1 = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
        )
        resp2 = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
        )
        assert hash(resp1) == hash(resp2)
        assert isinstance(hash(resp1), int)

    def test_llm_response_equality_uses_all_fields(self) -> None:
        """Two instances differing only in optional field are NOT equal."""
        resp1 = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
            latency_ms=100.0,
        )
        resp2 = LLMResponse(
            question_id="q1",
            raw_output="output",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
            latency_ms=200.0,
        )
        assert resp1 != resp2

    def test_llm_response_kw_only(self) -> None:
        """Positional construction raises TypeError (must use keyword arguments)."""
        with pytest.raises(TypeError):
            LLMResponse("q1", "output", "A", "gpt-4o", "openai")  # type: ignore[misc]

    def test_llm_response_round_trip(self) -> None:
        """to_dict -> JSON -> from_dict -> equality (None optional fields as null)."""
        resp = LLMResponse(
            question_id="q1",
            raw_output="The answer is A",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
        )
        d = resp.to_dict()
        # None values should be present in dict
        assert "latency_ms" in d
        assert d["latency_ms"] is None
        restored = LLMResponse.from_dict(json.loads(json.dumps(d)))
        assert resp == restored

    def test_llm_response_round_trip_with_optionals(self) -> None:
        """Same round-trip with all optional fields populated."""
        resp = LLMResponse(
            question_id="q1",
            raw_output="The answer is A",
            extracted_answer="A",
            model="gpt-4o",
            provider="openai",
            latency_ms=150.5,
            prompt_tokens=100,
            completion_tokens=50,
        )
        restored = LLMResponse.from_dict(json.loads(json.dumps(resp.to_dict())))
        assert resp == restored
        assert restored.latency_ms == 150.5
        assert restored.prompt_tokens == 100
        assert restored.completion_tokens == 50


# --- ScoredResponse tests ---


class TestScoredResponse:
    """Tests for the ScoredResponse frozen dataclass."""

    def test_scored_response_construction(self) -> None:
        """Create with question_id, is_correct, score, scoring_method."""
        sr = ScoredResponse(
            question_id="q1",
            is_correct=True,
            score=1.0,
            scoring_method="exact_match",
        )
        assert sr.question_id == "q1"
        assert sr.is_correct is True
        assert sr.score == 1.0
        assert sr.scoring_method == "exact_match"

    def test_scored_response_frozen(self) -> None:
        """Mutation raises FrozenInstanceError."""
        sr = ScoredResponse(
            question_id="q1",
            is_correct=True,
            score=1.0,
            scoring_method="exact_match",
        )
        with pytest.raises(AttributeError):
            sr.score = 0.5  # type: ignore[misc]

    def test_scored_response_hashable(self) -> None:
        """hash works."""
        sr = ScoredResponse(
            question_id="q1",
            is_correct=True,
            score=1.0,
            scoring_method="exact_match",
        )
        assert isinstance(hash(sr), int)

    def test_scored_response_round_trip(self) -> None:
        """to_dict -> JSON -> from_dict -> equality."""
        sr = ScoredResponse(
            question_id="q1",
            is_correct=False,
            score=0.75,
            scoring_method="semantic_similarity",
        )
        restored = ScoredResponse.from_dict(json.loads(json.dumps(sr.to_dict())))
        assert sr == restored
