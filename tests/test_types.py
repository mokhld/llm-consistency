"""Tests for llm_consistency.types module."""

from __future__ import annotations

import json

import pytest

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.types import MCOption, MCQuestion, PerturbationType


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
        with pytest.raises(ValidationError, match="[Dd]uplicate"):
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
        q = MCQuestion(
            id="q1", stem="Capital of France?", options=valid_options
        )
        restored = MCQuestion.from_dict(json.loads(json.dumps(q.to_dict())))
        assert q == restored
        assert hash(q) == hash(restored)
