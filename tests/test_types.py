"""Tests for llm_consistency.types module."""

from __future__ import annotations

import json

import pytest

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.types import PerturbationType


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
