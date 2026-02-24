"""Tests for llm_consistency.providers._batch_result module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from llm_consistency.providers._batch_result import BatchResult
from llm_consistency.types import LLMResponse


def _make_response(question_id: str = "q1") -> LLMResponse:
    """Helper to create a minimal LLMResponse."""
    return LLMResponse(
        question_id=question_id,
        raw_output="The answer is A",
        extracted_answer="A",
        model="test-model",
        provider="test",
    )


class TestBatchResult:
    """Tests for BatchResult frozen dataclass."""

    def test_is_frozen(self) -> None:
        result = BatchResult(responses=(), errors=())
        with pytest.raises(FrozenInstanceError):
            result.responses = ()  # type: ignore[misc]

    def test_attempted_equals_responses_plus_errors(self) -> None:
        result = BatchResult(
            responses=(
                _make_response("q1"),
                _make_response("q2"),
            ),
            errors=(("q3", "timeout"),),
        )
        assert result.attempted == 3

    def test_completed_equals_len_responses(self) -> None:
        result = BatchResult(
            responses=(
                _make_response("q1"),
                _make_response("q2"),
            ),
            errors=(),
        )
        assert result.completed == 2

    def test_failed_equals_len_errors(self) -> None:
        result = BatchResult(
            responses=(),
            errors=(
                ("q1", "rate limit"),
                ("q2", "timeout"),
            ),
        )
        assert result.failed == 2

    def test_empty_batch_result_has_zero_counts(self) -> None:
        result = BatchResult(responses=(), errors=())
        assert result.attempted == 0
        assert result.completed == 0
        assert result.failed == 0

    def test_mixed_responses_and_errors(self) -> None:
        result = BatchResult(
            responses=(
                _make_response("q1"),
                _make_response("q2"),
                _make_response("q3"),
            ),
            errors=(
                ("q4", "connection error"),
                ("q5", "budget exceeded"),
            ),
        )
        assert result.attempted == 5
        assert result.completed == 3
        assert result.failed == 2

    def test_errors_are_question_id_message_pairs(self) -> None:
        errors = (("q1", "timeout"), ("q2", "rate limit"))
        result = BatchResult(responses=(), errors=errors)
        assert result.errors[0] == ("q1", "timeout")
        assert result.errors[1] == ("q2", "rate limit")
