"""Tests for ``MCDataset.load_from_hub`` -- fully mocked, no real Hub calls."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_consistency._exceptions import ValidationError
from llm_consistency.datasets import MCDataset
from llm_consistency.types import MCQuestion


def _mmlu_row(
    *,
    question: str = "What is 2 + 2?",
    choices: list[str] | None = None,
    answer: Any = 1,
) -> dict[str, Any]:
    """Return a single row matching the cais/mmlu schema."""
    return {
        "question": question,
        "choices": list(choices) if choices is not None else ["3", "4", "5", "6"],
        "answer": answer,
    }


def _install_fake_datasets(load_dataset: MagicMock) -> dict[str, ModuleType]:
    """Build a fake ``datasets`` module whose ``load_dataset`` is the given mock."""
    fake = ModuleType("datasets")
    fake.load_dataset = load_dataset  # type: ignore[attr-defined]
    return {"datasets": fake}


# ---------------------------------------------------------------------------
# ImportError guard
# ---------------------------------------------------------------------------


class TestImportError:
    """A missing ``datasets`` package surfaces a clear install hint."""

    def test_import_error_message(self) -> None:
        with (
            patch.dict(sys.modules, {"datasets": None}),
            pytest.raises(ImportError, match=r"llm-consistency\[huggingface\]"),
        ):
            MCDataset.load_from_hub("cais/mmlu")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """MMLU-style schema maps cleanly to MCQuestion."""

    def test_mmlu_int_answer(self) -> None:
        rows = [
            _mmlu_row(question="Q1?", choices=["a", "b", "c", "d"], answer=0),
            _mmlu_row(question="Q2?", choices=["a", "b", "c", "d"], answer=2),
        ]
        mock_load = MagicMock(return_value=rows)
        with patch.dict(sys.modules, _install_fake_datasets(mock_load)):
            ds = MCDataset.load_from_hub("cais/mmlu")

        assert len(ds) == 2
        first, second = ds.questions
        assert isinstance(first, MCQuestion)
        assert first.id == "row_0"
        assert first.stem == "Q1?"
        assert [o.label for o in first.options] == ["A", "B", "C", "D"]
        assert [o.text for o in first.options] == ["a", "b", "c", "d"]
        correct_a = [o for o in first.options if o.is_correct]
        assert len(correct_a) == 1
        assert correct_a[0].label == "A"

        correct_b = [o for o in second.options if o.is_correct]
        assert correct_b[0].label == "C"

    def test_answer_as_letter_label(self) -> None:
        rows = [_mmlu_row(choices=["a", "b", "c", "d"], answer="B")]
        with patch.dict(
            sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
        ):
            ds = MCDataset.load_from_hub("ex/mc")

        correct = [o for o in ds.questions[0].options if o.is_correct]
        assert correct[0].label == "B"

    def test_answer_as_choice_text(self) -> None:
        rows = [_mmlu_row(choices=["red", "blue", "green"], answer="blue")]
        with patch.dict(
            sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
        ):
            ds = MCDataset.load_from_hub("ex/mc")

        correct = [o for o in ds.questions[0].options if o.is_correct]
        assert correct[0].label == "B"
        assert correct[0].text == "blue"

    def test_answer_letter_lowercase(self) -> None:
        rows = [_mmlu_row(choices=["a", "b", "c", "d"], answer="d")]
        with patch.dict(
            sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
        ):
            ds = MCDataset.load_from_hub("ex/mc")

        correct = [o for o in ds.questions[0].options if o.is_correct]
        assert correct[0].label == "D"


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------


class TestArgumentForwarding:
    """Optional kwargs are passed through to ``datasets.load_dataset``."""

    def test_default_call_signature(self) -> None:
        mock_load = MagicMock(return_value=[_mmlu_row()])
        with patch.dict(sys.modules, _install_fake_datasets(mock_load)):
            MCDataset.load_from_hub("cais/mmlu")

        mock_load.assert_called_once_with(
            "cais/mmlu", name=None, split="train", token=None
        )

    def test_forwards_split_name_token_and_kwargs(self) -> None:
        mock_load = MagicMock(return_value=[_mmlu_row()])
        with patch.dict(sys.modules, _install_fake_datasets(mock_load)):
            MCDataset.load_from_hub(
                "cais/mmlu",
                split="validation",
                name="abstract_algebra",
                token="hf_token",
                streaming=False,
                revision="main",
            )

        mock_load.assert_called_once_with(
            "cais/mmlu",
            name="abstract_algebra",
            split="validation",
            token="hf_token",
            streaming=False,
            revision="main",
        )

    def test_custom_column_mapping(self) -> None:
        rows = [
            {
                "prompt": "Pick one",
                "options": ["x", "y", "z"],
                "label": 2,
                "qid": "custom-1",
            }
        ]
        with patch.dict(
            sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
        ):
            ds = MCDataset.load_from_hub(
                "ex/mc",
                question_col="prompt",
                choices_col="options",
                answer_col="label",
                id_col="qid",
            )

        q = ds.questions[0]
        assert q.id == "custom-1"
        assert q.stem == "Pick one"
        assert [o.text for o in q.options] == ["x", "y", "z"]
        assert next(o for o in q.options if o.is_correct).label == "C"

    def test_subset_name_in_validation_error_message(self) -> None:
        rows = [{"question": "Q?", "choices": [], "answer": 0}]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match=r"cais/mmlu\[abstract_algebra\]"),
        ):
            MCDataset.load_from_hub("cais/mmlu", name="abstract_algebra")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Malformed rows raise ``ValidationError`` with a useful message."""

    def test_missing_question_column(self) -> None:
        rows = [{"choices": ["a", "b"], "answer": 0}]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="'question' not found"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_missing_choices_column(self) -> None:
        rows = [{"question": "Q?", "answer": 0}]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="'choices' not found"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_missing_answer_column(self) -> None:
        rows = [{"question": "Q?", "choices": ["a", "b"]}]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="'answer' not found"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_missing_id_column(self) -> None:
        rows = [_mmlu_row()]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="'qid' not found"),
        ):
            MCDataset.load_from_hub("ex/mc", id_col="qid")

    def test_choices_not_a_list(self) -> None:
        rows = [{"question": "Q?", "choices": "abcd", "answer": 0}]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="expected a list of strings"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_empty_choices(self) -> None:
        rows = [{"question": "Q?", "choices": [], "answer": 0}]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="has no choices"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_too_many_choices(self) -> None:
        rows = [
            {
                "question": "Q?",
                "choices": [f"opt_{i}" for i in range(27)],
                "answer": 0,
            }
        ]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="only up to 26"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_answer_index_out_of_range(self) -> None:
        rows = [_mmlu_row(choices=["a", "b"], answer=5)]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="out of range"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_answer_label_out_of_range(self) -> None:
        rows = [_mmlu_row(choices=["a", "b"], answer="D")]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="out of range"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_answer_text_no_match(self) -> None:
        rows = [_mmlu_row(choices=["red", "blue"], answer="green")]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="does not match any choice"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_answer_bool_rejected(self) -> None:
        # Bool is an int subclass; ensure it's caught explicitly.
        rows = [_mmlu_row(choices=["a", "b"], answer=True)]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="bool"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_answer_unsupported_type(self) -> None:
        rows = [_mmlu_row(choices=["a", "b"], answer=1.5)]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="Unsupported answer type"),
        ):
            MCDataset.load_from_hub("ex/mc")

    def test_duplicate_ids_via_id_col(self) -> None:
        rows = [
            {"question": "Q1?", "choices": ["a", "b"], "answer": 0, "qid": "dup"},
            {"question": "Q2?", "choices": ["a", "b"], "answer": 1, "qid": "dup"},
        ]
        with (
            patch.dict(
                sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
            ),
            pytest.raises(ValidationError, match="Duplicate question IDs"),
        ):
            MCDataset.load_from_hub("ex/mc", id_col="qid")


# ---------------------------------------------------------------------------
# MCQuestion invariants are enforced
# ---------------------------------------------------------------------------


class TestMCQuestionInvariants:
    """Validation inherited from MCQuestion's ``__post_init__``."""

    def test_exactly_one_correct_per_question(self) -> None:
        rows = [_mmlu_row(choices=["a", "b", "c"], answer=2)]
        with patch.dict(
            sys.modules, _install_fake_datasets(MagicMock(return_value=rows))
        ):
            ds = MCDataset.load_from_hub("ex/mc")

        q = ds.questions[0]
        correct = [o for o in q.options if o.is_correct]
        assert len(correct) == 1
        assert correct[0].label == "C"
