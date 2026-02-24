"""Tests for dataset loaders and validation utilities."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from llm_consistency._exceptions import ValidationError
from llm_consistency.datasets import (
    BaseDataset,
    CustomDataset,
    MCDataset,
    OpenEndedDataset,
    detect_format,
    validate_unique_ids,
)
from llm_consistency.types import MCOption, MCQuestion, OpenEndedQuestion

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mc_question_dict(
    qid: str = "q1",
    stem: str = "What is the capital of France?",
) -> dict[str, object]:
    """Return a dict matching the JSON schema for an MC question."""
    return {
        "id": qid,
        "stem": stem,
        "options": [
            {"label": "A", "text": "London", "is_correct": False},
            {"label": "B", "text": "Paris", "is_correct": True},
        ],
    }


def _oe_question_dict(
    qid: str = "oe1",
    stem: str = "Name the capital of France.",
) -> dict[str, object]:
    """Return a dict matching the JSON schema for an open-ended question."""
    return {
        "id": qid,
        "stem": stem,
        "reference_answers": ["Paris", "paris"],
    }


# ===========================================================================
# BaseDataset ABC
# ===========================================================================


class TestBaseDataset:
    """BaseDataset cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseDataset()  # type: ignore[abstract]

    def test_subclass_must_implement_questions_and_load(self) -> None:
        """A subclass that does not implement abstract members raises TypeError."""

        class IncompleteDataset(BaseDataset):  # type: ignore[abstract]
            pass

        with pytest.raises(TypeError):
            IncompleteDataset()  # type: ignore[abstract]


# ===========================================================================
# MCDataset.load()
# ===========================================================================


class TestMCDatasetJSON:
    """MCDataset loading from JSON files."""

    def test_load_json_returns_mc_questions(self, tmp_path: Path) -> None:
        data = {"questions": [_mc_question_dict("q1"), _mc_question_dict("q2")]}
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))

        ds = MCDataset.load(fp)
        assert len(ds) == 2
        assert all(isinstance(q, MCQuestion) for q in ds)

    def test_load_json_question_fields(self, tmp_path: Path) -> None:
        data = {"questions": [_mc_question_dict("q1")]}
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))

        ds = MCDataset.load(fp)
        q = ds.questions[0]
        assert q.id == "q1"
        assert q.stem == "What is the capital of France?"
        assert len(q.options) == 2
        assert q.options[1].is_correct is True
        assert q.options[1].text == "Paris"


class TestMCDatasetJSONL:
    """MCDataset loading from JSONL files."""

    def test_load_jsonl_returns_mc_questions(self, tmp_path: Path) -> None:
        fp = tmp_path / "mc.jsonl"
        lines = [json.dumps(_mc_question_dict(f"q{i}")) for i in range(3)]
        fp.write_text("\n".join(lines))

        ds = MCDataset.load(fp)
        assert len(ds) == 3
        assert all(isinstance(q, MCQuestion) for q in ds)


class TestMCDatasetCSV:
    """MCDataset loading from CSV files."""

    def test_load_csv_returns_mc_questions(self, tmp_path: Path) -> None:
        fp = tmp_path / "mc.csv"
        with fp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "stem", "option_a", "option_b", "correct"])
            writer.writerow(["q1", "Capital of France?", "London", "Paris", "B"])

        ds = MCDataset.load(fp)
        assert len(ds) == 1
        q = ds.questions[0]
        assert isinstance(q, MCQuestion)
        assert q.id == "q1"
        # B should be marked correct
        correct_opt = [o for o in q.options if o.is_correct]
        assert len(correct_opt) == 1
        assert correct_opt[0].label == "B"

    def test_csv_variable_option_counts(self, tmp_path: Path) -> None:
        """Trailing empty option columns are ignored."""
        fp = tmp_path / "mc.csv"
        with fp.open("w", newline="") as f:
            writer = csv.writer(f)
            cols = [
                "id",
                "stem",
                "option_a",
                "option_b",
                "option_c",
                "option_d",
                "correct",
            ]
            writer.writerow(cols)
            writer.writerow(["q1", "Q?", "Yes", "No", "", "", "A"])

        ds = MCDataset.load(fp)
        q = ds.questions[0]
        # Empty option_c and option_d should be skipped
        assert len(q.options) == 2


class TestMCDatasetInterface:
    """MCDataset __len__ and __iter__."""

    def test_len_returns_count(self, tmp_path: Path) -> None:
        data = {"questions": [_mc_question_dict(f"q{i}") for i in range(5)]}
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))

        ds = MCDataset.load(fp)
        assert len(ds) == 5

    def test_iter_yields_questions(self, tmp_path: Path) -> None:
        data = {"questions": [_mc_question_dict(f"q{i}") for i in range(3)]}
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))

        ds = MCDataset.load(fp)
        items = list(ds)
        assert len(items) == 3
        assert all(isinstance(q, MCQuestion) for q in items)

    def test_load_from_str_path(self, tmp_path: Path) -> None:
        data = {"questions": [_mc_question_dict("q1")]}
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))

        ds = MCDataset.load(str(fp))
        assert len(ds) == 1


# ===========================================================================
# OpenEndedDataset.load()
# ===========================================================================


class TestOpenEndedDatasetJSON:
    """OpenEndedDataset loading from JSON files."""

    def test_load_json_returns_oe_questions(self, tmp_path: Path) -> None:
        data = {"questions": [_oe_question_dict("oe1"), _oe_question_dict("oe2")]}
        fp = tmp_path / "oe.json"
        fp.write_text(json.dumps(data))

        ds = OpenEndedDataset.load(fp)
        assert len(ds) == 2
        assert all(isinstance(q, OpenEndedQuestion) for q in ds)

    def test_load_json_question_fields(self, tmp_path: Path) -> None:
        data = {"questions": [_oe_question_dict("oe1")]}
        fp = tmp_path / "oe.json"
        fp.write_text(json.dumps(data))

        ds = OpenEndedDataset.load(fp)
        q = ds.questions[0]
        assert q.id == "oe1"
        assert q.stem == "Name the capital of France."
        assert q.reference_answers == ("Paris", "paris")


class TestOpenEndedDatasetJSONL:
    """OpenEndedDataset loading from JSONL files."""

    def test_load_jsonl_returns_oe_questions(self, tmp_path: Path) -> None:
        fp = tmp_path / "oe.jsonl"
        lines = [json.dumps(_oe_question_dict(f"oe{i}")) for i in range(3)]
        fp.write_text("\n".join(lines))

        ds = OpenEndedDataset.load(fp)
        assert len(ds) == 3
        assert all(isinstance(q, OpenEndedQuestion) for q in ds)


class TestOpenEndedDatasetCSV:
    """OpenEndedDataset loading from CSV files."""

    def test_load_csv_with_pipe_delimited_answers(self, tmp_path: Path) -> None:
        fp = tmp_path / "oe.csv"
        with fp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "stem", "reference_answers"])
            writer.writerow(["oe1", "Capital of France?", "Paris|paris"])

        ds = OpenEndedDataset.load(fp)
        assert len(ds) == 1
        q = ds.questions[0]
        assert isinstance(q, OpenEndedQuestion)
        assert q.reference_answers == ("Paris", "paris")


# ===========================================================================
# CustomDataset
# ===========================================================================


class TestCustomDataset:
    """CustomDataset wraps user-supplied question sequences."""

    def test_wraps_mc_questions(self) -> None:
        q1 = MCQuestion(
            id="q1",
            stem="Test?",
            options=(
                MCOption(label="A", text="Yes", is_correct=True),
                MCOption(label="B", text="No", is_correct=False),
            ),
        )
        ds = CustomDataset([q1])
        assert len(ds) == 1
        assert list(ds) == [q1]

    def test_wraps_oe_questions(self) -> None:
        q1 = OpenEndedQuestion(
            id="oe1",
            stem="Test?",
            reference_answers=("ans1",),
        )
        ds = CustomDataset([q1])
        assert len(ds) == 1
        assert list(ds) == [q1]

    def test_len_and_iter(self) -> None:
        qs = [
            MCQuestion(
                id=f"q{i}",
                stem=f"Q{i}?",
                options=(
                    MCOption(label="A", text="Yes", is_correct=True),
                    MCOption(label="B", text="No", is_correct=False),
                ),
            )
            for i in range(4)
        ]
        ds = CustomDataset(qs)
        assert len(ds) == 4
        assert list(ds) == qs

    def test_load_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            CustomDataset.load("some_path")


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    """Validation utilities for dataset loading."""

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        fp = tmp_path / "data.xml"
        fp.write_text("<data/>")
        with pytest.raises(ValidationError, match="supported"):
            detect_format(fp)

    def test_missing_stem_raises(self, tmp_path: Path) -> None:
        data = {
            "questions": [
                {
                    "id": "q1",
                    "options": [
                        {"label": "A", "text": "Yes", "is_correct": True},
                        {"label": "B", "text": "No", "is_correct": False},
                    ],
                }
            ]
        }
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))
        with pytest.raises(ValidationError, match="stem"):
            MCDataset.load(fp)

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        data = {
            "questions": [
                {
                    "stem": "Test?",
                    "options": [
                        {"label": "A", "text": "Yes", "is_correct": True},
                        {"label": "B", "text": "No", "is_correct": False},
                    ],
                }
            ]
        }
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))
        with pytest.raises(ValidationError, match="id"):
            MCDataset.load(fp)

    def test_mc_zero_options_raises(self, tmp_path: Path) -> None:
        data = {"questions": [{"id": "q1", "stem": "Test?", "options": []}]}
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            MCDataset.load(fp)

    def test_duplicate_ids_raises(self, tmp_path: Path) -> None:
        data = {
            "questions": [
                _mc_question_dict("q1"),
                _mc_question_dict("q1"),  # duplicate
            ]
        }
        fp = tmp_path / "mc.json"
        fp.write_text(json.dumps(data))
        with pytest.raises(ValidationError, match="q1"):
            MCDataset.load(fp)

    def test_csv_invalid_correct_label_raises(self, tmp_path: Path) -> None:
        fp = tmp_path / "mc.csv"
        with fp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "stem", "option_a", "option_b", "correct"])
            writer.writerow(["q1", "Test?", "Yes", "No", "Z"])  # Z not valid

        with pytest.raises(ValidationError, match="correct"):
            MCDataset.load(fp)

    def test_validate_unique_ids_mc(self) -> None:
        q1 = MCQuestion(
            id="q1",
            stem="Test?",
            options=(
                MCOption(label="A", text="Yes", is_correct=True),
                MCOption(label="B", text="No", is_correct=False),
            ),
        )
        q2 = MCQuestion(
            id="q1",
            stem="Other?",
            options=(
                MCOption(label="A", text="Y", is_correct=True),
                MCOption(label="B", text="N", is_correct=False),
            ),
        )
        with pytest.raises(ValidationError, match="q1"):
            validate_unique_ids([q1, q2], "test-source")

    def test_detect_format_valid(self, tmp_path: Path) -> None:
        for ext, expected in [(".json", "json"), (".jsonl", "jsonl"), (".csv", "csv")]:
            fp = tmp_path / f"data{ext}"
            fp.write_text("")
            assert detect_format(fp) == expected
