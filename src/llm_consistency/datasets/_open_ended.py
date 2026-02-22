"""OpenEndedDataset loader for JSON/JSONL/CSV open-ended question files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

from llm_consistency._exceptions import ValidationError
from llm_consistency.datasets._base import BaseDataset
from llm_consistency.datasets._validation import detect_format, validate_unique_ids
from llm_consistency.types import OpenEndedQuestion

if TYPE_CHECKING:
    from typing import Any


class OpenEndedDataset(BaseDataset):
    """Dataset of open-ended questions loaded from file.

    Supports JSON, JSONL, and CSV file formats.  After loading,
    validates that all question IDs are unique.

    Args:
        questions: Tuple of OpenEndedQuestion instances.
    """

    def __init__(self, questions: tuple[OpenEndedQuestion, ...]) -> None:
        self._questions = questions

    @property
    def questions(self) -> tuple[OpenEndedQuestion, ...]:
        """Return the tuple of OpenEndedQuestion instances."""
        return self._questions

    @classmethod
    def load(cls, path: str | Path) -> OpenEndedDataset:
        """Load open-ended questions from a JSON, JSONL, or CSV file.

        Args:
            path: File path to load from.

        Returns:
            A new OpenEndedDataset instance.

        Raises:
            ValidationError: On unsupported format, missing fields,
                duplicate IDs, or invalid data.
        """
        p = Path(path)
        fmt = detect_format(p)
        if fmt == "json":
            questions = cls._load_json(p)
        elif fmt == "jsonl":
            questions = cls._load_jsonl(p)
        else:
            questions = cls._load_csv(p)
        validate_unique_ids(questions, str(p))
        return cls(questions)

    @classmethod
    def _load_json(cls, path: Path) -> tuple[OpenEndedQuestion, ...]:
        """Load from JSON format: ``{"questions": [...]}``."""
        with path.open() as f:
            data = json.load(f)
        raw_questions: list[dict[str, Any]] = data["questions"]
        results: list[OpenEndedQuestion] = []
        for idx, qdict in enumerate(raw_questions):
            results.append(cls._parse_question(qdict, str(path), idx))
        return tuple(results)

    @classmethod
    def _load_jsonl(cls, path: Path) -> tuple[OpenEndedQuestion, ...]:
        """Load from JSONL format: one question per line."""
        results: list[OpenEndedQuestion] = []
        with path.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    qdict: dict[str, Any] = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    msg = f"Invalid JSON at line {line_num} in {path}: {exc}"
                    raise ValidationError(msg) from exc
                results.append(cls._parse_question(qdict, str(path), line_num))
        return tuple(results)

    @classmethod
    def _load_csv(cls, path: Path) -> tuple[OpenEndedQuestion, ...]:
        """Load from CSV with pipe-delimited reference_answers."""
        results: list[OpenEndedQuestion] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                qid = row.get("id", "").strip()
                stem = row.get("stem", "").strip()
                raw_answers = row.get("reference_answers", "").strip()

                if not qid:
                    msg = f"Missing required field 'id' at row {row_idx} in {path}"
                    raise ValidationError(msg)
                if not stem:
                    msg = f"Missing required field 'stem' at row {row_idx} in {path}"
                    raise ValidationError(msg)

                answers = tuple(a.strip() for a in raw_answers.split("|") if a.strip())
                results.append(
                    OpenEndedQuestion(
                        id=qid,
                        stem=stem,
                        reference_answers=answers,
                    )
                )
        return tuple(results)

    @classmethod
    def _parse_question(
        cls,
        qdict: dict[str, Any],
        source: str,
        index: int,
    ) -> OpenEndedQuestion:
        """Parse a single open-ended question dict.

        Args:
            qdict: Raw question dictionary.
            source: File path for error messages.
            index: Question index/line number for error messages.

        Returns:
            An OpenEndedQuestion instance.

        Raises:
            ValidationError: On missing or invalid fields.
        """
        try:
            qid = qdict["id"]
        except KeyError:
            msg = f"Missing required field 'id' at index {index} in {source}"
            raise ValidationError(msg) from None
        try:
            stem = qdict["stem"]
        except KeyError:
            msg = f"Missing required field 'stem' at index {index} in {source}"
            raise ValidationError(msg) from None

        raw_answers = qdict.get("reference_answers", [])
        return OpenEndedQuestion(
            id=str(qid),
            stem=str(stem),
            reference_answers=tuple(str(a) for a in raw_answers),
        )
