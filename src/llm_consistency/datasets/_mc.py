"""MCDataset loader for JSON/JSONL/CSV multiple-choice question files."""

from __future__ import annotations

import csv
import json
import string
from pathlib import Path
from typing import TYPE_CHECKING

from llm_consistency._exceptions import ValidationError
from llm_consistency.datasets._base import BaseDataset
from llm_consistency.datasets._validation import detect_format, validate_unique_ids
from llm_consistency.types import MCOption, MCQuestion

if TYPE_CHECKING:
    from typing import Any


class MCDataset(BaseDataset):
    """Dataset of multiple-choice questions loaded from file.

    Supports JSON, JSONL, and CSV file formats.  After loading,
    validates that all question IDs are unique.

    Args:
        questions: Tuple of MCQuestion instances.
    """

    def __init__(self, questions: tuple[MCQuestion, ...]) -> None:
        self._questions = questions

    @property
    def questions(self) -> tuple[MCQuestion, ...]:
        """Return the tuple of MCQuestion instances."""
        return self._questions

    @classmethod
    def load(cls, path: str | Path) -> MCDataset:
        """Load MC questions from a JSON, JSONL, or CSV file.

        Args:
            path: File path to load from.

        Returns:
            A new MCDataset instance.

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
    def _load_json(cls, path: Path) -> tuple[MCQuestion, ...]:
        """Load from JSON format: ``{"questions": [...]}``."""
        with path.open() as f:
            data = json.load(f)
        raw_questions: list[dict[str, Any]] = data["questions"]
        results: list[MCQuestion] = []
        for idx, qdict in enumerate(raw_questions):
            results.append(cls._parse_question(qdict, str(path), idx))
        return tuple(results)

    @classmethod
    def _load_jsonl(cls, path: Path) -> tuple[MCQuestion, ...]:
        """Load from JSONL format: one question per line."""
        results: list[MCQuestion] = []
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
    def _load_csv(cls, path: Path) -> tuple[MCQuestion, ...]:
        """Load from CSV with option_a..option_z columns."""
        results: list[MCQuestion] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                qid = row.get("id", "").strip()
                stem = row.get("stem", "").strip()
                correct_label = row.get("correct", "").strip().upper()

                if not qid:
                    msg = f"Missing required field 'id' at row {row_idx} in {path}"
                    raise ValidationError(msg)
                if not stem:
                    msg = f"Missing required field 'stem' at row {row_idx} in {path}"
                    raise ValidationError(msg)

                options: list[MCOption] = []
                for letter in string.ascii_lowercase:
                    col_name = f"option_{letter}"
                    val = row.get(col_name, "").strip()
                    if not val:
                        continue
                    label = letter.upper()
                    options.append(
                        MCOption(
                            label=label,
                            text=val,
                            is_correct=(label == correct_label),
                        )
                    )

                if not options:
                    msg = f"No options found at row {row_idx} in {path}"
                    raise ValidationError(msg)

                option_labels = {o.label for o in options}
                if correct_label not in option_labels:
                    msg = (
                        f"Invalid correct label '{correct_label}' at "
                        f"row {row_idx} in {path}. "
                        f"Available: {sorted(option_labels)}"
                    )
                    raise ValidationError(msg)

                results.append(
                    MCQuestion(
                        id=qid,
                        stem=stem,
                        options=tuple(options),
                    )
                )
        return tuple(results)

    @classmethod
    def _parse_question(
        cls,
        qdict: dict[str, Any],
        source: str,
        index: int,
    ) -> MCQuestion:
        """Parse a single MC question dict into an MCQuestion.

        Args:
            qdict: Raw question dictionary.
            source: File path for error messages.
            index: Question index/line number for error messages.

        Returns:
            An MCQuestion instance.

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
        try:
            raw_options = qdict["options"]
        except KeyError:
            msg = f"Missing required field 'options' at index {index} in {source}"
            raise ValidationError(msg) from None

        options = tuple(
            MCOption(
                label=str(opt["label"]),
                text=str(opt["text"]),
                is_correct=bool(opt["is_correct"]),
            )
            for opt in raw_options
        )
        return MCQuestion(id=str(qid), stem=str(stem), options=options)
