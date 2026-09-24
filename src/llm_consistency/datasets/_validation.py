"""Shared validation utilities and format auto-detection for dataset loading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llm_consistency._exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from llm_consistency.types import MCQuestion, OpenEndedQuestion

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".csv": "csv",
}


def detect_format(path: Path) -> str:
    """Detect dataset file format from its extension.

    Args:
        path: Path to the dataset file.

    Returns:
        One of ``"json"``, ``"jsonl"``, or ``"csv"``.

    Raises:
        ValidationError: If the file extension is not supported.
    """
    suffix = path.suffix.lower()
    fmt = _SUPPORTED_EXTENSIONS.get(suffix)
    if fmt is None:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        msg = (
            f"Unsupported file extension '{suffix}' for {path}. "
            f"Supported formats: {supported}"
        )
        raise ValidationError(msg)
    return fmt


def load_json_questions(path: Path) -> list[Any]:
    """Read the ``questions`` list from a JSON dataset file.

    The file is read as UTF-8, with or without a byte order mark.

    Args:
        path: Path to a JSON file of the form ``{"questions": [...]}``.

    Returns:
        The raw ``questions`` list.

    Raises:
        ValidationError: If the file is not valid JSON, or is not an
            object with a ``questions`` list.
    """
    try:
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        msg = (
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        )
        raise ValidationError(msg) from exc
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        msg = f"Expected a JSON object with a 'questions' list in {path}"
        raise ValidationError(msg)
    questions: list[Any] = data["questions"]
    return questions


def validate_unique_ids(
    questions: Sequence[MCQuestion | OpenEndedQuestion],
    source: str,
) -> None:
    """Validate that all question IDs are unique.

    Args:
        questions: Sequence of questions to check.
        source: File path or description for error messages.

    Raises:
        ValidationError: If duplicate IDs are found, listing them.
    """
    seen: dict[str, int] = {}
    for q in questions:
        seen[q.id] = seen.get(q.id, 0) + 1
    duplicates = [qid for qid, count in seen.items() if count > 1]
    if duplicates:
        dup_list = ", ".join(sorted(duplicates))
        msg = f"Duplicate question IDs in {source}: {dup_list}"
        raise ValidationError(msg)
