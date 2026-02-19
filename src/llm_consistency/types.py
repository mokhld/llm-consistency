"""Core data types for llm-consistency evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from llm_consistency._exceptions import ValidationError


class PerturbationType(Enum):
    """Categories of perturbation applied to questions.

    Members:
        OPTION_REORDER: Shuffle MC answer option ordering.
        FORMAT_CHANGE: Change question formatting/template.
        SEPARATOR_CHANGE: Modify delimiters between options.
        PARAPHRASE: LLM-powered semantic rephrasing.
        INSTRUCTION_REPHRASE: System prompt variants.
    """

    OPTION_REORDER = "option_reorder"
    FORMAT_CHANGE = "format_change"
    SEPARATOR_CHANGE = "separator_change"
    PARAPHRASE = "paraphrase"
    INSTRUCTION_REPHRASE = "instruction_rephrase"


@dataclass(frozen=True)
class MCOption:
    """A single option in a multiple-choice question.

    Attributes:
        label: The option identifier (e.g., "A", "B", "C", "D").
        text: The option text content.
        is_correct: Whether this option is the correct answer.
    """

    label: str
    text: str
    is_correct: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "label": self.label,
            "text": self.text,
            "is_correct": self.is_correct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCOption:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with 'label', 'text', and 'is_correct' keys.

        Returns:
            A new MCOption instance.
        """
        return cls(
            label=str(data["label"]),
            text=str(data["text"]),
            is_correct=bool(data["is_correct"]),
        )


@dataclass(frozen=True)
class MCQuestion:
    """A multiple-choice question with exactly one correct answer.

    Validates at construction time that exactly one option has
    ``is_correct=True``.  Attempting to mutate any field raises
    ``FrozenInstanceError``.

    Attributes:
        id: Unique identifier for the question.
        stem: The question text.
        options: Tuple of MCOption instances (exactly one must be correct).
    """

    id: str
    stem: str
    options: tuple[MCOption, ...]

    def __post_init__(self) -> None:
        """Validate construction-time invariants."""
        if not self.id:
            msg = "MCQuestion.id must be a non-empty string"
            raise ValidationError(msg)
        if not self.stem:
            msg = "MCQuestion.stem must be a non-empty string"
            raise ValidationError(msg)
        if not self.options:
            msg = "MCQuestion must have at least one option"
            raise ValidationError(msg)
        labels = [o.label for o in self.options]
        if len(labels) != len(set(labels)):
            msg = "Duplicate option labels are not allowed"
            raise ValidationError(msg)
        correct_count = sum(1 for o in self.options if o.is_correct)
        if correct_count != 1:
            msg = f"Exactly one option must be correct, got {correct_count}"
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "stem": self.stem,
            "options": [o.to_dict() for o in self.options],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCQuestion:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with 'id', 'stem', and 'options' keys.

        Returns:
            A new MCQuestion instance.
        """
        options = tuple(MCOption.from_dict(o) for o in data["options"])
        return cls(id=str(data["id"]), stem=str(data["stem"]), options=options)
