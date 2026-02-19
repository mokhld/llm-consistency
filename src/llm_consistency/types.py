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


@dataclass(frozen=True)
class OpenEndedQuestion:
    """An open-ended question with reference answers.

    Validates at construction time that ``id`` and ``stem`` are non-empty.

    Attributes:
        id: Unique identifier for the question.
        stem: The question text.
        reference_answers: Tuple of acceptable reference answers.
    """

    id: str
    stem: str
    reference_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate construction-time invariants."""
        if not self.id:
            msg = "OpenEndedQuestion.id must be a non-empty string"
            raise ValidationError(msg)
        if not self.stem:
            msg = "OpenEndedQuestion.stem must be a non-empty string"
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "stem": self.stem,
            "reference_answers": list(self.reference_answers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenEndedQuestion:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with 'id', 'stem', and 'reference_answers' keys.

        Returns:
            A new OpenEndedQuestion instance.
        """
        return cls(
            id=str(data["id"]),
            stem=str(data["stem"]),
            reference_answers=tuple(str(a) for a in data["reference_answers"]),
        )


@dataclass(frozen=True, kw_only=True)
class PerturbedVariant:
    """A perturbed variant of a question with full provenance.

    Carries the perturbation type, seed, and variant index for
    reproducibility analysis.  Supports both MC (with options) and
    open-ended (options=None) variants.

    Attributes:
        original_question_id: ID of the original question this variant was
            generated from.
        perturbation_type: The type of perturbation applied.
        seed: Random seed used to generate this variant (for reproducibility).
        variant_index: Zero-indexed variant number.
        stem: The perturbed question text.
        options: Tuple of MCOption instances for MC variants, None for
            open-ended variants.
    """

    original_question_id: str
    perturbation_type: PerturbationType
    seed: int
    variant_index: int
    stem: str
    options: tuple[MCOption, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        The ``perturbation_type`` is serialized as its UPPER_CASE name string.
        """
        return {
            "original_question_id": self.original_question_id,
            "perturbation_type": self.perturbation_type.name,
            "seed": self.seed,
            "variant_index": self.variant_index,
            "stem": self.stem,
            "options": (
                [o.to_dict() for o in self.options]
                if self.options is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerturbedVariant:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with variant fields.  ``perturbation_type`` is
                expected as an UPPER_CASE name string.

        Returns:
            A new PerturbedVariant instance.
        """
        raw_options = data.get("options")
        options: tuple[MCOption, ...] | None = (
            tuple(MCOption.from_dict(o) for o in raw_options)
            if raw_options is not None
            else None
        )
        return cls(
            original_question_id=str(data["original_question_id"]),
            perturbation_type=PerturbationType[data["perturbation_type"]],
            seed=int(data["seed"]),
            variant_index=int(data["variant_index"]),
            stem=str(data["stem"]),
            options=options,
        )
