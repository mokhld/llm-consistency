"""Response scoring strategies."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_consistency._exceptions import ValidationError
from llm_consistency.types import LLMResponse, MCQuestion, ScoredResponse

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseScorer(ABC):
    """Abstract base class for response scorers.

    Subclasses must implement:
    - ``name`` (property): returns the scorer's string identifier
    - ``score``: evaluates a single LLM response against a question
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The scorer's identifier (e.g., 'exact_match')."""
        ...

    @abstractmethod
    def score(
        self,
        response: LLMResponse,
        question: MCQuestion,
    ) -> ScoredResponse:
        """Score a single LLM response against the given question.

        Args:
            response: The raw LLM response to score.
            question: The original question (provides correct answer).

        Returns:
            A ScoredResponse with is_correct, score, and scoring_method.
        """
        ...


def _get_correct_label(question: MCQuestion) -> str:
    """Get the correct answer label from an MCQuestion.

    MCQuestion validates exactly-one-correct at construction time,
    so this always succeeds.

    Args:
        question: The multiple-choice question.

    Returns:
        The label of the correct option.
    """
    for option in question.options:
        if option.is_correct:
            return option.label
    # Unreachable due to MCQuestion validation, but satisfies mypy
    msg = "No correct option found"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _extract_mc_answer(
    raw_output: str,
    valid_labels: frozenset[str],
) -> str | None:
    """Extract an MC answer label from raw LLM output.

    Tries patterns from most-specific to least-specific:
    1. ``"Answer: X"`` or ``"answer: (X)"`` format
    2. ``"The answer is X"`` or ``"the answer is (X)"`` format
    3. First valid label appearing as a standalone word
    4. Single-character output after stripping whitespace

    Args:
        raw_output: The complete raw text from the LLM.
        valid_labels: Set of acceptable answer labels
            (e.g., ``{"A", "B", "C", "D"}``).

    Returns:
        The extracted label in uppercase, or ``None`` if no valid
        answer found.
    """
    # Minimal normalization: strip whitespace and markdown bold markers
    text = raw_output.strip().replace("**", "")
    labels_alt = "|".join(sorted(valid_labels))

    # Strategy 1: "Answer: X" pattern (with optional parens, colon variants)
    pattern1 = rf"[Aa]nswer\s*[:]\s*\(?({labels_alt})\)?"
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 2: "The answer is X" pattern
    pattern2 = rf"[Tt]he\s+answer\s+is\s*\(?({labels_alt})\)?"
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 3: First standalone valid label (word boundary)
    pattern3 = rf"\b({labels_alt})\b"
    match = re.search(pattern3, text)
    if match:
        return match.group(1).upper()

    # Strategy 4: Single character after stripping
    if len(text) == 1 and text.upper() in valid_labels:
        return text.upper()

    return None


class ExactMatchScorer(BaseScorer):
    """Scorer that extracts MC answer labels and checks exact match.

    Uses a cascading regex extraction strategy to extract the answer
    label from the raw LLM output, then compares it against the
    correct option label from the question.

    Extraction strategies (tried in order):
    1. ``"Answer: X"`` format
    2. ``"The answer is X"`` format
    3. First standalone valid label
    4. Single-character output
    """

    @property
    def name(self) -> str:
        """Return ``'exact_match'``."""
        return "exact_match"

    def score(
        self,
        response: LLMResponse,
        question: MCQuestion,
    ) -> ScoredResponse:
        """Score a response by extracting and matching the MC answer.

        Always re-extracts from ``response.raw_output`` (not from
        ``response.extracted_answer``) for deterministic, scorer-owned
        extraction logic.

        Args:
            response: The raw LLM response to score.
            question: The original MC question (provides valid labels
                and the correct answer).

        Returns:
            A ScoredResponse with ``is_correct=True`` and ``score=1.0``
            if the extracted answer matches the correct label, or
            ``is_correct=False`` and ``score=0.0`` otherwise.
        """
        valid_labels = frozenset(o.label for o in question.options)
        correct_label = _get_correct_label(question)
        extracted = _extract_mc_answer(response.raw_output, valid_labels)
        is_correct = extracted is not None and extracted == correct_label
        return ScoredResponse(
            question_id=response.question_id,
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            scoring_method=self.name,
        )


class CustomScorerAdapter(BaseScorer):
    """Adapter that wraps a user-supplied callable as a scorer.

    Supports two callable signatures:

    **Full signature** (default, ``simple=False``):
        ``(LLMResponse, MCQuestion) -> ScoredResponse``
        The callable receives the full response and question objects
        and must return a ``ScoredResponse`` directly.

    **Simple signature** (``simple=True``):
        ``(str, str) -> bool``
        The callable receives ``(extracted_answer, correct_label)``
        strings. The adapter handles MC answer extraction and wraps
        the boolean result into a ``ScoredResponse``.

    Args:
        fn: The scoring callable to wrap.
        name: Scorer identifier (default ``"custom"``).
        simple: If ``True``, treat *fn* as a simple
            ``(str, str) -> bool`` callable.

    Example::

        # Full signature
        def my_scorer(resp, q):
            return ScoredResponse(...)
        adapter = CustomScorerAdapter(fn=my_scorer, name="my_scorer")

        # Simple signature
        adapter = CustomScorerAdapter(
            fn=lambda extracted, correct: extracted == correct,
            name="simple_match",
            simple=True,
        )
    """

    def __init__(
        self,
        fn: Callable[..., ScoredResponse | bool],
        *,
        name: str = "custom",
        simple: bool = False,
    ) -> None:
        """Initialize with a scoring callable.

        Args:
            fn: The callable to wrap as a scorer.
            name: Scorer identifier (default ``"custom"``).
            simple: If ``True``, *fn* is treated as
                ``(str, str) -> bool``.
        """
        self._fn = fn
        self._name = name
        self._simple = simple

    @property
    def name(self) -> str:
        """Return the user-supplied scorer name."""
        return self._name

    def score(
        self,
        response: LLMResponse,
        question: MCQuestion,
    ) -> ScoredResponse:
        """Score a response using the wrapped callable.

        Args:
            response: The raw LLM response to score.
            question: The original MC question.

        Returns:
            A ``ScoredResponse`` -- either directly from a full-signature
            callable or wrapped from a simple callable's boolean result.

        Raises:
            TypeError: If a full-signature callable returns something
                other than ``ScoredResponse``.
        """
        if self._simple:
            valid_labels = frozenset(o.label for o in question.options)
            extracted = _extract_mc_answer(response.raw_output, valid_labels)
            correct_label = _get_correct_label(question)
            is_correct = bool(self._fn(extracted or "", correct_label))
            return ScoredResponse(
                question_id=response.question_id,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                scoring_method=self._name,
            )

        result = self._fn(response, question)
        if not isinstance(result, ScoredResponse):
            msg = (
                f"Full-signature scorer must return ScoredResponse, "
                f"got {type(result).__name__}"
            )
            raise TypeError(msg)
        return result


_BUILTIN_SCORERS: dict[str, Callable[[], BaseScorer]] = {
    "exact_match": ExactMatchScorer,
}


def get_scorer(name: str) -> BaseScorer:
    """Return a built-in scorer by name.

    Args:
        name: Scorer identifier (e.g. ``"exact_match"``).

    Returns:
        A new scorer instance.

    Raises:
        ValidationError: If *name* is not a recognised built-in scorer.
    """
    factory = _BUILTIN_SCORERS.get(name)
    if factory is None:
        known = sorted(_BUILTIN_SCORERS)
        msg = f"Unknown scorer {name!r}. Known scorers: {known}"
        raise ValidationError(msg)
    return factory()
