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
            question: The question as the model saw it: the runners pass
                the variant's presented options, so the labels match the
                prompt (provides the valid labels and the correct answer).

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


# Text just before a label that negates it, as in "not A" or "isn't (B)".
_NEGATION_BEFORE = re.compile(r"(?:\bnot|n't)\s*\(?$", re.IGNORECASE)

# The next word after an "A" or "I" that starts a sentence.
_NEXT_WORD = re.compile(r"[ \t]+([a-z]+)")


def _reads_as_word(text: str, match: re.Match[str]) -> bool:
    """Return True when a matched "A" or "I" is the English word, not a label.

    That is the case when it starts a sentence and a lowercase word follows,
    as in "A good choice is C" or "I think B".  "is" and "was" do not count,
    because "A is correct" names the option.
    """
    if match.group(1) not in ("A", "I"):
        return False
    before = text[: match.start()].rstrip(" \t")
    if before and before[-1] not in ".!?\n":
        return False
    next_word = _NEXT_WORD.match(text, match.end())
    return next_word is not None and next_word.group(1) not in ("is", "was")


def _extract_mc_answer(
    raw_output: str,
    valid_labels: frozenset[str],
) -> str | None:
    """Extract an MC answer label from raw LLM output.

    Tries patterns from most-specific to least-specific:
    1. ``"Answer: X"`` / ``"answer: (X)"`` or ``"The answer is X"``
       format; the last such statement in the text wins
    2. First valid label appearing as a standalone word
    3. Single-character output after stripping whitespace

    Labels must match case-sensitively and end at a word boundary, so
    "Answer: Definitely C" does not read the "D" of "Definitely".  The one
    exception is strategy 3: an output that is a single character matches
    a label in either case.  Numeric labels such as ``"2"`` work the same
    way as letters.

    Strategy 1 takes the last match because a model that revises itself
    states its final answer last.  Strategy 2 keeps the first label,
    but skips labels directly negated ("not A", "isn't A"), so "The answer
    is not A, it is C." gives C.  It also skips a sentence-initial "A" or
    "I" used as a word ("A good choice here is C." gives C) unless no
    other label is found.

    Args:
        raw_output: The complete raw text from the LLM.
        valid_labels: Set of acceptable answer labels
            (e.g., ``{"A", "B", "C", "D"}`` or ``{"1", "2", "3", "4"}``).

    Returns:
        The extracted label as spelled in *valid_labels*, or ``None`` if
        no valid answer found.
    """
    # Minimal normalization: strip whitespace and markdown bold markers
    text = raw_output.strip().replace("**", "")
    # Longest first, so "10" is tried before "1".
    labels_alt = "|".join(
        re.escape(label) for label in sorted(valid_labels, key=len, reverse=True)
    )

    # Strategy 1: the last "Answer: X" or "The answer is X" statement
    stated = list(
        re.finditer(
            rf"(?i:answer\s*:|the\s+answer\s+is)\s*\(?({labels_alt})\b",
            text,
        )
    )
    if stated:
        return stated[-1].group(1)

    # Strategy 2: First standalone valid label (word boundary)
    word_like: list[str] = []
    for match in re.finditer(rf"\b({labels_alt})\b", text):
        if _NEGATION_BEFORE.search(text, 0, match.start()):
            continue
        if _reads_as_word(text, match):
            word_like.append(match.group(1))
            continue
        return match.group(1)
    if word_like:
        return word_like[0]

    # Strategy 3: Single character after stripping, in either case
    if len(text) == 1:
        return next((lab for lab in valid_labels if lab.lower() == text.lower()), None)

    return None


class ExactMatchScorer(BaseScorer):
    """Scorer that extracts MC answer labels and checks exact match.

    Uses a cascading regex extraction strategy to extract the answer
    label from the raw LLM output, then compares it against the
    correct option label from the question.

    Extraction strategies (tried in order, see :func:`_extract_mc_answer`):
    1. ``"Answer: X"`` format
    2. ``"The answer is X"`` format
    3. First standalone valid label that is not negated
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
            question: The MC question as presented to the model
                (provides valid labels and the correct answer).

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
