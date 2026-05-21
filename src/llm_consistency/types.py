"""Core data types for llm-consistency evaluation framework."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_consistency._exceptions import ValidationError


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return _dt.datetime.now(_dt.UTC).isoformat()


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


@dataclass(frozen=True, kw_only=True)
class LLMResponse:
    """Raw LLM response with metadata.

    Captures the raw output from an LLM provider along with the extracted
    answer and optional performance metadata.  Frozen and hashable for
    safe use in sets and as dict keys.

    Attributes:
        question_id: Back-reference to the originating question.
        raw_output: The complete raw text returned by the LLM.
        extracted_answer: The parsed/extracted answer label or text.
        model: The model identifier (e.g., ``"gpt-4o"``).
        provider: The provider identifier (e.g., ``"openai"``).
        latency_ms: Response latency in milliseconds, or ``None``.
        prompt_tokens: Number of prompt tokens, or ``None``.
        completion_tokens: Number of completion tokens, or ``None``.
    """

    question_id: str
    raw_output: str
    extracted_answer: str
    model: str
    provider: str
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        ``None`` optional fields are included as explicit null values.
        """
        return {
            "question_id": self.question_id,
            "raw_output": self.raw_output,
            "extracted_answer": self.extracted_answer,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with LLMResponse field keys.

        Returns:
            A new LLMResponse instance.
        """
        latency_raw = data.get("latency_ms")
        prompt_raw = data.get("prompt_tokens")
        completion_raw = data.get("completion_tokens")
        return cls(
            question_id=str(data["question_id"]),
            raw_output=str(data["raw_output"]),
            extracted_answer=str(data["extracted_answer"]),
            model=str(data["model"]),
            provider=str(data["provider"]),
            latency_ms=float(latency_raw) if latency_raw is not None else None,
            prompt_tokens=int(prompt_raw) if prompt_raw is not None else None,
            completion_tokens=(
                int(completion_raw) if completion_raw is not None else None
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ScoredResponse:
    """A scored response with correctness and score.

    Captures whether a response was correct, the numeric score, and the
    scoring method used.  Carries a ``question_id`` back-reference.

    Attributes:
        question_id: Back-reference to the originating question.
        is_correct: Whether the response was judged correct.
        score: Numeric score (0.0 to 1.0 typical range).
        scoring_method: Name of the scorer that produced this result.
        perturbation_type: The perturbation type that produced the
            variant this response came from (``PerturbationType.value``
            string, e.g. ``"option_reorder"``).  ``None`` for responses
            constructed outside the runner pipeline.
    """

    question_id: str
    is_correct: bool
    score: float
    scoring_method: str
    perturbation_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "question_id": self.question_id,
            "is_correct": self.is_correct,
            "score": self.score,
            "scoring_method": self.scoring_method,
            "perturbation_type": self.perturbation_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoredResponse:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with ScoredResponse field keys.

        Returns:
            A new ScoredResponse instance.
        """
        pt = data.get("perturbation_type")
        return cls(
            question_id=str(data["question_id"]),
            is_correct=bool(data["is_correct"]),
            score=float(data["score"]),
            scoring_method=str(data["scoring_method"]),
            perturbation_type=str(pt) if pt is not None else None,
        )


@dataclass(frozen=True, kw_only=True)
class MetricResult:
    """A point-estimate with a bootstrap confidence interval.

    Returned by ``*_with_ci`` metric functions. The interval is
    *one* of several methods, named by :attr:`method`.

    Attributes:
        value: The point estimate of the metric.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        n_samples: Number of samples (questions) the metric was
            computed over.
        confidence: Confidence level (e.g. ``0.95`` for 95% CI).
        method: Bootstrap method used, e.g. ``"bca"`` or ``"percentile"``.
    """

    value: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    confidence: float
    method: str

    def __post_init__(self) -> None:
        """Validate construction-time invariants."""
        if self.n_samples < 0:
            msg = "MetricResult.n_samples must be non-negative"
            raise ValidationError(msg)
        if not (0.0 < self.confidence < 1.0):
            msg = "MetricResult.confidence must be in (0.0, 1.0)"
            raise ValidationError(msg)
        if self.ci_lower > self.ci_upper:
            msg = (
                f"MetricResult.ci_lower ({self.ci_lower}) must be "
                f"<= ci_upper ({self.ci_upper})"
            )
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
            "confidence": self.confidence,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        """Deserialize from a dictionary."""
        return cls(
            value=float(data["value"]),
            ci_lower=float(data["ci_lower"]),
            ci_upper=float(data["ci_upper"]),
            n_samples=int(data["n_samples"]),
            confidence=float(data["confidence"]),
            method=str(data["method"]),
        )


@dataclass(frozen=True, kw_only=True)
class PairedTestResult:
    """Result of a paired statistical test on two models' per-question outcomes.

    Returned by :func:`llm_consistency.metrics.compare_mca_paired` for
    McNemar's exact binomial test on paired binary outcomes.

    Attributes:
        statistic: Test statistic — for McNemar's exact test, the smaller
            of the two discordant counts (i.e. ``min(b, c)`` where ``b``
            is "A passes, B fails" and ``c`` is "A fails, B passes").
        p_value: Two-sided p-value, in ``[0.0, 1.0]``.
        n_discordant: Total number of discordant pairs (``b + c``).
            A small ``n_discordant`` means the test has little power
            regardless of ``p_value``.
        method: Identifier for the test used (e.g. ``"mcnemar_exact"``).
    """

    statistic: float
    p_value: float
    n_discordant: int
    method: str

    def __post_init__(self) -> None:
        """Validate construction-time invariants."""
        if not 0.0 <= self.p_value <= 1.0:
            msg = "PairedTestResult.p_value must be in [0.0, 1.0]"
            raise ValidationError(msg)
        if self.n_discordant < 0:
            msg = "PairedTestResult.n_discordant must be non-negative"
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_discordant": self.n_discordant,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairedTestResult:
        """Deserialize from a dictionary."""
        return cls(
            statistic=float(data["statistic"]),
            p_value=float(data["p_value"]),
            n_discordant=int(data["n_discordant"]),
            method=str(data["method"]),
        )


@dataclass(frozen=True, kw_only=True)
class QuestionConsistencyResult:
    """Two-axis consistency result for a single question.

    Carries both the CAT-faithful correctness rate (``rc_correct``) and
    the answer agreement rate (``rc_agree``), enabling analysis of
    stable-but-wrong vs unstable-but-sometimes-right patterns.

    The ``answer_distribution`` field uses ``hash=False`` so the dict
    is excluded from the hash (making the instance hashable) while
    still participating in equality comparisons.

    Attributes:
        question_id: Back-reference to the originating question.
        rc_correct: CAT-faithful correctness rate (correct / total).
        rc_agree: Answer agreement rate (modal frequency / total).
        total_variants: Total number of perturbed variants evaluated.
        correct_count: Number of variants answered correctly.
        answer_distribution: Mapping of extracted answer to count.
            Excluded from hash but included in equality.
        scored_responses: Tuple of ScoredResponse instances for full
            response data embedding.
    """

    question_id: str
    rc_correct: float
    rc_agree: float
    total_variants: int
    correct_count: int
    answer_distribution: dict[str, int] = field(default_factory=dict, hash=False)
    scored_responses: tuple[ScoredResponse, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        ``answer_distribution`` is serialized as a plain dict.
        ``scored_responses`` is serialized as a list of dicts.
        """
        return {
            "question_id": self.question_id,
            "rc_correct": self.rc_correct,
            "rc_agree": self.rc_agree,
            "total_variants": self.total_variants,
            "correct_count": self.correct_count,
            "answer_distribution": dict(self.answer_distribution),
            "scored_responses": [sr.to_dict() for sr in self.scored_responses],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionConsistencyResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with QuestionConsistencyResult field keys.

        Returns:
            A new QuestionConsistencyResult instance.
        """
        raw_responses = data.get("scored_responses", [])
        scored_responses = tuple(ScoredResponse.from_dict(sr) for sr in raw_responses)
        raw_distribution = data.get("answer_distribution", {})
        answer_distribution = {str(k): int(v) for k, v in raw_distribution.items()}
        return cls(
            question_id=str(data["question_id"]),
            rc_correct=float(data["rc_correct"]),
            rc_agree=float(data["rc_agree"]),
            total_variants=int(data["total_variants"]),
            correct_count=int(data["correct_count"]),
            answer_distribution=answer_distribution,
            scored_responses=scored_responses,
        )


KNOWN_SCORERS: frozenset[str] = frozenset(
    {"exact_match", "semantic_similarity", "llm_judge"}
)


@dataclass(frozen=True, kw_only=True)
class EvaluationConfig:
    """User-facing configuration for an LLM consistency evaluation run.

    Eagerly validates all fields at construction time (fail-fast).
    Perturbation types are validated against the ``PerturbationType`` enum
    and scorer names against ``KNOWN_SCORERS``.

    Attributes:
        model: The LLM model identifier (e.g., ``"gpt-4o"``).
        provider: The provider identifier (e.g., ``"openai"``).
        perturbation_types: Tuple of perturbation categories to apply.
        scorer: Name of the scoring method (must be in ``KNOWN_SCORERS``).
        num_variants: Number of perturbation variants per question.
        concurrency: Maximum concurrent provider API calls.
        max_budget_usd: Spending cap in USD, or ``None`` for unlimited.
        mca_threshold: MCA threshold for CI pass/fail (0.0 to 1.0).
        core_threshold: Minimum CORE score for CI pass/fail, or ``None``.
        ci_mode: Whether to return exit code based on thresholds.
    """

    model: str
    provider: str
    perturbation_types: tuple[PerturbationType, ...]
    scorer: str
    num_variants: int = 5
    concurrency: int = 10
    max_budget_usd: float | None = None
    mca_threshold: float = 1.0
    core_threshold: float | None = None
    ci_mode: bool = False

    def __post_init__(self) -> None:
        """Validate construction-time invariants (eager validation)."""
        if not self.model:
            msg = "EvaluationConfig.model must be a non-empty string"
            raise ValidationError(msg)
        if not self.provider:
            msg = "EvaluationConfig.provider must be a non-empty string"
            raise ValidationError(msg)
        if not self.perturbation_types:
            msg = "EvaluationConfig.perturbation_types must be non-empty"
            raise ValidationError(msg)
        if self.scorer not in KNOWN_SCORERS:
            sorted_scorers = sorted(KNOWN_SCORERS)
            msg = f"Unknown scorer '{self.scorer}'. Known scorers: {sorted_scorers}"
            raise ValidationError(msg)
        if self.num_variants < 1:
            msg = "EvaluationConfig.num_variants must be >= 1"
            raise ValidationError(msg)
        if self.concurrency < 1:
            msg = "EvaluationConfig.concurrency must be >= 1"
            raise ValidationError(msg)
        if not (0.0 <= self.mca_threshold <= 1.0):
            msg = "EvaluationConfig.mca_threshold must be between 0.0 and 1.0"
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        ``perturbation_types`` is serialized as a list of UPPER_CASE name
        strings.
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "perturbation_types": [pt.name for pt in self.perturbation_types],
            "scorer": self.scorer,
            "num_variants": self.num_variants,
            "concurrency": self.concurrency,
            "max_budget_usd": self.max_budget_usd,
            "mca_threshold": self.mca_threshold,
            "core_threshold": self.core_threshold,
            "ci_mode": self.ci_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with EvaluationConfig field keys.
                ``perturbation_types`` is expected as a list of UPPER_CASE
                name strings.

        Returns:
            A new EvaluationConfig instance.
        """
        perturbation_types = tuple(
            PerturbationType[name] for name in data["perturbation_types"]
        )
        core_raw = data.get("core_threshold")
        budget_raw = data.get("max_budget_usd")
        return cls(
            model=str(data["model"]),
            provider=str(data["provider"]),
            perturbation_types=perturbation_types,
            scorer=str(data["scorer"]),
            num_variants=int(data.get("num_variants", 5)),
            concurrency=int(data.get("concurrency", 10)),
            max_budget_usd=float(budget_raw) if budget_raw is not None else None,
            mca_threshold=float(data.get("mca_threshold", 1.0)),
            core_threshold=float(core_raw) if core_raw is not None else None,
            ci_mode=bool(data.get("ci_mode", False)),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationReport:
    """Complete evaluation report with configuration, results, and metadata.

    Embeds the full tuple of ``QuestionConsistencyResult`` instances (not
    just aggregate metrics) per the locked design decision.  The
    ``created_at`` field auto-generates an ISO 8601 UTC timestamp if not
    provided.

    Attributes:
        config: The evaluation configuration used for this run.
        results: Full tuple of per-question consistency results.
        total_questions: Total number of questions evaluated.
        total_variants: Total number of perturbed variants across all
            questions.
        mean_rc_correct: Average correctness rate across all questions.
        mean_rc_agree: Average agreement rate across all questions.
        created_at: ISO 8601 timestamp of report creation.
    """

    config: EvaluationConfig
    results: tuple[QuestionConsistencyResult, ...]
    total_questions: int
    total_variants: int
    mean_rc_correct: float
    mean_rc_agree: float
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Nested ``config`` and ``results`` are recursively serialized via
        their own ``to_dict()`` methods.
        """
        return {
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "total_questions": self.total_questions,
            "total_variants": self.total_variants,
            "mean_rc_correct": self.mean_rc_correct,
            "mean_rc_agree": self.mean_rc_agree,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with EvaluationReport field keys.  Nested
                ``config`` and ``results`` are deserialized recursively.

        Returns:
            A new EvaluationReport instance.
        """
        config = EvaluationConfig.from_dict(data["config"])
        results = tuple(QuestionConsistencyResult.from_dict(r) for r in data["results"])
        return cls(
            config=config,
            results=results,
            total_questions=int(data["total_questions"]),
            total_variants=int(data["total_variants"]),
            mean_rc_correct=float(data["mean_rc_correct"]),
            mean_rc_agree=float(data["mean_rc_agree"]),
            created_at=str(data["created_at"]),
        )
