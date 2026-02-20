"""LLM Consistency evaluation framework."""

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency._version import __version__, __version_tuple__
from llm_consistency.metrics import (
    agreement_gated_accuracy,
    bootstrap_ci,
    build_question_consistency_result,
    car_curve,
    core_index,
    dtw_distance,
    mca,
    normalized_dtw,
    trapezoidal_auc,
)
from llm_consistency.types import (
    KNOWN_SCORERS,
    EvaluationConfig,
    EvaluationReport,
    LLMResponse,
    MCOption,
    MCQuestion,
    OpenEndedQuestion,
    PerturbationType,
    PerturbedVariant,
    QuestionConsistencyResult,
    ScoredResponse,
)

__all__ = [
    "KNOWN_SCORERS",
    "EvaluationConfig",
    "EvaluationReport",
    "LLMConsistencyError",
    "LLMResponse",
    "MCOption",
    "MCQuestion",
    "OpenEndedQuestion",
    "PerturbationType",
    "PerturbedVariant",
    "QuestionConsistencyResult",
    "ScoredResponse",
    "ValidationError",
    "__version__",
    "__version_tuple__",
    "agreement_gated_accuracy",
    "bootstrap_ci",
    "build_question_consistency_result",
    "car_curve",
    "core_index",
    "dtw_distance",
    "mca",
    "normalized_dtw",
    "trapezoidal_auc",
]
