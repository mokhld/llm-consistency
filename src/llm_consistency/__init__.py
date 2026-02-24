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
from llm_consistency.perturbations import (
    BasePerturbation,
    FormatChangePerturbation,
    OptionReorderPerturbation,
    SeparatorChangePerturbation,
)
from llm_consistency.perturbations import (
    get as get_perturbation,
)
from llm_consistency.perturbations import (
    list_registered as list_registered_perturbations,
)
from llm_consistency.perturbations import (
    register as register_perturbation,
)
from llm_consistency.providers import (
    BaseLLMProvider,
    BatchResult,
    BudgetExceededError,
    estimate_cost,
    get_provider,
)
from llm_consistency.scoring import (
    BaseScorer,
    CustomScorerAdapter,
    ExactMatchScorer,
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
    "BaseLLMProvider",
    "BasePerturbation",
    "BaseScorer",
    "BatchResult",
    "BudgetExceededError",
    "CustomScorerAdapter",
    "EvaluationConfig",
    "EvaluationReport",
    "ExactMatchScorer",
    "FormatChangePerturbation",
    "LLMConsistencyError",
    "LLMResponse",
    "MCOption",
    "MCQuestion",
    "OpenEndedQuestion",
    "OptionReorderPerturbation",
    "PerturbationType",
    "PerturbedVariant",
    "QuestionConsistencyResult",
    "ScoredResponse",
    "SeparatorChangePerturbation",
    "ValidationError",
    "__version__",
    "__version_tuple__",
    "agreement_gated_accuracy",
    "bootstrap_ci",
    "build_question_consistency_result",
    "car_curve",
    "core_index",
    "dtw_distance",
    "estimate_cost",
    "get_perturbation",
    "get_provider",
    "list_registered_perturbations",
    "mca",
    "normalized_dtw",
    "register_perturbation",
    "trapezoidal_auc",
]
