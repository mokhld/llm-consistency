"""LLM Consistency evaluation framework."""

from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency._version import __version__, __version_tuple__
from llm_consistency.datasets import (
    BaseDataset,
    CustomDataset,
    MCDataset,
    OpenEndedDataset,
)
from llm_consistency.metrics import (
    agreement_gated_accuracy,
    agreement_gated_accuracy_with_ci,
    bootstrap_ci,
    bootstrap_ci_bca,
    build_question_consistency_result,
    car_curve,
    car_curve_with_ci,
    core_index,
    core_index_with_ci,
    dtw_distance,
    mca,
    mca_with_ci,
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
from llm_consistency.reports import (
    ConsoleReporter,
    export_csv,
    export_html,
    export_json,
    export_markdown,
    render_car_ascii,
)
from llm_consistency.runners import (
    BatchRunner,
    CIRunner,
    RunMetadata,
    StreamingRunner,
)
from llm_consistency.scoring import (
    BaseScorer,
    CustomScorerAdapter,
    ExactMatchScorer,
    get_scorer,
)
from llm_consistency.types import (
    KNOWN_SCORERS,
    EvaluationConfig,
    EvaluationReport,
    LLMResponse,
    MCOption,
    MCQuestion,
    MetricResult,
    OpenEndedQuestion,
    PerturbationType,
    PerturbedVariant,
    QuestionConsistencyResult,
    ScoredResponse,
)

__all__ = [
    "KNOWN_SCORERS",
    "BaseDataset",
    "BaseLLMProvider",
    "BasePerturbation",
    "BaseScorer",
    "BatchResult",
    "BatchRunner",
    "BudgetExceededError",
    "CIRunner",
    "ConsoleReporter",
    "CustomDataset",
    "CustomScorerAdapter",
    "EvaluationConfig",
    "EvaluationReport",
    "ExactMatchScorer",
    "FormatChangePerturbation",
    "LLMConsistencyError",
    "LLMResponse",
    "MCDataset",
    "MCOption",
    "MCQuestion",
    "MetricResult",
    "OpenEndedDataset",
    "OpenEndedQuestion",
    "OptionReorderPerturbation",
    "PerturbationType",
    "PerturbedVariant",
    "QuestionConsistencyResult",
    "RunMetadata",
    "ScoredResponse",
    "SeparatorChangePerturbation",
    "StreamingRunner",
    "ValidationError",
    "__version__",
    "__version_tuple__",
    "agreement_gated_accuracy",
    "agreement_gated_accuracy_with_ci",
    "bootstrap_ci",
    "bootstrap_ci_bca",
    "build_question_consistency_result",
    "car_curve",
    "car_curve_with_ci",
    "core_index",
    "core_index_with_ci",
    "dtw_distance",
    "estimate_cost",
    "export_csv",
    "export_html",
    "export_json",
    "export_markdown",
    "get_perturbation",
    "get_provider",
    "get_scorer",
    "list_registered_perturbations",
    "mca",
    "mca_with_ci",
    "normalized_dtw",
    "register_perturbation",
    "render_car_ascii",
    "trapezoidal_auc",
]
