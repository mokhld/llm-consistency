"""CIRunner: pass/fail exit codes based on metric thresholds.

Wraps :class:`BatchRunner` and checks MCA and CORE thresholds from
the evaluation config, returning exit code ``0`` (pass) or ``1`` (fail)
for CI/CD integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_consistency.metrics import core_index, mca
from llm_consistency.runners._batch import BatchRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_consistency.datasets._base import BaseDataset
    from llm_consistency.providers._base import BaseLLMProvider
    from llm_consistency.runners._metadata import RunMetadata
    from llm_consistency.scoring import BaseScorer
    from llm_consistency.types import (
        EvaluationConfig,
        EvaluationReport,
        QuestionConsistencyResult,
    )

_logger = logging.getLogger(__name__)


def count_error_variants(results: Sequence[QuestionConsistencyResult]) -> int:
    """Count variants whose provider call failed.

    The runners record a failed call as a ``ScoredResponse`` whose
    ``scoring_method`` starts with ``"error:"``.
    """
    return sum(
        1
        for qcr in results
        for sr in qcr.scored_responses
        if sr.scoring_method.startswith("error:")
    )


def gate_failures(report: EvaluationReport) -> tuple[str, ...]:
    """Apply the CI pass/fail rule to *report*.

    The report passes when all of these hold:

    - ``mca(results, config.mca_threshold) >= config.min_mca``;
    - ``core_index(results) >= config.core_threshold``, when a CORE
      threshold is set;
    - no variant failed with a provider error.

    Returns:
        One message per failed check. An empty tuple means the report
        passes.
    """
    config = report.config
    failures: list[str] = []

    mca_value = mca(report.results, config.mca_threshold)
    if mca_value < config.min_mca:
        failures.append(
            f"MCA check failed: MCA(threshold={config.mca_threshold:.3f}) "
            f"= {mca_value:.3f}, expected >= {config.min_mca:.3f}"
        )

    if config.core_threshold is not None:
        core_value = core_index(report.results)
        if core_value < config.core_threshold:
            failures.append(
                f"CORE check failed: CORE = {core_value:.3f}, "
                f"expected >= {config.core_threshold:.3f}"
            )

    errors = count_error_variants(report.results)
    if errors:
        failures.append(
            f"Error check failed: {errors} of {report.total_variants} "
            "variants failed with provider errors"
        )

    return tuple(failures)


class CIRunner:
    """CI/CD evaluation runner returning pass/fail exit codes.

    Wraps :class:`BatchRunner` to run the full evaluation pipeline,
    then applies :func:`gate_failures` to the report. Returns ``0`` if
    every check passes, ``1`` otherwise.

    MCA check: ``mca(results, config.mca_threshold)`` must be at least
    ``config.min_mca``. With the default ``min_mca=1.0`` every question
    must reach ``rc_correct >= mca_threshold``.

    CORE check: only performed when ``config.core_threshold is not None``.
    The CORE index must meet or exceed the threshold.

    Error check: any variant whose provider call failed fails the run.

    When a check fails the failure is logged at WARNING level with the
    actual vs target value, and recorded on the ``failures`` attribute so
    callers can inspect the result without re-parsing log output. The
    report and run metadata are kept on ``last_report`` and
    ``last_metadata`` so callers can export them.

    Examples:
        ::

            runner = CIRunner()
            exit_code = await runner.run(dataset, config, provider, scorer)
            sys.exit(exit_code)
    """

    def __init__(self) -> None:
        self.failures: tuple[str, ...] = ()
        self.last_report: EvaluationReport | None = None
        self.last_metadata: RunMetadata | None = None

    async def run(
        self,
        dataset: BaseDataset,
        config: EvaluationConfig,
        provider: BaseLLMProvider,
        scorer: BaseScorer,
        *,
        seed: int = 42,
    ) -> int:
        """Run evaluation and return exit code based on metric thresholds.

        Args:
            dataset: Dataset of questions to evaluate.
            config: Evaluation configuration with threshold settings.
            provider: LLM provider for querying.
            scorer: Scorer for evaluating responses.
            seed: Random seed for reproducible perturbation generation.

        Returns:
            ``0`` if all checks pass, ``1`` if any check fails.
        """
        batch_runner = BatchRunner()
        report = await batch_runner.run(dataset, config, provider, scorer, seed=seed)
        self.last_report = report
        self.last_metadata = batch_runner.last_metadata

        self.failures = gate_failures(report)
        for msg in self.failures:
            _logger.warning(msg)
        return 0 if not self.failures else 1
