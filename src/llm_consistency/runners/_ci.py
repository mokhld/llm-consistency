"""CIRunner: pass/fail exit codes based on metric thresholds.

Wraps :class:`BatchRunner` and checks MCA and CORE thresholds from
the evaluation config, returning exit code ``0`` (pass) or ``1`` (fail)
for CI/CD integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_consistency.metrics import core_index, mca
from llm_consistency.runners._batch import BatchRunner

if TYPE_CHECKING:
    from llm_consistency.datasets._base import BaseDataset
    from llm_consistency.providers._base import BaseLLMProvider
    from llm_consistency.scoring import BaseScorer
    from llm_consistency.types import EvaluationConfig


class CIRunner:
    """CI/CD evaluation runner returning pass/fail exit codes.

    Wraps :class:`BatchRunner` to run the full evaluation pipeline,
    then checks MCA and CORE metric thresholds from the config.
    Returns ``0`` if all thresholds pass, ``1`` otherwise.

    MCA check: at the configured ``mca_threshold``, all questions must
    meet the threshold (i.e., ``mca(results, threshold) == 1.0``).

    CORE check: only performed when ``config.core_threshold is not None``.
    The CORE index must meet or exceed the threshold.

    Examples:
        ::

            runner = CIRunner()
            exit_code = await runner.run(dataset, config, provider, scorer)
            sys.exit(exit_code)
    """

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
            ``0`` if all thresholds pass, ``1`` if any threshold fails.
        """
        # 1. Run the full batch evaluation
        batch_runner = BatchRunner()
        report = await batch_runner.run(dataset, config, provider, scorer, seed=seed)

        # 2. Check MCA threshold
        passed = True
        mca_value = mca(report.results, config.mca_threshold)
        if mca_value < 1.0:
            passed = False

        # 3. Check CORE threshold (only when configured)
        if config.core_threshold is not None:
            core_value = core_index(report.results)
            if core_value < config.core_threshold:
                passed = False

        return 0 if passed else 1
