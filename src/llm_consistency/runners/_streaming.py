"""StreamingRunner: async iterator yielding per-question results progressively.

Yields :class:`QuestionConsistencyResult` objects one at a time, in dataset
order, enabling progressive display during long evaluations.
Reuses the same pipeline helpers as :class:`BatchRunner`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

from llm_consistency.runners._pipeline import (
    error_summary,
    has_error_variants,
    process_question,
    warn_if_budget_not_enforced,
)
from llm_consistency.types import MCQuestion, QuestionConsistencyResult

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llm_consistency.datasets._base import BaseDataset
    from llm_consistency.providers._base import BaseLLMProvider
    from llm_consistency.scoring import BaseScorer
    from llm_consistency.types import EvaluationConfig


class StreamingRunner:
    """Streaming evaluation runner yielding per-question results.

    Unlike :class:`BatchRunner`, which collects all results into an
    :class:`EvaluationReport`, the streaming runner yields each
    :class:`QuestionConsistencyResult` as soon as it is computed.
    This enables progressive display and early termination.

    The pipeline stages are identical to :class:`BatchRunner`:
    generate variants -> render prompts -> query LLM ->
    score responses -> build QCR.

    Examples:
        ::

            runner = StreamingRunner()
            async for qcr in runner.run_stream(dataset, config, provider, scorer):
                print(f"{qcr.question_id}: rc_correct={qcr.rc_correct:.2f}")
    """

    async def run_stream(
        self,
        dataset: BaseDataset,
        config: EvaluationConfig,
        provider: BaseLLMProvider,
        scorer: BaseScorer,
        *,
        seed: int = 42,
    ) -> AsyncIterator[QuestionConsistencyResult]:
        """Yield per-question consistency results as an async iterator.

        For each question in *dataset*, runs the full evaluation pipeline
        and yields the resulting :class:`QuestionConsistencyResult`.
        Up to ``config.concurrency`` questions are evaluated ahead of the
        consumer; results are yielded in dataset order.

        Args:
            dataset: Dataset of questions to evaluate.
            config: Evaluation configuration.
            provider: LLM provider for querying.
            scorer: Scorer for evaluating responses.
            seed: Random seed for reproducible perturbation generation.

        Yields:
            A :class:`QuestionConsistencyResult` for each question.

        Raises:
            BudgetExceededError: If the provider's budget cap is reached.
                Questions still in flight are cancelled.
        """
        semaphore = asyncio.Semaphore(config.concurrency)
        warn_if_budget_not_enforced(config, provider, _logger)
        in_flight: deque[asyncio.Future[QuestionConsistencyResult]] = deque()
        failed: list[QuestionConsistencyResult] = []
        skipped = 0

        try:
            for question in dataset:
                if not isinstance(question, MCQuestion):
                    skipped += 1
                    continue
                in_flight.append(
                    asyncio.ensure_future(
                        process_question(
                            question, config, provider, scorer, semaphore, seed
                        )
                    )
                )
                if len(in_flight) < config.concurrency:
                    continue
                qcr = await in_flight.popleft()
                if has_error_variants(qcr):
                    failed.append(qcr)
                yield qcr
            while in_flight:
                qcr = await in_flight.popleft()
                if has_error_variants(qcr):
                    failed.append(qcr)
                yield qcr
        finally:
            # Runs on normal completion, on error, and when the consumer
            # closes the stream early: stop the questions still in flight.
            for task in in_flight:
                task.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
            if skipped:
                _logger.warning(
                    "StreamingRunner skipped %d non-MCQuestion items "
                    "(open-ended not yet supported)",
                    skipped,
                )
            summary = error_summary(failed)
            if summary is not None:
                _logger.warning("StreamingRunner: %s.", summary)
