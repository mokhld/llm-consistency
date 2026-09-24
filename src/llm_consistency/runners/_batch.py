"""BatchRunner: full pipeline orchestration with async concurrency.

Orchestrates the complete evaluation pipeline:
perturb -> query -> score -> analyze -> report.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from llm_consistency._exceptions import ValidationError
from llm_consistency.runners._checkpoint import CheckpointWriter, read_checkpoint
from llm_consistency.runners._metadata import RunMetadata
from llm_consistency.runners._pipeline import (
    error_summary,
    has_error_variants,
    process_question,
    warn_if_budget_not_enforced,
)
from llm_consistency.types import EvaluationReport, MCQuestion

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from rich.progress import Progress

    from llm_consistency.datasets._base import BaseDataset
    from llm_consistency.providers._base import BaseLLMProvider
    from llm_consistency.scoring import BaseScorer
    from llm_consistency.types import (
        EvaluationConfig,
        QuestionConsistencyResult,
    )


class BatchRunner:
    """Batch evaluation runner with full pipeline orchestration.

    Orchestrates: generate variants -> render prompts -> query LLM ->
    score responses -> build QCRs -> aggregate into EvaluationReport.

    Questions are processed concurrently, and the number of provider calls
    in flight is bounded by ``config.concurrency`` via an
    ``asyncio.Semaphore``.  ``report.results`` stays in dataset order.
    An optional Rich ``Progress`` instance enables live progress display
    during batch execution.

    After :meth:`run` completes, the :attr:`last_metadata` attribute
    holds the :class:`RunMetadata` captured at the start of the run.
    """

    def __init__(self) -> None:
        self.last_metadata: RunMetadata | None = None

    async def run(
        self,
        dataset: BaseDataset,
        config: EvaluationConfig,
        provider: BaseLLMProvider,
        scorer: BaseScorer,
        *,
        progress: Progress | None = None,
        seed: int = 42,
        checkpoint_path: str | Path | None = None,
    ) -> EvaluationReport:
        """Run the full evaluation pipeline on a dataset.

        Args:
            dataset: Dataset of questions to evaluate.
            config: Evaluation configuration.
            provider: LLM provider for querying.
            scorer: Scorer for evaluating responses.
            progress: Optional Rich Progress instance for live display.
            seed: Random seed for reproducible perturbation generation.
            checkpoint_path: If set, persist per-question results to a
                JSONL file. If the file already exists, it must have
                been written for the same ``config`` + ``seed`` — those
                question IDs are then loaded and skipped on this run,
                making long evaluations resumable across crashes. The
                dataset is *not* hashed into the checkpoint; users are
                responsible for keeping the dataset stable across
                resumes.  Questions with a failed variant are not
                written to the checkpoint, so a resume retries them.

        Returns:
            A complete :class:`EvaluationReport` with per-question
            consistency results and aggregate metrics.

        Raises:
            BudgetExceededError: If the provider's budget cap is reached.
                The run is aborted and outstanding queries are cancelled;
                questions finished before that are already checkpointed.
        """
        semaphore = asyncio.Semaphore(config.concurrency)
        warn_if_budget_not_enforced(config, provider, _logger)
        self.last_metadata = RunMetadata.capture(config, seed)

        items = list(dataset)
        # Non-MC questions (open-ended) are not yet supported.
        questions = [q for q in items if isinstance(q, MCQuestion)]
        skipped = len(items) - len(questions)

        ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
        slots = _resume_slots(ckpt_path, questions, config, seed)
        resumed = sum(1 for qcr in slots if qcr is not None)

        # Set up optional Rich progress task
        advance: Callable[[], None] | None = None
        if progress is not None:
            task_id = progress.add_task(
                "Evaluating questions...",
                total=len(questions),
            )
            if resumed:
                progress.advance(task_id, advance=resumed)
            advance = partial(progress.advance, task_id)

        async def _evaluate(i: int) -> QuestionConsistencyResult:
            return await process_question(
                questions[i], config, provider, scorer, semaphore, seed
            )

        with ExitStack() as stack:
            writer: CheckpointWriter | None = None
            if ckpt_path is not None:
                writer = stack.enter_context(
                    CheckpointWriter(ckpt_path, config=config, seed=seed)
                )
            unsaved = await _fill_slots(
                slots, _evaluate, config.concurrency, writer, advance
            )

        if skipped:
            _logger.warning(
                "BatchRunner skipped %d non-MCQuestion items "
                "(open-ended not yet supported)",
                skipped,
            )

        results = [qcr for qcr in slots if qcr is not None]
        summary = error_summary(results)
        if summary is not None:
            if unsaved:
                summary += (
                    f"; {unsaved} question(s) with errors were not checkpointed "
                    "and will be retried on resume"
                )
            _logger.warning("BatchRunner: %s.", summary)

        # Compute aggregates
        total_questions = len(results)
        if total_questions == 0:
            msg = (
                "BatchRunner produced zero results: the dataset contained no "
                "MCQuestion items. An EvaluationReport over zero questions is "
                "not meaningful."
            )
            raise ValidationError(msg)
        total_variants = sum(r.total_variants for r in results)
        mean_rc_correct = sum(r.rc_correct for r in results) / total_questions
        mean_rc_agree = sum(r.rc_agree for r in results) / total_questions

        return EvaluationReport(
            config=config,
            results=tuple(results),
            total_questions=total_questions,
            total_variants=total_variants,
            mean_rc_correct=mean_rc_correct,
            mean_rc_agree=mean_rc_agree,
        )


def _resume_slots(
    ckpt_path: Path | None,
    questions: list[MCQuestion],
    config: EvaluationConfig,
    seed: int,
) -> list[QuestionConsistencyResult | None]:
    """Return one slot per question, filled from the checkpoint when present.

    Checkpointed results for ids that are no longer in the dataset are
    dropped, and a repeated id resolves to its last record.
    """
    prior_by_id: dict[str, QuestionConsistencyResult] = {}
    if ckpt_path is not None and ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        _, prior = read_checkpoint(
            ckpt_path,
            config=config,
            seed=seed,
            question_ids={q.id for q in questions},
        )
        prior_by_id = {qcr.question_id: qcr for qcr in prior}
    slots = [prior_by_id.get(q.id) for q in questions]
    resumed = sum(1 for qcr in slots if qcr is not None)
    if resumed:
        _logger.info(
            "BatchRunner resuming from checkpoint %s: %d question(s) "
            "already complete, will be skipped.",
            ckpt_path,
            resumed,
        )
    return slots


async def _fill_slots(
    slots: list[QuestionConsistencyResult | None],
    evaluate: Callable[[int], Awaitable[QuestionConsistencyResult]],
    max_running: int,
    writer: CheckpointWriter | None,
    advance: Callable[[], None] | None,
) -> int:
    """Evaluate every empty slot, at most *max_running* questions at a time.

    Questions start as others finish, so a slow question does not hold up
    the rest, and memory stays bounded on large datasets.  Results are
    stored as they finish.  This loop is the only writer of the checkpoint,
    so appends stay serialized.  A question with a failed variant is kept
    in *slots* but not checkpointed.  If an evaluation raises, the results
    that finished with it are stored, the others are cancelled, and the
    exception propagates.

    Returns:
        The number of finished questions left out of the checkpoint.
    """
    todo = deque(i for i, qcr in enumerate(slots) if qcr is None)
    running: dict[asyncio.Future[QuestionConsistencyResult], int] = {}
    unsaved = 0
    try:
        while todo or running:
            while todo and len(running) < max_running:
                i = todo.popleft()
                running[asyncio.ensure_future(evaluate(i))] = i
            done, _ = await asyncio.wait(
                running.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            # Questions that finish together are stored in dataset order.
            for i, task in sorted((running.pop(t), t) for t in done):
                if task.exception() is not None:
                    continue
                qcr = slots[i] = task.result()
                if writer is not None:
                    if has_error_variants(qcr):
                        unsaved += 1
                    else:
                        writer.append(qcr)
                if advance is not None:
                    advance()
            for task in done:
                task.result()  # re-raise the first failure, if any
    except BaseException:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    return unsaved
