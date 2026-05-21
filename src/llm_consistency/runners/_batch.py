"""BatchRunner: full pipeline orchestration with async concurrency.

Orchestrates the complete evaluation pipeline:
perturb -> query -> score -> analyze -> report.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from llm_consistency._exceptions import ValidationError
from llm_consistency.runners._checkpoint import CheckpointWriter, read_checkpoint
from llm_consistency.runners._metadata import RunMetadata
from llm_consistency.runners._pipeline import (
    build_scored_qcr,
    generate_variants_for_question,
    render_prompt,
)
from llm_consistency.scoring import _extract_mc_answer
from llm_consistency.types import (
    EvaluationReport,
    LLMResponse,
    MCQuestion,
    ScoredResponse,
)

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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

    Async concurrency is bounded by ``config.concurrency`` via an
    ``asyncio.Semaphore``.  An optional Rich ``Progress`` instance
    enables live progress display during batch execution.

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
                resumes.

        Returns:
            A complete :class:`EvaluationReport` with per-question
            consistency results and aggregate metrics.
        """
        semaphore = asyncio.Semaphore(config.concurrency)
        self.last_metadata = RunMetadata.capture(config, seed)

        ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
        completed_ids: set[str] = set()
        results: list[QuestionConsistencyResult] = []

        if (
            ckpt_path is not None
            and ckpt_path.exists()
            and ckpt_path.stat().st_size > 0
        ):
            _, prior = read_checkpoint(ckpt_path, config=config, seed=seed)
            results.extend(prior)
            completed_ids = {qcr.question_id for qcr in prior}
            if completed_ids:
                _logger.info(
                    "BatchRunner resuming from checkpoint %s: %d question(s) "
                    "already complete, will be skipped.",
                    ckpt_path,
                    len(completed_ids),
                )

        # Set up optional Rich progress task
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "Evaluating questions...",
                total=len(dataset),
            )
            if completed_ids:
                progress.advance(task_id, advance=len(completed_ids))

        skipped = 0

        with ExitStack() as stack:
            writer: CheckpointWriter | None = None
            if ckpt_path is not None:
                writer = stack.enter_context(
                    CheckpointWriter(ckpt_path, config=config, seed=seed)
                )

            for question in dataset:
                if not isinstance(question, MCQuestion):
                    # Non-MC questions (open-ended) are not yet supported.
                    skipped += 1
                    continue

                if question.id in completed_ids:
                    # Already in the checkpoint; do not re-query the provider.
                    continue

                qcr = await self._process_question(
                    question, config, provider, scorer, semaphore, seed
                )
                results.append(qcr)
                if writer is not None:
                    writer.append(qcr)

                if progress is not None and task_id is not None:
                    progress.advance(task_id)

        if skipped:
            _logger.warning(
                "BatchRunner skipped %d non-MCQuestion items "
                "(open-ended not yet supported)",
                skipped,
            )

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

    async def _process_question(
        self,
        question: MCQuestion,
        config: EvaluationConfig,
        provider: BaseLLMProvider,
        scorer: BaseScorer,
        semaphore: asyncio.Semaphore,
        seed: int,
    ) -> QuestionConsistencyResult:
        """Process a single question through the pipeline.

        Args:
            question: The MC question to evaluate.
            config: Evaluation configuration.
            provider: LLM provider.
            scorer: Response scorer.
            semaphore: Concurrency semaphore.
            seed: Random seed.

        Returns:
            A QuestionConsistencyResult for this question.
        """
        # 1. Generate variants
        variants = generate_variants_for_question(question, config, seed)

        # 2. Render prompts and build query pairs
        prompts: list[tuple[str, str]] = []
        for i, variant in enumerate(variants):
            prompt_text = render_prompt(variant)
            variant_qid = f"{question.id}_v{i}"
            prompts.append((prompt_text, variant_qid))

        # 3. Query LLM concurrently (bounded by semaphore).
        # Per-variant errors are captured so one failure does not tear down
        # the whole batch; failed variants are recorded as an empty raw
        # output plus an ``error`` ScoredResponse so they participate in
        # rc_correct (always False) and rc_agree (as a distinct sentinel).
        async def _bounded_query(prompt: str, qid: str) -> tuple[str, str, str | None]:
            """Query with semaphore and return (qid, raw_output, error)."""
            async with semaphore:
                try:
                    resp = await provider.query(prompt, qid)
                except Exception as exc:
                    # Capture all per-variant failures so one bad call does not
                    # tear down the batch; the error is recorded on the
                    # ScoredResponse downstream.
                    return (qid, "", f"{type(exc).__name__}: {exc}")
                return (qid, resp.raw_output, None)

        tasks = [_bounded_query(p, qid) for p, qid in prompts]
        query_results = await asyncio.gather(*tasks)

        # 4. Score each response against the ORIGINAL question
        valid_labels = frozenset(o.label for o in question.options)
        scored_responses: list[ScoredResponse] = []
        variant_data: list[tuple[str, bool]] = []

        for variant, (variant_qid, raw_output, error) in zip(
            variants, query_results, strict=True
        ):
            pt_value = variant.perturbation_type.value
            if error is not None:
                scored_responses.append(
                    ScoredResponse(
                        question_id=variant_qid,
                        is_correct=False,
                        score=0.0,
                        scoring_method=f"error:{error}",
                        perturbation_type=pt_value,
                    )
                )
                variant_data.append((f"<error:{variant_qid}>", False))
                continue

            # Build an LLMResponse for the scorer
            response = LLMResponse(
                question_id=variant_qid,
                raw_output=raw_output,
                extracted_answer="",
                model=config.model,
                provider=config.provider,
            )

            sr = scorer.score(response, question)
            sr = replace(sr, perturbation_type=pt_value)
            scored_responses.append(sr)

            # Extract answer label for variant_data (answer distribution)
            extracted = _extract_mc_answer(raw_output, valid_labels)
            answer_str = extracted if extracted is not None else raw_output.strip()
            variant_data.append((answer_str, sr.is_correct))

        # 5. Build QCR with scored_responses populated
        return build_scored_qcr(question.id, variant_data, tuple(scored_responses))
