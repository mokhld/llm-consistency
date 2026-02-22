"""BatchRunner: full pipeline orchestration with async concurrency.

Orchestrates the complete evaluation pipeline:
perturb -> query -> score -> analyze -> report.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

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
    ) -> EvaluationReport:
        """Run the full evaluation pipeline on a dataset.

        Args:
            dataset: Dataset of questions to evaluate.
            config: Evaluation configuration.
            provider: LLM provider for querying.
            scorer: Scorer for evaluating responses.
            progress: Optional Rich Progress instance for live display.
            seed: Random seed for reproducible perturbation generation.

        Returns:
            A complete :class:`EvaluationReport` with per-question
            consistency results and aggregate metrics.
        """
        semaphore = asyncio.Semaphore(config.concurrency)
        self.last_metadata = RunMetadata.capture(config, seed)

        # Set up optional Rich progress task
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "Evaluating questions...",
                total=len(dataset),
            )

        results: list[QuestionConsistencyResult] = []

        for question in dataset:
            if not isinstance(question, MCQuestion):
                continue  # Skip non-MC questions (open-ended not yet supported)

            qcr = await self._process_question(
                question, config, provider, scorer, semaphore, seed
            )
            results.append(qcr)

            if progress is not None and task_id is not None:
                progress.advance(task_id)

        # Compute aggregates
        total_questions = len(results)
        total_variants = sum(r.total_variants for r in results)
        mean_rc_correct = (
            sum(r.rc_correct for r in results) / total_questions
            if total_questions > 0
            else 0.0
        )
        mean_rc_agree = (
            sum(r.rc_agree for r in results) / total_questions
            if total_questions > 0
            else 0.0
        )

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

        # 3. Query LLM concurrently (bounded by semaphore)
        async def _bounded_query(prompt: str, qid: str) -> tuple[str, str]:
            """Query with semaphore and return (qid, raw_output)."""
            async with semaphore:
                resp = await provider.query(prompt, qid)
                return (qid, resp.raw_output)

        tasks = [_bounded_query(p, qid) for p, qid in prompts]
        query_results = await asyncio.gather(*tasks)

        # 4. Score each response against the ORIGINAL question
        valid_labels = frozenset(o.label for o in question.options)
        scored_responses: list[ScoredResponse] = []
        variant_data: list[tuple[str, bool]] = []

        for variant_qid, raw_output in query_results:
            # Build an LLMResponse for the scorer
            response = LLMResponse(
                question_id=variant_qid,
                raw_output=raw_output,
                extracted_answer="",
                model=config.model,
                provider=config.provider,
            )

            sr = scorer.score(response, question)
            scored_responses.append(sr)

            # Extract answer label for variant_data (answer distribution)
            extracted = _extract_mc_answer(raw_output, valid_labels)
            answer_str = extracted if extracted is not None else raw_output.strip()
            variant_data.append((answer_str, sr.is_correct))

        # 5. Build QCR with scored_responses populated
        return build_scored_qcr(question.id, variant_data, tuple(scored_responses))
