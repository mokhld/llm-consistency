"""StreamingRunner: async iterator yielding per-question results progressively.

Yields :class:`QuestionConsistencyResult` objects one-at-a-time as each
question completes, enabling progressive display during long evaluations.
Reuses the same pipeline helpers as :class:`BatchRunner`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from llm_consistency.runners._pipeline import (
    build_scored_qcr,
    generate_variants_for_question,
    render_prompt,
)
from llm_consistency.scoring import _extract_mc_answer
from llm_consistency.types import (
    LLMResponse,
    MCQuestion,
    QuestionConsistencyResult,
    ScoredResponse,
)

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

        Args:
            dataset: Dataset of questions to evaluate.
            config: Evaluation configuration.
            provider: LLM provider for querying.
            scorer: Scorer for evaluating responses.
            seed: Random seed for reproducible perturbation generation.

        Yields:
            A :class:`QuestionConsistencyResult` for each question.
        """
        semaphore = asyncio.Semaphore(config.concurrency)
        skipped = 0

        try:
            for question in dataset:
                if not isinstance(question, MCQuestion):
                    skipped += 1
                    continue

                qcr = await self._process_question(
                    question, config, provider, scorer, semaphore, seed
                )
                yield qcr
        finally:
            # Best-effort: when the consumer breaks early (GeneratorExit /
            # asyncio cancellation), let any in-flight per-question tasks
            # observe cancellation. _process_question's inner gather() is
            # awaited synchronously so there is nothing for us to cancel
            # here directly, but we log skip counts and any unexpected
            # in-flight tasks scheduled on the running loop.
            if skipped:
                _logger.warning(
                    "StreamingRunner skipped %d non-MCQuestion items "
                    "(open-ended not yet supported)",
                    skipped,
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
        """Process a single question through the evaluation pipeline.

        Identical logic to :meth:`BatchRunner._process_question`, reusing
        the shared pipeline helpers.

        Args:
            question: The MC question to evaluate.
            config: Evaluation configuration.
            provider: LLM provider.
            scorer: Response scorer.
            semaphore: Concurrency semaphore.
            seed: Random seed.

        Returns:
            A :class:`QuestionConsistencyResult` for this question.
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
        # Per-variant errors captured (see BatchRunner for rationale).
        async def _bounded_query(prompt: str, qid: str) -> tuple[str, str, str | None]:
            async with semaphore:
                try:
                    resp = await provider.query(prompt, qid)
                except Exception as exc:
                    return (qid, "", f"{type(exc).__name__}: {exc}")
                return (qid, resp.raw_output, None)

        tasks = [_bounded_query(p, qid) for p, qid in prompts]
        query_results = await asyncio.gather(*tasks)

        # 4. Score each response against the ORIGINAL question
        valid_labels = frozenset(o.label for o in question.options)
        scored_responses: list[ScoredResponse] = []
        variant_data: list[tuple[str, bool]] = []

        for variant_qid, raw_output, error in query_results:
            if error is not None:
                scored_responses.append(
                    ScoredResponse(
                        question_id=variant_qid,
                        is_correct=False,
                        score=0.0,
                        scoring_method=f"error:{error}",
                    )
                )
                variant_data.append((f"<error:{variant_qid}>", False))
                continue

            response = LLMResponse(
                question_id=variant_qid,
                raw_output=raw_output,
                extracted_answer="",
                model=config.model,
                provider=config.provider,
            )
            sr = scorer.score(response, question)
            scored_responses.append(sr)

            extracted = _extract_mc_answer(raw_output, valid_labels)
            answer_str = extracted if extracted is not None else raw_output.strip()
            variant_data.append((answer_str, sr.is_correct))

        # 5. Build QCR with scored_responses populated
        return build_scored_qcr(question.id, variant_data, tuple(scored_responses))
