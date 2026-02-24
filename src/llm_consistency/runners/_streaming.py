"""StreamingRunner: async iterator yielding per-question results progressively.

Yields :class:`QuestionConsistencyResult` objects one-at-a-time as each
question completes, enabling progressive display during long evaluations.
Reuses the same pipeline helpers as :class:`BatchRunner`.
"""

from __future__ import annotations

import asyncio
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

        try:
            for question in dataset:
                if not isinstance(question, MCQuestion):
                    continue  # Skip non-MC questions (open-ended not yet supported)

                qcr = await self._process_question(
                    question, config, provider, scorer, semaphore, seed
                )
                yield qcr
        finally:
            # Ensure clean shutdown on GeneratorExit (early break from async for)
            pass

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

        # 3. Query LLM concurrently (bounded by semaphore)
        async def _bounded_query(prompt: str, qid: str) -> tuple[str, str]:
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
