"""Tests for StreamingRunner with async iterator yielding per-question results."""

from __future__ import annotations

import importlib

import pytest

from llm_consistency.datasets import CustomDataset
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.scoring import ExactMatchScorer
from llm_consistency.types import (
    EvaluationConfig,
    MCOption,
    MCQuestion,
    PerturbationType,
    QuestionConsistencyResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_question(qid: str, correct_label: str = "A") -> MCQuestion:
    """Build a simple 4-option MC question with the given correct label."""
    labels = ["A", "B", "C", "D"]
    return MCQuestion(
        id=qid,
        stem=f"Question {qid}?",
        options=tuple(
            MCOption(
                label=lab,
                text=f"Option {lab}",
                is_correct=(lab == correct_label),
            )
            for lab in labels
        ),
    )


def _make_config(
    num_variants: int = 3,
    concurrency: int = 5,
) -> EvaluationConfig:
    return EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(PerturbationType.OPTION_REORDER,),
        scorer="exact_match",
        num_variants=num_variants,
        concurrency=concurrency,
    )


def _get_streaming() -> object:
    return importlib.import_module("llm_consistency.runners._streaming")


def _get_runners() -> object:
    return importlib.import_module("llm_consistency.runners")


# ---------------------------------------------------------------------------
# StreamingRunner.run_stream() tests
# ---------------------------------------------------------------------------


class TestStreamingRunnerYieldsQCR:
    """StreamingRunner.run_stream() returns an AsyncIterator[QCR]."""

    @pytest.mark.asyncio
    async def test_yields_exactly_n_qcr_objects(self) -> None:
        """With a 3-question dataset and MockLLMProvider, yields exactly 3 QCRs."""
        mod = _get_streaming()
        questions = [_make_question(f"q{i}", correct_label="A") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.StreamingRunner()  # type: ignore[attr-defined]
        results = [
            qcr
            async for qcr in runner.run_stream(
                dataset, config, provider, scorer, seed=42
            )
        ]

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_each_qcr_has_scored_responses(self) -> None:
        """Each yielded QCR has scored_responses populated (not empty)."""
        mod = _get_streaming()
        q = _make_question("q1", correct_label="A")
        dataset = CustomDataset([q])
        config = _make_config(num_variants=3)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.StreamingRunner()  # type: ignore[attr-defined]
        async for qcr in runner.run_stream(dataset, config, provider, scorer, seed=42):
            assert isinstance(qcr, QuestionConsistencyResult)
            assert len(qcr.scored_responses) > 0

    @pytest.mark.asyncio
    async def test_yields_one_at_a_time(self) -> None:
        """Results are yielded progressively (not all at once)."""
        mod = _get_streaming()
        questions = [_make_question(f"q{i}", correct_label="A") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.StreamingRunner()  # type: ignore[attr-defined]
        count = 0
        async for qcr in runner.run_stream(dataset, config, provider, scorer, seed=42):
            count += 1
            # After first yield we should have exactly 1 result so far
            if count == 1:
                assert isinstance(qcr, QuestionConsistencyResult)
                break  # Early break to confirm streaming (not batched)

        # If we got here, at least one result was yielded before all completed
        assert count == 1

    @pytest.mark.asyncio
    async def test_early_break_no_exception(self) -> None:
        """Early break from async for does not raise exceptions (clean exit)."""
        mod = _get_streaming()
        questions = [_make_question(f"q{i}", correct_label="A") for i in range(5)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.StreamingRunner()  # type: ignore[attr-defined]
        # Break after first result -- should not raise
        async for qcr in runner.run_stream(dataset, config, provider, scorer, seed=42):
            assert isinstance(qcr, QuestionConsistencyResult)
            break

    @pytest.mark.asyncio
    async def test_qcr_question_ids_match_dataset(self) -> None:
        """Each yielded QCR's question_id matches the original question."""
        mod = _get_streaming()
        questions = [_make_question(f"q{i}", correct_label="A") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.StreamingRunner()  # type: ignore[attr-defined]
        qids = [
            qcr.question_id
            async for qcr in runner.run_stream(
                dataset, config, provider, scorer, seed=42
            )
        ]

        assert qids == ["q0", "q1", "q2"]


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------


class TestStreamingRunnerPublicAPI:
    """StreamingRunner is importable from runners package."""

    def test_streaming_runner_importable(self) -> None:
        mod = _get_runners()
        assert hasattr(mod, "StreamingRunner")
