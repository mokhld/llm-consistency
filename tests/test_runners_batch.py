"""Tests for batch runner, pipeline helpers, and run metadata."""

from __future__ import annotations

import importlib
import platform

import pytest

from llm_consistency.datasets import CustomDataset
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.scoring import ExactMatchScorer
from llm_consistency.types import (
    EvaluationConfig,
    EvaluationReport,
    MCOption,
    MCQuestion,
    PerturbationType,
    PerturbedVariant,
    QuestionConsistencyResult,
    ScoredResponse,
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


def _get_pipeline() -> object:
    return importlib.import_module("llm_consistency.runners._pipeline")


def _get_metadata() -> object:
    return importlib.import_module("llm_consistency.runners._metadata")


def _get_batch() -> object:
    return importlib.import_module("llm_consistency.runners._batch")


def _get_runners() -> object:
    return importlib.import_module("llm_consistency.runners")


# ---------------------------------------------------------------------------
# Pipeline helpers tests
# ---------------------------------------------------------------------------


class TestGenerateVariantsForQuestion:
    """Tests for generate_variants_for_question pipeline helper."""

    def test_returns_variants_list(self) -> None:
        mod = _get_pipeline()
        q = _make_question("q1")
        config = _make_config(num_variants=3)
        variants = mod.generate_variants_for_question(q, config, seed=42)  # type: ignore[attr-defined]
        assert isinstance(variants, list)
        assert len(variants) == 3

    def test_respects_num_variants(self) -> None:
        mod = _get_pipeline()
        q = _make_question("q1")
        config = _make_config(num_variants=2)
        variants = mod.generate_variants_for_question(q, config, seed=42)  # type: ignore[attr-defined]
        assert len(variants) == 2

    def test_variant_original_question_id(self) -> None:
        mod = _get_pipeline()
        q = _make_question("q1")
        config = _make_config(num_variants=1)
        variants = mod.generate_variants_for_question(q, config, seed=42)  # type: ignore[attr-defined]
        assert variants[0].original_question_id == "q1"


class TestRenderPrompt:
    """Tests for render_prompt pipeline helper."""

    def test_with_options_renders_stem_plus_options(self) -> None:
        mod = _get_pipeline()
        q = _make_question("q1")
        config = _make_config(num_variants=1)
        variants = mod.generate_variants_for_question(q, config, seed=42)  # type: ignore[attr-defined]
        text = mod.render_prompt(variants[0])  # type: ignore[attr-defined]
        assert "Question q1?" in text
        assert "A." in text or "B." in text

    def test_without_options_returns_stem(self) -> None:
        mod = _get_pipeline()
        variant = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.FORMAT_CHANGE,
            seed=42,
            variant_index=0,
            stem="Formatted question text\nA. opt1\nB. opt2",
            options=None,
        )
        text = mod.render_prompt(variant)  # type: ignore[attr-defined]
        assert text == "Formatted question text\nA. opt1\nB. opt2"


class TestBuildScoredQcr:
    """Tests for build_scored_qcr pipeline helper."""

    def test_populates_scored_responses(self) -> None:
        mod = _get_pipeline()
        scored = (
            ScoredResponse(
                question_id="q1_v0",
                is_correct=True,
                score=1.0,
                scoring_method="exact_match",
            ),
            ScoredResponse(
                question_id="q1_v1",
                is_correct=False,
                score=0.0,
                scoring_method="exact_match",
            ),
        )
        variant_data = [("A", True), ("B", False)]
        qcr = mod.build_scored_qcr("q1", variant_data, scored)  # type: ignore[attr-defined]

        assert qcr.question_id == "q1"
        assert qcr.scored_responses == scored
        assert len(qcr.scored_responses) == 2
        assert qcr.total_variants == 2
        assert qcr.rc_correct == 0.5


# ---------------------------------------------------------------------------
# RunMetadata tests
# ---------------------------------------------------------------------------


class TestRunMetadata:
    """Tests for RunMetadata frozen dataclass."""

    def test_capture_creates_metadata(self) -> None:
        mod = _get_metadata()
        config = _make_config()
        meta = mod.RunMetadata.capture(config, seed=42)  # type: ignore[attr-defined]
        assert meta.perturbation_seed == 42
        assert meta.model == "mock"
        assert meta.provider == "mock"
        assert meta.python_version == platform.python_version()
        assert "T" in meta.timestamp  # ISO 8601

    def test_capture_is_frozen(self) -> None:
        mod = _get_metadata()
        config = _make_config()
        meta = mod.RunMetadata.capture(config, seed=42)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            meta.model = "changed"

    def test_to_dict(self) -> None:
        mod = _get_metadata()
        config = _make_config()
        meta = mod.RunMetadata.capture(config, seed=42)  # type: ignore[attr-defined]
        d = meta.to_dict()
        assert "package_version" in d
        assert "python_version" in d
        assert "timestamp" in d
        assert "config_snapshot" in d
        assert d["perturbation_seed"] == 42
        assert d["model"] == "mock"
        assert d["provider"] == "mock"


# ---------------------------------------------------------------------------
# BatchRunner.run() tests
# ---------------------------------------------------------------------------


class TestBatchRunnerEndToEnd:
    """End-to-end tests for BatchRunner with MockLLMProvider."""

    async def test_produces_evaluation_report(self) -> None:
        mod = _get_batch()
        q1 = _make_question("q1", correct_label="A")
        q2 = _make_question("q2", correct_label="A")
        dataset = CustomDataset([q1, q2])
        config = _make_config(num_variants=3, concurrency=5)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        report = await runner.run(dataset, config, provider, scorer, seed=42)

        assert isinstance(report, EvaluationReport)
        assert report.total_questions == 2
        assert len(report.results) == 2
        for qcr in report.results:
            assert isinstance(qcr, QuestionConsistencyResult)
            assert len(qcr.scored_responses) > 0

    async def test_report_aggregates(self) -> None:
        mod = _get_batch()
        q1 = _make_question("q1", correct_label="A")
        q2 = _make_question("q2", correct_label="A")
        dataset = CustomDataset([q1, q2])
        config = _make_config(num_variants=3, concurrency=5)
        # MockLLMProvider returns "A" by default, correct_label is "A"
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        report = await runner.run(dataset, config, provider, scorer, seed=42)

        # All responses are "A", which is correct
        assert report.mean_rc_correct > 0.0
        assert report.mean_rc_agree > 0.0

    async def test_concurrency_semaphore(self) -> None:
        mod = _get_batch()
        questions = [_make_question(f"q{i}", correct_label="A") for i in range(4)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2, concurrency=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        report = await runner.run(dataset, config, provider, scorer, seed=42)

        assert report.total_questions == 4
        assert len(report.results) == 4

    async def test_no_progress_no_crash(self) -> None:
        mod = _get_batch()
        q = _make_question("q1", correct_label="A")
        dataset = CustomDataset([q])
        config = _make_config(num_variants=2, concurrency=5)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        # progress=None (default) should not crash
        report = await runner.run(
            dataset, config, provider, scorer, progress=None, seed=42
        )
        assert report.total_questions == 1


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------


class TestRunnersPublicAPI:
    """Tests for runners package __init__.py re-exports."""

    def test_batch_runner_importable(self) -> None:
        mod = _get_runners()
        assert hasattr(mod, "BatchRunner")

    def test_run_metadata_importable(self) -> None:
        mod = _get_runners()
        assert hasattr(mod, "RunMetadata")

    def test_pipeline_helpers_importable(self) -> None:
        mod = _get_runners()
        assert hasattr(mod, "generate_variants_for_question")
        assert hasattr(mod, "render_prompt")
        assert hasattr(mod, "build_scored_qcr")
