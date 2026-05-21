"""Tests for batch runner, pipeline helpers, and run metadata."""

from __future__ import annotations

import importlib
import json
import platform
from typing import TYPE_CHECKING

import pytest

from llm_consistency._exceptions import ValidationError
from llm_consistency.datasets import CustomDataset
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.scoring import ExactMatchScorer
from llm_consistency.types import (
    EvaluationConfig,
    EvaluationReport,
    MCOption,
    MCQuestion,
    OpenEndedQuestion,
    PerturbationType,
    PerturbedVariant,
    QuestionConsistencyResult,
    ScoredResponse,
)

if TYPE_CHECKING:
    from pathlib import Path

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


# ---------------------------------------------------------------------------
# Empty-dataset guard
# ---------------------------------------------------------------------------


class TestBatchRunnerEmptyDataset:
    """BatchRunner.run() raises ValidationError on zero MC questions."""

    @pytest.mark.asyncio
    async def test_empty_dataset_raises(self) -> None:
        mod = _get_batch()
        dataset = CustomDataset([])
        config = _make_config()
        provider = MockLLMProvider(model="mock", default_response="A")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        with pytest.raises(ValidationError, match="zero results"):
            await runner.run(dataset, config, provider, scorer, seed=42)


# ---------------------------------------------------------------------------
# Per-variant error capture
# ---------------------------------------------------------------------------


class _FailingProvider(MockLLMProvider):
    """MockLLMProvider variant that raises on every query."""

    async def query(  # type: ignore[override]
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
    ) -> object:
        msg = f"simulated provider failure for {question_id}"
        raise RuntimeError(msg)


class TestBatchRunnerErrorCapture:
    """Provider failures are captured per-variant, not propagated."""

    @pytest.mark.asyncio
    async def test_provider_failure_does_not_abort_batch(self) -> None:
        mod = _get_batch()
        q = _make_question("q1", correct_label="A")
        dataset = CustomDataset([q])
        config = _make_config(num_variants=2)
        provider = _FailingProvider(model="mock")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        report = await runner.run(dataset, config, provider, scorer, seed=42)

        # The run completed; the one question is in the report.
        assert report.total_questions == 1
        (qcr,) = report.results
        # All variants failed -> rc_correct is 0 and scoring_method tagged 'error'.
        assert qcr.rc_correct == 0.0
        assert all(
            sr.scoring_method.startswith("error:") for sr in qcr.scored_responses
        )


class TestBatchRunnerSkippedQuestions:
    """Non-MCQuestion items are skipped with a warning log."""

    @pytest.mark.asyncio
    async def test_non_mc_questions_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mod = _get_batch()
        oe = OpenEndedQuestion(id="o1", stem="Why?", reference_answers=("because",))
        mc = _make_question("q1", correct_label="A")
        dataset = CustomDataset([oe, mc])
        config = _make_config()
        provider = MockLLMProvider(model="mock", default_response="A")
        scorer = ExactMatchScorer()

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        with caplog.at_level("WARNING", logger="llm_consistency.runners._batch"):
            report = await runner.run(dataset, config, provider, scorer, seed=42)

        assert report.total_questions == 1
        assert any("skipped 1" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


class _CountingProvider(MockLLMProvider):
    """MockLLMProvider that records which question_ids it was queried for."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.queried_ids: list[str] = []

    async def query(  # type: ignore[override]
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
    ) -> object:
        self.queried_ids.append(question_id)
        return await super().query(prompt, question_id, system=system)


def _read_qcr_ids_from_checkpoint(path: Path) -> list[str]:
    """Helper: list question IDs present in a checkpoint file (qcr lines only)."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "qcr":
            ids.append(obj["qcr"]["question_id"])
    return ids


class TestBatchRunnerCheckpointResume:
    """Checkpoint persistence and resume semantics in BatchRunner.run()."""

    @pytest.mark.asyncio
    async def test_fresh_run_with_checkpoint_writes_all_qcrs(
        self,
        tmp_path: Path,
    ) -> None:
        mod = _get_batch()
        questions = [_make_question(f"q{i}") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()
        ckpt = tmp_path / "run.jsonl"

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        report = await runner.run(
            dataset, config, provider, scorer, seed=42, checkpoint_path=ckpt
        )

        assert report.total_questions == 3
        assert _read_qcr_ids_from_checkpoint(ckpt) == ["q0", "q1", "q2"]

    @pytest.mark.asyncio
    async def test_resume_skips_completed_questions(
        self,
        tmp_path: Path,
    ) -> None:
        mod = _get_batch()
        questions = [_make_question(f"q{i}") for i in range(4)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        scorer = ExactMatchScorer()
        ckpt = tmp_path / "run.jsonl"

        # First run: complete the first 2 questions only.
        partial_provider = _CountingProvider(model="mock")
        partial_runner = mod.BatchRunner()  # type: ignore[attr-defined]
        await partial_runner.run(
            CustomDataset(questions[:2]),
            config,
            partial_provider,
            scorer,
            seed=42,
            checkpoint_path=ckpt,
        )
        assert _read_qcr_ids_from_checkpoint(ckpt) == ["q0", "q1"]

        # Resume against the full dataset.
        resume_provider = _CountingProvider(model="mock")
        resume_runner = mod.BatchRunner()  # type: ignore[attr-defined]
        report = await resume_runner.run(
            dataset,
            config,
            resume_provider,
            scorer,
            seed=42,
            checkpoint_path=ckpt,
        )

        # Only q2 and q3 should have been re-queried; q0/q1 came from the
        # checkpoint, which means no variant of q0/q1 hit the provider.
        for qid in resume_provider.queried_ids:
            assert qid.startswith(("q2_", "q3_")), (
                f"resume provider unexpectedly queried {qid}"
            )

        # Final report should cover all 4 questions in dataset order.
        assert report.total_questions == 4
        assert [r.question_id for r in report.results] == ["q0", "q1", "q2", "q3"]
        assert _read_qcr_ids_from_checkpoint(ckpt) == ["q0", "q1", "q2", "q3"]

    @pytest.mark.asyncio
    async def test_resume_with_mismatched_config_raises(
        self,
        tmp_path: Path,
    ) -> None:
        mod = _get_batch()
        questions = [_make_question(f"q{i}") for i in range(2)]
        dataset = CustomDataset(questions)
        config_a = _make_config(num_variants=2)
        config_b = _make_config(num_variants=5)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()
        ckpt = tmp_path / "run.jsonl"

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        await runner.run(
            dataset, config_a, provider, scorer, seed=42, checkpoint_path=ckpt
        )

        with pytest.raises(ValidationError, match="different config"):
            await runner.run(
                dataset, config_b, provider, scorer, seed=42, checkpoint_path=ckpt
            )

    @pytest.mark.asyncio
    async def test_resume_with_different_seed_raises(
        self,
        tmp_path: Path,
    ) -> None:
        mod = _get_batch()
        questions = [_make_question(f"q{i}") for i in range(2)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        provider = MockLLMProvider(model="mock")
        scorer = ExactMatchScorer()
        ckpt = tmp_path / "run.jsonl"

        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        await runner.run(
            dataset, config, provider, scorer, seed=42, checkpoint_path=ckpt
        )

        with pytest.raises(ValidationError, match="different config"):
            await runner.run(
                dataset, config, provider, scorer, seed=99, checkpoint_path=ckpt
            )

    @pytest.mark.asyncio
    async def test_full_resume_no_new_queries_required(
        self,
        tmp_path: Path,
    ) -> None:
        """If the checkpoint already covers the whole dataset, the
        provider must not be queried at all on resume."""
        mod = _get_batch()
        questions = [_make_question(f"q{i}") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        scorer = ExactMatchScorer()
        ckpt = tmp_path / "run.jsonl"

        first_provider = MockLLMProvider(model="mock")
        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        original = await runner.run(
            dataset, config, first_provider, scorer, seed=42, checkpoint_path=ckpt
        )

        # Resume with a counting provider; expect zero queries.
        second_provider = _CountingProvider(model="mock")
        resumed = await runner.run(
            dataset, config, second_provider, scorer, seed=42, checkpoint_path=ckpt
        )

        assert second_provider.queried_ids == []
        assert resumed.total_questions == original.total_questions
        assert [r.question_id for r in resumed.results] == [
            r.question_id for r in original.results
        ]

    @pytest.mark.asyncio
    async def test_resume_truncated_last_line_does_not_corrupt_run(
        self,
        tmp_path: Path,
    ) -> None:
        """A crash-truncated final QCR line is skipped on resume; the
        affected question gets re-queried and the report still has all
        results."""
        mod = _get_batch()
        questions = [_make_question(f"q{i}") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(num_variants=2)
        scorer = ExactMatchScorer()
        ckpt = tmp_path / "run.jsonl"

        first_provider = MockLLMProvider(model="mock")
        runner = mod.BatchRunner()  # type: ignore[attr-defined]
        await runner.run(
            CustomDataset(questions[:2]),
            config,
            first_provider,
            scorer,
            seed=42,
            checkpoint_path=ckpt,
        )

        # Simulate a partial write by appending an unterminated qcr line.
        with ckpt.open("a", encoding="utf-8") as fh:
            fh.write('{"type": "qcr", "qcr": {"question_id": "q9"')  # no newline

        # Resume with the full dataset; the bogus line is dropped.
        resume_provider = _CountingProvider(model="mock")
        report = await runner.run(
            dataset, config, resume_provider, scorer, seed=42, checkpoint_path=ckpt
        )

        # q0 and q1 came from the checkpoint; only q2 should have been re-queried.
        for qid in resume_provider.queried_ids:
            assert qid.startswith("q2_"), (
                f"unexpected requery of {qid}; partial-write recovery is broken"
            )
        assert report.total_questions == 3
        assert [r.question_id for r in report.results] == ["q0", "q1", "q2"]
