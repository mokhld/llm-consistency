"""Tests for CIRunner with pass/fail exit codes based on metric thresholds."""

from __future__ import annotations

import importlib

import pytest

from llm_consistency.datasets import CustomDataset
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.scoring import ExactMatchScorer
from llm_consistency.types import (
    EvaluationConfig,
    GenerationParams,
    LLMResponse,
    MCOption,
    MCQuestion,
    PerturbationType,
    QuestionConsistencyResult,
    ScoredResponse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_question(qid: str, correct_label: str = "B") -> MCQuestion:
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
    mca_threshold: float = 1.0,
    core_threshold: float | None = None,
    num_variants: int = 3,
    min_mca: float = 1.0,
) -> EvaluationConfig:
    # separator_change keeps option labels in place, so a fixed answer is
    # correct on every variant and these tests exercise only the gate.
    return EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(PerturbationType.SEPARATOR_CHANGE,),
        scorer="exact_match",
        num_variants=num_variants,
        concurrency=5,
        mca_threshold=mca_threshold,
        min_mca=min_mca,
        core_threshold=core_threshold,
    )


def _get_ci() -> object:
    return importlib.import_module("llm_consistency.runners._ci")


def _get_runners() -> object:
    return importlib.import_module("llm_consistency.runners")


# ---------------------------------------------------------------------------
# CIRunner.run() tests
# ---------------------------------------------------------------------------


class TestCIRunnerExitCodes:
    """CIRunner.run() returns exit code 0 (pass) or 1 (fail)."""

    @pytest.mark.asyncio
    async def test_all_correct_returns_zero(self) -> None:
        """MockLLMProvider always answers 'B', correct_label='B' -> exit 0."""
        mod = _get_ci()
        questions = [_make_question(f"q{i}", correct_label="B") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(mca_threshold=1.0)
        provider = MockLLMProvider(model="mock", default_response="B")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_all_incorrect_returns_one(self) -> None:
        """MockLLMProvider always answers 'X' (invalid) -> exit 1."""
        mod = _get_ci()
        questions = [_make_question(f"q{i}", correct_label="B") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(mca_threshold=1.0)
        provider = MockLLMProvider(model="mock", default_response="X")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_core_threshold_fail(self) -> None:
        """When CORE score is below core_threshold, returns 1."""
        mod = _get_ci()
        questions = [_make_question(f"q{i}", correct_label="B") for i in range(3)]
        dataset = CustomDataset(questions)
        # Set an impossibly high CORE threshold
        config = _make_config(
            mca_threshold=0.0,  # low MCA threshold -> pass MCA check
            core_threshold=0.999,
        )
        # 'X' always incorrect -> CORE will be low
        provider = MockLLMProvider(model="mock", default_response="X")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_core_threshold_none_ignores_core(self) -> None:
        """When core_threshold is None, only MCA is checked."""
        mod = _get_ci()
        questions = [_make_question(f"q{i}", correct_label="B") for i in range(3)]
        dataset = CustomDataset(questions)
        # No core_threshold + low MCA threshold -> should pass even with bad answers
        config = _make_config(
            mca_threshold=0.0,  # All questions pass at threshold 0
            core_threshold=None,
        )
        provider = MockLLMProvider(model="mock", default_response="B")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_return_type_is_int(self) -> None:
        """CIRunner.run() returns an int, not an EvaluationReport."""
        mod = _get_ci()
        q = _make_question("q1", correct_label="B")
        dataset = CustomDataset([q])
        config = _make_config()
        provider = MockLLMProvider(model="mock", default_response="B")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        result = await runner.run(dataset, config, provider, scorer, seed=42)

        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------


class TestCIRunnerPublicAPI:
    """CIRunner is importable from runners package."""

    def test_ci_runner_importable(self) -> None:
        mod = _get_runners()
        assert hasattr(mod, "CIRunner")


class TestCIRunnerFailures:
    """CIRunner.failures records which thresholds failed and why."""

    @pytest.mark.asyncio
    async def test_mca_failure_recorded(self) -> None:
        mod = _get_ci()
        questions = [_make_question(f"q{i}", correct_label="B") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(mca_threshold=1.0)
        provider = MockLLMProvider(model="mock", default_response="X")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 1
        assert any("MCA check failed" in f for f in runner.failures)

    @pytest.mark.asyncio
    async def test_core_failure_recorded(self) -> None:
        mod = _get_ci()
        questions = [_make_question(f"q{i}", correct_label="B") for i in range(3)]
        dataset = CustomDataset(questions)
        config = _make_config(mca_threshold=0.0, core_threshold=0.999)
        provider = MockLLMProvider(model="mock", default_response="X")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 1
        assert any("CORE check failed" in f for f in runner.failures)

    @pytest.mark.asyncio
    async def test_no_failures_on_pass(self) -> None:
        mod = _get_ci()
        q = _make_question("q1", correct_label="B")
        dataset = CustomDataset([q])
        config = _make_config()
        provider = MockLLMProvider(model="mock", default_response="B")
        scorer = ExactMatchScorer()

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(dataset, config, provider, scorer, seed=42)

        assert exit_code == 0
        assert runner.failures == ()


class _FailOnStemProvider(MockLLMProvider):
    """Mock provider that raises for prompts containing *stem*."""

    def __init__(self, stem: str, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._stem = stem

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> LLMResponse:
        if self._stem in prompt:
            msg = "simulated provider outage"
            raise RuntimeError(msg)
        return await super().query(
            prompt, question_id, system=system, generation=generation
        )


class TestCIRunnerMinMca:
    """The pass target is min_mca, separate from the consistency level c."""

    @pytest.mark.asyncio
    async def test_partial_mca_fails_by_default_and_passes_with_min_mca(
        self,
    ) -> None:
        mod = _get_ci()
        # q0..q2 answer B correctly; q3 has correct label C, so it fails.
        questions = [_make_question(f"q{i}") for i in range(3)]
        questions.append(_make_question("q3", correct_label="C"))
        dataset = CustomDataset(questions)
        provider = MockLLMProvider(model="mock", default_response="B")
        scorer = ExactMatchScorer()

        strict = mod.CIRunner()  # type: ignore[attr-defined]
        assert await strict.run(dataset, _make_config(), provider, scorer) == 1
        assert any("expected >= 1.000" in f for f in strict.failures)

        lenient = mod.CIRunner()  # type: ignore[attr-defined]
        config = _make_config(min_mca=0.75)
        assert await lenient.run(dataset, config, provider, scorer) == 0
        assert lenient.failures == ()

    @pytest.mark.asyncio
    async def test_report_and_metadata_kept_for_export(self) -> None:
        mod = _get_ci()
        dataset = CustomDataset([_make_question("q1")])
        provider = MockLLMProvider(model="mock", default_response="X")

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        exit_code = await runner.run(
            dataset, _make_config(), provider, ExactMatchScorer(), seed=7
        )

        assert exit_code == 1
        assert runner.last_report is not None
        assert runner.last_report.total_questions == 1
        assert runner.last_metadata is not None
        assert runner.last_metadata.perturbation_seed == 7


class TestCIRunnerErrorVariants:
    """Variants that failed with a provider error fail the gate."""

    @pytest.mark.asyncio
    async def test_errored_variants_fail_even_when_thresholds_pass(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mod = _get_ci()
        questions = [_make_question(f"q{i}") for i in range(3)]
        dataset = CustomDataset(questions)
        provider = _FailOnStemProvider(
            "Question q0?", model="mock", default_response="B"
        )
        # Thresholds that pass on their own: every question counts at c=0.
        config = _make_config(mca_threshold=0.0)

        runner = mod.CIRunner()  # type: ignore[attr-defined]
        with caplog.at_level("WARNING"):
            exit_code = await runner.run(dataset, config, provider, ExactMatchScorer())

        assert exit_code == 1
        assert runner.failures == (
            "Error check failed: 3 of 9 variants failed with provider errors",
        )
        assert "3 of 9 variants failed with provider errors" in caplog.text

    def test_count_error_variants_uses_error_prefix(self) -> None:
        mod = _get_ci()
        qcr = QuestionConsistencyResult(
            question_id="q1",
            rc_correct=0.0,
            rc_agree=0.5,
            total_variants=3,
            correct_count=0,
            answer_distribution={"": 2, "A": 1},
            scored_responses=tuple(
                ScoredResponse(
                    question_id="q1",
                    is_correct=False,
                    score=0.0,
                    scoring_method=method,
                )
                for method in ("error:TimeoutError", "error:", "exact_match")
            ),
        )
        assert mod.count_error_variants([qcr]) == 2  # type: ignore[attr-defined]
