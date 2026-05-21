"""Tests for CIRunner with pass/fail exit codes based on metric thresholds."""

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
) -> EvaluationConfig:
    return EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(PerturbationType.OPTION_REORDER,),
        scorer="exact_match",
        num_variants=num_variants,
        concurrency=5,
        mca_threshold=mca_threshold,
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
