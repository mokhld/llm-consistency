"""Pipeline semantics shared by BatchRunner, StreamingRunner and CIRunner.

Covers per-variant scoring (review item A1), the prompt contract and
decoding settings (B1), provider error handling (A2, A22) and
question-level concurrency (A9).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from typing import TYPE_CHECKING

import pytest

from llm_consistency.datasets import CustomDataset
from llm_consistency.providers import BudgetExceededError
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.runners import (
    DEFAULT_PROMPT_TEMPLATE,
    BatchRunner,
    CIRunner,
    StreamingRunner,
)
from llm_consistency.runners._pipeline import (
    describe_error,
    presented_options,
    query_kwargs,
)
from llm_consistency.scoring import ExactMatchScorer
from llm_consistency.types import (
    EvaluationConfig,
    GenerationParams,
    LLMResponse,
    MCOption,
    MCQuestion,
    OpenEndedQuestion,
    PerturbationType,
    PerturbedVariant,
    QuestionConsistencyResult,
)
from prompt_aware_providers import (
    OracleProvider,
    PositionBiasedProvider,
    ProseOracleProvider,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _question(qid: str, stem: str, texts: str, correct: str) -> MCQuestion:
    labels = "ABCDE"
    return MCQuestion(
        id=qid,
        stem=stem,
        options=tuple(
            MCOption(label=labels[i], text=text, is_correct=labels[i] == correct)
            for i, text in enumerate(texts.split(","))
        ),
    )


# Correct answers sit at different positions, and one question has 5 options.
_QUESTIONS = (
    _question(
        "q1", "What is the capital of France?", "London,Paris,Berlin,Madrid", "B"
    ),
    _question(
        "q2", "Which planet is closest to the Sun?", "Venus,Mars,Mercury,Earth", "C"
    ),
    _question(
        "q3", "What is the largest ocean?", "Pacific,Atlantic,Indian,Arctic", "A"
    ),
    _question(
        "q4",
        "What is two plus two?",
        "three,five,six,four,twenty-two",
        "D",
    ),
)

_BUILTIN_TYPES = (
    PerturbationType.OPTION_REORDER,
    PerturbationType.FORMAT_CHANGE,
    PerturbationType.SEPARATOR_CHANGE,
)


def _config(
    *types: PerturbationType,
    num_variants: int = 30,
    concurrency: int = 5,
) -> EvaluationConfig:
    return EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=types or (PerturbationType.OPTION_REORDER,),
        scorer="exact_match",
        num_variants=num_variants,
        concurrency=concurrency,
    )


def _simple_question(qid: str) -> MCQuestion:
    return _question(qid, f"Question {qid}?", "one,two,three,four", "A")


async def _collect(stream: object) -> list[QuestionConsistencyResult]:
    return [qcr async for qcr in stream]  # type: ignore[attr-defined]


def _correct_label(question: MCQuestion) -> str:
    return next(o.label for o in question.options if o.is_correct)


# ---------------------------------------------------------------------------
# Per-variant scoring (A1)
# ---------------------------------------------------------------------------

_TYPE_SETS = [(pt,) for pt in _BUILTIN_TYPES] + [_BUILTIN_TYPES]
_TYPE_IDS = [pt.value for pt in _BUILTIN_TYPES] + ["all"]


class TestOracleScoresPerfectly:
    """A model that always picks the correct option text scores 1.0 everywhere."""

    @pytest.mark.parametrize("types", _TYPE_SETS, ids=_TYPE_IDS)
    async def test_batch_runner(self, types: tuple[PerturbationType, ...]) -> None:
        report = await BatchRunner().run(
            CustomDataset(list(_QUESTIONS)),
            _config(*types),
            OracleProvider(_QUESTIONS),
            ExactMatchScorer(),
        )
        for question, qcr in zip(_QUESTIONS, report.results, strict=True):
            assert qcr.rc_correct == 1.0, qcr
            assert qcr.rc_agree == 1.0, qcr
            # Agreement is keyed by the option's original label.
            assert qcr.answer_distribution == {
                _correct_label(question): qcr.total_variants
            }

    @pytest.mark.parametrize("types", _TYPE_SETS, ids=_TYPE_IDS)
    async def test_streaming_runner(self, types: tuple[PerturbationType, ...]) -> None:
        results = await _collect(
            StreamingRunner().run_stream(
                CustomDataset(list(_QUESTIONS)),
                _config(*types),
                OracleProvider(_QUESTIONS),
                ExactMatchScorer(),
            )
        )
        assert [r.question_id for r in results] == [q.id for q in _QUESTIONS]
        for qcr in results:
            assert qcr.rc_correct == 1.0, qcr
            assert qcr.rc_agree == 1.0, qcr

    async def test_numbered_template_is_scored(self) -> None:
        """The numbered layout's "1".."n" answers are read and scored."""
        report = await BatchRunner().run(
            CustomDataset(list(_QUESTIONS)),
            _config(PerturbationType.FORMAT_CHANGE),
            OracleProvider(_QUESTIONS),
            ExactMatchScorer(),
        )
        for qcr in report.results:
            assert qcr.total_variants == 7
            assert all(sr.is_correct for sr in qcr.scored_responses)

    @pytest.mark.parametrize("types", _TYPE_SETS, ids=_TYPE_IDS)
    async def test_answer_line_after_reasoning(
        self, types: tuple[PerturbationType, ...]
    ) -> None:
        """ "Answer: X" after prose that names a wrong option first."""
        report = await BatchRunner().run(
            CustomDataset(list(_QUESTIONS)),
            _config(*types),
            ProseOracleProvider(_QUESTIONS),
            ExactMatchScorer(),
        )
        for qcr in report.results:
            assert qcr.rc_correct == 1.0, qcr
            assert qcr.rc_agree == 1.0, qcr


class TestPositionBiasedAgreement:
    """Always answering the first listed option is not consistent content."""

    async def test_option_reorder_breaks_agreement(self) -> None:
        report = await BatchRunner().run(
            CustomDataset(list(_QUESTIONS)),
            _config(PerturbationType.OPTION_REORDER),
            PositionBiasedProvider(_QUESTIONS),
            ExactMatchScorer(),
        )
        assert report.mean_rc_agree < 1.0
        for qcr in report.results:
            assert qcr.rc_agree < 1.0
            assert len(qcr.answer_distribution) > 1
        # Correct only when the correct option happens to come first.
        assert 0.0 < report.mean_rc_correct < 1.0

    @pytest.mark.parametrize(
        "pt",
        [PerturbationType.FORMAT_CHANGE, PerturbationType.SEPARATOR_CHANGE],
        ids=lambda pt: pt.value,
    )
    async def test_layout_changes_keep_agreement(self, pt: PerturbationType) -> None:
        """Without reordering, "A" and "1" both mean the first option."""
        report = await BatchRunner().run(
            CustomDataset(list(_QUESTIONS)),
            _config(pt),
            PositionBiasedProvider(_QUESTIONS),
            ExactMatchScorer(),
        )
        for question, qcr in zip(_QUESTIONS, report.results, strict=True):
            assert qcr.rc_agree == 1.0
            assert set(qcr.answer_distribution) == {"A"}
            expected = 1.0 if _correct_label(question) == "A" else 0.0
            assert qcr.rc_correct == expected


class TestPresentedOptionsFallback:
    """Variants from perturbations that do not declare presented_options."""

    def test_uses_variant_options(self) -> None:
        question = _simple_question("q1")
        shown = tuple(reversed(question.options))
        variant = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.OPTION_REORDER,
            seed=0,
            variant_index=0,
            stem=question.stem,
            options=shown,
        )
        with pytest.warns(UserWarning, match="presented_options"):
            presented = presented_options(variant, question)
        assert [o.text for o in presented] == [o.text for o in shown]
        assert [o.original_label for o in presented] == [o.label for o in shown]

    def test_uses_question_options_when_rendered_into_stem(self) -> None:
        question = _simple_question("q1")
        variant = PerturbedVariant(
            original_question_id="q1",
            perturbation_type=PerturbationType.FORMAT_CHANGE,
            seed=0,
            variant_index=0,
            stem="custom rendering",
        )
        with pytest.warns(UserWarning, match="presented_options"):
            presented = presented_options(variant, question)
        assert [o.label for o in presented] == ["A", "B", "C", "D"]
        assert [o.original_label for o in presented] == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Prompt contract and decoding settings (B1)
# ---------------------------------------------------------------------------


class _RecordingProvider(MockLLMProvider):
    """Answers "A" and records each prompt with the keyword arguments sent."""

    def __init__(self) -> None:
        super().__init__(model="mock")
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def query(  # type: ignore[override]
        self, prompt: str, question_id: str, **kwargs: object
    ) -> LLMResponse:
        self.calls.append((question_id, prompt, kwargs))
        return await super().query(prompt, question_id)


def _instruction(labels: str) -> str:
    """The default template's text after ``{question}``, for *labels*."""
    return DEFAULT_PROMPT_TEMPLATE.split("{question}", 1)[1].format(labels=labels)


async def _run_recorded(
    config: EvaluationConfig, runner_kind: str = "batch"
) -> _RecordingProvider:
    provider = _RecordingProvider()
    dataset = CustomDataset(list(_QUESTIONS))
    if runner_kind == "batch":
        await BatchRunner().run(dataset, config, provider, ExactMatchScorer())
    else:
        await _collect(
            StreamingRunner().run_stream(dataset, config, provider, ExactMatchScorer())
        )
    return provider


class TestPromptContract:
    @pytest.mark.parametrize("pt", _BUILTIN_TYPES, ids=lambda pt: pt.value)
    async def test_default_prompt_asks_for_an_answer_line(
        self, pt: PerturbationType
    ) -> None:
        """Every prompt ends with the instruction and the labels it shows."""
        provider = await _run_recorded(_config(pt))
        by_id = {q.id: q for q in _QUESTIONS}
        label_lists: set[str] = set()
        for qid, prompt, _ in provider.calls:
            question = by_id[qid.rsplit("_v", 1)[0]]
            assert prompt.startswith(question.stem)
            n = len(question.options)
            numbered = re.search(r"^1\. ", prompt, re.MULTILINE) is not None
            labels = ", ".join(
                [str(i + 1) for i in range(n)] if numbered else "ABCDE"[:n]
            )
            assert prompt.endswith(_instruction(labels)), prompt
            label_lists.add(labels)
        if pt is PerturbationType.FORMAT_CHANGE:
            assert {"1, 2, 3, 4", "1, 2, 3, 4, 5"} <= label_lists
        assert {"A, B, C, D", "A, B, C, D, E"} <= label_lists

    async def test_custom_template(self) -> None:
        config = dataclasses.replace(
            _config(num_variants=1),
            prompt_template="Q: {question}\nReply with one of {labels} only.",
        )
        provider = await _run_recorded(config)
        _, prompt, _ = provider.calls[0]
        assert prompt.startswith(f"Q: {_QUESTIONS[0].stem}\nA. ")
        assert prompt.endswith("\nReply with one of A, B, C, D only.")

    async def test_bare_template_sends_the_question_alone(self) -> None:
        config = dataclasses.replace(
            _config(PerturbationType.SEPARATOR_CHANGE, num_variants=1),
            prompt_template="{question}",
        )
        provider = await _run_recorded(config)
        _, prompt, _ = provider.calls[0]
        assert prompt.startswith(_QUESTIONS[0].stem)
        assert "Answer" not in prompt


class TestQueryArguments:
    """System prompt and generation settings reach provider.query."""

    @pytest.mark.parametrize("runner_kind", ["batch", "stream"])
    async def test_sent_when_set(self, runner_kind: str) -> None:
        config = dataclasses.replace(
            _config(num_variants=2),
            system_prompt="Be careful.",
            temperature=0.0,
            max_tokens=64,
            generation_seed=7,
        )
        provider = await _run_recorded(config, runner_kind)
        expected = {
            "system": "Be careful.",
            "generation": GenerationParams(temperature=0.0, max_tokens=64, seed=7),
        }
        assert provider.calls
        assert all(kwargs == expected for _, _, kwargs in provider.calls)

    async def test_nothing_extra_sent_by_default(self) -> None:
        provider = await _run_recorded(_config(num_variants=2))
        assert all(kwargs == {} for _, _, kwargs in provider.calls)

    def test_zero_temperature_counts_as_set(self) -> None:
        config = dataclasses.replace(_config(), temperature=0.0)
        assert query_kwargs(config) == {"generation": GenerationParams(temperature=0.0)}

    async def test_query_override_without_the_keywords_still_works(self) -> None:
        """A provider written before B1 runs unchanged when nothing is set."""

        class _OldProvider(MockLLMProvider):
            async def query(  # type: ignore[override]
                self, prompt: str, question_id: str
            ) -> LLMResponse:
                return await super().query(prompt, question_id)

        report = await BatchRunner().run(
            CustomDataset([_simple_question("q0")]),
            _config(num_variants=2),
            _OldProvider(model="mock"),
            ExactMatchScorer(),
        )
        (qcr,) = report.results
        assert qcr.total_variants == 2
        assert not any(
            sr.scoring_method.startswith("error:") for sr in qcr.scored_responses
        )


# ---------------------------------------------------------------------------
# Provider errors (A2, A22)
# ---------------------------------------------------------------------------


class _ScriptedProvider(MockLLMProvider):
    """Answers "A", or raises the exception mapped to the question id."""

    def __init__(self, errors: dict[str, BaseException] | None = None) -> None:
        super().__init__(model="mock")
        self.errors = errors or {}
        self.queried_ids: list[str] = []

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> LLMResponse:
        self.queried_ids.append(question_id)
        error = self.errors.get(question_id.rsplit("_v", 1)[0])
        if error is not None:
            raise error
        return await super().query(
            prompt, question_id, system=system, generation=generation
        )


def _budget_error() -> BudgetExceededError:
    return BudgetExceededError(spent=1.0, estimated=0.1, limit=1.0)


class TestBudgetExceededPropagates:
    """BudgetExceededError aborts the run instead of becoming a wrong answer."""

    async def test_batch_runner(self) -> None:
        dataset = CustomDataset([_simple_question("q0"), _simple_question("q1")])
        provider = _ScriptedProvider({"q1": _budget_error()})
        with pytest.raises(BudgetExceededError):
            await BatchRunner().run(
                dataset, _config(num_variants=2), provider, ExactMatchScorer()
            )

    async def test_streaming_runner(self) -> None:
        dataset = CustomDataset([_simple_question("q0"), _simple_question("q1")])
        provider = _ScriptedProvider({"q1": _budget_error()})
        with pytest.raises(BudgetExceededError):
            await _collect(
                StreamingRunner().run_stream(
                    dataset, _config(num_variants=2), provider, ExactMatchScorer()
                )
            )

    async def test_ci_runner(self) -> None:
        dataset = CustomDataset([_simple_question("q0")])
        provider = _ScriptedProvider({"q0": _budget_error()})
        with pytest.raises(BudgetExceededError):
            await CIRunner().run(
                dataset, _config(num_variants=2), provider, ExactMatchScorer()
            )

    @pytest.mark.parametrize("runner_kind", ["batch", "stream"])
    async def test_outstanding_queries_are_cancelled(self, runner_kind: str) -> None:
        """Queries still waiting when the budget trips do not keep running."""
        started: list[str] = []
        cancelled: list[str] = []

        class _Provider(MockLLMProvider):
            async def query(
                self,
                prompt: str,
                question_id: str,
                *,
                system: str | None = None,
                generation: GenerationParams | None = None,
            ) -> LLMResponse:
                started.append(question_id)
                if question_id == "q0_v0":
                    await asyncio.sleep(0.01)
                    raise _budget_error()
                try:
                    await asyncio.Event().wait()  # never answers
                except asyncio.CancelledError:
                    cancelled.append(question_id)
                    raise
                raise AssertionError  # pragma: no cover

        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(4)])
        config = _config(num_variants=2, concurrency=4)
        provider = _Provider(model="mock")
        if runner_kind == "batch":
            run = BatchRunner().run(dataset, config, provider, ExactMatchScorer())
        else:
            run = _collect(
                StreamingRunner().run_stream(
                    dataset, config, provider, ExactMatchScorer()
                )
            )
        with pytest.raises(BudgetExceededError):
            await asyncio.wait_for(run, timeout=2.0)
        # Every query that started, apart from the one that raised, was
        # cancelled, and nothing is left running or waiting to start.
        assert sorted(cancelled) == sorted(set(started) - {"q0_v0"})
        assert len(started) < 8
        await asyncio.sleep(0)
        leftover = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert leftover == []


class TestErrorVariants:
    """Other provider errors are recorded per variant and summarised once."""

    async def test_batch_logs_one_warning_with_type_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(3)])
        provider = _ScriptedProvider(
            {"q1": RuntimeError("boom"), "q2": TimeoutError("slow")}
        )
        with caplog.at_level("WARNING", logger="llm_consistency.runners._batch"):
            report = await BatchRunner().run(
                dataset, _config(num_variants=2), provider, ExactMatchScorer()
            )

        warnings = [r for r in caplog.records if "failed with provider" in r.message]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "4 variant(s)" in message
        assert "RuntimeError: 2" in message
        assert "TimeoutError: 2" in message
        failed = [
            sr
            for qcr in report.results
            for sr in qcr.scored_responses
            if sr.scoring_method.startswith("error:")
        ]
        assert len(failed) == 4

    async def test_streaming_logs_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(3)])
        provider = _ScriptedProvider({"q0": ValueError("bad")})
        with caplog.at_level("WARNING", logger="llm_consistency.runners._streaming"):
            await _collect(
                StreamingRunner().run_stream(
                    dataset, _config(num_variants=2), provider, ExactMatchScorer()
                )
            )
        warnings = [r for r in caplog.records if "failed with provider" in r.message]
        assert len(warnings) == 1
        assert "ValueError: 2" in warnings[0].getMessage()

    async def test_streaming_counts_errors_behind_a_full_window(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(3)])
        provider = _ScriptedProvider({"q0": ValueError("bad"), "q2": KeyError("x")})
        with caplog.at_level("WARNING", logger="llm_consistency.runners._streaming"):
            await _collect(
                StreamingRunner().run_stream(
                    dataset,
                    _config(num_variants=2, concurrency=1),
                    provider,
                    ExactMatchScorer(),
                )
            )
        (warning,) = [r for r in caplog.records if "failed with provider" in r.message]
        assert "ValueError: 2" in warning.getMessage()
        assert "KeyError: 2" in warning.getMessage()

    async def test_no_warning_without_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question("q0")])
        with caplog.at_level("WARNING"):
            await BatchRunner().run(
                dataset,
                _config(num_variants=2),
                _ScriptedProvider(),
                ExactMatchScorer(),
            )
        assert not [r for r in caplog.records if "failed with provider" in r.message]

    async def test_error_text_is_redacted_and_truncated(self) -> None:
        secret = "sk-ant-api03-AbCdEf1234567890"
        provider = _ScriptedProvider(
            {"q0": RuntimeError(f"Invalid key {secret}. " + "x" * 500)}
        )
        report = await BatchRunner().run(
            CustomDataset([_simple_question("q0")]),
            _config(num_variants=2),
            provider,
            ExactMatchScorer(),
        )
        for sr in report.results[0].scored_responses:
            assert sr.scoring_method.startswith("error:RuntimeError: Invalid key ")
            assert "AbCdEf" not in sr.scoring_method
            assert "[REDACTED]" in sr.scoring_method
            assert len(sr.scoring_method) == len("error:") + 200
            assert sr.scoring_method.endswith("...")


class TestDescribeError:
    """describe_error redacts key shapes and caps the length."""

    @pytest.mark.parametrize(
        "text",
        [
            "Incorrect API key provided: sk-proj-abc123***********wxyz.",
            "key sk-ant-api03-ZZZZZZZZZZ rejected",
            "header Authorization: Bearer eyJhbGciOi.payload.sig",
        ],
    )
    def test_redacts(self, text: str) -> None:
        described = describe_error(PermissionError(text))
        assert described.startswith("PermissionError: ")
        assert "[REDACTED]" in described
        for leaked in ("abc123", "wxyz", "ZZZZ", "eyJhbGciOi"):
            assert leaked not in described

    def test_leaves_ordinary_words(self) -> None:
        assert describe_error(RuntimeError("task-sk-1 failed")) == (
            "RuntimeError: task-sk-1 failed"
        )

    def test_short_text_unchanged(self) -> None:
        assert describe_error(TimeoutError("slow")) == "TimeoutError: slow"


class TestCheckpointSkipsErrors:
    """A question with a failed variant is not checkpointed, so resume retries it."""

    @staticmethod
    def _checkpoint_ids(path: Path) -> list[str]:
        return [
            obj["qcr"]["question_id"]
            for obj in map(json.loads, path.read_text().splitlines())
            if obj.get("type") == "qcr"
        ]

    async def test_resume_retries_failed_question(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        questions = [_simple_question(f"q{i}") for i in range(3)]
        config = _config(num_variants=2)
        ckpt = tmp_path / "run.jsonl"

        with caplog.at_level("WARNING", logger="llm_consistency.runners._batch"):
            first = await BatchRunner().run(
                CustomDataset(questions),
                config,
                _ScriptedProvider({"q1": ConnectionError("reset")}),
                ExactMatchScorer(),
                checkpoint_path=ckpt,
            )
        assert [r.question_id for r in first.results] == ["q0", "q1", "q2"]
        assert sorted(self._checkpoint_ids(ckpt)) == ["q0", "q2"]
        assert any("not checkpointed" in r.getMessage() for r in caplog.records)

        retry = _ScriptedProvider()
        second = await BatchRunner().run(
            CustomDataset(questions),
            config,
            retry,
            ExactMatchScorer(),
            checkpoint_path=ckpt,
        )
        assert {qid.rsplit("_v", 1)[0] for qid in retry.queried_ids} == {"q1"}
        assert [r.question_id for r in second.results] == ["q0", "q1", "q2"]
        assert not any(
            sr.scoring_method.startswith("error:")
            for sr in second.results[1].scored_responses
        )
        assert sorted(self._checkpoint_ids(ckpt)) == ["q0", "q1", "q2"]

    async def test_resume_drops_ids_not_in_dataset(self, tmp_path: Path) -> None:
        config = _config(num_variants=2)
        ckpt = tmp_path / "run.jsonl"
        await BatchRunner().run(
            CustomDataset([_simple_question("q0"), _simple_question("q1")]),
            config,
            _ScriptedProvider(),
            ExactMatchScorer(),
            checkpoint_path=ckpt,
        )

        provider = _ScriptedProvider()
        report = await BatchRunner().run(
            CustomDataset([_simple_question("q1"), _simple_question("q2")]),
            config,
            provider,
            ExactMatchScorer(),
            checkpoint_path=ckpt,
        )
        assert [r.question_id for r in report.results] == ["q1", "q2"]
        assert {qid.rsplit("_v", 1)[0] for qid in provider.queried_ids} == {"q2"}


# ---------------------------------------------------------------------------
# Question-level concurrency (A9)
# ---------------------------------------------------------------------------


class _TrackingProvider(MockLLMProvider):
    """Records how many queries are in flight; q0 answers slowest."""

    def __init__(self) -> None:
        super().__init__(model="mock")
        self.in_flight = 0
        self.peak = 0

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> LLMResponse:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.05 if question_id.startswith("q0_") else 0.005)
        finally:
            self.in_flight -= 1
        return await super().query(
            prompt, question_id, system=system, generation=generation
        )


class _FakeProgress:
    def __init__(self) -> None:
        self.total: int | None = None
        self.completed = 0

    def add_task(self, description: str, total: int) -> int:
        self.total = total
        return 1

    def advance(self, task_id: int, advance: int = 1) -> None:
        self.completed += advance


class TestQuestionConcurrency:
    async def test_batch_overlaps_questions_and_keeps_order(self) -> None:
        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(6)])
        provider = _TrackingProvider()
        report = await BatchRunner().run(
            dataset,
            _config(num_variants=2, concurrency=6),
            provider,
            ExactMatchScorer(),
        )
        # One question alone has only 2 variants in flight.
        assert 2 < provider.peak <= 6
        assert [r.question_id for r in report.results] == [f"q{i}" for i in range(6)]

    async def test_concurrency_bound_respected(self) -> None:
        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(6)])
        provider = _TrackingProvider()
        await BatchRunner().run(
            dataset,
            _config(num_variants=3, concurrency=4),
            provider,
            ExactMatchScorer(),
        )
        assert provider.peak == 4

    async def test_progress_advances_per_question(self) -> None:
        progress = _FakeProgress()
        await BatchRunner().run(
            CustomDataset([_simple_question(f"q{i}") for i in range(4)]),
            _config(num_variants=2),
            _ScriptedProvider(),
            ExactMatchScorer(),
            progress=progress,  # type: ignore[arg-type]
        )
        assert progress.total == 4
        assert progress.completed == 4

    async def test_progress_reaches_total(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "run.jsonl"
        config = _config(num_variants=2)
        await BatchRunner().run(
            CustomDataset([_simple_question("q0")]),
            config,
            _ScriptedProvider(),
            ExactMatchScorer(),
            checkpoint_path=ckpt,
        )
        oe = OpenEndedQuestion(id="o1", stem="Why?", reference_answers=("because",))
        dataset = CustomDataset([oe] + [_simple_question(f"q{i}") for i in range(3)])
        progress = _FakeProgress()
        await BatchRunner().run(
            dataset,
            config,
            _ScriptedProvider(),
            ExactMatchScorer(),
            progress=progress,  # type: ignore[arg-type]
            checkpoint_path=ckpt,
        )
        assert progress.total == 3
        assert progress.completed == 3

    async def test_streaming_overlaps_questions_and_keeps_order(self) -> None:
        dataset = CustomDataset([_simple_question(f"q{i}") for i in range(6)])
        provider = _TrackingProvider()
        results = await _collect(
            StreamingRunner().run_stream(
                dataset,
                _config(num_variants=2, concurrency=6),
                provider,
                ExactMatchScorer(),
            )
        )
        assert provider.peak > 2
        assert [r.question_id for r in results] == [f"q{i}" for i in range(6)]

    async def test_streaming_close_cancels_in_flight(self) -> None:
        """Closing the stream early stops the questions evaluated ahead."""
        in_flight: set[str] = set()

        class _Provider(MockLLMProvider):
            async def query(
                self,
                prompt: str,
                question_id: str,
                *,
                system: str | None = None,
                generation: GenerationParams | None = None,
            ) -> LLMResponse:
                if not question_id.startswith("q0_"):
                    in_flight.add(question_id)
                    try:
                        await asyncio.Event().wait()  # never answers
                    finally:
                        in_flight.discard(question_id)
                return await super().query(
                    prompt, question_id, system=system, generation=generation
                )

        stream = StreamingRunner().run_stream(
            CustomDataset([_simple_question(f"q{i}") for i in range(8)]),
            _config(num_variants=2, concurrency=4),
            _Provider(model="mock"),
            ExactMatchScorer(),
        )
        first = await anext(stream)
        assert first.question_id == "q0"
        assert in_flight  # later questions were started ahead of the consumer
        await stream.aclose()
        assert in_flight == set()


# ---------------------------------------------------------------------------
# Budget set on the config but not on the provider
# ---------------------------------------------------------------------------


class TestBudgetNotEnforcedWarning:
    """A budget on EvaluationConfig alone is not enforced, so the runners warn."""

    @staticmethod
    def _budget_config() -> EvaluationConfig:
        return EvaluationConfig(
            model="mock",
            provider="mock",
            perturbation_types=(PerturbationType.OPTION_REORDER,),
            scorer="exact_match",
            num_variants=2,
            max_budget_usd=1.0,
        )

    async def test_batch_warns_when_provider_has_no_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question("q1")])
        await BatchRunner().run(
            dataset, self._budget_config(), MockLLMProvider(), ExactMatchScorer()
        )
        assert "max_budget_usd=1.0 is not enforced" in caplog.text

    async def test_streaming_warns_when_provider_has_no_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question("q1")])
        await _collect(
            StreamingRunner().run_stream(
                dataset, self._budget_config(), MockLLMProvider(), ExactMatchScorer()
            )
        )
        assert "is not enforced" in caplog.text

    async def test_no_warning_when_provider_enforces_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dataset = CustomDataset([_simple_question("q1")])
        provider = MockLLMProvider(max_budget_usd=1.0)
        assert provider.max_budget_usd == 1.0
        await BatchRunner().run(
            dataset, self._budget_config(), provider, ExactMatchScorer()
        )
        assert "is not enforced" not in caplog.text
