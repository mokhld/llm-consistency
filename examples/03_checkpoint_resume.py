"""Crash mid-run, resume from JSONL checkpoint.

BatchRunner persists each QuestionConsistencyResult to a JSONL file as
soon as it's computed. If the run dies (network failure, OOM, host
reboot, ctrl-c), restart with the same checkpoint path and the
previously-completed questions are skipped — the provider is not
re-queried for them.

This example:

1. Runs the first pass with a *flaky* provider that raises a
   ``BaseException`` after the first two questions complete, simulating
   a hard crash. (Regular ``Exception`` subclasses are captured
   per-variant by the runner and the batch keeps going — we need
   something that bypasses ``except Exception`` to actually stop it.)
2. Re-runs against the same checkpoint file with a non-flaky provider.
3. Demonstrates that the second run starts from question 3, not from
   scratch.

Run from the repo root::

    uv run python examples/03_checkpoint_resume.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _helpers import BaseIdMockProvider

from llm_consistency import (
    BatchRunner,
    EvaluationConfig,
    ExactMatchScorer,
    LLMResponse,
    MCDataset,
    PerturbationType,
)
from llm_consistency.runners._checkpoint import read_checkpoint


class SimulatedCrash(BaseException):
    """A BaseException so the runner's `except Exception` lets us through."""


class FlakyMockProvider(BaseIdMockProvider):
    """Mock provider that hard-crashes after N successful questions."""

    def __init__(self, *, fail_after: int, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._fail_after = fail_after
        self._questions_seen: set[str] = set()

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
    ) -> LLMResponse:
        base_qid = question_id.split("_v")[0]
        self._questions_seen.add(base_qid)
        if len(self._questions_seen) > self._fail_after:
            # BaseException bypasses the runner's `except Exception` so
            # the batch genuinely halts mid-flight, like a SIGKILL.
            msg = f"simulated crash on {base_qid}"
            raise SimulatedCrash(msg)
        return await super().query(prompt, question_id, system=system)


async def run_pass(
    label: str,
    provider: BaseIdMockProvider,
    dataset: MCDataset,
    config: EvaluationConfig,
    checkpoint_path: Path,
) -> None:
    print(f"\n--- {label} ---")
    runner = BatchRunner()
    try:
        report = await runner.run(
            dataset,
            config,
            provider,
            ExactMatchScorer(),
            seed=42,
            checkpoint_path=checkpoint_path,
        )
        print(f"Completed {report.total_questions} question(s).")
        print(f"  mean RC_correct: {report.mean_rc_correct:.3f}")
        print(f"  mean RC_agree:   {report.mean_rc_agree:.3f}")
    except SimulatedCrash as exc:
        print(f"Run was interrupted: {exc}")

    _, completed = read_checkpoint(checkpoint_path, config=config, seed=42)
    print(f"Checkpoint now holds {len(completed)} completed question(s):")
    for qcr in completed:
        print(f"  - {qcr.question_id} (rc_correct={qcr.rc_correct:.2f})")


async def main() -> None:
    dataset = MCDataset.load(Path(__file__).parent / "datasets" / "sample.jsonl")
    config = EvaluationConfig(
        model="mock-model",
        provider="mock",
        scorer="exact_match",
        perturbation_types=(PerturbationType.OPTION_REORDER,),
        num_variants=2,
        concurrency=2,
    )
    responses = {"q1": "B", "q2": "C", "q3": "B", "q4": "C", "q5": "C"}

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "resume_demo.jsonl"

        flaky = FlakyMockProvider(
            model="mock-model",
            responses=responses,
            fail_after=2,
        )
        await run_pass(
            "Pass 1 (flaky provider — crash after 2 questions)",
            flaky,
            dataset,
            config,
            ckpt,
        )

        stable = BaseIdMockProvider(model="mock-model", responses=responses)
        await run_pass(
            "Pass 2 (stable provider — resumes from checkpoint)",
            stable,
            dataset,
            config,
            ckpt,
        )


if __name__ == "__main__":
    asyncio.run(main())
