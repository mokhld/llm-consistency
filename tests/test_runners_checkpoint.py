"""Tests for the checkpoint/resume support module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from llm_consistency._exceptions import ValidationError
from llm_consistency.runners._checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointHeader,
    CheckpointWriter,
    compute_config_hash,
    read_checkpoint,
)
from llm_consistency.types import (
    EvaluationConfig,
    PerturbationType,
    QuestionConsistencyResult,
    ScoredResponse,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    perturbation: PerturbationType = PerturbationType.OPTION_REORDER,
    num_variants: int = 3,
) -> EvaluationConfig:
    return EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(perturbation,),
        scorer="exact_match",
        num_variants=num_variants,
        concurrency=2,
    )


def _make_qcr(qid: str, rc_correct: float = 1.0) -> QuestionConsistencyResult:
    sr = ScoredResponse(
        question_id=f"{qid}_v0",
        is_correct=rc_correct == 1.0,
        score=rc_correct,
        scoring_method="exact_match",
    )
    return QuestionConsistencyResult(
        question_id=qid,
        rc_correct=rc_correct,
        rc_agree=1.0,
        total_variants=1,
        correct_count=int(rc_correct),
        answer_distribution={"A": 1},
        scored_responses=(sr,),
    )


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_stable_for_same_inputs(self) -> None:
        config = _make_config()
        assert compute_config_hash(config, 42) == compute_config_hash(config, 42)

    def test_changes_with_seed(self) -> None:
        config = _make_config()
        assert compute_config_hash(config, 42) != compute_config_hash(config, 43)

    def test_changes_with_num_variants(self) -> None:
        a = _make_config(num_variants=3)
        b = _make_config(num_variants=5)
        assert compute_config_hash(a, 42) != compute_config_hash(b, 42)

    def test_changes_with_perturbation_type(self) -> None:
        a = _make_config(perturbation=PerturbationType.OPTION_REORDER)
        b = _make_config(perturbation=PerturbationType.FORMAT_CHANGE)
        assert compute_config_hash(a, 42) != compute_config_hash(b, 42)


# ---------------------------------------------------------------------------
# CheckpointWriter
# ---------------------------------------------------------------------------


class TestCheckpointWriter:
    def test_writes_header_on_new_file(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"

        with CheckpointWriter(path, config=config, seed=42):
            pass

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert header["version"] == CHECKPOINT_VERSION
        assert header["config_hash"] == compute_config_hash(config, 42)
        assert header["seed"] == 42

    def test_appends_qcrs(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        qcr1 = _make_qcr("q1")
        qcr2 = _make_qcr("q2", rc_correct=0.5)

        with CheckpointWriter(path, config=config, seed=42) as writer:
            writer.append(qcr1)
            writer.append(qcr2)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        record1 = json.loads(lines[1])
        record2 = json.loads(lines[2])
        assert record1["type"] == "qcr"
        assert record1["qcr"]["question_id"] == "q1"
        assert record2["qcr"]["question_id"] == "q2"
        assert record2["qcr"]["rc_correct"] == 0.5

    def test_reopens_existing_file_for_append(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"

        with CheckpointWriter(path, config=config, seed=42) as w1:
            w1.append(_make_qcr("q1"))
        with CheckpointWriter(path, config=config, seed=42) as w2:
            w2.append(_make_qcr("q2"))

        lines = path.read_text(encoding="utf-8").splitlines()
        # 1 header + 2 qcrs
        assert len(lines) == 3
        assert json.loads(lines[1])["qcr"]["question_id"] == "q1"
        assert json.loads(lines[2])["qcr"]["question_id"] == "q2"

    def test_rejects_mismatched_config_hash_on_reopen(self, tmp_path: Path) -> None:
        config_a = _make_config(num_variants=3)
        config_b = _make_config(num_variants=5)
        path = tmp_path / "ckpt.jsonl"

        with CheckpointWriter(path, config=config_a, seed=42) as writer:
            writer.append(_make_qcr("q1"))

        with pytest.raises(ValidationError, match="different config"):  # noqa: SIM117
            with CheckpointWriter(path, config=config_b, seed=42):
                pass

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "nested" / "deeper" / "ckpt.jsonl"

        with CheckpointWriter(path, config=config, seed=42):
            pass

        assert path.exists()

    def test_append_outside_context_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        writer = CheckpointWriter(tmp_path / "ckpt.jsonl", config=config, seed=42)
        with pytest.raises(RuntimeError, match="outside of its context manager"):
            writer.append(_make_qcr("q1"))


# ---------------------------------------------------------------------------
# read_checkpoint
# ---------------------------------------------------------------------------


class TestReadCheckpoint:
    def test_round_trip_writer_to_reader(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        original = [_make_qcr("q1"), _make_qcr("q2", rc_correct=0.0)]

        with CheckpointWriter(path, config=config, seed=42) as writer:
            for qcr in original:
                writer.append(qcr)

        header, replayed = read_checkpoint(path, config=config, seed=42)
        assert isinstance(header, CheckpointHeader)
        assert header.config_hash == compute_config_hash(config, 42)
        assert len(replayed) == 2
        assert replayed[0].question_id == "q1"
        assert replayed[1].question_id == "q2"
        assert replayed[1].rc_correct == 0.0

    def test_header_only_returns_empty_results(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        with CheckpointWriter(path, config=config, seed=42):
            pass

        _, results = read_checkpoint(path, config=config, seed=42)
        assert results == ()

    def test_mismatched_hash_raises(self, tmp_path: Path) -> None:
        config_a = _make_config()
        config_b = _make_config(num_variants=7)
        path = tmp_path / "ckpt.jsonl"
        with CheckpointWriter(path, config=config_a, seed=42):
            pass

        with pytest.raises(ValidationError, match="different config"):
            read_checkpoint(path, config=config_b, seed=42)

    def test_mismatched_seed_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        with CheckpointWriter(path, config=config, seed=42):
            pass

        with pytest.raises(ValidationError, match="different config"):
            read_checkpoint(path, config=config, seed=43)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ValidationError, match="empty"):
            read_checkpoint(path, config=config, seed=42)

    def test_garbage_header_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        path.write_text("not json at all\n", encoding="utf-8")

        with pytest.raises(ValidationError, match="not valid JSON"):
            read_checkpoint(path, config=config, seed=42)

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        # Build a header with an unsupported future version.
        bogus = {
            "type": "header",
            "version": 999,
            "config_hash": compute_config_hash(config, 42),
            "created_at": "2026-01-01T00:00:00+00:00",
            "package_version": "0.0.0",
            "python_version": "3.12.0",
            "seed": 42,
            "config_snapshot": config.to_dict(),
        }
        path.write_text(json.dumps(bogus) + "\n", encoding="utf-8")

        with pytest.raises(ValidationError, match="version 999"):
            read_checkpoint(path, config=config, seed=42)

    def test_truncated_last_line_is_skipped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"

        with CheckpointWriter(path, config=config, seed=42) as writer:
            writer.append(_make_qcr("q1"))
            writer.append(_make_qcr("q2"))

        # Simulate a partial write: corrupt the last line.
        text = path.read_text(encoding="utf-8")
        truncated = text + '{"type": "qcr", "qcr": {"question'
        path.write_text(truncated, encoding="utf-8")

        with caplog.at_level("WARNING", logger="llm_consistency.runners._checkpoint"):
            _, results = read_checkpoint(path, config=config, seed=42)

        assert [r.question_id for r in results] == ["q1", "q2"]
        assert any("truncated" in rec.message for rec in caplog.records)

    def test_malformed_intermediate_line_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        with CheckpointWriter(path, config=config, seed=42) as writer:
            writer.append(_make_qcr("q1"))
            writer.append(_make_qcr("q2"))
        text = path.read_text(encoding="utf-8")
        # Corrupt the middle qcr line.
        lines = text.splitlines()
        lines[1] = "{garbage"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ValidationError, match="malformed JSON"):
            read_checkpoint(path, config=config, seed=42)

    def test_qcr_line_with_wrong_type_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        with CheckpointWriter(path, config=config, seed=42):
            pass
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "weird", "qcr": {}}) + "\n")

        with pytest.raises(ValidationError, match="type='qcr'"):
            read_checkpoint(path, config=config, seed=42)

    def test_qcr_line_missing_qcr_field_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "ckpt.jsonl"
        with CheckpointWriter(path, config=config, seed=42):
            pass
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "qcr"}) + "\n")

        with pytest.raises(ValidationError, match="'qcr' field missing"):
            read_checkpoint(path, config=config, seed=42)


# ---------------------------------------------------------------------------
# CheckpointHeader.from_dict edge cases
# ---------------------------------------------------------------------------


class TestCheckpointHeader:
    def test_from_dict_rejects_wrong_type(self) -> None:
        with pytest.raises(ValidationError, match="type='header'"):
            CheckpointHeader.from_dict({"type": "qcr"})

    def test_from_dict_rejects_missing_field(self) -> None:
        with pytest.raises(ValidationError, match="missing required field"):
            CheckpointHeader.from_dict({"type": "header"})

    def test_to_dict_round_trip(self) -> None:
        config = _make_config()
        original = CheckpointHeader(
            version=CHECKPOINT_VERSION,
            config_hash=compute_config_hash(config, 42),
            created_at="2026-05-21T00:00:00+00:00",
            package_version="1.2.3",
            python_version="3.12.0",
            seed=42,
            config_snapshot=config.to_dict(),
        )
        restored = CheckpointHeader.from_dict(original.to_dict())
        assert restored == original
