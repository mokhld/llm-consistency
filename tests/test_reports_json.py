"""Tests for JSON report export (llm_consistency.reports._json_export)."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

from llm_consistency.runners._metadata import RunMetadata
from llm_consistency.types import (
    EvaluationConfig,
    EvaluationReport,
    PerturbationType,
    QuestionConsistencyResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_report() -> EvaluationReport:
    """Build a minimal EvaluationReport for testing."""
    qcr1 = QuestionConsistencyResult(
        question_id="q1",
        rc_correct=1.0,
        rc_agree=1.0,
        total_variants=3,
        correct_count=3,
        answer_distribution={"B": 3},
        scored_responses=(),
    )
    config = EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(PerturbationType.OPTION_REORDER,),
        scorer="exact_match",
    )
    return EvaluationReport(
        config=config,
        results=(qcr1,),
        total_questions=1,
        total_variants=3,
        mean_rc_correct=1.0,
        mean_rc_agree=1.0,
    )


def _make_metadata() -> RunMetadata:
    """Build a minimal RunMetadata for testing."""
    return RunMetadata(
        package_version="0.1.0",
        python_version="3.12.0",
        timestamp="2026-01-01T00:00:00+00:00",
        config_snapshot={"model": "mock"},
        perturbation_seed=42,
        model="mock",
        provider="mock",
    )


def _get_export_json():
    """Import export_json via importlib to avoid PLC0415."""
    mod = importlib.import_module("llm_consistency.reports")
    return mod.export_json


def test_export_json_writes_valid_json_file(tmp_path: Path) -> None:
    """export_json() writes a file containing valid JSON."""
    export_json = _get_export_json()

    report = _make_report()
    out = tmp_path / "report.json"
    export_json(report, out)

    raw = out.read_text()
    data = json.loads(raw)  # Must not raise
    assert isinstance(data, dict)


def test_export_json_contains_config_key(tmp_path: Path) -> None:
    """JSON file contains a 'config' key with serialized EvaluationConfig."""
    export_json = _get_export_json()

    report = _make_report()
    out = tmp_path / "report.json"
    export_json(report, out)

    data = json.loads(out.read_text())
    assert "config" in data
    assert data["config"]["model"] == "mock"
    assert data["config"]["provider"] == "mock"


def test_export_json_contains_results_key(tmp_path: Path) -> None:
    """JSON file contains a 'results' key as a list of per-question QCR dicts."""
    export_json = _get_export_json()

    report = _make_report()
    out = tmp_path / "report.json"
    export_json(report, out)

    data = json.loads(out.read_text())
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 1
    assert data["results"][0]["question_id"] == "q1"


def test_export_json_contains_aggregate_key(tmp_path: Path) -> None:
    """JSON file contains 'aggregate' key with core_index, mca, car_curve."""
    export_json = _get_export_json()

    report = _make_report()
    out = tmp_path / "report.json"
    export_json(report, out)

    data = json.loads(out.read_text())
    assert "aggregate" in data
    agg = data["aggregate"]
    assert "core_index" in agg
    assert "mca_at_threshold" in agg
    assert "car_curve" in agg


def test_export_json_with_metadata(tmp_path: Path) -> None:
    """JSON file contains 'metadata' key when metadata is provided."""
    export_json = _get_export_json()

    report = _make_report()
    metadata = _make_metadata()
    out = tmp_path / "report.json"
    export_json(report, out, metadata=metadata)

    data = json.loads(out.read_text())
    assert "metadata" in data
    assert data["metadata"]["model"] == "mock"
    assert data["metadata"]["perturbation_seed"] == 42


def test_export_json_without_metadata_omits_key(tmp_path: Path) -> None:
    """export_json with metadata=None omits the metadata key or sets it null."""
    export_json = _get_export_json()

    report = _make_report()
    out = tmp_path / "report.json"
    export_json(report, out)

    data = json.loads(out.read_text())
    # Either key absent or value is null
    assert "metadata" not in data or data["metadata"] is None


def test_export_json_file_is_parseable(tmp_path: Path) -> None:
    """File content is valid JSON parseable by json.loads()."""
    export_json = _get_export_json()

    report = _make_report()
    out = tmp_path / "report.json"
    export_json(report, out)

    raw = out.read_text()
    # Should not raise json.JSONDecodeError
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    # Verify it has indentation (pretty-printed)
    assert "\n" in raw
