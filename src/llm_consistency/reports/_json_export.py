"""JSON report export for evaluation results.

Serializes an ``EvaluationReport`` to a JSON file with per-question
results, aggregate metrics (CORE, MCA, CAR curve), and optional
run metadata.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llm_consistency.metrics import car_curve, core_index, mca

if TYPE_CHECKING:
    from pathlib import Path

    from llm_consistency.runners._metadata import RunMetadata
    from llm_consistency.types import EvaluationReport


def export_json(
    report: EvaluationReport,
    path: Path,
    *,
    metadata: RunMetadata | None = None,
) -> None:
    """Export an evaluation report to a JSON file.

    Serializes the report via ``report.to_dict()``, computes aggregate
    metrics (CORE index, MCA at threshold, CAR curve), and writes the
    result as pretty-printed JSON.

    Args:
        report: The evaluation report to export.
        path: Output file path.
        metadata: Optional run metadata to include.
    """
    data: dict[str, Any] = report.to_dict()

    # Compute aggregate metrics
    core_val = core_index(report.results)
    mca_val = mca(report.results, report.config.mca_threshold)
    curve = car_curve(report.results)

    data["aggregate"] = {
        "core_index": core_val,
        "mca_at_threshold": mca_val,
        "car_curve": [[c, m] for c, m in curve],
    }

    if metadata is not None:
        data["metadata"] = metadata.to_dict()

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
