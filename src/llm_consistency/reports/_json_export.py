"""JSON report export for evaluation results.

Serializes an ``EvaluationReport`` to a JSON file with per-question
results, aggregate metrics (CORE, MCA, CAR curve), and optional
run metadata.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_consistency.metrics import (
    car_curve,
    car_curve_with_ci,
    core_index,
    core_index_with_ci,
    mca,
    mca_with_ci,
)

if TYPE_CHECKING:
    from llm_consistency.runners._metadata import RunMetadata
    from llm_consistency.types import EvaluationReport


def export_json(
    report: EvaluationReport,
    path: Path,
    *,
    metadata: RunMetadata | None = None,
    include_ci: bool = True,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    bootstrap_seed: int | None = 42,
) -> None:
    """Export an evaluation report to a JSON file.

    Serializes the report via ``report.to_dict()``, computes aggregate
    metrics (CORE index, MCA at threshold, CAR curve), and writes the
    result as pretty-printed UTF-8 JSON.

    The write is atomic: data is first written to a temporary file in
    the same directory, then renamed into place. An interrupted run
    therefore leaves either the previous file or no file, never a
    partially-written one. Explicit ``utf-8`` encoding ensures the
    output is portable across platforms regardless of the system locale.

    When *include_ci* is true (the default), the aggregate block also
    carries BCa bootstrap confidence intervals for CORE, MCA at the
    configured threshold, and each point on the CAR curve. The
    bootstrap is reproducible via *bootstrap_seed*.

    Args:
        report: The evaluation report to export.
        path: Output file path.
        metadata: Optional run metadata to include.
        include_ci: If True, include BCa bootstrap CIs alongside each
            aggregate metric.
        n_bootstrap: Number of bootstrap resamples when *include_ci*.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        bootstrap_seed: Random seed for reproducible bootstrap CIs.
    """
    data: dict[str, Any] = report.to_dict()

    # Compute aggregate metrics
    core_val = core_index(report.results)
    mca_val = mca(report.results, report.config.mca_threshold)
    curve = car_curve(report.results)

    aggregate: dict[str, Any] = {
        "core_index": core_val,
        "mca_at_threshold": mca_val,
        "car_curve": [[c, m] for c, m in curve],
    }

    if include_ci and report.results:
        core_ci = core_index_with_ci(
            report.results,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=bootstrap_seed,
        )
        mca_ci = mca_with_ci(
            report.results,
            report.config.mca_threshold,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=bootstrap_seed,
        )
        curve_ci = car_curve_with_ci(
            report.results,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=bootstrap_seed,
        )
        aggregate["core_index_ci"] = core_ci.to_dict()
        aggregate["mca_at_threshold_ci"] = mca_ci.to_dict()
        aggregate["car_curve_ci"] = [[c, mr.to_dict()] for c, mr in curve_ci]

    data["aggregate"] = aggregate

    if metadata is not None:
        data["metadata"] = metadata.to_dict()

    payload = json.dumps(data, indent=2, ensure_ascii=False)

    parent = path.parent if str(path.parent) else None
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=parent,
    )
    tmp_path = Path(tmp_name)
    try:
        # mkstemp creates files with mode 0o600 by default. Match the user's
        # umask so the final file has conventional 0o644-ish permissions
        # rather than being owner-only.
        umask = os.umask(0)
        os.umask(umask)
        with contextlib.suppress(OSError):
            tmp_path.chmod(0o666 & ~umask)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        tmp_path.replace(path)
    except Exception:
        # Best-effort cleanup of the temp file if the replace failed.
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
