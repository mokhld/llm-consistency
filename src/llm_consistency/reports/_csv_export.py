"""CSV report export for evaluation results.

One row per :class:`QuestionConsistencyResult` with the headline
per-question metrics. For aggregate metrics (CORE, MCA at threshold,
CAR curve, bootstrap CIs), use :func:`export_json` — CSV is a flat
table format and not the right shape for nested objects.

The write is atomic in the same way as :func:`export_json`: a
temporary file in the same directory is written and renamed into
place, so an interrupted export leaves either the previous file or
nothing, never a half-written one.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_consistency.types import EvaluationReport


_CSV_FIELDS = (
    "question_id",
    "rc_correct",
    "rc_agree",
    "total_variants",
    "correct_count",
    "answer_distribution",
)


def export_csv(report: EvaluationReport, path: Path) -> None:
    """Export a report's per-question results as CSV (UTF-8, atomic).

    Columns: ``question_id``, ``rc_correct``, ``rc_agree``,
    ``total_variants``, ``correct_count``, ``answer_distribution``
    (the last is rendered as a ``"label=count; label=count"`` string
    so the file stays a flat table that opens cleanly in spreadsheets).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_FIELDS)
    for qcr in report.results:
        dist = "; ".join(
            f"{label}={count}"
            for label, count in sorted(qcr.answer_distribution.items())
        )
        writer.writerow(
            [
                qcr.question_id,
                f"{qcr.rc_correct:.6f}",
                f"{qcr.rc_agree:.6f}",
                qcr.total_variants,
                qcr.correct_count,
                dist,
            ]
        )

    payload = buf.getvalue()

    parent = path.parent if str(path.parent) else None
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=parent,
    )
    tmp_path = Path(tmp_name)
    try:
        umask = os.umask(0)
        os.umask(umask)
        with contextlib.suppress(OSError):
            tmp_path.chmod(0o666 & ~umask)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(payload)
        tmp_path.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
