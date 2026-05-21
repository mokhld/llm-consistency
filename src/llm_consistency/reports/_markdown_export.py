"""Markdown report export for evaluation results.

Produces a self-contained human-readable summary: title, optional run
metadata block, aggregate metrics, and a per-question table. Atomic
UTF-8 write following the same pattern as :func:`export_json` and
:func:`export_csv`.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from llm_consistency.metrics import (
    agreement_gated_accuracy,
    car_curve,
    core_index,
    mca,
)

if TYPE_CHECKING:
    from llm_consistency.runners._metadata import RunMetadata
    from llm_consistency.types import EvaluationReport


def export_markdown(
    report: EvaluationReport,
    path: Path,
    *,
    metadata: RunMetadata | None = None,
    tau_agree: float = 0.8,
) -> None:
    """Write a Markdown summary of an evaluation report.

    Sections rendered (in order):

    1. ``# LLM Consistency Report`` title.
    2. ``## Run`` block — model, provider, scorer, perturbations,
       configured thresholds, and (if *metadata* is given) package and
       Python versions, timestamp, perturbation seed.
    3. ``## Aggregate metrics`` table — total questions, total
       variants, mean ``rc_correct``, mean ``rc_agree``, CORE index,
       MCA at the configured threshold, agreement-gated accuracy at
       *tau_agree*.
    4. ``## CAR curve`` table — five evenly-spaced threshold points.
    5. ``## Per-question results`` table — one row per question.
    """
    buf = io.StringIO()
    buf.write("# LLM Consistency Report\n\n")

    _write_run_block(buf, report, metadata)
    _write_aggregate_metrics(buf, report, tau_agree=tau_agree)
    _write_car_curve(buf, report)
    _write_per_question(buf, report)

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
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        tmp_path.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _write_run_block(
    buf: io.StringIO,
    report: EvaluationReport,
    metadata: RunMetadata | None,
) -> None:
    config = report.config
    buf.write("## Run\n\n")
    buf.write(f"- **Model:** `{config.model}`\n")
    buf.write(f"- **Provider:** `{config.provider}`\n")
    buf.write(f"- **Scorer:** `{config.scorer}`\n")
    pert_str = ", ".join(f"`{pt.value}`" for pt in config.perturbation_types)
    buf.write(f"- **Perturbations:** {pert_str}\n")
    buf.write(f"- **Variants per question:** {config.num_variants}\n")
    buf.write(f"- **MCA threshold:** {config.mca_threshold:.3f}\n")
    if config.core_threshold is not None:
        buf.write(f"- **CORE threshold:** {config.core_threshold:.3f}\n")
    if config.max_budget_usd is not None:
        buf.write(f"- **Max budget USD:** {config.max_budget_usd}\n")
    if metadata is not None:
        buf.write(f"- **Package version:** `{metadata.package_version}`\n")
        buf.write(f"- **Python version:** `{metadata.python_version}`\n")
        buf.write(f"- **Timestamp (UTC):** `{metadata.timestamp}`\n")
        buf.write(f"- **Perturbation seed:** `{metadata.perturbation_seed}`\n")
    buf.write("\n")


def _write_aggregate_metrics(
    buf: io.StringIO,
    report: EvaluationReport,
    *,
    tau_agree: float,
) -> None:
    config = report.config
    core_val = core_index(report.results)
    mca_val = mca(report.results, config.mca_threshold)
    aga_val = agreement_gated_accuracy(report.results, tau_agree)

    buf.write("## Aggregate metrics\n\n")
    buf.write("| Metric | Value |\n")
    buf.write("|---|---|\n")
    buf.write(f"| Total questions | {report.total_questions} |\n")
    buf.write(f"| Total variants | {report.total_variants} |\n")
    buf.write(f"| Mean rc_correct | {report.mean_rc_correct:.4f} |\n")
    buf.write(f"| Mean rc_agree | {report.mean_rc_agree:.4f} |\n")
    buf.write(f"| CORE index | {core_val:.4f} |\n")
    buf.write(f"| MCA at threshold ({config.mca_threshold:.2f}) | {mca_val:.4f} |\n")
    buf.write(
        f"| Agreement-gated accuracy (τ_agree={tau_agree:.2f}) | {aga_val:.4f} |\n\n"
    )


def _write_car_curve(buf: io.StringIO, report: EvaluationReport) -> None:
    curve = car_curve(report.results)
    if not curve:
        return
    buf.write("## CAR curve\n\n")
    buf.write("| consistency threshold | accuracy |\n")
    buf.write("|---|---|\n")
    for c, accuracy in curve:
        buf.write(f"| {c:.2f} | {accuracy:.4f} |\n")
    buf.write("\n")


def _write_per_question(buf: io.StringIO, report: EvaluationReport) -> None:
    buf.write("## Per-question results\n\n")
    buf.write(
        "| question_id | rc_correct | rc_agree | variants | correct | "
        "answer distribution |\n"
    )
    buf.write("|---|---|---|---|---|---|\n")
    for qcr in report.results:
        dist = ", ".join(
            f"`{label}`={count}"
            for label, count in sorted(qcr.answer_distribution.items())
        )
        buf.write(
            f"| {qcr.question_id} | {qcr.rc_correct:.4f} | "
            f"{qcr.rc_agree:.4f} | {qcr.total_variants} | "
            f"{qcr.correct_count} | {dist} |\n"
        )
    buf.write("\n")
