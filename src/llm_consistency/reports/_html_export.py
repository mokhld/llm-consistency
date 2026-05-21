"""HTML report export for evaluation results.

Produces a self-contained, single-page HTML report — inline CSS, no
external assets, no JavaScript — so a report can be opened directly
from disk or attached to a PR/email without server-side rendering.
Atomic UTF-8 write following the same pattern as :func:`export_json`,
:func:`export_csv`, and :func:`export_markdown`.

All dynamic content (question IDs, model identifiers, timestamps,
answer labels) is HTML-escaped via :func:`html.escape` so a malicious
question ID or label cannot inject markup into the document.
"""

from __future__ import annotations

import contextlib
import html
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


_HTML_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  max-width: 960px;
  margin: 2rem auto;
  padding: 0 1.25rem;
  color: #1d1d1f;
  line-height: 1.5;
}
h1 {
  margin: 0 0 1.5rem 0;
  font-size: 1.75rem;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 0.5rem;
}
h2 {
  margin: 2rem 0 0.75rem 0;
  font-size: 1.25rem;
  color: #111827;
}
table {
  border-collapse: collapse;
  margin: 0.5rem 0 1.25rem 0;
  width: 100%;
  font-size: 0.95rem;
}
th, td {
  text-align: left;
  padding: 0.4rem 0.75rem;
  border-bottom: 1px solid #e5e7eb;
}
th {
  background: #f3f4f6;
  font-weight: 600;
}
tbody tr:hover {
  background: #fafafa;
}
code, .mono {
  font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
  font-size: 0.9em;
  background: #f3f4f6;
  padding: 0.05rem 0.3rem;
  border-radius: 3px;
}
ul.run {
  list-style: none;
  padding: 0;
  margin: 0;
}
ul.run li {
  padding: 0.2rem 0;
}
ul.run b {
  display: inline-block;
  min-width: 12rem;
  color: #374151;
}
.num {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
""".strip()


def export_html(
    report: EvaluationReport,
    path: Path,
    *,
    metadata: RunMetadata | None = None,
    tau_agree: float = 0.8,
) -> None:
    """Write a self-contained HTML summary of an evaluation report.

    Sections (in order):

    1. ``<h1>`` title.
    2. ``<h2>Run</h2>`` block — model, provider, scorer, perturbations,
       thresholds, and (if *metadata* is given) package and Python
       versions, timestamp, perturbation seed.
    3. ``<h2>Aggregate metrics</h2>`` table — total questions, total
       variants, mean ``rc_correct``, mean ``rc_agree``, CORE index,
       MCA at the configured threshold, agreement-gated accuracy at
       *tau_agree*.
    4. ``<h2>CAR curve</h2>`` table.
    5. ``<h2>Per-question results</h2>`` table.

    All dynamic content is HTML-escaped. The page bundles its CSS
    inline so the file works when opened directly from disk.
    """
    buf = io.StringIO()
    buf.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
    buf.write('<meta charset="utf-8">\n')
    buf.write('<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    buf.write("<title>LLM Consistency Report</title>\n")
    buf.write(f"<style>{_HTML_CSS}</style>\n")
    buf.write("</head>\n<body>\n")
    buf.write("<h1>LLM Consistency Report</h1>\n")

    _write_run_block(buf, report, metadata)
    _write_aggregate_metrics(buf, report, tau_agree=tau_agree)
    _write_car_curve(buf, report)
    _write_per_question(buf, report)

    buf.write("</body>\n</html>\n")
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
    buf.write('<h2>Run</h2>\n<ul class="run">\n')
    buf.write(f"<li><b>Model:</b> <code>{html.escape(config.model)}</code></li>\n")
    buf.write(
        f"<li><b>Provider:</b> <code>{html.escape(config.provider)}</code></li>\n"
    )
    buf.write(f"<li><b>Scorer:</b> <code>{html.escape(config.scorer)}</code></li>\n")
    pert_str = ", ".join(
        f"<code>{html.escape(pt.value)}</code>" for pt in config.perturbation_types
    )
    buf.write(f"<li><b>Perturbations:</b> {pert_str}</li>\n")
    buf.write(f"<li><b>Variants per question:</b> {config.num_variants}</li>\n")
    buf.write(f"<li><b>MCA threshold:</b> {config.mca_threshold:.3f}</li>\n")
    if config.core_threshold is not None:
        buf.write(f"<li><b>CORE threshold:</b> {config.core_threshold:.3f}</li>\n")
    if config.max_budget_usd is not None:
        buf.write(f"<li><b>Max budget USD:</b> {config.max_budget_usd}</li>\n")
    if metadata is not None:
        buf.write(
            "<li><b>Package version:</b> "
            f"<code>{html.escape(metadata.package_version)}</code></li>\n"
        )
        buf.write(
            "<li><b>Python version:</b> "
            f"<code>{html.escape(metadata.python_version)}</code></li>\n"
        )
        buf.write(
            "<li><b>Timestamp (UTC):</b> "
            f"<code>{html.escape(metadata.timestamp)}</code></li>\n"
        )
        buf.write(
            "<li><b>Perturbation seed:</b> "
            f"<code>{metadata.perturbation_seed}</code></li>\n"
        )
    buf.write("</ul>\n")


def _metric_row(label: str, value: str) -> str:
    return f'<tr><td>{label}</td><td class="num">{value}</td></tr>\n'


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

    buf.write("<h2>Aggregate metrics</h2>\n<table>\n")
    buf.write('<thead><tr><th>Metric</th><th class="num">Value</th></tr></thead>\n')
    buf.write("<tbody>\n")
    buf.write(_metric_row("Total questions", str(report.total_questions)))
    buf.write(_metric_row("Total variants", str(report.total_variants)))
    buf.write(_metric_row("Mean rc_correct", f"{report.mean_rc_correct:.4f}"))
    buf.write(_metric_row("Mean rc_agree", f"{report.mean_rc_agree:.4f}"))
    buf.write(_metric_row("CORE index", f"{core_val:.4f}"))
    buf.write(
        _metric_row(
            f"MCA at threshold ({config.mca_threshold:.2f})",
            f"{mca_val:.4f}",
        )
    )
    buf.write(
        _metric_row(
            f"Agreement-gated accuracy (τ_agree={tau_agree:.2f})",
            f"{aga_val:.4f}",
        )
    )
    buf.write("</tbody>\n</table>\n")


def _write_car_curve(buf: io.StringIO, report: EvaluationReport) -> None:
    curve = car_curve(report.results)
    if not curve:
        return
    buf.write("<h2>CAR curve</h2>\n<table>\n")
    buf.write(
        "<thead><tr><th>consistency threshold</th>"
        '<th class="num">accuracy</th></tr></thead>\n'
    )
    buf.write("<tbody>\n")
    for c, accuracy in curve:
        buf.write(
            f'<tr><td class="num">{c:.2f}</td>'
            f'<td class="num">{accuracy:.4f}</td></tr>\n'
        )
    buf.write("</tbody>\n</table>\n")


def _write_per_question(buf: io.StringIO, report: EvaluationReport) -> None:
    buf.write("<h2>Per-question results</h2>\n<table>\n")
    buf.write(
        "<thead><tr>"
        "<th>question_id</th>"
        '<th class="num">rc_correct</th>'
        '<th class="num">rc_agree</th>'
        '<th class="num">variants</th>'
        '<th class="num">correct</th>'
        "<th>answer distribution</th>"
        "</tr></thead>\n"
    )
    buf.write("<tbody>\n")
    for qcr in report.results:
        dist = ", ".join(
            f"<code>{html.escape(str(label))}</code>={count}"
            for label, count in sorted(qcr.answer_distribution.items())
        )
        buf.write(
            "<tr>"
            f"<td>{html.escape(qcr.question_id)}</td>"
            f'<td class="num">{qcr.rc_correct:.4f}</td>'
            f'<td class="num">{qcr.rc_agree:.4f}</td>'
            f'<td class="num">{qcr.total_variants}</td>'
            f'<td class="num">{qcr.correct_count}</td>'
            f"<td>{dist}</td>"
            "</tr>\n"
        )
    buf.write("</tbody>\n</table>\n")
