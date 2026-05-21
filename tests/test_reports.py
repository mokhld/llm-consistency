"""Tests for CSV and Markdown report exporters."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from llm_consistency.reports import export_csv, export_html, export_markdown
from llm_consistency.runners._metadata import RunMetadata
from llm_consistency.types import (
    EvaluationConfig,
    EvaluationReport,
    PerturbationType,
    QuestionConsistencyResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(num_questions: int = 3) -> EvaluationReport:
    """Synthesize a report with `num_questions` QCRs."""
    config = EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(PerturbationType.OPTION_REORDER,),
        scorer="exact_match",
        num_variants=2,
        concurrency=2,
        mca_threshold=0.8,
        core_threshold=0.5,
    )
    results = tuple(
        QuestionConsistencyResult(
            question_id=f"q{i}",
            rc_correct=1.0 - 0.25 * i,
            rc_agree=1.0 - 0.1 * i,
            total_variants=2,
            correct_count=2 - i,
            answer_distribution={"A": 2 - i, "B": i},
        )
        for i in range(num_questions)
    )
    total_variants = sum(r.total_variants for r in results)
    mean_rc_correct = sum(r.rc_correct for r in results) / num_questions
    mean_rc_agree = sum(r.rc_agree for r in results) / num_questions
    return EvaluationReport(
        config=config,
        results=results,
        total_questions=num_questions,
        total_variants=total_variants,
        mean_rc_correct=mean_rc_correct,
        mean_rc_agree=mean_rc_agree,
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=3)
        out_path = tmp_path / "report.csv"
        export_csv(report, out_path)

        with out_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))

        # header + 3 data rows
        assert len(rows) == 4
        assert rows[0] == [
            "question_id",
            "rc_correct",
            "rc_agree",
            "total_variants",
            "correct_count",
            "answer_distribution",
        ]
        assert rows[1][0] == "q0"
        assert float(rows[1][1]) == 1.0
        assert int(rows[1][3]) == 2

    def test_answer_distribution_rendering(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=2)
        out_path = tmp_path / "report.csv"
        export_csv(report, out_path)
        text = out_path.read_text(encoding="utf-8")
        # q0 has {A: 2, B: 0} -> rendered sorted by label
        # q1 has {A: 1, B: 1}
        assert "A=2; B=0" in text
        assert "A=1; B=1" in text

    def test_creates_atomic_with_no_temp_leftover(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.csv"
        export_csv(report, out_path)
        # No leftover *.tmp file beside the output
        assert list(tmp_path.glob("*.tmp")) == []
        assert out_path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        out_path = tmp_path / "report.csv"
        out_path.write_text("old contents", encoding="utf-8")
        export_csv(_make_report(num_questions=1), out_path)
        text = out_path.read_text(encoding="utf-8")
        assert "question_id" in text
        assert "old contents" not in text

    def test_empty_results_writes_header_only(self, tmp_path: Path) -> None:
        # Build a report with empty results — bypass BatchRunner since
        # that path raises; we want to exercise the exporter directly.
        config = EvaluationConfig(
            model="mock",
            provider="mock",
            perturbation_types=(PerturbationType.OPTION_REORDER,),
            scorer="exact_match",
            num_variants=2,
            concurrency=1,
        )
        report = EvaluationReport(
            config=config,
            results=(),
            total_questions=0,
            total_variants=0,
            mean_rc_correct=0.0,
            mean_rc_agree=0.0,
        )
        out_path = tmp_path / "empty.csv"
        export_csv(report, out_path)
        rows = list(csv.reader(out_path.open(encoding="utf-8", newline="")))
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


class TestExportMarkdown:
    def test_writes_title_and_sections(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=2)
        out_path = tmp_path / "report.md"
        export_markdown(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        assert text.startswith("# LLM Consistency Report")
        assert "## Run" in text
        assert "## Aggregate metrics" in text
        assert "## CAR curve" in text
        assert "## Per-question results" in text

    def test_includes_metadata_block_when_provided(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        metadata = RunMetadata.capture(report.config, seed=7)
        out_path = tmp_path / "report.md"
        export_markdown(report, out_path, metadata=metadata)

        text = out_path.read_text(encoding="utf-8")
        assert "**Package version:**" in text
        assert "**Perturbation seed:** `7`" in text

    def test_omits_metadata_when_not_provided(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.md"
        export_markdown(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        assert "**Package version:**" not in text
        assert "**Perturbation seed:**" not in text

    def test_per_question_table_rows(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=3)
        out_path = tmp_path / "report.md"
        export_markdown(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        # Expect markdown rows for q0..q2
        assert "| q0 |" in text
        assert "| q1 |" in text
        assert "| q2 |" in text

    def test_core_threshold_rendered_when_set(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.md"
        export_markdown(report, out_path)
        text = out_path.read_text(encoding="utf-8")
        # _make_report uses core_threshold=0.5
        assert "**CORE threshold:** 0.500" in text

    def test_no_temp_leftover(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.md"
        export_markdown(report, out_path)
        assert list(tmp_path.glob("*.tmp")) == []
        assert out_path.exists()


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


class TestExportHtml:
    def test_writes_doctype_and_sections(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=2)
        out_path = tmp_path / "report.html"
        export_html(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")
        assert "<title>LLM Consistency Report</title>" in text
        assert "<h1>LLM Consistency Report</h1>" in text
        assert "<h2>Run</h2>" in text
        assert "<h2>Aggregate metrics</h2>" in text
        assert "<h2>CAR curve</h2>" in text
        assert "<h2>Per-question results</h2>" in text
        assert text.rstrip().endswith("</html>")

    def test_includes_inline_css_no_external_assets(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.html"
        export_html(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        assert "<style>" in text
        assert "</style>" in text
        # No external CSS / JS / images.
        assert "<link " not in text
        assert "<script" not in text
        assert "<img " not in text

    def test_includes_metadata_block_when_provided(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        metadata = RunMetadata.capture(report.config, seed=7)
        out_path = tmp_path / "report.html"
        export_html(report, out_path, metadata=metadata)

        text = out_path.read_text(encoding="utf-8")
        assert "Package version:" in text
        assert "Perturbation seed:" in text
        assert "<code>7</code>" in text

    def test_omits_metadata_when_not_provided(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.html"
        export_html(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        assert "Package version:" not in text
        assert "Perturbation seed:" not in text

    def test_per_question_rows(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=3)
        out_path = tmp_path / "report.html"
        export_html(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        # Each question_id should appear in a <td>
        for i in range(3):
            assert f"<td>q{i}</td>" in text

    def test_html_escapes_question_ids(self, tmp_path: Path) -> None:
        """A malicious question_id with `<script>` must not inject markup."""
        from llm_consistency.types import (  # noqa: PLC0415
            EvaluationReport,
            QuestionConsistencyResult,
        )

        base = _make_report(num_questions=1)
        evil = QuestionConsistencyResult(
            question_id="<script>alert(1)</script>",
            rc_correct=1.0,
            rc_agree=1.0,
            total_variants=1,
            correct_count=1,
            answer_distribution={"<b>A</b>": 1},
        )
        report = EvaluationReport(
            config=base.config,
            results=(evil,),
            total_questions=1,
            total_variants=1,
            mean_rc_correct=1.0,
            mean_rc_agree=1.0,
        )

        out_path = tmp_path / "evil.html"
        export_html(report, out_path)

        text = out_path.read_text(encoding="utf-8")
        # Literal <script> tag must be escaped — not present as a real tag.
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
        # And the bold answer label must also be escaped.
        assert "<b>A</b>=1" not in text
        assert "&lt;b&gt;A&lt;/b&gt;" in text

    def test_core_threshold_rendered_when_set(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.html"
        export_html(report, out_path)
        text = out_path.read_text(encoding="utf-8")
        # _make_report uses core_threshold=0.5
        assert "CORE threshold:" in text
        assert "0.500" in text

    def test_no_temp_leftover(self, tmp_path: Path) -> None:
        report = _make_report(num_questions=1)
        out_path = tmp_path / "report.html"
        export_html(report, out_path)
        assert list(tmp_path.glob("*.tmp")) == []
        assert out_path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        out_path = tmp_path / "report.html"
        out_path.write_text("old contents", encoding="utf-8")
        export_html(_make_report(num_questions=1), out_path)
        text = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in text
        assert "old contents" not in text


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------


class TestReportsPublicAPI:
    def test_export_csv_top_level_import(self) -> None:
        from llm_consistency import export_csv as top_csv  # noqa: PLC0415

        assert top_csv is export_csv

    def test_export_markdown_top_level_import(self) -> None:
        from llm_consistency import export_markdown as top_md  # noqa: PLC0415

        assert top_md is export_markdown

    def test_export_html_top_level_import(self) -> None:
        from llm_consistency import export_html as top_html  # noqa: PLC0415

        assert top_html is export_html
