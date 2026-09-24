"""Tests for Rich console report (llm_consistency.reports._console)."""

from __future__ import annotations

import importlib
from io import StringIO

from rich.console import Console

from llm_consistency.types import (
    EvaluationConfig,
    EvaluationReport,
    PerturbationType,
    QuestionConsistencyResult,
    ScoredResponse,
)


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


def _get_console_reporter():
    """Import ConsoleReporter via importlib to avoid PLC0415."""
    mod = importlib.import_module("llm_consistency.reports")
    return mod.ConsoleReporter


def _get_render_car_ascii():
    """Import render_car_ascii via importlib to avoid PLC0415."""
    mod = importlib.import_module("llm_consistency.reports")
    return mod.render_car_ascii


def test_console_reporter_display_does_not_raise() -> None:
    """ConsoleReporter.display(report) does not raise."""
    cls = _get_console_reporter()

    buf = StringIO()
    console = Console(file=buf, force_terminal=True)
    reporter = cls(console=console)
    report = _make_report()

    # Should not raise any exception
    reporter.display(report)


def test_console_reporter_output_contains_core() -> None:
    """Output contains 'CORE' metric value."""
    cls = _get_console_reporter()

    buf = StringIO()
    console = Console(file=buf, force_terminal=True)
    reporter = cls(console=console)
    reporter.display(_make_report())

    output = buf.getvalue()
    assert "CORE" in output


def test_console_reporter_output_contains_mca() -> None:
    """Output contains 'MCA' metric value."""
    cls = _get_console_reporter()

    buf = StringIO()
    console = Console(file=buf, force_terminal=True)
    reporter = cls(console=console)
    reporter.display(_make_report())

    output = buf.getvalue()
    assert "MCA" in output


def test_console_reporter_output_contains_pass_fail() -> None:
    """Output contains pass/fail status."""
    cls = _get_console_reporter()

    buf = StringIO()
    console = Console(file=buf, force_terminal=True)
    reporter = cls(console=console)
    reporter.display(_make_report())

    output = buf.getvalue()
    assert "PASS" in output or "FAIL" in output


def test_render_car_ascii_returns_multiline_with_stars() -> None:
    """render_car_ascii(curve) returns a multi-line string with '*' data points."""
    render_car_ascii = _get_render_car_ascii()

    # Perfect curve: all MCA values = 1.0
    curve = [(i / 10, 1.0) for i in range(11)]
    result = render_car_ascii(curve)

    assert isinstance(result, str)
    assert "\n" in result
    assert "*" in result


def test_render_car_ascii_has_axes() -> None:
    """render_car_ascii(curve) returns a string with y-axis and x-axis."""
    render_car_ascii = _get_render_car_ascii()

    curve = [(i / 10, 1.0) for i in range(11)]
    result = render_car_ascii(curve)

    # Should have y-axis values (1.0, 0.0)
    assert "1.0" in result
    assert "0.0" in result


def test_console_reporter_with_string_io() -> None:
    """ConsoleReporter works with Console(file=StringIO()) for capturing output."""
    cls = _get_console_reporter()

    buf = StringIO()
    console = Console(file=buf, force_terminal=True)
    reporter = cls(console=console)
    reporter.display(_make_report())

    output = buf.getvalue()
    # Output should be non-empty and contain key metrics
    assert len(output) > 0
    assert "CORE" in output
    assert "MCA" in output


def _report_with(
    rc_corrects: list[float],
    *,
    mca_threshold: float = 1.0,
    min_mca: float = 1.0,
    core_threshold: float | None = None,
    errored_variants: int = 0,
) -> EvaluationReport:
    """Build a report with one QCR per rc_correct value.

    The first ``errored_variants`` scored responses of the first question
    are marked as provider errors (``scoring_method`` starting "error:").
    """
    results = []
    for i, rc in enumerate(rc_corrects):
        scored = tuple(
            ScoredResponse(
                question_id=f"q{i}",
                is_correct=False,
                score=0.0,
                scoring_method="error:RuntimeError"
                if i == 0 and v < errored_variants
                else "exact_match",
            )
            for v in range(2)
        )
        results.append(
            QuestionConsistencyResult(
                question_id=f"q{i}",
                rc_correct=rc,
                rc_agree=1.0,
                total_variants=2,
                correct_count=round(2 * rc),
                answer_distribution={"A": 2},
                scored_responses=scored,
            )
        )
    config = EvaluationConfig(
        model="mock",
        provider="mock",
        perturbation_types=(PerturbationType.OPTION_REORDER,),
        scorer="exact_match",
        mca_threshold=mca_threshold,
        min_mca=min_mca,
        core_threshold=core_threshold,
    )
    return EvaluationReport(
        config=config,
        results=tuple(results),
        total_questions=len(results),
        total_variants=2 * len(results),
        mean_rc_correct=sum(rc_corrects) / len(rc_corrects),
        mean_rc_agree=1.0,
    )


def _row(report: EvaluationReport, metric: str, **kwargs: float) -> str:
    """Render *report* and return the summary-table line for *metric*."""
    buf = StringIO()
    _get_console_reporter()(console=Console(file=buf, width=120)).display(
        report, **kwargs
    )
    lines = [line for line in buf.getvalue().splitlines() if metric in line]
    assert lines, f"no {metric!r} row in output"
    return lines[0]


def test_core_status_is_na_without_core_threshold() -> None:
    assert "n/a" in _row(_report_with([1.0]), "CORE")


def test_core_status_uses_core_threshold_when_set() -> None:
    assert "FAIL" in _row(_report_with([0.0], core_threshold=0.5), "CORE")
    assert "PASS" in _row(_report_with([1.0], core_threshold=0.5), "CORE")


def test_mca_status_uses_min_mca_like_the_ci_gate() -> None:
    """9 of 10 questions pass at c=0.8: MCA is 0.9."""
    ci = importlib.import_module("llm_consistency.runners._ci")
    rcs = [1.0] * 9 + [0.5]

    strict = _report_with(rcs, mca_threshold=0.8)
    assert "FAIL" in _row(strict, "MCA(0.80)")
    assert any("MCA check failed" in f for f in ci.gate_failures(strict))

    lenient = _report_with(rcs, mca_threshold=0.8, min_mca=0.9)
    assert "PASS" in _row(lenient, "MCA(0.80)")
    assert ci.gate_failures(lenient) == ()


def test_threshold_override_of_zero_is_honoured() -> None:
    report = _report_with([0.0], mca_threshold=1.0, min_mca=1.0)
    line = _row(report, "MCA(", threshold=0.0)
    assert "MCA(0.00)" in line
    assert "1.0000" in line
    assert "PASS" in line


def test_failed_variants_row_only_when_errors() -> None:
    buf = StringIO()
    reporter = _get_console_reporter()(console=Console(file=buf, width=120))
    reporter.display(_report_with([1.0, 1.0]))
    assert "Failed variants" not in buf.getvalue()

    line = _row(_report_with([0.0, 1.0], errored_variants=1), "Failed variants")
    assert "1 / 4" in line
    assert "FAIL" in line
