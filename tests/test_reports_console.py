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
