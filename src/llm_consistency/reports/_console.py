"""Rich console reporter for evaluation results.

Displays color-coded pass/fail summary tables and an ASCII CAR
curve in the terminal using the Rich library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_consistency.metrics import car_curve, core_index, mca
from llm_consistency.reports._car_ascii import render_car_ascii

if TYPE_CHECKING:
    from llm_consistency.types import EvaluationReport


class ConsoleReporter:
    """Rich-formatted terminal reporter for evaluation results.

    Displays a summary table with CORE, MCA, mean_rc_correct, and
    mean_rc_agree metrics, each with a color-coded pass/fail status.
    Also renders an ASCII CAR curve in a Rich Panel.

    Args:
        console: Optional Rich Console instance for output capture.
            Defaults to a new Console.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def display(
        self,
        report: EvaluationReport,
        *,
        threshold: float | None = None,
    ) -> None:
        """Display evaluation results in the terminal.

        Computes metrics from the report, builds a Rich Table with
        pass/fail status, and renders an ASCII CAR curve.

        Args:
            report: The evaluation report to display.
            threshold: Optional override for MCA threshold.
                Defaults to ``report.config.mca_threshold``.
        """
        mca_threshold = threshold or report.config.mca_threshold
        core_threshold = report.config.core_threshold

        # Compute metrics
        core_val = core_index(report.results)
        mca_val = mca(report.results, mca_threshold)

        # Build summary table
        table = Table(title="Evaluation Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        # CORE row
        core_status = _pass_fail(
            core_val, core_threshold if core_threshold is not None else 0.0
        )
        table.add_row("CORE", f"{core_val:.4f}", core_status)

        # MCA row
        mca_status = _pass_fail(mca_val, mca_threshold)
        table.add_row("MCA", f"{mca_val:.4f}", mca_status)

        # mean_rc_correct row
        table.add_row(
            "Mean RC Correct",
            f"{report.mean_rc_correct:.4f}",
            _pass_fail(report.mean_rc_correct, mca_threshold),
        )

        # mean_rc_agree row
        table.add_row(
            "Mean RC Agree",
            f"{report.mean_rc_agree:.4f}",
            _pass_fail(report.mean_rc_agree, 0.5),
        )

        self._console.print(table)

        # Render CAR curve
        curve = car_curve(report.results)
        ascii_curve = render_car_ascii(curve)
        self._console.print(Panel(ascii_curve, title="CAR Curve"))


def _pass_fail(value: float, threshold: float) -> str:
    """Return a Rich-styled PASS or FAIL string."""
    if value >= threshold:
        return "[green]PASS[/]"
    return "[red]FAIL[/]"
