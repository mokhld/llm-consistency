"""Hand-rolled ASCII CAR curve renderer.

Renders a Consistency-Accuracy Relationship curve as ASCII art
suitable for terminal display. No external dependencies.
"""

from __future__ import annotations


def render_car_ascii(
    curve: list[tuple[float, float]],
    width: int = 60,
    height: int = 15,
) -> str:
    """Render a CAR curve as an ASCII chart.

    Y-axis labels on the left (1.0 at top, 0.0 at bottom).
    X-axis labels at bottom (thresholds).
    Data points marked with ``*``.

    Args:
        curve: List of ``(threshold, mca_value)`` pairs.
        width: Character width of the plot area.
        height: Character height of the plot area.

    Returns:
        Multi-line ASCII string representing the CAR curve.
    """
    if not curve:
        return ""

    # Build character grid (rows x cols), filled with spaces
    grid: list[list[str]] = [[" "] * width for _ in range(height)]

    # Place data points
    for threshold, mca_val in curve:
        col = int(threshold * (width - 1))
        row = int((1.0 - mca_val) * (height - 1))
        # Clamp to grid bounds
        col = max(0, min(col, width - 1))
        row = max(0, min(row, height - 1))
        grid[row][col] = "*"

    # Build output with y-axis labels
    lines: list[str] = []
    for row_idx in range(height):
        y_val = 1.0 - (row_idx / (height - 1))
        label = f"{y_val:.1f} |"
        lines.append(label + "".join(grid[row_idx]))

    # X-axis separator
    y_label_width = len("0.0 |")
    lines.append(" " * (y_label_width - 1) + "+" + "-" * width)

    # X-axis labels
    x_labels = " " * y_label_width
    x_labels += "0.0"
    padding = width - 6  # space between 0.0 and 1.0
    if padding > 0:
        x_labels += " " * padding
    x_labels += "1.0"
    lines.append(x_labels)

    return "\n".join(lines)
