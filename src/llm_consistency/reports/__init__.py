"""Report generation, export, and display.

Re-exports the console reporter, ASCII CAR renderer, and the JSON / CSV
/ Markdown / HTML exporters for convenient public access via
``from llm_consistency.reports import ...``.
"""

from llm_consistency.reports._car_ascii import render_car_ascii
from llm_consistency.reports._console import ConsoleReporter
from llm_consistency.reports._csv_export import export_csv
from llm_consistency.reports._html_export import export_html
from llm_consistency.reports._json_export import export_json
from llm_consistency.reports._markdown_export import export_markdown

__all__ = [
    "ConsoleReporter",
    "export_csv",
    "export_html",
    "export_json",
    "export_markdown",
    "render_car_ascii",
]
