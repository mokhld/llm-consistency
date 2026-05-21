"""Export the same EvaluationReport to JSON, CSV, Markdown, and HTML.

All four exporters take an `EvaluationReport` plus a path. The CLI
auto-detects format from the `--output` extension; from Python you call
each exporter directly and choose the path.

This example writes all four side-by-side into a temp directory and
prints their sizes so you can inspect the artefacts afterwards.

Run from the repo root::

    uv run python examples/05_export_formats.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _helpers import BaseIdMockProvider

from llm_consistency import (
    BatchRunner,
    EvaluationConfig,
    ExactMatchScorer,
    MCDataset,
    PerturbationType,
    export_csv,
    export_html,
    export_json,
    export_markdown,
)


async def main() -> None:
    dataset = MCDataset.load(Path(__file__).parent / "datasets" / "sample.jsonl")
    config = EvaluationConfig(
        model="mock-model",
        provider="mock",
        scorer="exact_match",
        perturbation_types=(PerturbationType.FORMAT_CHANGE,),
        num_variants=3,
    )
    provider = BaseIdMockProvider(
        model="mock-model",
        responses={"q1": "B", "q2": "C", "q3": "B", "q4": "A", "q5": "C"},
    )

    runner = BatchRunner()
    report = await runner.run(dataset, config, provider, ExactMatchScorer(), seed=42)
    metadata = runner.last_metadata

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        targets = {
            "JSON": out / "report.json",
            "CSV": out / "report.csv",
            "Markdown": out / "report.md",
            "HTML": out / "report.html",
        }

        export_json(report, targets["JSON"], metadata=metadata)
        export_csv(report, targets["CSV"])
        export_markdown(report, targets["Markdown"], metadata=metadata)
        export_html(report, targets["HTML"], metadata=metadata)

        print(f"Wrote 4 reports to {out}:\n")
        for fmt, path in targets.items():
            size_kb = path.stat().st_size / 1024
            print(f"  {fmt:<9} {path.name:<14} {size_kb:6.2f} KB")

        print("\nFirst lines of each artefact:")
        for fmt, path in targets.items():
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            preview = first_line if len(first_line) <= 80 else first_line[:77] + "..."
            print(f"  {fmt:<9} {preview}")


if __name__ == "__main__":
    asyncio.run(main())
