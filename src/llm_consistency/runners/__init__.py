"""Evaluation pipeline runners.

Re-exports BatchRunner, RunMetadata, and shared pipeline helpers
for convenient public access.
"""

from llm_consistency.runners._batch import BatchRunner
from llm_consistency.runners._metadata import RunMetadata
from llm_consistency.runners._pipeline import (
    build_scored_qcr,
    generate_variants_for_question,
    render_prompt,
)

__all__ = [
    "BatchRunner",
    "RunMetadata",
    "build_scored_qcr",
    "generate_variants_for_question",
    "render_prompt",
]
