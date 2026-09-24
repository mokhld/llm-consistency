"""Evaluation pipeline runners.

Re-exports BatchRunner, StreamingRunner, CIRunner, RunMetadata, and
shared pipeline helpers for convenient public access.
"""

from llm_consistency.runners._batch import BatchRunner
from llm_consistency.runners._ci import CIRunner
from llm_consistency.runners._metadata import RunMetadata
from llm_consistency.runners._pipeline import (
    DEFAULT_PROMPT_TEMPLATE,
    build_scored_qcr,
    generate_variants_for_question,
    render_prompt,
)
from llm_consistency.runners._streaming import StreamingRunner

__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "BatchRunner",
    "CIRunner",
    "RunMetadata",
    "StreamingRunner",
    "build_scored_qcr",
    "generate_variants_for_question",
    "render_prompt",
]
