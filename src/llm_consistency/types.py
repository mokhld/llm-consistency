"""Core data types for llm-consistency evaluation framework."""

from __future__ import annotations

from enum import Enum


class PerturbationType(Enum):
    """Categories of perturbation applied to questions.

    Members:
        OPTION_REORDER: Shuffle MC answer option ordering.
        FORMAT_CHANGE: Change question formatting/template.
        SEPARATOR_CHANGE: Modify delimiters between options.
        PARAPHRASE: LLM-powered semantic rephrasing.
        INSTRUCTION_REPHRASE: System prompt variants.
    """

    OPTION_REORDER = "option_reorder"
    FORMAT_CHANGE = "format_change"
    SEPARATOR_CHANGE = "separator_change"
    PARAPHRASE = "paraphrase"
    INSTRUCTION_REPHRASE = "instruction_rephrase"
