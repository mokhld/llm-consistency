"""Dataset loading layer for the llm-consistency evaluation framework.

Re-exports all dataset classes and validation utilities for convenient
public access via ``from llm_consistency.datasets import ...``.
"""

from llm_consistency.datasets._base import BaseDataset
from llm_consistency.datasets._custom import CustomDataset
from llm_consistency.datasets._mc import MCDataset
from llm_consistency.datasets._open_ended import OpenEndedDataset
from llm_consistency.datasets._validation import detect_format, validate_unique_ids

__all__ = [
    "BaseDataset",
    "CustomDataset",
    "MCDataset",
    "OpenEndedDataset",
    "detect_format",
    "validate_unique_ids",
]
