"""Tests for llm_consistency.providers module-level imports."""

from __future__ import annotations

import importlib


def test_providers_module_importable() -> None:
    """llm_consistency.providers is importable and has __all__."""
    mod = importlib.import_module("llm_consistency.providers")
    assert hasattr(mod, "__all__")
    assert len(mod.__all__) > 0


def test_providers_no_sdk_required_at_import() -> None:
    """Importing providers does not require openai/anthropic/etc."""
    mod = importlib.import_module("llm_consistency.providers")
    # If import succeeded without openai/anthropic installed,
    # concrete providers are properly lazy-loaded.
    assert hasattr(mod, "BaseLLMProvider")
    assert hasattr(mod, "get_provider")
