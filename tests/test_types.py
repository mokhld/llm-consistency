"""Tests for llm_consistency.types module."""

import pytest


def test_placeholder_sync() -> None:
    """Verify sync test execution works."""
    assert True


@pytest.mark.asyncio
async def test_placeholder_async() -> None:
    """Verify async test execution works with pytest-asyncio auto mode."""
    assert True
