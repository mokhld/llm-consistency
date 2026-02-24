"""Tests for providers package public API surface."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests: providers package-level imports
# ---------------------------------------------------------------------------
class TestProvidersPackageImports:
    """All key symbols are importable from llm_consistency.providers."""

    def test_import_base_llm_provider(self) -> None:
        """BaseLLMProvider importable from providers package."""
        from llm_consistency.providers import BaseLLMProvider  # noqa: PLC0415

        assert BaseLLMProvider is not None

    def test_import_batch_result(self) -> None:
        """BatchResult importable from providers package."""
        from llm_consistency.providers import BatchResult  # noqa: PLC0415

        assert BatchResult is not None

    def test_import_budget_exceeded_error(self) -> None:
        """BudgetExceededError importable from providers package."""
        from llm_consistency.providers import BudgetExceededError  # noqa: PLC0415

        assert BudgetExceededError is not None

    def test_import_estimate_cost(self) -> None:
        """estimate_cost importable from providers package."""
        from llm_consistency.providers import estimate_cost  # noqa: PLC0415

        assert callable(estimate_cost)

    def test_import_get_provider(self) -> None:
        """get_provider importable from providers package."""
        from llm_consistency.providers import get_provider  # noqa: PLC0415

        assert callable(get_provider)

    def test_import_get_model_pricing(self) -> None:
        """get_model_pricing importable from providers package."""
        from llm_consistency.providers import get_model_pricing  # noqa: PLC0415

        assert callable(get_model_pricing)

    def test_import_cost_per_token(self) -> None:
        """CostPerToken importable from providers package."""
        from llm_consistency.providers import CostPerToken  # noqa: PLC0415

        assert CostPerToken is not None

    def test_import_async_token_bucket(self) -> None:
        """AsyncTokenBucket importable from providers package."""
        from llm_consistency.providers import AsyncTokenBucket  # noqa: PLC0415

        assert AsyncTokenBucket is not None


# ---------------------------------------------------------------------------
# Tests: top-level package imports
# ---------------------------------------------------------------------------
class TestTopLevelImports:
    """Key provider symbols importable from llm_consistency."""

    def test_import_base_llm_provider(self) -> None:
        """BaseLLMProvider importable from top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "BaseLLMProvider")

    def test_import_batch_result(self) -> None:
        """BatchResult importable from top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "BatchResult")

    def test_import_budget_exceeded_error(self) -> None:
        """BudgetExceededError importable from top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "BudgetExceededError")

    def test_import_estimate_cost(self) -> None:
        """estimate_cost importable from top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "estimate_cost")

    def test_import_get_provider(self) -> None:
        """get_provider importable from top-level."""
        mod = importlib.import_module("llm_consistency")
        assert hasattr(mod, "get_provider")


# ---------------------------------------------------------------------------
# Tests: get_provider() factory function
# ---------------------------------------------------------------------------
class TestGetProvider:
    """get_provider() routes to correct provider by name."""

    def test_unknown_provider_raises_valueerror(self) -> None:
        """get_provider with unknown name raises ValueError."""
        from llm_consistency.providers import get_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown")

    def test_unknown_provider_lists_available(self) -> None:
        """ValueError message lists available provider names."""
        from llm_consistency.providers import get_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="anthropic") as exc_info:
            get_provider("unknown")
        msg = str(exc_info.value)
        assert "openai" in msg
        assert "anthropic" in msg
        assert "ollama" in msg
        assert "litellm" in msg

    def test_get_provider_openai(self) -> None:
        """get_provider('openai') returns OpenAIProvider."""
        mock_client = MagicMock()
        with patch(
            "llm_consistency.providers._openai.AsyncOpenAI",
            return_value=mock_client,
            create=True,
        ):
            # Patch the openai import inside the provider module
            mock_openai_module = MagicMock()
            mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_client)
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                from llm_consistency.providers import (  # noqa: PLC0415
                    get_provider,
                )

                provider = get_provider("openai", model="gpt-4o")

                from llm_consistency.providers._openai import (  # noqa: PLC0415
                    OpenAIProvider,
                )

                assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# Tests: __all__ coverage
# ---------------------------------------------------------------------------
class TestProvidersDunderAll:
    """providers.__all__ exports the expected symbols."""

    def test_all_contains_expected_symbols(self) -> None:
        """__all__ has the documented public API."""
        from llm_consistency.providers import __all__  # noqa: PLC0415

        expected = {
            "AsyncTokenBucket",
            "BaseLLMProvider",
            "BatchResult",
            "BudgetExceededError",
            "CostPerToken",
            "estimate_cost",
            "get_model_pricing",
            "get_provider",
        }
        assert expected == set(__all__)

    def test_top_level_all_contains_provider_symbols(self) -> None:
        """Top-level __all__ includes provider re-exports."""
        mod = importlib.import_module("llm_consistency")
        all_names = set(mod.__all__)
        assert "BaseLLMProvider" in all_names
        assert "BatchResult" in all_names
        assert "BudgetExceededError" in all_names
        assert "estimate_cost" in all_names
        assert "get_provider" in all_names
