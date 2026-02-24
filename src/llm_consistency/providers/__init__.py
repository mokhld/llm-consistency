"""Async LLM provider interfaces.

Public API for the providers package. Concrete provider classes
(OpenAI, Anthropic, Ollama, LiteLLM) are NOT imported at package
level to avoid requiring all four SDKs. Use :func:`get_provider`
for lazy instantiation by name, or import concrete classes directly::

    from llm_consistency.providers._openai import OpenAIProvider
    from llm_consistency.providers._anthropic import AnthropicProvider
    from llm_consistency.providers._ollama import OllamaProvider
    from llm_consistency.providers._litellm import LiteLLMProvider
"""

from __future__ import annotations

from typing import Any

from llm_consistency.providers._base import BaseLLMProvider
from llm_consistency.providers._batch_result import BatchResult
from llm_consistency.providers._budget import BudgetExceededError, CostPerToken
from llm_consistency.providers._cost import estimate_cost, get_model_pricing
from llm_consistency.providers._rate_limit import AsyncTokenBucket

_PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "llm_consistency.providers._openai",
    "anthropic": "llm_consistency.providers._anthropic",
    "ollama": "llm_consistency.providers._ollama",
    "litellm": "llm_consistency.providers._litellm",
}

_PROVIDER_CLASS_NAMES: dict[str, str] = {
    "openai": "OpenAIProvider",
    "anthropic": "AnthropicProvider",
    "ollama": "OllamaProvider",
    "litellm": "LiteLLMProvider",
}


def get_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    """Create a provider instance by name with lazy import.

    Avoids importing all SDK dependencies at package level.
    Each provider SDK is imported only when its provider is
    requested.

    Args:
        name: Provider name (``"openai"``, ``"anthropic"``,
            ``"ollama"``, or ``"litellm"``).
        **kwargs: Forwarded to the provider constructor.

    Returns:
        A :class:`BaseLLMProvider` subclass instance.

    Raises:
        ValueError: If *name* is not a known provider.
    """
    module_path = _PROVIDER_REGISTRY.get(name)
    if module_path is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        msg = f"Unknown provider {name!r}. Available providers: {available}"
        raise ValueError(msg)

    import importlib  # noqa: PLC0415

    module = importlib.import_module(module_path)
    cls = getattr(module, _PROVIDER_CLASS_NAMES[name])
    return cls(**kwargs)  # type: ignore[no-any-return]


__all__ = [
    "AsyncTokenBucket",
    "BaseLLMProvider",
    "BatchResult",
    "BudgetExceededError",
    "CostPerToken",
    "estimate_cost",
    "get_model_pricing",
    "get_provider",
]
