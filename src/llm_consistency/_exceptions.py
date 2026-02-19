"""Custom exception hierarchy for the llm-consistency package."""


class LLMConsistencyError(Exception):
    """Base exception for the llm-consistency package."""


class ValidationError(LLMConsistencyError, ValueError):
    """Raised when a type's construction-time validation fails."""
