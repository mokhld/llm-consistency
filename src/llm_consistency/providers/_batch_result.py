"""BatchResult for tracking partial failures in batch LLM queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_consistency.types import LLMResponse


@dataclass(frozen=True)
class BatchResult:
    """Result of a batch query with partial failure tracking.

    Tracks both successful responses and errors, ensuring no failed
    samples are silently dropped.

    Attributes:
        responses: Tuple of successful LLM responses.
        errors: Tuple of ``(question_id, error_message)`` pairs for
            failed requests.
    """

    responses: tuple[LLMResponse, ...]
    errors: tuple[tuple[str, str], ...]

    @property
    def attempted(self) -> int:
        """Total number of requests attempted."""
        return len(self.responses) + len(self.errors)

    @property
    def completed(self) -> int:
        """Number of successfully completed requests."""
        return len(self.responses)

    @property
    def failed(self) -> int:
        """Number of failed requests."""
        return len(self.errors)
