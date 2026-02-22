"""MockLLMProvider for offline testing and user onboarding.

Returns deterministic responses without API keys or network access.
Supports three response modes: default, response map, and cycling list.
"""

from __future__ import annotations

from typing_extensions import override

from llm_consistency.providers._base import BaseLLMProvider, _RawResponse
from llm_consistency.types import LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Deterministic LLM provider for testing and onboarding.

    Returns predictable responses without external SDK dependencies
    or network access.  Supports three response modes:

    - **Default:** Always returns ``default_response`` (default ``"A"``).
    - **Response map:** Maps ``question_id`` to a specific answer.
    - **Cycling list:** Successive queries cycle through a list.

    Args:
        model: Model identifier (default ``"mock"``).
        responses: A dict mapping question_id to answer, a list of
            answers to cycle through, or ``None`` for default mode.
        default_response: Fallback response when no mapping exists.
        **kwargs: Forwarded to :class:`BaseLLMProvider`.

    Examples:
        Default mode::

            provider = MockLLMProvider(model="mock")
            resp = await provider.query("prompt", "q1")
            assert resp.raw_output == "A"

        Response map::

            provider = MockLLMProvider(
                model="mock",
                responses={"q1": "A", "q2": "B"},
            )

        Cycling list::

            provider = MockLLMProvider(
                model="mock",
                responses=["A", "B", "C"],
            )
    """

    def __init__(
        self,
        *,
        model: str = "mock",
        responses: dict[str, str] | list[str] | None = None,
        default_response: str = "A",
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, **kwargs)  # type: ignore[arg-type]
        self._response_map: dict[str, str] | None = (
            responses if isinstance(responses, dict) else None
        )
        self._response_list: list[str] | None = (
            list(responses) if isinstance(responses, list) else None
        )
        self._default = default_response
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        """Return ``'mock'``."""
        return "mock"

    @override
    async def _send_request(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> _RawResponse:
        """Satisfy the ABC contract.

        Not called when ``query()`` is overridden, but required by
        the abstract base class.
        """
        return _RawResponse(
            content=self._default,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=0.1,
        )

    @override
    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
    ) -> LLMResponse:
        """Return a deterministic response based on the configured mode.

        Bypasses rate limiting, retry, and budget (meaningless for a
        mock).  Directly constructs an :class:`LLMResponse`.

        Args:
            prompt: The user prompt (ignored for response selection).
            question_id: Back-reference for response-map lookup.
            system: Optional system message (ignored).

        Returns:
            An :class:`LLMResponse` with deterministic content.
        """
        if self._response_map is not None and question_id in self._response_map:
            content = self._response_map[question_id]
        elif self._response_list is not None:
            content = self._response_list[self._call_count % len(self._response_list)]
        else:
            content = self._default
        self._call_count += 1
        return LLMResponse(
            question_id=question_id,
            raw_output=content,
            extracted_answer="",
            model=self._model,
            provider=self.provider_name,
            latency_ms=0.1,
            prompt_tokens=10,
            completion_tokens=5,
        )
