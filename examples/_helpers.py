"""Shared helpers for the example scripts.

`BatchRunner` queries the provider with variant IDs like ``"q1_v0"``,
``"q1_v1"`` — not the bare ``"q1"``. The stock `MockLLMProvider`
response-map mode is exact-match, so a map keyed by base IDs would
miss every variant and fall back to the default response.

`BaseIdMockProvider` strips the ``_vN`` suffix before lookup so the
examples can stay terse: ``{"q1": "B", "q2": "C", ...}``.
"""

from __future__ import annotations

from llm_consistency import LLMResponse
from llm_consistency.providers._mock import MockLLMProvider


class BaseIdMockProvider(MockLLMProvider):
    """Mock provider that looks up responses by the *base* question ID."""

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
    ) -> LLMResponse:
        base_qid = question_id.split("_v")[0]
        return await super().query(prompt, base_qid, system=system)
