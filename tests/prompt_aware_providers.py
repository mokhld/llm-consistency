"""Test providers that read the prompt, for semantic scoring tests.

``MockLLMProvider`` ignores the prompt, so it cannot catch scoring bugs that
depend on what the model was shown. These providers find each option in the
rendered prompt and answer with the label printed next to it:

* :class:`OracleProvider` answers the label of the correct option text, so a
  correct pipeline scores it RC_correct == RC_agree == 1.0 under every
  perturbation.
* :class:`ProseOracleProvider` gives the same answer as "Answer: X" after
  reasoning text that names a wrong option first.
* :class:`PositionBiasedProvider` answers the label of the first option
  listed, so its answers change content whenever the options are reordered.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.types import LLMResponse, MCQuestion

if TYPE_CHECKING:
    from collections.abc import Iterable

    from llm_consistency.types import GenerationParams


def _find_option(prompt: str, text: str) -> re.Match[str]:
    """Locate ``<label><punctuation> <text>`` in a rendered prompt.

    Covers every built-in layout: ``A. x``, ``A) x``, ``(A) x``, ``1. x``,
    ``- A: x``, ``OPTION A: x``, ``  A  x`` and the separator variants.
    """
    pattern = rf"(?<!\w)\(?([A-Z]|\d+)\)?[.:]?\s+{re.escape(text)}(?!\w)"
    match = re.search(pattern, prompt)
    if match is None:
        msg = f"option text {text!r} not found in prompt:\n{prompt}"
        raise AssertionError(msg)
    return match


class _PromptAwareProvider(MockLLMProvider):
    """Base for providers that answer from the options in the prompt."""

    def __init__(self, questions: Iterable[MCQuestion], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._questions = {q.id: q for q in questions}

    def _answer(self, prompt: str, question: MCQuestion) -> str:
        raise NotImplementedError

    async def query(
        self,
        prompt: str,
        question_id: str,
        *,
        system: str | None = None,
        generation: GenerationParams | None = None,
    ) -> LLMResponse:
        # Runners send variant ids of the form "<question id>_v<index>".
        question = self._questions[question_id.rsplit("_v", 1)[0]]
        return LLMResponse(
            question_id=question_id,
            raw_output=self._answer(prompt, question),
            extracted_answer="",
            model=self._model,
            provider=self.provider_name,
        )


class OracleProvider(_PromptAwareProvider):
    """Always answers the label shown next to the correct option text."""

    def _answer(self, prompt: str, question: MCQuestion) -> str:
        correct = next(o.text for o in question.options if o.is_correct)
        return _find_option(prompt, correct).group(1)


class ProseOracleProvider(OracleProvider):
    """Reasons about a wrong option first, then states "Answer: X" last."""

    def _answer(self, prompt: str, question: MCQuestion) -> str:
        wrong = next(o.text for o in question.options if not o.is_correct)
        wrong_label = _find_option(prompt, wrong).group(1)
        return (
            f"Option {wrong_label} looks plausible, but it does not fit the "
            f"question.\n\nAnswer: {super()._answer(prompt, question)}"
        )


class PositionBiasedProvider(_PromptAwareProvider):
    """Always answers the label of the first option listed in the prompt."""

    def _answer(self, prompt: str, question: MCQuestion) -> str:
        matches = [_find_option(prompt, o.text) for o in question.options]
        return min(matches, key=lambda m: m.start()).group(1)
