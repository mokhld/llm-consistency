"""CustomDataset adapter wrapping user-supplied question sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_consistency.datasets._base import BaseDataset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from llm_consistency.types import MCQuestion, OpenEndedQuestion


class CustomDataset(BaseDataset):
    """Dataset adapter wrapping a user-supplied sequence of questions.

    Stores questions as an immutable tuple.  The :meth:`load` classmethod
    raises :class:`NotImplementedError` since custom datasets do not load
    from files.

    Args:
        questions: Sequence of MCQuestion or OpenEndedQuestion instances.
    """

    def __init__(
        self,
        questions: Sequence[MCQuestion | OpenEndedQuestion],
    ) -> None:
        self._questions: tuple[MCQuestion | OpenEndedQuestion, ...] = tuple(
            questions,
        )

    @property
    def questions(
        self,
    ) -> tuple[MCQuestion | OpenEndedQuestion, ...]:
        """Return the stored tuple of questions."""
        return self._questions

    @classmethod
    def load(cls, path: str | Path) -> CustomDataset:
        """Not supported -- raises NotImplementedError.

        Args:
            path: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        msg = "CustomDataset does not load from files"
        raise NotImplementedError(msg)
