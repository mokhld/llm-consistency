"""BaseDataset ABC with questions property, __len__, __iter__, load classmethod."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from llm_consistency.types import MCQuestion, OpenEndedQuestion


class BaseDataset(ABC):
    """Abstract base class for all dataset types.

    Subclasses must implement the :attr:`questions` property and the
    :meth:`load` classmethod.  Concrete ``__len__`` and ``__iter__``
    implementations delegate to :attr:`questions`.
    """

    @property
    @abstractmethod
    def questions(
        self,
    ) -> tuple[MCQuestion | OpenEndedQuestion, ...]:
        """Return the sequence of questions in this dataset."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseDataset:
        """Load a dataset from a file path.

        Args:
            path: File path (str or Path) to load from.

        Returns:
            A new dataset instance.
        """

    def __len__(self) -> int:
        """Return the number of questions in the dataset."""
        return len(self.questions)

    def __iter__(self) -> Iterator[MCQuestion | OpenEndedQuestion]:
        """Iterate over the questions in the dataset."""
        return iter(self.questions)
