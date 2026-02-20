"""Perturbation generators for input variation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_consistency.types import MCQuestion, PerturbationType, PerturbedVariant

# ---------------------------------------------------------------------------
# BasePerturbation ABC
# ---------------------------------------------------------------------------


class BasePerturbation(ABC):
    """Abstract base class for perturbation generators.

    Subclasses must implement:
    - ``perturbation_type`` (property): returns the ``PerturbationType`` enum value
    - ``generate_variants``: produces perturbed variants of a question
    """

    @property
    @abstractmethod
    def perturbation_type(self) -> PerturbationType:
        """The type of perturbation this generator applies."""
        ...

    @abstractmethod
    def generate_variants(
        self,
        question: MCQuestion,
        *,
        seed: int = 0,
        n: int | None = None,
    ) -> tuple[PerturbedVariant, ...]:
        """Generate perturbed variants of the given question.

        Args:
            question: The original multiple-choice question to perturb.
            seed: Random seed for reproducibility.
            n: Number of variants to generate.  If ``None``, the
                implementation chooses a sensible default.

        Returns:
            A tuple of ``PerturbedVariant`` instances.
        """
        ...


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BasePerturbation] = {}


def register(
    name: str,
    perturbation: BasePerturbation,
    *,
    force: bool = False,
) -> None:
    """Register a perturbation instance by name.

    Args:
        name: Unique string key for lookup via :func:`get`.
        perturbation: A ``BasePerturbation`` instance.
        force: If ``True``, overwrite an existing registration.

    Raises:
        TypeError: If *perturbation* is not a ``BasePerturbation`` instance.
        ValueError: If *name* is already registered and *force* is ``False``.
    """
    if not isinstance(perturbation, BasePerturbation):
        msg = (
            f"Expected a BasePerturbation instance, "
            f"got {type(perturbation).__name__}"
        )
        raise TypeError(msg)
    if name in _REGISTRY and not force:
        msg = f"Perturbation '{name}' is already registered"
        raise ValueError(msg)
    _REGISTRY[name] = perturbation


def get(name: str) -> BasePerturbation:
    """Retrieve a registered perturbation by name.

    Args:
        name: The registered string key.

    Returns:
        The ``BasePerturbation`` instance registered under *name*.

    Raises:
        KeyError: If *name* is not registered.  The error message
            includes the list of available perturbation names.
    """
    if name not in _REGISTRY:
        available = list_registered()
        msg = (
            f"Unknown perturbation '{name}'. "
            f"Available: {available}"
        )
        raise KeyError(msg)
    return _REGISTRY[name]


def list_registered() -> list[str]:
    """Return a sorted list of all registered perturbation names.

    Returns:
        Alphabetically sorted list of registered name strings.
    """
    return sorted(_REGISTRY.keys())


def _register_builtins() -> None:
    """Register built-in perturbation instances.

    Currently a no-op -- concrete perturbation classes will be registered
    here in plans 02-03 once they exist.
    """


def _reset_registry() -> None:
    """Clear the registry and re-register built-in perturbations.

    Intended for test isolation.  After clearing, calls
    :func:`_register_builtins` to restore the default set of
    perturbation generators.
    """
    _REGISTRY.clear()
    _register_builtins()
