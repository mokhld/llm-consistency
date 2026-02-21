"""Perturbation generators for input variation."""

from __future__ import annotations

import itertools
import random
import string
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_consistency.types import MCOption, PerturbationType, PerturbedVariant

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_consistency.types import MCQuestion

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
# OptionReorderPerturbation
# ---------------------------------------------------------------------------


class OptionReorderPerturbation(BasePerturbation):
    """Generate variants by reordering MC options with label reassignment.

    Produces all non-identity permutations of the option list.  Labels
    are reassigned to match position (A=first, B=second, ...) while
    ``is_correct`` follows the text content.

    When *n* is specified and smaller than the total number of
    non-identity permutations, a deterministic sample of size *n* is
    drawn using ``random.Random(seed)``.  If *n* is ``None`` or
    exceeds the available count, all non-identity permutations are
    returned.
    """

    @property
    def perturbation_type(self) -> PerturbationType:
        """Return :attr:`PerturbationType.OPTION_REORDER`."""
        return PerturbationType.OPTION_REORDER

    def generate_variants(
        self,
        question: MCQuestion,
        *,
        seed: int = 0,
        n: int | None = None,
    ) -> tuple[PerturbedVariant, ...]:
        """Generate reordered-option variants for *question*.

        Args:
            question: The original multiple-choice question to perturb.
            seed: Random seed for deterministic N-sampling.
            n: Number of variants to generate.  ``None`` returns all
                non-identity permutations.

        Returns:
            A tuple of ``PerturbedVariant`` instances.
        """
        num_options = len(question.options)
        labels = string.ascii_uppercase[:num_options]
        identity = tuple(range(num_options))

        # All permutations minus the identity
        all_perms = [p for p in itertools.permutations(identity) if p != identity]

        # N-sampling
        if n is not None and n < len(all_perms):
            selected = random.Random(seed).sample(all_perms, k=n)
        else:
            selected = all_perms

        variants: list[PerturbedVariant] = []
        for idx, perm in enumerate(selected):
            new_options = tuple(
                MCOption(
                    label=labels[new_pos],
                    text=question.options[orig_idx].text,
                    is_correct=question.options[orig_idx].is_correct,
                )
                for new_pos, orig_idx in enumerate(perm)
            )
            variants.append(
                PerturbedVariant(
                    original_question_id=question.id,
                    perturbation_type=PerturbationType.OPTION_REORDER,
                    seed=seed,
                    variant_index=idx,
                    stem=question.stem,
                    options=new_options,
                )
            )

        return tuple(variants)


# ---------------------------------------------------------------------------
# Format-change templates
# ---------------------------------------------------------------------------


def _fmt_dot(stem: str, options: tuple[MCOption, ...]) -> str:
    """Dot-separated: ``A. Paris``."""
    block = "\n".join(f"{o.label}. {o.text}" for o in options)
    return f"{stem}\n{block}"


def _fmt_paren(stem: str, options: tuple[MCOption, ...]) -> str:
    """Parenthetical: ``A) Paris``."""
    block = "\n".join(f"{o.label}) {o.text}" for o in options)
    return f"{stem}\n{block}"


def _fmt_enclosed(stem: str, options: tuple[MCOption, ...]) -> str:
    """Enclosed parenthetical: ``(A) Paris``."""
    block = "\n".join(f"({o.label}) {o.text}" for o in options)
    return f"{stem}\n{block}"


def _fmt_numbered(stem: str, options: tuple[MCOption, ...]) -> str:
    """Numbered (no letter labels): ``1. Paris``."""
    block = "\n".join(f"{i + 1}. {o.text}" for i, o in enumerate(options))
    return f"{stem}\n{block}"


def _fmt_markdown(stem: str, options: tuple[MCOption, ...]) -> str:
    """Markdown bullet: ``- A: Paris``."""
    block = "\n".join(f"- {o.label}: {o.text}" for o in options)
    return f"{stem}\n\n{block}"


def _fmt_verbose(stem: str, options: tuple[MCOption, ...]) -> str:
    """Verbose uppercase: ``OPTION A: Paris``."""
    block = "\n".join(f"OPTION {o.label}: {o.text}" for o in options)
    return f"{stem}\n{block}"


def _fmt_tabular(stem: str, options: tuple[MCOption, ...]) -> str:
    """Tabular aligned: ``  A  Paris``."""
    block = "\n".join(f"  {o.label}  {o.text}" for o in options)
    return f"{stem}\n{block}"


_TEMPLATES: tuple[Callable[[str, tuple[MCOption, ...]], str], ...] = (
    _fmt_dot,
    _fmt_paren,
    _fmt_enclosed,
    _fmt_numbered,
    _fmt_markdown,
    _fmt_verbose,
    _fmt_tabular,
)


# ---------------------------------------------------------------------------
# FormatChangePerturbation
# ---------------------------------------------------------------------------


class FormatChangePerturbation(BasePerturbation):
    """Generate variants by applying different formatting templates.

    Produces one variant per template from :data:`_TEMPLATES` (minimum 6).
    Each template renders the full question (stem + options) into the
    variant's ``stem`` field, with ``options=None`` because the options
    have been rendered into a presentation string.

    When *n* is specified and smaller than the total number of templates,
    a deterministic sample of size *n* is drawn using ``random.Random(seed)``.
    If *n* is ``None`` or exceeds the available count, all templates are used.
    """

    @property
    def perturbation_type(self) -> PerturbationType:
        """Return :attr:`PerturbationType.FORMAT_CHANGE`."""
        return PerturbationType.FORMAT_CHANGE

    def generate_variants(
        self,
        question: MCQuestion,
        *,
        seed: int = 0,
        n: int | None = None,
    ) -> tuple[PerturbedVariant, ...]:
        """Generate format-changed variants for *question*.

        Args:
            question: The original multiple-choice question to perturb.
            seed: Random seed for deterministic N-sampling.
            n: Number of variants to generate.  ``None`` returns all
                template variants.

        Returns:
            A tuple of ``PerturbedVariant`` instances.
        """
        all_rendered = [tmpl(question.stem, question.options) for tmpl in _TEMPLATES]

        # N-sampling
        if n is not None and n < len(all_rendered):
            indices = list(range(len(all_rendered)))
            selected_indices = random.Random(seed).sample(indices, k=n)
            selected = [all_rendered[i] for i in selected_indices]
        else:
            selected = all_rendered

        return tuple(
            PerturbedVariant(
                original_question_id=question.id,
                perturbation_type=PerturbationType.FORMAT_CHANGE,
                seed=seed,
                variant_index=idx,
                stem=rendered,
                options=None,
            )
            for idx, rendered in enumerate(selected)
        )


# ---------------------------------------------------------------------------
# Separator variants
# ---------------------------------------------------------------------------

_SEPARATORS: tuple[str, ...] = (
    "\n",  # newline
    "\n\n",  # double newline
    "; ",  # semicolon
    ", ",  # comma
    " | ",  # pipe
    " - ",  # dash
    "\t",  # tab
    " / ",  # slash
)


# ---------------------------------------------------------------------------
# SeparatorChangePerturbation
# ---------------------------------------------------------------------------


class SeparatorChangePerturbation(BasePerturbation):
    """Generate variants by varying the delimiter between options.

    Uses a default rendering format (``A. text``) and varies only the
    separator between options.  Produces one variant per separator from
    :data:`_SEPARATORS` (minimum 8).

    Each variant renders the full question (stem + separated options) into
    the variant's ``stem`` field, with ``options=None``.

    When *n* is specified and smaller than the total number of separators,
    a deterministic sample of size *n* is drawn using ``random.Random(seed)``.
    """

    @property
    def perturbation_type(self) -> PerturbationType:
        """Return :attr:`PerturbationType.SEPARATOR_CHANGE`."""
        return PerturbationType.SEPARATOR_CHANGE

    def generate_variants(
        self,
        question: MCQuestion,
        *,
        seed: int = 0,
        n: int | None = None,
    ) -> tuple[PerturbedVariant, ...]:
        """Generate separator-changed variants for *question*.

        Args:
            question: The original multiple-choice question to perturb.
            seed: Random seed for deterministic N-sampling.
            n: Number of variants to generate.  ``None`` returns all
                separator variants.

        Returns:
            A tuple of ``PerturbedVariant`` instances.
        """
        option_parts = [f"{o.label}. {o.text}" for o in question.options]

        all_rendered = [
            f"{question.stem}\n{sep.join(option_parts)}" for sep in _SEPARATORS
        ]

        # N-sampling
        if n is not None and n < len(all_rendered):
            indices = list(range(len(all_rendered)))
            selected_indices = random.Random(seed).sample(indices, k=n)
            selected = [all_rendered[i] for i in selected_indices]
        else:
            selected = all_rendered

        return tuple(
            PerturbedVariant(
                original_question_id=question.id,
                perturbation_type=PerturbationType.SEPARATOR_CHANGE,
                seed=seed,
                variant_index=idx,
                stem=rendered,
                options=None,
            )
            for idx, rendered in enumerate(selected)
        )


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
        msg = f"Expected a BasePerturbation instance, got {type(perturbation).__name__}"
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
        msg = f"Unknown perturbation '{name}'. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[name]


def list_registered() -> list[str]:
    """Return a sorted list of all registered perturbation names.

    Returns:
        Alphabetically sorted list of registered name strings.
    """
    return sorted(_REGISTRY.keys())


def _register_builtins() -> None:
    """Register built-in perturbation instances."""
    register(PerturbationType.OPTION_REORDER.value, OptionReorderPerturbation())
    register(PerturbationType.FORMAT_CHANGE.value, FormatChangePerturbation())
    register(PerturbationType.SEPARATOR_CHANGE.value, SeparatorChangePerturbation())


def _reset_registry() -> None:
    """Clear the registry and re-register built-in perturbations.

    Intended for test isolation.  After clearing, calls
    :func:`_register_builtins` to restore the default set of
    perturbation generators.
    """
    _REGISTRY.clear()
    _register_builtins()


# Auto-register built-in perturbations on import
_register_builtins()
