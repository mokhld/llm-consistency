"""Shared pipeline helpers for variant generation, prompt rendering, and QCR building.

Reusable across BatchRunner, streaming runner, and CI runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_consistency import perturbations
from llm_consistency.metrics import build_question_consistency_result
from llm_consistency.types import QuestionConsistencyResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_consistency.types import (
        EvaluationConfig,
        MCQuestion,
        PerturbedVariant,
        ScoredResponse,
    )


def generate_variants_for_question(
    question: MCQuestion,
    config: EvaluationConfig,
    seed: int = 42,
) -> list[PerturbedVariant]:
    """Generate perturbed variants for a question using configured perturbation types.

    For each perturbation type in ``config.perturbation_types``, retrieves
    the registered perturbation and generates ``config.num_variants`` variants.
    All variants are flattened into a single list.

    Args:
        question: The original MC question to perturb.
        config: Evaluation configuration with perturbation types and variant count.
        seed: Random seed for reproducible perturbation generation.

    Returns:
        Flat list of all perturbed variants across all perturbation types.
    """
    all_variants: list[PerturbedVariant] = []
    for pt in config.perturbation_types:
        perturbation = perturbations.get(pt.value)
        variants = perturbation.generate_variants(
            question, seed=seed, n=config.num_variants
        )
        all_variants.extend(variants)
    return all_variants


def render_prompt(variant: PerturbedVariant) -> str:
    """Render a perturbed variant into a prompt string.

    For variants with ``options is not None`` (e.g., option_reorder),
    renders the stem plus options in ``A. text`` format.  For variants
    with ``options is None`` (e.g., format_change, separator_change),
    returns ``variant.stem`` directly since options are already embedded.

    Args:
        variant: The perturbed variant to render.

    Returns:
        The prompt string ready to send to an LLM.
    """
    if variant.options is not None:
        lines = [f"{o.label}. {o.text}" for o in variant.options]
        return f"{variant.stem}\n{chr(10).join(lines)}"
    return variant.stem


def build_scored_qcr(
    question_id: str,
    variant_data: Sequence[tuple[str, bool]],
    scored_responses: tuple[ScoredResponse, ...],
) -> QuestionConsistencyResult:
    """Build a QCR with populated scored_responses field.

    Extends :func:`~llm_consistency.metrics.build_question_consistency_result`
    by attaching the ``scored_responses`` tuple to the result, rather than
    leaving it as the default empty tuple.

    Args:
        question_id: The question identifier.
        variant_data: Sequence of ``(extracted_answer, is_correct)`` pairs.
        scored_responses: Tuple of ScoredResponse instances to attach.

    Returns:
        A fully computed :class:`QuestionConsistencyResult` with
        ``scored_responses`` populated.
    """
    qcr = build_question_consistency_result(question_id, variant_data)
    return QuestionConsistencyResult(
        question_id=qcr.question_id,
        rc_correct=qcr.rc_correct,
        rc_agree=qcr.rc_agree,
        total_variants=qcr.total_variants,
        correct_count=qcr.correct_count,
        answer_distribution=qcr.answer_distribution,
        scored_responses=scored_responses,
    )
