"""Shared pipeline helpers for variant generation, prompt rendering, and QCR building.

Reusable across BatchRunner, streaming runner, and CI runner.
"""

from __future__ import annotations

import asyncio
import re
import warnings
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar

from llm_consistency import perturbations
from llm_consistency.metrics import build_question_consistency_result
from llm_consistency.providers._budget import BudgetExceededError
from llm_consistency.scoring import _extract_mc_answer
from llm_consistency.types import (
    GenerationParams,
    LLMResponse,
    MCQuestion,
    PresentedOption,
    QuestionConsistencyResult,
    ScoredResponse,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Awaitable, Iterable, Sequence

    from llm_consistency.providers._base import BaseLLMProvider
    from llm_consistency.scoring import BaseScorer
    from llm_consistency.types import EvaluationConfig, PerturbedVariant

_T = TypeVar("_T")

# Failed variants are recorded with ``scoring_method="error:<Type>: <message>"``.
_ERROR_PREFIX = "error:"
_MAX_ERROR_CHARS = 200
# API keys in the shapes the supported SDKs echo back ("sk-...", "sk-ant-...",
# masked "sk-abc***xyz") and bearer tokens.
_SECRET = re.compile(r"(?<![\w-])sk-[\w*-]{8,}|Bearer\s+\S+")

# Prompt template used when ``EvaluationConfig.prompt_template`` is None.
# ``{question}`` is the rendered question and options, ``{labels}`` the
# option labels shown, comma separated. It is modelled on the fixed
# first-line answer instruction of the CAT paper (section 4.1).
DEFAULT_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Answer with the label of the correct option. The first line of your "
    'response must be "Answer: X", where X is one of {labels}.'
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


def render_prompt(
    variant: PerturbedVariant,
    template: str | None = None,
    labels: Sequence[str] | None = None,
) -> str:
    """Render a perturbed variant into a prompt string.

    The question part is the stem plus options in ``A. text`` format for
    variants with ``options is not None`` (e.g., option_reorder), and
    ``variant.stem`` for variants with ``options is None`` (e.g.,
    format_change, separator_change), whose options are already embedded.
    It is then placed in *template*.

    Args:
        variant: The perturbed variant to render.
        template: Prompt template with a ``{question}`` placeholder and an
            optional ``{labels}`` placeholder.  ``None`` uses
            :data:`DEFAULT_PROMPT_TEMPLATE`; pass ``"{question}"`` for
            the question alone.
        labels: The option labels shown to the model, joined with ``", "``
            for ``{labels}``.  Defaults to the labels of
            ``variant.presented_options``, or of ``variant.options``.

    Returns:
        The prompt string ready to send to an LLM.
    """
    if variant.options is not None:
        lines = [f"{o.label}. {o.text}" for o in variant.options]
        question = f"{variant.stem}\n{chr(10).join(lines)}"
    else:
        question = variant.stem
    if labels is None:
        shown = variant.presented_options or variant.options or ()
        labels = [o.label for o in shown]
    if template is None:
        template = DEFAULT_PROMPT_TEMPLATE
    return template.format(question=question, labels=", ".join(labels))


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


def presented_options(
    variant: PerturbedVariant, question: MCQuestion
) -> tuple[PresentedOption, ...]:
    """Return the options the model saw in *variant*.

    Perturbations that do not set ``presented_options`` fall back to
    ``variant.options`` (what :func:`render_prompt` shows), or to the
    original options when the options are rendered into the stem.  The
    labels of those options are then treated as the original labels,
    which is only right if the perturbation kept every option under its
    original label.  A ``UserWarning`` is issued once per perturbation
    type when this fallback is used.

    Args:
        variant: The perturbed variant that was sent to the model.
        question: The original question the variant came from.

    Returns:
        The presented options, each with its label in the original question.
    """
    if variant.presented_options is not None:
        return variant.presented_options
    warnings.warn(
        f"Variants of perturbation type {variant.perturbation_type.value!r} do "
        "not set presented_options, so answers are scored as if every option "
        "kept its original label. Set PerturbedVariant.presented_options if "
        "the perturbation relabels or reorders options.",
        UserWarning,
        stacklevel=2,
    )
    options = variant.options if variant.options is not None else question.options
    return tuple(
        PresentedOption(
            label=o.label,
            text=o.text,
            is_correct=o.is_correct,
            original_label=o.label,
        )
        for o in options
    )


def query_kwargs(config: EvaluationConfig) -> dict[str, Any]:
    """Return the ``system`` and ``generation`` arguments for ``provider.query``.

    Each is left out when the config does not set it, so a provider that
    overrides ``query`` without these keyword arguments keeps working.
    """
    kwargs: dict[str, Any] = {}
    if config.system_prompt is not None:
        kwargs["system"] = config.system_prompt
    generation = GenerationParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.generation_seed,
    )
    if generation != GenerationParams():
        kwargs["generation"] = generation
    return kwargs


def describe_error(exc: BaseException) -> str:
    """Summarise a provider exception for storage in a report.

    API keys and bearer tokens are redacted and the text is capped at
    200 characters, because it is written to reports and checkpoints.
    """
    text = _SECRET.sub("[REDACTED]", f"{type(exc).__name__}: {exc}")
    if len(text) > _MAX_ERROR_CHARS:
        text = text[: _MAX_ERROR_CHARS - 3] + "..."
    return text


def warn_if_budget_not_enforced(
    config: EvaluationConfig, provider: BaseLLMProvider, logger: logging.Logger
) -> None:
    """Warn when the config sets a budget that the provider does not enforce.

    The budget is enforced by the provider, so ``config.max_budget_usd``
    only takes effect when it is also passed to :func:`get_provider`.
    """
    enforced = getattr(provider, "max_budget_usd", None)
    if config.max_budget_usd is not None and enforced is None:
        logger.warning(
            "EvaluationConfig.max_budget_usd=%s is not enforced: the provider "
            "was created without max_budget_usd. Pass max_budget_usd to "
            "get_provider() to cap spending.",
            config.max_budget_usd,
        )


def has_error_variants(qcr: QuestionConsistencyResult) -> bool:
    """Return True when any variant of *qcr* failed with a provider error."""
    return any(
        sr.scoring_method.startswith(_ERROR_PREFIX) for sr in qcr.scored_responses
    )


def error_summary(results: Iterable[QuestionConsistencyResult]) -> str | None:
    """Describe the failed variants in *results*, or return None if none failed.

    Returns:
        Text such as ``"3 variant(s) failed with provider errors and were
        scored as incorrect (RateLimitError: 2, TimeoutError: 1)"``.
    """
    types = Counter(
        sr.scoring_method[len(_ERROR_PREFIX) :].split(":", 1)[0]
        for qcr in results
        for sr in qcr.scored_responses
        if sr.scoring_method.startswith(_ERROR_PREFIX)
    )
    if not types:
        return None
    detail = ", ".join(f"{name}: {count}" for name, count in types.most_common())
    return (
        f"{types.total()} variant(s) failed with provider errors and were "
        f"scored as incorrect ({detail})"
    )


async def gather_or_cancel(aws: Iterable[Awaitable[_T]]) -> list[_T]:
    """Await *aws* concurrently and return their results in order.

    Unlike :func:`asyncio.gather`, when one of them raises the others
    are cancelled before the exception propagates, so no provider calls
    keep running after an aborted run.
    """
    tasks = [asyncio.ensure_future(aw) for aw in aws]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def process_question(
    question: MCQuestion,
    config: EvaluationConfig,
    provider: BaseLLMProvider,
    scorer: BaseScorer,
    semaphore: asyncio.Semaphore,
    seed: int,
) -> QuestionConsistencyResult:
    """Run one question through the pipeline: perturb, query, score, build QCR.

    Each response is scored against the options as presented in its own
    variant.  Agreement is keyed by the option's label in the original
    question, so two variants agree when the model picked the same option
    content, whatever label it was shown under.

    Provider errors are captured per variant: the variant is recorded as
    incorrect, with ``scoring_method`` ``"error:<Type>: <message>"`` and a
    distinct answer key so it never counts as agreement.
    :class:`~llm_consistency.providers.BudgetExceededError` is not captured;
    it propagates and cancels the question's other queries.

    Args:
        question: The MC question to evaluate.
        config: Evaluation configuration.
        provider: LLM provider.
        scorer: Response scorer.
        semaphore: Bounds concurrent provider calls across the whole run.
        seed: Random seed for variant generation.

    Returns:
        A QuestionConsistencyResult for this question.
    """
    variants = generate_variants_for_question(question, config, seed)
    variant_qids = [f"{question.id}_v{i}" for i in range(len(variants))]
    variant_options = [presented_options(v, question) for v in variants]
    kwargs = query_kwargs(config)

    async def _query(prompt: str, qid: str) -> tuple[str, str | None]:
        """Return ``(raw_output, error)`` for one variant."""
        async with semaphore:
            try:
                resp = await provider.query(prompt, qid, **kwargs)
            except BudgetExceededError:
                raise
            except Exception as exc:
                # Any other failure is recorded on this variant so one bad
                # call does not tear down the run.
                return ("", describe_error(exc))
        return (resp.raw_output, None)

    outputs = await gather_or_cancel(
        _query(render_prompt(v, config.prompt_template, [o.label for o in opts]), qid)
        for v, opts, qid in zip(variants, variant_options, variant_qids, strict=True)
    )

    scored_responses: list[ScoredResponse] = []
    variant_data: list[tuple[str, bool]] = []
    for variant, options, variant_qid, (raw_output, error) in zip(
        variants, variant_options, variant_qids, outputs, strict=True
    ):
        pt_value = variant.perturbation_type.value
        if error is not None:
            scored_responses.append(
                ScoredResponse(
                    question_id=variant_qid,
                    is_correct=False,
                    score=0.0,
                    scoring_method=f"{_ERROR_PREFIX}{error}",
                    perturbation_type=pt_value,
                )
            )
            variant_data.append((f"<error:{variant_qid}>", False))
            continue

        response = LLMResponse(
            question_id=variant_qid,
            raw_output=raw_output,
            extracted_answer="",
            model=config.model,
            provider=config.provider,
        )
        shown = MCQuestion(id=question.id, stem=question.stem, options=options)
        sr = replace(scorer.score(response, shown), perturbation_type=pt_value)
        scored_responses.append(sr)

        # Key the answer by the option's original label (option identity).
        original_label = {o.label: o.original_label for o in options}
        extracted = _extract_mc_answer(raw_output, frozenset(original_label))
        answer_key = (
            original_label[extracted] if extracted is not None else raw_output.strip()
        )
        variant_data.append((answer_key, sr.is_correct))

    return build_scored_qcr(question.id, variant_data, tuple(scored_responses))
