"""Command-line interface for llm-consistency."""

from __future__ import annotations

import asyncio
import functools
import re
from typing import TYPE_CHECKING, Any

import click

from llm_consistency._config_loader import (
    check_config_keys,
    load_config_file,
    run_defaults_from_config,
)
from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.datasets import MCDataset
from llm_consistency.metrics import compare_mca_paired, core_index, mca
from llm_consistency.perturbations import (
    list_registered as list_registered_perturbations,
)
from llm_consistency.providers import get_provider
from llm_consistency.providers._cost import estimate_cost
from llm_consistency.reports import ConsoleReporter, export_json
from llm_consistency.runners import BatchRunner, CIRunner
from llm_consistency.runners._pipeline import (
    generate_variants_for_question,
    presented_options,
    render_prompt,
)
from llm_consistency.scoring import get_scorer
from llm_consistency.types import EvaluationConfig, MCQuestion, PerturbationType

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from llm_consistency.providers import BaseLLMProvider
    from llm_consistency.types import EvaluationReport

# Keys accepted in a ``compare`` config file.
_COMPARE_KEYS = frozenset(
    {
        "models",
        "dataset",
        "perturbations",
        "num_variants",
        "concurrency",
        "scorer",
        "seed",
        "mca_threshold",
        "min_mca",
        "core_threshold",
        "max_budget_usd",
        "rpm",
        "prompt_template",
        "system_prompt",
        "temperature",
        "max_tokens",
        "generation_seed",
    }
)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that catches domain exceptions and re-raises as ClickException.

    Converts domain-specific exceptions into user-friendly CLI error
    messages, preventing raw tracebacks from reaching the terminal.

    Args:
        func: The Click command function to wrap.

    Returns:
        Wrapped function with error handling.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except SystemExit:
            raise
        except ValidationError as exc:
            raise click.ClickException(f"Validation error: {exc}") from None
        except LLMConsistencyError as exc:
            raise click.ClickException(str(exc)) from None
        except FileNotFoundError as exc:
            raise click.ClickException(f"File not found: {exc}") from None
        except KeyError as exc:
            raise click.ClickException(
                f"Configuration error: missing key {exc}"
            ) from None
        except TypeError as exc:
            raise click.ClickException(f"Invalid data format: {exc}") from None
        except ImportError as exc:
            # Raised by providers whose optional SDK is not installed.
            raise click.ClickException(str(exc)) from None
        except ValueError as exc:
            # Unknown provider names and malformed dataset files.
            raise click.ClickException(str(exc)) from None

    return wrapper


def _load_config_callback(
    ctx: click.Context,
    _param: click.Parameter,
    value: str | None,
) -> None:
    """Click callback to load config file and set as default_map.

    Loads a YAML or TOML config file and merges values into the Click
    context's ``default_map``, allowing config file values to serve
    as defaults that CLI flags can override. Settings may sit under a
    ``run`` section or at the top level; unknown keys are rejected.

    Args:
        ctx: Click context.
        _param: Click parameter (unused).
        value: Path to the config file, or ``None``.
    """
    if value is None:
        return
    from pathlib import Path  # noqa: PLC0415

    param_names = [p.name for p in ctx.command.params if p.expose_value and p.name]
    try:
        defaults = run_defaults_from_config(load_config_file(Path(value)), param_names)
    except ValidationError as exc:
        raise click.ClickException(f"Config error: {exc}") from None
    ctx.default_map = ctx.default_map or {}
    ctx.default_map.update(defaults)


def _parse_perturbation_types(
    names: tuple[str, ...],
) -> tuple[PerturbationType, ...]:
    """Parse perturbation type names into enum values.

    Accepts both lowercase (``option_reorder``) and uppercase
    (``OPTION_REORDER``) names. Raises a ClickException if a name
    cannot be resolved or has no registered generator.

    Args:
        names: Tuple of perturbation type name strings.

    Returns:
        Tuple of resolved PerturbationType enum members.

    Raises:
        click.ClickException: If a name is not a valid perturbation type,
            or no perturbation is registered under its value.
    """
    registered = list_registered_perturbations()
    result: list[PerturbationType] = []
    for name in names:
        try:
            pt = PerturbationType[name.upper()]
        except KeyError:
            try:
                pt = PerturbationType(name.lower())
            except ValueError:
                msg = (
                    f"Unknown perturbation type: {name!r}. "
                    f"Registered perturbations: {registered}"
                )
                raise click.ClickException(msg) from None
        if pt.value not in registered:
            msg = (
                f"Perturbation type {pt.value!r} has no registered generator. "
                f"Registered perturbations: {registered}"
            )
            raise click.ClickException(msg)
        result.append(pt)
    return tuple(result)


def _dry_run_report(
    config: EvaluationConfig,
    dataset: MCDataset,
    provider: Any,
    scorer: Any,
    *,
    seed: int,
) -> None:
    """Print a dry-run summary without spending provider tokens.

    Validates that variants can be generated and a prompt rendered for
    the first MCQuestion in the dataset (so users learn about pipeline
    wiring bugs before paying for them), then prints a summary with an
    estimated cost for known models. The call count is the number of
    variants the run will generate: each perturbation type yields at
    most ``num_variants`` per question, and fewer when it has fewer
    distinct variants (for example, a 2-option question has one
    reordering). Warns when the estimate exceeds ``max_budget_usd``.
    Ends with the generation settings, the system prompt and the full
    prompt for the first variant, as they will be sent.
    """
    mc_questions = [q for q in dataset if isinstance(q, MCQuestion)]
    if not mc_questions:
        msg = "Dataset contains no MCQuestion items; nothing to evaluate."
        raise click.ClickException(msg)

    sample = mc_questions[0]
    variants = generate_variants_for_question(sample, config, seed)
    if not variants:
        msg = "Pipeline produced zero variants for the first question."
        raise click.ClickException(msg)
    labels = [o.label for o in presented_options(variants[0], sample)]
    sample_prompt = render_prompt(variants[0], config.prompt_template, labels)

    num_questions = len(mc_questions)
    num_calls = sum(
        len(generate_variants_for_question(q, config, seed)) for q in mc_questions
    )
    estimated_usd = estimate_cost(config.model, num_calls)
    cost_str = (
        f"~${estimated_usd:.4f}" if estimated_usd > 0 else "unknown (model not priced)"
    )

    click.echo("Dry run — no provider calls made.")
    click.echo(f"  model:               {config.model}")
    click.echo(f"  provider:            {config.provider}")
    click.echo(f"  scorer:              {config.scorer}")
    click.echo(
        "  perturbations:       "
        + ", ".join(pt.value for pt in config.perturbation_types)
    )
    click.echo(f"  questions (MC):      {num_questions}")
    click.echo(f"  variants per type:   {config.num_variants} (max)")
    click.echo(f"  total provider calls:{num_calls}")
    click.echo(f"  estimated cost:      {cost_str}")
    if config.max_budget_usd is not None:
        click.echo(f"  budget:              ${config.max_budget_usd:.4f}")
    click.echo(f"  provider class:      {type(provider).__name__}")
    click.echo(f"  scorer class:        {type(scorer).__name__}")
    click.echo(f"  temperature:         {_setting(config.temperature)}")
    click.echo(f"  max tokens:          {_setting(config.max_tokens)}")
    click.echo(f"  generation seed:     {_setting(config.generation_seed)}")
    if config.max_budget_usd is not None and estimated_usd > config.max_budget_usd:
        click.echo(
            f"Warning: the estimated cost (~${estimated_usd:.4f}) exceeds "
            f"--max-budget-usd (${config.max_budget_usd:.4f}). The run will "
            "stop with an error when the budget is reached."
        )
    click.echo("")
    if config.system_prompt is None:
        click.echo("System prompt: none")
    else:
        click.echo("System prompt:")
        click.echo("  " + config.system_prompt.replace("\n", "\n  "))
    click.echo("")
    click.echo("Sample prompt (variant 0 of first question):")
    click.echo("  " + sample_prompt.replace("\n", "\n  "))


def _setting(value: float | None) -> str:
    """Format a generation setting for the dry-run summary."""
    return "not sent (provider default)" if value is None else str(value)


def _export_report(
    report: Any,
    path: Path,
    *,
    metadata: Any = None,
) -> None:
    """Route an :class:`EvaluationReport` to the right exporter by extension.

    Recognises ``.csv``, ``.md``/``.markdown``, and ``.html``/``.htm``;
    everything else falls back to JSON, preserving the historical default.
    """
    from llm_consistency.reports import (  # noqa: PLC0415
        export_csv,
        export_html,
        export_markdown,
    )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        export_csv(report, path)
    elif suffix in {".md", ".markdown"}:
        export_markdown(report, path, metadata=metadata)
    elif suffix in {".html", ".htm"}:
        export_html(report, path, metadata=metadata)
    else:
        export_json(report, path, metadata=metadata)


@click.group(invoke_without_command=True)
@click.version_option(package_name="llm-consistency")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """LLM Consistency evaluation framework."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option("--model", "-m", required=True, help="LLM model identifier")
@click.option(
    "--provider",
    "-p",
    required=True,
    help="Provider name (openai, anthropic, ollama, litellm, mock)",
)
@click.option(
    "--dataset",
    "-d",
    "dataset_path",
    required=True,
    type=click.Path(exists=True),
    help="Dataset file path",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    is_eager=True,
    callback=_load_config_callback,
    expose_value=False,
    help="Config file (YAML/TOML)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help=(
        "Report output path. Format follows the extension: .csv, .md, "
        ".html, otherwise JSON"
    ),
)
@click.option(
    "--perturbations",
    multiple=True,
    default=("option_reorder",),
    help="Perturbation types to apply",
)
@click.option(
    "--num-variants",
    type=click.IntRange(min=1),
    default=5,
    help="Variants per question (>=1)",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=10,
    help="Max concurrent API calls (>=1)",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    help="Random seed (>=0)",
)
@click.option("--scorer", default="exact_match", help="Scoring method")
@click.option(
    "--mca-threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1.0,
    help=(
        "Consistency level c for MCA: a question passes when RC_correct >= c (0.0-1.0)"
    ),
)
@click.option(
    "--min-mca",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1.0,
    help=(
        "Minimum MCA at --mca-threshold for a pass (0.0-1.0). The default "
        "1.0 requires every question to pass"
    ),
)
@click.option(
    "--core-threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help="CORE threshold for pass/fail (0.0-1.0)",
)
@click.option(
    "--max-budget-usd",
    type=click.FloatRange(min=0.0),
    default=None,
    help=(
        "Spending cap in USD (>=0). The run stops with an error before a "
        "request that would exceed it"
    ),
)
@click.option(
    "--rpm",
    type=click.IntRange(min=1),
    default=60,
    help="Provider rate limit in requests per minute (>=1)",
)
@click.option(
    "--prompt-template",
    default=None,
    help=(
        "Prompt template. {question} (required) is replaced by the question "
        "and options, {labels} by the option labels shown. Default: the "
        'question plus an instruction to answer on the first line as "Answer: X"'
    ),
)
@click.option(
    "--system-prompt",
    default=None,
    help="System prompt sent with every request",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0, max=2.0),
    default=None,
    help=(
        "Sampling temperature (0.0-2.0). Not sent when unset, so the "
        "provider default applies"
    ),
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum output tokens per response (>=1). Not sent when unset",
)
@click.option(
    "--generation-seed",
    type=int,
    default=None,
    help=(
        "Provider sampling seed, separate from --seed. Not sent when unset; "
        "Anthropic ignores it"
    ),
)
@click.option(
    "--ci",
    is_flag=True,
    help=(
        "CI mode: skip the console summary and exit 1 when a threshold "
        "fails or any variant failed with a provider error"
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=(
        "Validate dataset, config, and provider without spending tokens. "
        "Renders one prompt for the first question to prove the pipeline "
        "wires up end-to-end."
    ),
)
@_handle_errors
def run(
    model: str,
    provider: str,
    dataset_path: str,
    output: str | None,
    perturbations: tuple[str, ...],
    num_variants: int,
    concurrency: int,
    seed: int,
    scorer: str,
    mca_threshold: float,
    min_mca: float,
    core_threshold: float | None,
    max_budget_usd: float | None,
    rpm: int,
    prompt_template: str | None,
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int | None,
    generation_seed: int | None,
    ci: bool,
    dry_run: bool,
) -> None:
    """Execute an evaluation run."""
    from pathlib import Path  # noqa: PLC0415

    pert_types = _parse_perturbation_types(perturbations)

    config = EvaluationConfig(
        model=model,
        provider=provider,
        perturbation_types=pert_types,
        scorer=scorer,
        num_variants=num_variants,
        concurrency=concurrency,
        max_budget_usd=max_budget_usd,
        mca_threshold=mca_threshold,
        min_mca=min_mca,
        core_threshold=core_threshold,
        ci_mode=ci,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        generation_seed=generation_seed,
    )

    prov = get_provider(
        provider,
        model=model,
        max_budget_usd=max_budget_usd,
        requests_per_minute=rpm,
    )
    ds = MCDataset.load(dataset_path)
    scoring = get_scorer(scorer)

    if dry_run:
        _dry_run_report(config, ds, prov, scoring, seed=seed)
        return

    if ci:
        # CIRunner logs failed checks via the standard logging module
        # (visible by default thanks to Python's lastResort handler) and
        # exposes them on .failures for programmatic access. The report
        # is written before exiting so a failed build still has it.
        ci_runner = CIRunner()
        exit_code = asyncio.run(ci_runner.run(ds, config, prov, scoring, seed=seed))
        if output and ci_runner.last_report is not None:
            _export_report(
                ci_runner.last_report,
                Path(output),
                metadata=ci_runner.last_metadata,
            )
        raise SystemExit(exit_code)

    runner = BatchRunner()
    report = asyncio.run(runner.run(ds, config, prov, scoring, seed=seed))
    ConsoleReporter().display(report, threshold=mca_threshold)

    if output:
        _export_report(report, Path(output), metadata=runner.last_metadata)


@cli.group()
def perturbations() -> None:
    """Manage perturbation types."""


@perturbations.command("list")
def perturbations_list() -> None:
    """Show all registered perturbation types."""
    from llm_consistency.perturbations import list_registered  # noqa: PLC0415

    names = list_registered()
    if not names:
        click.echo("No perturbations registered.")
        return
    click.echo("Available perturbation types:")
    for name in names:
        click.echo(f"  - {name}")


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Config file with models list (YAML/TOML)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for per-model reports",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "md", "html"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Per-model report file format when --output is set.",
)
@_handle_errors
def compare(config: str, output: str | None, output_format: str) -> None:
    """Compare multiple models on the same evaluation."""
    from pathlib import Path  # noqa: PLC0415

    try:
        data = load_config_file(Path(config))
        check_config_keys(data, _COMPARE_KEYS)
    except ValidationError as exc:
        raise click.ClickException(f"Config error: {exc}") from None

    # Validate models key
    models = data.get("models")
    if not models or not isinstance(models, list):
        msg = "Config must contain 'models' list with 'model' and 'provider' keys"
        raise click.ClickException(msg)
    for entry in models:
        if (
            not isinstance(entry, dict)
            or "model" not in entry
            or "provider" not in entry
        ):
            msg = "Config must contain 'models' list with 'model' and 'provider' keys"
            raise click.ClickException(msg)

    # Extract shared config
    dataset_path = data.get("dataset")
    if not dataset_path:
        msg = "Config must contain 'dataset' path"
        raise click.ClickException(msg)
    seed: int = int(data.get("seed", 42))
    mca_threshold: float = float(data.get("mca_threshold", 1.0))

    # Load dataset once
    ds = MCDataset.load(dataset_path)
    runs = _compare_runs(data, models)
    scoring = get_scorer(runs[0][0].scorer)

    out_dir: Path | None = None
    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
    stems = _report_file_stems([cfg.model for cfg, _ in runs])

    # Run per model sequentially, exporting each report as soon as it is
    # ready so a later failure does not lose earlier results.
    reports: list[EvaluationReport] = []
    for (eval_config, prov), stem in zip(runs, stems, strict=True):
        runner = BatchRunner()
        report = asyncio.run(runner.run(ds, eval_config, prov, scoring, seed=seed))
        reports.append(report)

        click.echo(f"\nModel: {eval_config.model} ({eval_config.provider})")
        ConsoleReporter().display(report)
        if out_dir is not None:
            path = out_dir / f"{stem}.{output_format.lower()}"
            _export_report(report, path, metadata=runner.last_metadata)
            click.echo(f"Report written to {path}")

    _print_comparison(reports, mca_threshold)


def _compare_runs(
    data: dict[str, Any],
    models: list[dict[str, Any]],
) -> list[tuple[EvaluationConfig, BaseLLMProvider]]:
    """Build the config and provider for each model in a compare config.

    Everything is built before the first paid call, so an unknown
    provider, a missing SDK or an unpriced model fails fast.
    """
    pert_names = tuple(data.get("perturbations", ["option_reorder"]))
    pert_types = _parse_perturbation_types(pert_names)
    num_variants: int = int(data.get("num_variants", 5))
    concurrency: int = int(data.get("concurrency", 10))
    scorer_name: str = str(data.get("scorer", "exact_match"))
    mca_threshold: float = float(data.get("mca_threshold", 1.0))
    min_mca: float = float(data.get("min_mca", 1.0))
    core_threshold_raw = data.get("core_threshold")
    core_threshold: float | None = (
        float(core_threshold_raw) if core_threshold_raw is not None else None
    )
    max_budget_raw = data.get("max_budget_usd")
    max_budget_usd: float | None = (
        float(max_budget_raw) if max_budget_raw is not None else None
    )
    rpm: int = int(data.get("rpm", 60))
    if rpm < 1:
        msg = "Config error: 'rpm' must be >= 1"
        raise click.ClickException(msg)
    template_raw = data.get("prompt_template")
    system_raw = data.get("system_prompt")
    temperature_raw = data.get("temperature")
    max_tokens_raw = data.get("max_tokens")
    generation_seed_raw = data.get("generation_seed")

    runs: list[tuple[EvaluationConfig, BaseLLMProvider]] = []
    for entry in models:
        model_name = str(entry["model"])
        provider_name = str(entry["provider"])
        eval_config = EvaluationConfig(
            model=model_name,
            provider=provider_name,
            perturbation_types=pert_types,
            scorer=scorer_name,
            num_variants=num_variants,
            concurrency=concurrency,
            max_budget_usd=max_budget_usd,
            mca_threshold=mca_threshold,
            min_mca=min_mca,
            core_threshold=core_threshold,
            prompt_template=str(template_raw) if template_raw is not None else None,
            system_prompt=str(system_raw) if system_raw is not None else None,
            temperature=(
                float(temperature_raw) if temperature_raw is not None else None
            ),
            max_tokens=int(max_tokens_raw) if max_tokens_raw is not None else None,
            generation_seed=(
                int(generation_seed_raw) if generation_seed_raw is not None else None
            ),
        )
        prov = get_provider(
            provider_name,
            model=model_name,
            max_budget_usd=max_budget_usd,
            requests_per_minute=rpm,
        )
        runs.append((eval_config, prov))
    return runs


def _report_file_stems(model_names: list[str]) -> list[str]:
    """Return a distinct, filesystem-safe file stem for each model name.

    Runs of characters outside ``[A-Za-z0-9._-]`` become ``_`` and
    leading dots are dropped, so ``openai/gpt-4o-mini`` becomes
    ``openai_gpt-4o-mini``. Stems that collide, compared
    case-insensitively for macOS and Windows, get ``-2``, ``-3``, ...
    suffixes.
    """
    stems: list[str] = []
    seen: set[str] = set()
    for name in model_names:
        base = _UNSAFE_FILENAME_CHARS.sub("_", name).lstrip(".") or "model"
        stem = base
        suffix = 1
        while stem.casefold() in seen:
            suffix += 1
            stem = f"{base}-{suffix}"
        seen.add(stem.casefold())
        stems.append(stem)
    return stems


def _print_comparison(reports: list[EvaluationReport], threshold: float) -> None:
    """Print one row of headline metrics per model.

    Columns are CORE, MCA at *threshold*, mean RC_correct, mean
    RC_agree, and the p-value of McNemar's exact test
    (:func:`compare_mca_paired`) against the first model.
    """
    mca_header = f"MCA({threshold:.2f})"
    width = max(len("Model"), *(len(r.config.model) for r in reports))
    header = (
        f"{'Model':<{width}}  {'CORE':>6}  {mca_header:>9}  "
        f"{'RC_correct':>10}  {'RC_agree':>8}  {'p-value':>7}"
    )
    click.echo(
        f"\nComparison (p-value: McNemar exact test on MCA({threshold:.2f}) "
        f"pass/fail against {reports[0].config.model})"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for i, report in enumerate(reports):
        if i == 0:
            p_value = "-"
        else:
            paired = compare_mca_paired(reports[0].results, report.results, threshold)
            p_value = f"{paired.p_value:.4f}"
        click.echo(
            f"{report.config.model:<{width}}  {core_index(report.results):>6.4f}  "
            f"{mca(report.results, threshold):>9.4f}  "
            f"{report.mean_rc_correct:>10.4f}  {report.mean_rc_agree:>8.4f}  "
            f"{p_value:>7}"
        )


@cli.group()
def dataset() -> None:
    """Dataset management commands."""


@dataset.command("validate")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--type",
    "dataset_type",
    type=click.Choice(["mc", "open-ended"]),
    default="mc",
    help="Dataset type",
)
@_handle_errors
def dataset_validate(path: str, dataset_type: str) -> None:
    """Validate a dataset file format."""
    from llm_consistency.datasets import (  # noqa: PLC0415
        OpenEndedDataset,
    )

    ds = MCDataset.load(path) if dataset_type == "mc" else OpenEndedDataset.load(path)
    click.echo(f"Valid {dataset_type} dataset: {len(ds)} questions")


def main() -> None:
    """Entry point for the CLI."""
    cli()
