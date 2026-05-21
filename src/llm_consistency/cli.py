"""Command-line interface for llm-consistency."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any

import click

from llm_consistency._config_loader import load_config_file
from llm_consistency._exceptions import LLMConsistencyError, ValidationError
from llm_consistency.datasets import MCDataset
from llm_consistency.providers import get_provider
from llm_consistency.providers._cost import estimate_cost
from llm_consistency.reports import ConsoleReporter, export_json
from llm_consistency.runners import BatchRunner, CIRunner
from llm_consistency.runners._pipeline import (
    generate_variants_for_question,
    render_prompt,
)
from llm_consistency.scoring import get_scorer
from llm_consistency.types import EvaluationConfig, MCQuestion, PerturbationType

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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

    return wrapper


def _load_config_callback(
    ctx: click.Context,
    _param: click.Parameter,
    value: str | None,
) -> None:
    """Click callback to load config file and set as default_map.

    Loads a YAML or TOML config file and merges values into the Click
    context's ``default_map``, allowing config file values to serve
    as defaults that CLI flags can override.

    Args:
        ctx: Click context.
        _param: Click parameter (unused).
        value: Path to the config file, or ``None``.
    """
    if value is None:
        return
    from pathlib import Path  # noqa: PLC0415

    try:
        data = load_config_file(Path(value))
    except ValidationError as exc:
        raise click.ClickException(f"Config error: {exc}") from None
    ctx.default_map = ctx.default_map or {}
    ctx.default_map.update(data)


def _parse_perturbation_types(
    names: tuple[str, ...],
) -> tuple[PerturbationType, ...]:
    """Parse perturbation type names into enum values.

    Accepts both lowercase (``option_reorder``) and uppercase
    (``OPTION_REORDER``) names. Raises a ClickException if a name
    cannot be resolved.

    Args:
        names: Tuple of perturbation type name strings.

    Returns:
        Tuple of resolved PerturbationType enum members.

    Raises:
        click.ClickException: If a name is not a valid perturbation type.
    """
    result: list[PerturbationType] = []
    for name in names:
        try:
            result.append(PerturbationType[name.upper()])
        except KeyError:
            try:
                result.append(PerturbationType(name.lower()))
            except ValueError:
                valid = [pt.value for pt in PerturbationType]
                msg = f"Unknown perturbation type: {name!r}. Valid types: {valid}"
                raise click.ClickException(msg) from None
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
    estimated cost for known models.
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
    sample_prompt = render_prompt(variants[0])

    num_questions = len(mc_questions)
    num_calls = num_questions * config.num_variants
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
    click.echo(f"  variants per Q:      {config.num_variants}")
    click.echo(f"  total provider calls:{num_calls}")
    click.echo(f"  estimated cost:      {cost_str}")
    click.echo(f"  provider class:      {type(provider).__name__}")
    click.echo(f"  scorer class:        {type(scorer).__name__}")
    click.echo("")
    click.echo("Sample prompt (variant 0 of first question):")
    click.echo("  " + sample_prompt.replace("\n", "\n  "))


def _export_report(
    report: Any,
    path: Path,
    *,
    metadata: Any = None,
) -> None:
    """Route an :class:`EvaluationReport` to the right exporter by extension.

    Recognises ``.csv`` and ``.md``/``.markdown``; everything else falls
    back to JSON, preserving the historical default.
    """
    from llm_consistency.reports import (  # noqa: PLC0415
        export_csv,
        export_markdown,
    )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        export_csv(report, path)
    elif suffix in {".md", ".markdown"}:
        export_markdown(report, path, metadata=metadata)
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
    help="JSON output path",
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
    help="MCA threshold for pass/fail (0.0-1.0)",
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
    help="Budget ceiling in USD (>=0)",
)
@click.option("--ci", is_flag=True, help="CI mode: exit 1 on threshold failure")
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
    core_threshold: float | None,
    max_budget_usd: float | None,
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
        core_threshold=core_threshold,
        ci_mode=ci,
    )

    prov = get_provider(provider, model=model)
    ds = MCDataset.load(dataset_path)
    scoring = get_scorer(scorer)

    if dry_run:
        _dry_run_report(config, ds, prov, scoring, seed=seed)
        return

    if ci:
        # CIRunner logs failed thresholds via the standard logging module
        # (visible by default thanks to Python's lastResort handler) and
        # exposes them on .failures for programmatic access.
        ci_runner = CIRunner()
        exit_code = asyncio.run(ci_runner.run(ds, config, prov, scoring, seed=seed))
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
    type=click.Choice(["json", "csv", "md"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Per-model report file format when --output is set.",
)
@_handle_errors
def compare(config: str, output: str | None, output_format: str) -> None:
    """Compare multiple models on the same evaluation."""
    from pathlib import Path  # noqa: PLC0415

    data = load_config_file(Path(config))

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

    pert_names = tuple(data.get("perturbations", ["option_reorder"]))
    pert_types = _parse_perturbation_types(pert_names)
    num_variants: int = int(data.get("num_variants", 5))
    concurrency: int = int(data.get("concurrency", 10))
    scorer_name: str = str(data.get("scorer", "exact_match"))
    seed: int = int(data.get("seed", 42))
    mca_threshold: float = float(data.get("mca_threshold", 1.0))
    core_threshold_raw = data.get("core_threshold")
    core_threshold: float | None = (
        float(core_threshold_raw) if core_threshold_raw is not None else None
    )
    max_budget_raw = data.get("max_budget_usd")
    max_budget_usd: float | None = (
        float(max_budget_raw) if max_budget_raw is not None else None
    )

    # Load dataset once
    ds = MCDataset.load(dataset_path)
    scoring = get_scorer(scorer_name)

    # Run per model sequentially
    results_list: list[tuple[str, Any]] = []
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
            core_threshold=core_threshold,
        )

        prov = get_provider(provider_name, model=model_name)
        runner = BatchRunner()
        report = asyncio.run(runner.run(ds, eval_config, prov, scoring, seed=seed))
        results_list.append((model_name, report))

    # Display comparison summary
    click.echo("\n--- Comparison Results ---\n")
    for model_name, report in results_list:
        click.echo(f"Model: {model_name}")
        ConsoleReporter().display(report, threshold=mca_threshold)
        click.echo("")

    # Export per-model reports in the requested format if --output is set.
    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = {"json": ".json", "csv": ".csv", "md": ".md"}[output_format.lower()]
        for model_name, report in results_list:
            _export_report(report, out_dir / f"{model_name}{ext}")


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
