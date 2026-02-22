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
from llm_consistency.reports import ConsoleReporter, export_json
from llm_consistency.runners import BatchRunner, CIRunner
from llm_consistency.scoring import ExactMatchScorer
from llm_consistency.types import EvaluationConfig, PerturbationType

if TYPE_CHECKING:
    from collections.abc import Callable


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

    data = load_config_file(Path(value))
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
    type=int,
    default=5,
    help="Variants per question",
)
@click.option(
    "--concurrency",
    type=int,
    default=10,
    help="Max concurrent API calls",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--scorer", default="exact_match", help="Scoring method")
@click.option(
    "--mca-threshold",
    type=float,
    default=1.0,
    help="MCA threshold for pass/fail",
)
@click.option(
    "--core-threshold",
    type=float,
    default=None,
    help="CORE threshold for pass/fail",
)
@click.option(
    "--max-budget-usd",
    type=float,
    default=None,
    help="Budget ceiling in USD",
)
@click.option("--ci", is_flag=True, help="CI mode: exit 1 on threshold failure")
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
    scoring = ExactMatchScorer()

    if ci:
        exit_code = asyncio.run(CIRunner().run(ds, config, prov, scoring, seed=seed))
        raise SystemExit(exit_code)

    runner = BatchRunner()
    report = asyncio.run(runner.run(ds, config, prov, scoring, seed=seed))
    ConsoleReporter().display(report, threshold=mca_threshold)

    if output:
        export_json(report, Path(output), metadata=runner.last_metadata)


def main() -> None:
    """Entry point for the CLI."""
    cli()
