"""Tests for llm_consistency.cli module."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import llm_consistency.cli as cli_module
from llm_consistency._exceptions import ValidationError
from llm_consistency.cli import cli
from llm_consistency.providers import BudgetExceededError
from llm_consistency.providers._mock import MockLLMProvider
from llm_consistency.runners import BatchRunner
from llm_consistency.types import LLMResponse


def _create_mc_dataset(tmp_path: Path) -> Path:
    """Create a valid MC dataset JSON file for testing."""
    dataset = {
        "questions": [
            {
                "id": "q1",
                "stem": "What is 1+1?",
                "options": [
                    {"label": "A", "text": "1", "is_correct": False},
                    {"label": "B", "text": "2", "is_correct": True},
                ],
            },
            {
                "id": "q2",
                "stem": "What is 2+2?",
                "options": [
                    {"label": "A", "text": "3", "is_correct": False},
                    {"label": "B", "text": "4", "is_correct": True},
                ],
            },
        ]
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    return dataset_path


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "LLM Consistency" in result.output
    assert "run" in result.output


def test_cli_no_args_shows_help() -> None:
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert "LLM Consistency" in result.output or "run" in result.output


def test_cli_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_run_help() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    for flag in [
        "--model",
        "--provider",
        "--dataset",
        "--config",
        "--output",
        "--perturbations",
        "--num-variants",
        "--seed",
        "--ci",
    ]:
        assert flag in result.output, f"Missing flag: {flag}"


def test_run_with_mock_provider(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--perturbations",
            "option_reorder",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"


def test_run_ci_mode_exit_code(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--perturbations",
            "option_reorder",
            "--ci",
        ],
    )
    # CI mode may exit 0 or 1 depending on mock results; should NOT crash
    assert result.exit_code in (0, 1), f"CLI error: {result.output}"


def test_run_with_json_output(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    output_path = tmp_path / "results.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--perturbations",
            "option_reorder",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert output_path.exists(), "JSON output file not created"
    data = json.loads(output_path.read_text())
    assert "results" in data


def test_run_with_yaml_config(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "model: test-model\nprovider: mock\n"
        "perturbations:\n  - option_reorder\n"
        "num_variants: 3\nseed: 99\n"
    )
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
        ],
    )
    assert result.exit_code == 0, f"CLI with config failed: {result.output}"


def test_run_with_toml_config(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        'model = "test-model"\n'
        'provider = "mock"\n'
        "num_variants = 3\n"
        "seed = 99\n"
        'perturbations = ["option_reorder"]\n'
    )
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
        ],
    )
    assert result.exit_code == 0, f"CLI with TOML config failed: {result.output}"


def test_run_cli_overrides_config(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "model: config-model\nprovider: mock\n"
        "perturbations:\n  - option_reorder\n"
        "num_variants: 3\n"
    )
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, f"CLI override failed: {result.output}"


def test_run_missing_dataset_file() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "mock",
            "--dataset",
            "/nonexistent/path.json",
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_run_invalid_provider(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "nonexistent_provider",
            "--dataset",
            str(dataset_path),
            "--perturbations",
            "option_reorder",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Unknown provider 'nonexistent_provider'" in result.output


def test_run_invalid_perturbation_type(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--perturbations",
            "nonexistent_pert",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown perturbation type" in result.output


def test_run_invalid_scorer(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "test",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--perturbations",
            "option_reorder",
            "--scorer",
            "bad_scorer",
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_main_entry_point() -> None:
    import importlib  # noqa: PLC0415

    mod = importlib.import_module("llm_consistency.cli")
    main_fn = mod.main
    runner = CliRunner()
    # main() calls cli(), which is the Click group
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Verify main is callable (entry point)
    assert callable(main_fn)


def test_compare_help() -> None:
    result = CliRunner().invoke(cli, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "models" in result.output.lower()


def test_compare_with_config(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: test-model-1\n"
        "    provider: mock\n"
        "  - model: test-model-2\n"
        "    provider: mock\n"
        f"dataset: {dataset_path}\n"
        "perturbations:\n"
        "  - option_reorder\n"
        "num_variants: 3\n"
        "seed: 42\n"
    )
    result = CliRunner().invoke(cli, ["compare", "--config", str(config_path)])
    assert result.exit_code == 0, f"Compare failed: {result.output}"
    assert "test-model-1" in result.output
    assert "test-model-2" in result.output


def test_compare_no_config_errors() -> None:
    result = CliRunner().invoke(cli, ["compare"])
    assert result.exit_code != 0


def test_compare_invalid_config_no_models(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("dataset: something.json\n")
    result = CliRunner().invoke(cli, ["compare", "--config", str(config_path)])
    assert result.exit_code != 0


def test_perturbations_list() -> None:
    result = CliRunner().invoke(cli, ["perturbations", "list"])
    assert result.exit_code == 0
    assert "option_reorder" in result.output
    assert "format_change" in result.output
    assert "separator_change" in result.output


def test_perturbations_help() -> None:
    result = CliRunner().invoke(cli, ["perturbations", "--help"])
    assert result.exit_code == 0


def _create_open_ended_dataset(tmp_path: Path) -> Path:
    """Create a valid open-ended dataset JSON file for testing."""
    dataset = {
        "questions": [
            {
                "id": "q1",
                "stem": "Explain gravity",
                "reference_answers": [
                    "Force of attraction between masses",
                ],
            },
            {
                "id": "q2",
                "stem": "What is photosynthesis?",
                "reference_answers": [
                    "Process by which plants convert light to energy",
                ],
            },
        ]
    }
    path = tmp_path / "open_ended.json"
    path.write_text(json.dumps(dataset))
    return path


def test_dataset_validate_mc_valid(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(cli, ["dataset", "validate", str(dataset_path)])
    assert result.exit_code == 0, f"Validate failed: {result.output}"
    assert "Valid" in result.output
    assert "2 questions" in result.output


def test_dataset_validate_mc_with_type_flag(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli, ["dataset", "validate", "--type", "mc", str(dataset_path)]
    )
    assert result.exit_code == 0


def test_dataset_validate_open_ended(tmp_path: Path) -> None:
    dataset_path = _create_open_ended_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["dataset", "validate", "--type", "open-ended", str(dataset_path)],
    )
    assert result.exit_code == 0
    assert "Valid" in result.output


def test_dataset_validate_invalid_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([{"bad": "data"}]))
    result = CliRunner().invoke(cli, ["dataset", "validate", str(bad_path)])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_dataset_validate_nonexistent_file() -> None:
    result = CliRunner().invoke(cli, ["dataset", "validate", "/nonexistent/path.json"])
    assert result.exit_code != 0


def test_dataset_help() -> None:
    result = CliRunner().invoke(cli, ["dataset", "--help"])
    assert result.exit_code == 0


def test_run_rejects_negative_num_variants(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--num-variants",
            "-1",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "-1" in result.output


def test_run_rejects_out_of_range_mca_threshold(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--mca-threshold",
            "2.5",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "2.5" in result.output


def test_run_rejects_unknown_scorer(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--scorer",
            "not_a_real_scorer",
        ],
    )
    # The scorer flag now goes through EvaluationConfig validation which
    # raises ValidationError for unknown scorers.
    assert result.exit_code != 0


def test_run_happy_path_with_mock(tmp_path: Path) -> None:
    """End-to-end smoke test of `run` with mock provider."""
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["total_questions"] == 2


# ---------------------------------------------------------------------------
# --dry-run flag
# ---------------------------------------------------------------------------


def test_dry_run_in_help() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output


def test_dry_run_unknown_model_reports_unknown_cost(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--num-variants",
            "3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "questions (MC):      2" in result.output
    assert "variants per type:   3 (max)" in result.output
    # A 2-option question has a single non-identity reordering.
    assert "total provider calls:2" in result.output
    assert "unknown (model not priced)" in result.output
    assert "Sample prompt" in result.output


def test_dry_run_known_model_shows_estimated_cost(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "gpt-4o",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--num-variants",
            "2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "estimated cost:      ~$" in result.output


def test_dry_run_does_not_write_output(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not out_path.exists(), "dry-run must not write the output file"


# ---------------------------------------------------------------------------
# --output format routing
# ---------------------------------------------------------------------------


def test_run_output_csv_by_extension(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.csv"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    text = out_path.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("question_id,rc_correct,")
    assert "q1," in text and "q2," in text


def test_run_output_markdown_by_extension(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.md"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("# LLM Consistency Report")
    assert "## Aggregate metrics" in text
    assert "| q1 |" in text


def test_run_output_html_by_extension(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.html"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "<h1>LLM Consistency Report</h1>" in text
    assert "<td>q1</td>" in text


def test_run_output_htm_alias_routes_to_html(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.htm"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")


def test_run_output_unknown_extension_falls_back_to_json(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.xyz"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "mock",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "-o",
            str(out_path),
            "--num-variants",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["total_questions"] == 2


# ---------------------------------------------------------------------------
# Helpers for wiring tests
# ---------------------------------------------------------------------------


def _create_four_option_dataset(tmp_path: Path, correct: tuple[str, ...]) -> Path:
    """One 4-option question per entry in *correct* (its correct label)."""
    questions = [
        {
            "id": f"q{i}",
            "stem": f"Question {i}?",
            "options": [
                {"label": lab, "text": f"Option {lab}", "is_correct": lab == right}
                for lab in "ABCD"
            ],
        }
        for i, right in enumerate(correct)
    ]
    path = tmp_path / "four.json"
    path.write_text(json.dumps({"questions": questions}))
    return path


def _capture_get_provider(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Replace the CLI's get_provider; record each call's arguments.

    Returns a MockLLMProvider that answers ``responses[model]`` (default
    "A"), so tests do not depend on provider-side budget checks.
    """
    calls: list[dict[str, Any]] = []

    def fake(name: str, **kwargs: Any) -> MockLLMProvider:
        calls.append({"name": name, **kwargs})
        answer = (responses or {}).get(kwargs["model"], "A")
        return MockLLMProvider(model=kwargs["model"], default_response=answer)

    monkeypatch.setattr(cli_module, "get_provider", fake)
    return calls


class _OverBudgetProvider(MockLLMProvider):
    """Mock provider whose every query trips the budget cap."""

    async def query(
        self, prompt: str, question_id: str, *, system: str | None = None
    ) -> LLMResponse:
        raise BudgetExceededError(spent=0.01, estimated=0.005, limit=0.01)


def _readme_block(pattern: str) -> str:
    """Return the first README code block matching *pattern*, verbatim."""
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    match = re.search(pattern, readme, re.DOTALL)
    assert match, f"README block not found: {pattern}"
    return match.group(1)


# ---------------------------------------------------------------------------
# Budget and rate limit wiring (A3)
# ---------------------------------------------------------------------------


def test_run_passes_budget_and_rpm_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_get_provider(monkeypatch)
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "gpt-4o",
            "-p",
            "openai",
            "-d",
            str(dataset_path),
            "--max-budget-usd",
            "0.5",
            "--rpm",
            "120",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "name": "openai",
            "model": "gpt-4o",
            "max_budget_usd": 0.5,
            "requests_per_minute": 120,
        }
    ]


def test_run_rpm_defaults_to_60(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_get_provider(monkeypatch)
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli, ["run", "-m", "m", "-p", "mock", "-d", str(dataset_path)]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["requests_per_minute"] == 60
    assert calls[0]["max_budget_usd"] is None


def test_run_rejects_zero_rpm(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli, ["run", "-m", "m", "-p", "mock", "-d", str(dataset_path), "--rpm", "0"]
    )
    assert result.exit_code == 2
    assert "--rpm" in result.output


@pytest.mark.parametrize("ci_flag", [[], ["--ci"]])
def test_budget_exceeded_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ci_flag: list[str]
) -> None:
    monkeypatch.setattr(
        cli_module,
        "get_provider",
        lambda _name, **kw: _OverBudgetProvider(model=kw["model"]),
    )
    dataset_path = _create_mc_dataset(tmp_path)
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "gpt-4o",
            "-p",
            "openai",
            "-d",
            str(dataset_path),
            "--max-budget-usd",
            "0.01",
            "-o",
            str(out_path),
            *ci_flag,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: Budget exceeded" in result.output
    assert not out_path.exists()


def test_provider_validation_error_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_name: str, **_kw: Any) -> MockLLMProvider:
        msg = "No price known for model 'x'; cannot enforce max_budget_usd"
        raise ValidationError(msg)

    monkeypatch.setattr(cli_module, "get_provider", refuse)
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "x",
            "-p",
            "openai",
            "-d",
            str(dataset_path),
            "--max-budget-usd",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Validation error: No price known for model 'x'" in result.output


def test_missing_provider_sdk_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_name: str, **_kw: Any) -> MockLLMProvider:
        msg = "Install llm-consistency[openai] to use the OpenAI provider"
        raise ImportError(msg)

    monkeypatch.setattr(cli_module, "get_provider", missing)
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli, ["run", "-m", "gpt-4o", "-p", "openai", "-d", str(dataset_path)]
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: Install llm-consistency[openai]" in result.output


def test_malformed_dataset_json_is_a_clean_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"questions": [')
    for args in (
        ["run", "-m", "m", "-p", "mock", "-d", str(bad)],
        ["dataset", "validate", str(bad)],
    ):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 1, args
        assert isinstance(result.exception, SystemExit), args
        assert "Error:" in result.output


def test_unregistered_perturbation_lists_registered_ones(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "m",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--perturbations",
            "paraphrase",
        ],
    )
    assert result.exit_code == 1
    assert "'paraphrase' has no registered generator" in result.output
    assert "option_reorder" in result.output
    assert "Configuration error" not in result.output


# ---------------------------------------------------------------------------
# CI gate (A5)
# ---------------------------------------------------------------------------


def test_ci_writes_report_even_when_failing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock answers "A"; q1 and q2 are both "B", so the strict gate fails.
    dataset_path = _create_four_option_dataset(tmp_path, ("B", "B"))
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "m",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--perturbations",
            "separator_change",
            "--ci",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 1
    # CI failures are logged at WARNING (stderr via logging's lastResort
    # handler in a real shell; captured by caplog under pytest).
    assert "MCA check failed" in caplog.text
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["total_questions"] == 2
    assert data["metadata"]["perturbation_seed"] == 42


def test_ci_min_mca_sets_the_pass_target(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock answers "A": q0 is right on every variant, q1 never. MCA = 0.5.
    dataset_path = _create_four_option_dataset(tmp_path, ("A", "B"))
    base = ["run", "-m", "m", "-p", "mock", "-d", str(dataset_path)]
    base += ["--perturbations", "separator_change", "--mca-threshold", "0.8"]

    strict = CliRunner().invoke(cli, [*base, "--ci"])
    assert strict.exit_code == 1
    assert "expected >= 1.000" in caplog.text

    lenient = CliRunner().invoke(cli, [*base, "--ci", "--min-mca", "0.5"])
    assert lenient.exit_code == 0, lenient.output

    # The console applies the same rule.
    console = CliRunner().invoke(cli, [*base, "--min-mca", "0.5"])
    mca_line = next(ln for ln in console.output.splitlines() if "MCA(0.80)" in ln)
    assert "PASS" in mca_line


def test_ci_fails_when_variants_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _Broken(MockLLMProvider):
        async def query(
            self, prompt: str, question_id: str, *, system: str | None = None
        ) -> LLMResponse:
            if "Question 1?" in prompt:
                msg = "simulated outage"
                raise RuntimeError(msg)
            return await super().query(prompt, question_id, system=system)

    monkeypatch.setattr(
        cli_module, "get_provider", lambda _n, **kw: _Broken(model=kw["model"])
    )
    dataset_path = _create_four_option_dataset(tmp_path, ("A", "A"))
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "m",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--perturbations",
            "separator_change",
            "--num-variants",
            "2",
            "--mca-threshold",
            "0.0",
            "--ci",
        ],
    )
    assert result.exit_code == 1
    assert "2 of 4 variants failed with provider errors" in caplog.text


def test_console_core_status_is_na_without_threshold(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    result = CliRunner().invoke(
        cli, ["run", "-m", "m", "-p", "mock", "-d", str(dataset_path)]
    )
    core_line = next(ln for ln in result.output.splitlines() if "CORE" in ln)
    assert "n/a" in core_line


def test_console_core_status_uses_core_threshold_flag(tmp_path: Path) -> None:
    # Mock answers "A", both questions are "B": CORE is 0.
    dataset_path = _create_four_option_dataset(tmp_path, ("B", "B"))
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "m",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--perturbations",
            "separator_change",
            "--core-threshold",
            "0.5",
        ],
    )
    core_line = next(ln for ln in result.output.splitlines() if "CORE" in ln)
    assert "FAIL" in core_line


# ---------------------------------------------------------------------------
# Config files (A6)
# ---------------------------------------------------------------------------


def _assert_readme_run_config_applied(
    tmp_path: Path, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_get_provider(monkeypatch)
    dataset_path = _create_four_option_dataset(tmp_path, ("A", "B"))
    out_path = tmp_path / "report.json"
    # -p mock overrides the file's provider: explicit flags beat config.
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-c",
            str(config_path),
            "-d",
            str(dataset_path),
            "-p",
            "mock",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output

    data = json.loads(out_path.read_text(encoding="utf-8"))
    config = data["config"]
    assert config["model"] == "gpt-5-mini"
    assert config["provider"] == "mock"
    assert config["perturbation_types"] == ["OPTION_REORDER", "FORMAT_CHANGE"]
    assert config["num_variants"] == 3
    assert config["concurrency"] == 5
    assert config["mca_threshold"] == 0.8
    assert config["max_budget_usd"] == 1.0
    assert data["metadata"]["perturbation_seed"] == 42
    assert calls[0]["max_budget_usd"] == 1.0


def test_readme_yaml_config_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_readme_block(r"```yaml\n(run:\n.*?)```"))
    _assert_readme_run_config_applied(tmp_path, config_path, monkeypatch)


def test_readme_toml_config_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_readme_block(r"```toml\n(\[run\]\n.*?)```"))
    _assert_readme_run_config_applied(tmp_path, config_path, monkeypatch)


def test_config_run_section_sets_seed_and_dataset(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"run:\n  model: m\n  provider: mock\n  seed: 7\n  dataset: {dataset_path}\n"
    )
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli, ["run", "-c", str(config_path), "-o", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["metadata"]["perturbation_seed"] == 7
    assert data["total_questions"] == 2


def test_cli_flag_beats_config_value(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        '[run]\nmodel = "m"\nprovider = "mock"\nnum_variants = 3\nseed = 7\n'
    )
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-c",
            str(config_path),
            "-d",
            str(dataset_path),
            "--num-variants",
            "2",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["config"]["num_variants"] == 2
    assert data["metadata"]["perturbation_seed"] == 7


def test_config_unknown_key_is_rejected(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("run:\n  model: m\n  provider: mock\n  num_variant: 3\n")
    result = CliRunner().invoke(
        cli, ["run", "-c", str(config_path), "-d", str(dataset_path)]
    )
    assert result.exit_code == 1
    assert "Unknown config key(s): ['num_variant']" in result.output
    assert "'num_variants'" in result.output


def test_config_min_mca_and_rpm_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_get_provider(monkeypatch)
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("model: m\nprovider: mock\nmin_mca: 0.5\nrpm: 30\n")
    out_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        ["run", "-c", str(config_path), "-d", str(dataset_path), "-o", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["requests_per_minute"] == 30
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["config"]["min_mca"] == 0.5


# ---------------------------------------------------------------------------
# Dry-run call counting (A14)
# ---------------------------------------------------------------------------


def test_dry_run_counts_calls_per_perturbation_type(tmp_path: Path) -> None:
    dataset_path = _create_four_option_dataset(tmp_path, ("A", "B", "C"))
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "-m",
            "m",
            "-p",
            "mock",
            "-d",
            str(dataset_path),
            "--dry-run",
            "--perturbations",
            "option_reorder",
            "--perturbations",
            "format_change",
            "--perturbations",
            "separator_change",
            "--num-variants",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    # Per question: 10 reorders (of 23), 7 format templates, 8 separators.
    assert "total provider calls:75" in result.output


def test_dry_run_warns_when_estimate_exceeds_budget(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    base = ["run", "-m", "gpt-4o", "-p", "mock", "-d", str(dataset_path)]
    over = CliRunner().invoke(cli, [*base, "--dry-run", "--max-budget-usd", "0"])
    assert over.exit_code == 0, over.output
    assert "Warning: the estimated cost" in over.output

    under = CliRunner().invoke(cli, [*base, "--dry-run", "--max-budget-usd", "100"])
    assert under.exit_code == 0, under.output
    assert "Warning" not in under.output


# ---------------------------------------------------------------------------
# compare (A11)
# ---------------------------------------------------------------------------


def _write_compare_config(tmp_path: Path, dataset_path: Path, models: str) -> Path:
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        f"models:\n{models}"
        f"dataset: {dataset_path}\n"
        "perturbations: [separator_change]\n"
        "num_variants: 3\n"
        "max_budget_usd: 2.5\n"
        "rpm: 90\n"
    )
    return config_path


def test_compare_passes_budget_and_rpm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_get_provider(monkeypatch)
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = _write_compare_config(
        tmp_path,
        dataset_path,
        "  - {model: a, provider: openai}\n  - {model: b, provider: anthropic}\n",
    )
    result = CliRunner().invoke(cli, ["compare", "-c", str(config_path)])
    assert result.exit_code == 0, result.output
    assert [(c["name"], c["model"]) for c in calls] == [
        ("openai", "a"),
        ("anthropic", "b"),
    ]
    assert all(c["max_budget_usd"] == 2.5 for c in calls)
    assert all(c["requests_per_minute"] == 90 for c in calls)


def test_compare_sanitises_and_dedupes_file_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture_get_provider(monkeypatch)
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = _write_compare_config(
        tmp_path,
        dataset_path,
        "  - {model: openai/gpt-4o-mini, provider: litellm}\n"
        "  - {model: openai/gpt-4o-mini, provider: litellm}\n"
        "  - {model: ../escape, provider: mock}\n",
    )
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        cli, ["compare", "-c", str(config_path), "-o", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "_escape.json",
        "openai_gpt-4o-mini-2.json",
        "openai_gpt-4o-mini.json",
    ]
    assert not (tmp_path / "escape.json").exists()
    data = json.loads((out_dir / "openai_gpt-4o-mini.json").read_text())
    assert data["metadata"]["model"] == "openai/gpt-4o-mini"
    assert data["metadata"]["perturbation_seed"] == 42


def test_compare_exports_each_model_before_a_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture_get_provider(monkeypatch)

    class _FailingRunner(BatchRunner):
        async def run(self, dataset: Any, config: Any, *args: Any, **kw: Any) -> Any:
            if config.model == "boom":
                raise BudgetExceededError(spent=2.5, estimated=0.01, limit=2.5)
            return await super().run(dataset, config, *args, **kw)

    monkeypatch.setattr(cli_module, "BatchRunner", _FailingRunner)
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = _write_compare_config(
        tmp_path,
        dataset_path,
        "  - {model: first, provider: mock}\n  - {model: boom, provider: mock}\n",
    )
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        cli, ["compare", "-c", str(config_path), "-o", str(out_dir), "--format", "csv"]
    )
    assert result.exit_code == 1
    assert "Budget exceeded" in result.output
    assert (out_dir / "first.csv").exists()
    assert not (out_dir / "boom.csv").exists()


def test_compare_prints_table_with_mcnemar_p_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both questions have "B" correct; "good" always answers B, "bad" A.
    _capture_get_provider(monkeypatch, responses={"good": "B", "bad": "A"})
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = _write_compare_config(
        tmp_path,
        dataset_path,
        "  - {model: good, provider: mock}\n  - {model: bad, provider: mock}\n",
    )
    result = CliRunner().invoke(cli, ["compare", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    lines = result.output.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.startswith("Model "))
    assert lines[header].split() == [
        "Model",
        "CORE",
        "MCA(1.00)",
        "RC_correct",
        "RC_agree",
        "p-value",
    ]
    good = lines[header + 2].split()
    bad = lines[header + 3].split()
    assert good[0] == "good"
    assert good[2:4] == ["1.0000", "1.0000"]
    assert good[-1] == "-"
    assert bad[0] == "bad"
    assert bad[2:4] == ["0.0000", "0.0000"]
    # b=2 discordant pairs, c=0: exact two-sided p = 2 * 0.5**2
    assert bad[-1] == "0.5000"


def test_compare_rejects_unknown_config_key(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        f"models:\n  - {{model: a, provider: mock}}\ndataset: {dataset_path}\n"
        "num_variant: 3\n"
    )
    result = CliRunner().invoke(cli, ["compare", "-c", str(config_path)])
    assert result.exit_code == 1
    assert "Unknown config key(s): ['num_variant']" in result.output


def test_compare_rejects_zero_rpm(tmp_path: Path) -> None:
    dataset_path = _create_mc_dataset(tmp_path)
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        f"models:\n  - {{model: a, provider: mock}}\ndataset: {dataset_path}\nrpm: 0\n"
    )
    result = CliRunner().invoke(cli, ["compare", "-c", str(config_path)])
    assert result.exit_code == 1
    assert "'rpm' must be >= 1" in result.output


def test_report_file_stems() -> None:
    assert cli_module._report_file_stems(
        ["openai/gpt-4o", "GPT-4o", "gpt-4o", "a b:c", "...", "gpt-4o-2"]
    ) == ["openai_gpt-4o", "GPT-4o", "gpt-4o-2", "a_b_c", "model", "gpt-4o-2-2"]
