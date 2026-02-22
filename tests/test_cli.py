"""Tests for llm_consistency.cli module."""

import json
from pathlib import Path

from click.testing import CliRunner

from llm_consistency.cli import cli


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


# --- CLI group and version tests (CLI-01) ---


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


# --- run subcommand tests (CLI-02) ---


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


# --- Config file support tests (CLI-06) ---


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


# --- Error handling tests (CLI-07) ---


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
    assert result.exit_code != 0
    assert "Traceback" not in result.output


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
