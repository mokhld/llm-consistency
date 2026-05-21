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
    assert "variants per Q:      3" in result.output
    assert "total provider calls:6" in result.output
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
