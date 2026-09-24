"""Tests for llm_consistency._config_loader module."""

from pathlib import Path

import click
import pytest

from llm_consistency._config_loader import (
    check_config_keys,
    load_config_file,
    run_defaults_from_config,
)
from llm_consistency._exceptions import ValidationError


def test_load_yaml_config(tmp_path: Path) -> None:
    config_file = tmp_path / "eval.yaml"
    config_file.write_text(
        "model: gpt-4o\nprovider: openai\nperturbations:\n"
        "  - option_reorder\nnum_variants: 5\n"
    )
    result = load_config_file(config_file)
    assert result["model"] == "gpt-4o"
    assert result["provider"] == "openai"
    assert result["perturbations"] == ["option_reorder"]
    assert result["num_variants"] == 5


def test_load_toml_config(tmp_path: Path) -> None:
    config_file = tmp_path / "eval.toml"
    config_file.write_text(
        'model = "gpt-4o"\nprovider = "openai"\n'
        'perturbations = ["option_reorder"]\nnum_variants = 5\n'
    )
    result = load_config_file(config_file)
    assert result["model"] == "gpt-4o"
    assert result["provider"] == "openai"
    assert result["perturbations"] == ["option_reorder"]
    assert result["num_variants"] == 5


def test_load_yml_extension(tmp_path: Path) -> None:
    config_file = tmp_path / "eval.yml"
    config_file.write_text("model: gpt-4o\nprovider: openai\n")
    result = load_config_file(config_file)
    assert result["model"] == "gpt-4o"
    assert result["provider"] == "openai"


def test_unsupported_extension(tmp_path: Path) -> None:
    config_file = tmp_path / "eval.ini"
    config_file.write_text("[section]\nkey=value\n")
    with pytest.raises(click.BadParameter, match="Unsupported config format"):
        load_config_file(config_file)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config_file(Path("/nonexistent/path/config.yaml"))


def test_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    result = load_config_file(config_file)
    assert result == {}


def test_invalid_yaml_raises_validation_error(tmp_path: Path) -> None:
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("key: : : invalid : : :\n  - nested\n")
    with pytest.raises(ValidationError, match="Failed to parse YAML"):
        load_config_file(config_file)


def test_non_mapping_yaml_raises_validation_error(tmp_path: Path) -> None:
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- 1\n- 2\n- 3\n")
    with pytest.raises(ValidationError, match="must contain a mapping"):
        load_config_file(config_file)


def test_invalid_toml_raises_validation_error(tmp_path: Path) -> None:
    config_file = tmp_path / "broken.toml"
    config_file.write_text("not = valid = toml\n")
    with pytest.raises(ValidationError, match="Failed to parse TOML"):
        load_config_file(config_file)


_RUN_PARAMS = ("model", "provider", "dataset_path", "num_variants", "seed")


def test_run_section_is_unwrapped() -> None:
    data = {"run": {"model": "m", "num_variants": 3}}
    assert run_defaults_from_config(data, _RUN_PARAMS) == {
        "model": "m",
        "num_variants": 3,
    }


def test_flat_keys_are_accepted() -> None:
    data = {"model": "m", "seed": 7}
    assert run_defaults_from_config(data, _RUN_PARAMS) == data


def test_dataset_key_maps_to_dataset_path() -> None:
    data = {"run": {"dataset": "qs.json"}}
    assert run_defaults_from_config(data, _RUN_PARAMS) == {"dataset_path": "qs.json"}


def test_unknown_key_lists_valid_keys() -> None:
    with pytest.raises(ValidationError) as exc_info:
        run_defaults_from_config({"run": {"num_variant": 3}}, _RUN_PARAMS)
    message = str(exc_info.value)
    assert "num_variant" in message
    assert "'num_variants'" in message
    assert "'dataset'" in message


def test_keys_beside_run_section_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"\['seed'\].*inside the 'run'"):
        run_defaults_from_config({"run": {"model": "m"}, "seed": 1}, _RUN_PARAMS)


def test_run_section_must_be_a_mapping() -> None:
    with pytest.raises(ValidationError, match="must be a mapping"):
        run_defaults_from_config({"run": ["model"]}, _RUN_PARAMS)


def test_check_config_keys_accepts_known_and_rejects_unknown() -> None:
    check_config_keys({"a": 1}, {"a", "b"})
    with pytest.raises(ValidationError, match=r"\['c'\].*Valid keys: \['a', 'b'\]"):
        check_config_keys({"a": 1, "c": 2}, {"a", "b"})
