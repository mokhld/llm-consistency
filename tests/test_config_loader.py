"""Tests for llm_consistency._config_loader module."""

from pathlib import Path

import click
import pytest

from llm_consistency._config_loader import load_config_file
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
