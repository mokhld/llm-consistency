"""YAML and TOML configuration file loader."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

import click
import yaml

from llm_consistency._exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a configuration file (YAML or TOML).

    Detects file format by extension and parses accordingly.
    Supports ``.yaml``, ``.yml``, and ``.toml`` extensions.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        click.BadParameter: If the file format is unsupported.
        FileNotFoundError: If the file does not exist.
        ValidationError: If the file fails to parse or its top-level
            value is not a mapping.
    """
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as f:
            try:
                return tomllib.load(f)
            except tomllib.TOMLDecodeError as exc:
                msg = f"Failed to parse TOML config {path}: {exc}"
                raise ValidationError(msg) from exc
    if suffix in (".yaml", ".yml"):
        with path.open() as f:
            try:
                result = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                msg = f"Failed to parse YAML config {path}: {exc}"
                raise ValidationError(msg) from exc
        if result is None:
            return {}
        if not isinstance(result, dict):
            msg = (
                f"YAML config {path} must contain a mapping at the top level, "
                f"got {type(result).__name__}"
            )
            raise ValidationError(msg)
        return result
    msg = f"Unsupported config format: '{suffix}'. Use .yaml, .yml, or .toml"
    raise click.BadParameter(msg)
