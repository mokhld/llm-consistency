"""YAML and TOML configuration file loader."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

import click
import yaml

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
    """
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as f:
            return tomllib.load(f)
    if suffix in (".yaml", ".yml"):
        with path.open() as f:
            result = yaml.safe_load(f)
            return result if isinstance(result, dict) else {}
    msg = f"Unsupported config format: '{suffix}'. Use .yaml, .yml, or .toml"
    raise click.BadParameter(msg)
