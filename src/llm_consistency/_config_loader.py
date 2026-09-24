"""YAML and TOML configuration file loader."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

import click
import yaml

from llm_consistency._exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

# Config keys that differ from the Click parameter they set.
_KEY_ALIASES: dict[str, str] = {"dataset": "dataset_path"}


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


def check_config_keys(data: dict[str, Any], valid_keys: Collection[str]) -> None:
    """Reject keys that are not in *valid_keys*.

    Raises:
        ValidationError: If *data* has any unknown key. The message lists
            the unknown keys and the valid ones.
    """
    unknown = sorted(set(data) - set(valid_keys))
    if unknown:
        msg = f"Unknown config key(s): {unknown}. Valid keys: {sorted(valid_keys)}"
        raise ValidationError(msg)


def run_defaults_from_config(
    data: dict[str, Any],
    param_names: Collection[str],
) -> dict[str, Any]:
    """Turn a parsed config file into ``default_map`` entries for ``run``.

    Settings may sit under a ``run`` section (YAML ``run:`` or TOML
    ``[run]``) or at the top level. A ``dataset`` key sets the
    ``dataset_path`` parameter.

    Args:
        data: Parsed config file from :func:`load_config_file`.
        param_names: Names of the Click parameters the file may set.

    Returns:
        Mapping from Click parameter name to value.

    Raises:
        ValidationError: If ``run`` is not a mapping, if other keys sit
            beside a ``run`` section, or if a key is unknown.
    """
    if "run" in data:
        others = sorted(k for k in data if k != "run")
        if others:
            msg = f"Config keys {others} must go inside the 'run' section"
            raise ValidationError(msg)
        section = data["run"]
        if not isinstance(section, dict):
            msg = (
                f"Config 'run' section must be a mapping, got {type(section).__name__}"
            )
            raise ValidationError(msg)
    else:
        section = data

    check_config_keys(section, set(param_names) | set(_KEY_ALIASES))
    return {_KEY_ALIASES.get(key, key): value for key, value in section.items()}
