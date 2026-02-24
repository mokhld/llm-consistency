"""RunMetadata frozen dataclass for reproducibility tracking.

Captures package version, Python version, timestamp, config snapshot,
and seed at the start of each evaluation run.
"""

from __future__ import annotations

import datetime as dt
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_consistency._version import __version__

if TYPE_CHECKING:
    from llm_consistency.types import EvaluationConfig


@dataclass(frozen=True)
class RunMetadata:
    """Frozen metadata snapshot for evaluation run reproducibility.

    Attributes:
        package_version: The llm-consistency package version string.
        python_version: The Python interpreter version.
        timestamp: ISO 8601 UTC timestamp of run start.
        config_snapshot: Serialized evaluation configuration.
        perturbation_seed: Random seed used for perturbation generation.
        model: The LLM model identifier.
        provider: The LLM provider identifier.
    """

    package_version: str
    python_version: str
    timestamp: str
    config_snapshot: dict[str, Any] = field(hash=False)
    perturbation_seed: int
    model: str
    provider: str

    @classmethod
    def capture(cls, config: EvaluationConfig, seed: int) -> RunMetadata:
        """Capture a metadata snapshot from the current environment.

        Args:
            config: The evaluation configuration for this run.
            seed: The perturbation seed used.

        Returns:
            A frozen RunMetadata instance.
        """
        return cls(
            package_version=__version__,
            python_version=platform.python_version(),
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
            config_snapshot=config.to_dict(),
            perturbation_seed=seed,
            model=config.model,
            provider=config.provider,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a JSON-compatible dictionary.

        Returns:
            Dictionary with all metadata fields.
        """
        return {
            "package_version": self.package_version,
            "python_version": self.python_version,
            "timestamp": self.timestamp,
            "config_snapshot": self.config_snapshot,
            "perturbation_seed": self.perturbation_seed,
            "model": self.model,
            "provider": self.provider,
        }
