"""JSONL checkpoint/resume support for BatchRunner.

Persists per-question results as they complete so long-running evals can
be resumed after a crash without losing finished work.

File format (line-delimited JSON, UTF-8):

* Line 1: header object — ``{"type": "header", "version": 1,
  "config_hash": "...", "created_at": "...", "package_version": "...",
  "python_version": "...", "config_snapshot": {...}, "seed": int}``.
* Line 2+: per-question results — ``{"type": "qcr", "qcr": {...}}`` where
  the inner mapping is :meth:`QuestionConsistencyResult.to_dict`.

Each append is followed by ``flush`` + ``os.fsync`` so a crash leaves at
most a single truncated final line, which the reader detects and skips.

The header's ``config_hash`` covers the :class:`EvaluationConfig` plus
the run ``seed``. Resuming with a different config or seed raises
:class:`ValidationError`. The *dataset* is intentionally not hashed —
users are responsible for keeping the dataset stable across resumes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any

from llm_consistency._exceptions import ValidationError
from llm_consistency._version import __version__
from llm_consistency.types import QuestionConsistencyResult

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from llm_consistency.types import EvaluationConfig

_logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def compute_config_hash(config: EvaluationConfig, seed: int) -> str:
    """Stable SHA-256 over the run's identity-defining config + seed.

    Any change to model, provider, scorer, perturbation types,
    ``num_variants``, threshold settings, or ``seed`` invalidates the
    checkpoint. The dataset is not hashed.
    """
    payload = {"config": config.to_dict(), "seed": seed}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointHeader:
    """Header line for a checkpoint file."""

    version: int
    config_hash: str
    created_at: str
    package_version: str
    python_version: str
    seed: int
    config_snapshot: dict[str, Any] = field(hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "header",
            "version": self.version,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "seed": self.seed,
            "config_snapshot": self.config_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointHeader:
        if data.get("type") != "header":
            msg = (
                "Checkpoint header line missing type='header' marker; "
                "file may be corrupt or not a checkpoint."
            )
            raise ValidationError(msg)
        try:
            return cls(
                version=int(data["version"]),
                config_hash=str(data["config_hash"]),
                created_at=str(data["created_at"]),
                package_version=str(data["package_version"]),
                python_version=str(data["python_version"]),
                seed=int(data["seed"]),
                config_snapshot=dict(data.get("config_snapshot", {})),
            )
        except KeyError as exc:
            msg = f"Checkpoint header missing required field: {exc.args[0]!r}"
            raise ValidationError(msg) from exc


def _build_header(config: EvaluationConfig, seed: int) -> CheckpointHeader:
    return CheckpointHeader(
        version=CHECKPOINT_VERSION,
        config_hash=compute_config_hash(config, seed),
        created_at=dt.datetime.now(dt.UTC).isoformat(),
        package_version=__version__,
        python_version=platform.python_version(),
        seed=seed,
        config_snapshot=config.to_dict(),
    )


def read_checkpoint(
    path: Path,
    *,
    config: EvaluationConfig,
    seed: int,
) -> tuple[CheckpointHeader, tuple[QuestionConsistencyResult, ...]]:
    """Read an existing checkpoint and validate it against ``config``/``seed``.

    Returns the parsed header plus a tuple of all completed
    :class:`QuestionConsistencyResult` instances, in the order they were
    written.

    Raises:
        ValidationError: file is empty, header is malformed, header
            version is unsupported, or the header's ``config_hash``
            does not match the current run.

    A trailing line that fails to parse as JSON is treated as a partial
    write from a prior crash: it is logged at WARNING and skipped, with
    all earlier lines kept.
    """
    expected_hash = compute_config_hash(config, seed)

    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if not lines:
        msg = (
            f"Checkpoint file {path} is empty. Delete it and start a fresh "
            "run, or point at a different path."
        )
        raise ValidationError(msg)

    header = _parse_header_line(path, lines[0])

    if header.version != CHECKPOINT_VERSION:
        msg = (
            f"Checkpoint at {path} uses version {header.version}, but this "
            f"runtime only supports version {CHECKPOINT_VERSION}."
        )
        raise ValidationError(msg)

    if header.config_hash != expected_hash:
        msg = (
            f"Checkpoint at {path} was written for a different config (hash "
            f"{header.config_hash[:12]}… vs current {expected_hash[:12]}…). "
            "Resuming would mix results from incompatible runs. Delete the "
            "checkpoint file or point at a new one to start fresh."
        )
        raise ValidationError(msg)

    results: list[QuestionConsistencyResult] = []
    for line_no, raw in enumerate(lines[1:], start=2):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            if line_no == len(lines):
                _logger.warning(
                    "Checkpoint %s: skipping truncated final line %d "
                    "(likely a partial write from a prior crash).",
                    path,
                    line_no,
                )
                break
            msg = (
                f"Checkpoint {path} has malformed JSON on line {line_no}; "
                "file is corrupt."
            )
            raise ValidationError(msg) from None

        if obj.get("type") != "qcr":
            msg = (
                f"Checkpoint {path} line {line_no}: expected "
                f"type='qcr', got type={obj.get('type')!r}."
            )
            raise ValidationError(msg)

        qcr_data = obj.get("qcr")
        if not isinstance(qcr_data, dict):
            msg = (
                f"Checkpoint {path} line {line_no}: 'qcr' field missing or "
                "not an object."
            )
            raise ValidationError(msg)

        results.append(QuestionConsistencyResult.from_dict(qcr_data))

    return header, tuple(results)


def _parse_header_line(path: Path, raw: str) -> CheckpointHeader:
    stripped = raw.strip()
    if not stripped:
        msg = f"Checkpoint {path}: first line is blank, expected a header."
        raise ValidationError(msg)
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        msg = f"Checkpoint {path}: header line is not valid JSON ({exc.msg})."
        raise ValidationError(msg) from exc
    return CheckpointHeader.from_dict(obj)


class CheckpointWriter:
    """Append-only writer for a JSONL checkpoint file.

    Intended use as a context manager::

        with CheckpointWriter(path, config=config, seed=seed) as writer:
            for qcr in compute_qcrs():
                writer.append(qcr)

    On entry, if the target file does not already exist (or is empty),
    a header line is written. If the file already exists with content,
    the existing header is validated against ``config`` and ``seed``;
    on mismatch :class:`ValidationError` is raised before any new data
    is written.
    """

    def __init__(
        self,
        path: Path,
        *,
        config: EvaluationConfig,
        seed: int,
    ) -> None:
        self.path = path
        self._config = config
        self._seed = seed
        self._fh: IO[str] | None = None

    def __enter__(self) -> CheckpointWriter:
        existing_size = self.path.stat().st_size if self.path.exists() else 0
        if existing_size > 0:
            # Validate the existing header before opening for append.
            read_checkpoint(self.path, config=self._config, seed=self._seed)
            self._fh = self.path.open("a", encoding="utf-8")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")
            header = _build_header(self._config, self._seed)
            self._write_line(header.to_dict())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def append(self, qcr: QuestionConsistencyResult) -> None:
        """Append one QCR to the checkpoint and fsync to disk."""
        self._write_line({"type": "qcr", "qcr": qcr.to_dict()})

    def _write_line(self, obj: dict[str, Any]) -> None:
        if self._fh is None:
            msg = "CheckpointWriter used outside of its context manager."
            raise RuntimeError(msg)
        self._fh.write(json.dumps(obj, separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
