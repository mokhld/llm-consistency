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
most a single truncated final line. The reader skips it, and the writer
cuts it off before appending so the next record starts on a fresh line.

The header's ``config_hash`` covers the :class:`EvaluationConfig` fields
named in :data:`CONFIG_HASH_FIELDS` plus the run ``seed``: the settings
that change what a run produces. Resuming with a different value for
any of them raises :class:`ValidationError`; changing anything else
(``concurrency``, ``max_budget_usd``, pass/fail thresholds, ``ci_mode``)
is allowed. Version 1 checkpoints, written by releases before 1.1, are
rejected: those releases scored ``option_reorder`` variants against the
wrong labels, so their results cannot be mixed with new ones.

The *dataset* is not hashed. :func:`read_checkpoint` can drop results
for question IDs that are no longer in the dataset (``question_ids``),
and keeps only the last record for each question ID.
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
    from collections.abc import Collection, Mapping
    from pathlib import Path
    from types import TracebackType

    from llm_consistency.types import EvaluationConfig

_logger = logging.getLogger(__name__)

# Version 2: variants are scored against the options they presented (1.1).
CHECKPOINT_VERSION = 2

# EvaluationConfig.to_dict() keys that change what a run produces. Only
# these and the seed go into the checkpoint hash. Add new result-affecting
# fields here; each must be a to_dict() key.
CONFIG_HASH_FIELDS: tuple[str, ...] = (
    "model",
    "provider",
    "perturbation_types",
    "scorer",
    "num_variants",
    "prompt_template",
    "system_prompt",
    "temperature",
    "max_tokens",
    "generation_seed",
)


def compute_config_hash(config: EvaluationConfig, seed: int) -> str:
    """Stable SHA-256 over the result-affecting config fields + seed.

    Covers the fields in :data:`CONFIG_HASH_FIELDS` and ``seed``. Fields
    that do not change results, such as ``concurrency``,
    ``max_budget_usd`` and the pass/fail thresholds, are not hashed. The
    dataset is not hashed.
    """
    snapshot = config.to_dict()
    return _hash_payload({name: snapshot[name] for name in CONFIG_HASH_FIELDS}, seed)


def _hash_payload(config_fields: Mapping[str, Any], seed: int) -> str:
    payload = {"config": config_fields, "seed": seed}
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
    question_ids: Collection[str] | None = None,
) -> tuple[CheckpointHeader, tuple[QuestionConsistencyResult, ...]]:
    """Read an existing checkpoint and validate it against ``config``/``seed``.

    Returns the parsed header plus a tuple of the completed
    :class:`QuestionConsistencyResult` instances, in the order they were
    written. If a question ID has more than one record, the last record
    wins (at the position of the first).

    Args:
        path: Checkpoint file.
        config: Configuration of the run being resumed.
        seed: Seed of the run being resumed.
        question_ids: If given, results for question IDs not in this
            collection (for example, questions removed from the dataset
            since the checkpoint was written) are dropped and logged.
            Pass the current dataset's IDs when resuming.

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

    if header.version < CHECKPOINT_VERSION:
        msg = (
            f"Checkpoint at {path} was written by an older release "
            f"(checkpoint version {header.version}, package "
            f"{header.package_version}) that scored option_reorder variants "
            "against the wrong labels. Its results cannot be resumed; delete "
            "the file or point at a new one to start fresh."
        )
        raise ValidationError(msg)
    if header.version != CHECKPOINT_VERSION:
        msg = (
            f"Checkpoint at {path} uses version {header.version}, but this "
            f"runtime only supports version {CHECKPOINT_VERSION}."
        )
        raise ValidationError(msg)

    if header.config_hash != expected_hash:
        msg = (
            f"Checkpoint at {path} was written for a different config (hash "
            f"{header.config_hash[:12]}… vs current {expected_hash[:12]}…): "
            f"one of {', '.join(CONFIG_HASH_FIELDS)} or the seed differs. "
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

    return header, _latest_by_id(path, results, question_ids)


def _latest_by_id(
    path: Path,
    results: list[QuestionConsistencyResult],
    question_ids: Collection[str] | None,
) -> tuple[QuestionConsistencyResult, ...]:
    """Keep the last record per question ID, optionally only for *question_ids*."""
    by_id = {qcr.question_id: qcr for qcr in results}
    if len(by_id) < len(results):
        _logger.warning(
            "Checkpoint %s: %d duplicate record(s); keeping the last record "
            "for each question ID.",
            path,
            len(results) - len(by_id),
        )
    if question_ids is not None:
        wanted = set(question_ids)
        stale = [qid for qid in by_id if qid not in wanted]
        if stale:
            _logger.warning(
                "Checkpoint %s: ignoring %d result(s) for question IDs not in "
                "the dataset.",
                path,
                len(stale),
            )
            for qid in stale:
                del by_id[qid]
    return tuple(by_id.values())


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
    is written. An unterminated last line left by a crash is then cut
    off (or, if it is a complete record, terminated) so appends start on
    a fresh line.
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
            _repair_last_line(self.path)
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


def _repair_last_line(path: Path) -> None:
    """Make *path* end with a newline so the next append starts a new line.

    A crash can leave the last line without its newline. If that line
    parses as JSON it is a complete record and only the newline is
    added. Otherwise it is a partial write, which :func:`read_checkpoint`
    skips, and it is removed.
    """
    with path.open("rb+") as fh:
        data = fh.read()
        if data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except ValueError:
            fh.truncate(start)
        else:
            fh.write(b"\n")
        fh.flush()
        os.fsync(fh.fileno())
