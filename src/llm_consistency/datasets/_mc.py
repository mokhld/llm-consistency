"""MCDataset loader for JSON/JSONL/CSV multiple-choice question files."""

from __future__ import annotations

import csv
import json
import string
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from llm_consistency._exceptions import ValidationError
from llm_consistency.datasets._base import BaseDataset
from llm_consistency.datasets._validation import (
    detect_format,
    load_json_questions,
    validate_unique_ids,
)
from llm_consistency.types import MCOption, MCQuestion

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

_MAX_MC_LABELS = len(string.ascii_uppercase)

AnswerFormat = Literal["auto", "index", "label", "text"]


class MCDataset(BaseDataset):
    """Dataset of multiple-choice questions loaded from file.

    Supports JSON, JSONL, and CSV file formats.  After loading,
    validates that all question IDs are unique.

    Args:
        questions: Tuple of MCQuestion instances.
    """

    def __init__(self, questions: tuple[MCQuestion, ...]) -> None:
        self._questions = questions

    @property
    def questions(self) -> tuple[MCQuestion, ...]:
        """Return the tuple of MCQuestion instances."""
        return self._questions

    @classmethod
    def load(cls, path: str | Path) -> MCDataset:
        """Load MC questions from a JSON, JSONL, or CSV file.

        Args:
            path: File path to load from.

        Returns:
            A new MCDataset instance.

        Raises:
            ValidationError: On unsupported format, missing fields,
                duplicate IDs, or invalid data.
        """
        p = Path(path)
        fmt = detect_format(p)
        if fmt == "json":
            questions = cls._load_json(p)
        elif fmt == "jsonl":
            questions = cls._load_jsonl(p)
        else:
            questions = cls._load_csv(p)
        validate_unique_ids(questions, str(p))
        return cls(questions)

    @classmethod
    def load_from_hub(
        cls,
        repo_id: str,
        *,
        split: str = "train",
        name: str | None = None,
        token: str | None = None,
        question_col: str = "question",
        choices_col: str = "choices",
        answer_col: str = "answer",
        id_col: str | None = None,
        answer_format: AnswerFormat = "auto",
        **kwargs: Any,
    ) -> MCDataset:
        """Load multiple-choice questions from a HuggingFace Hub dataset.

        Requires the optional ``huggingface`` extra
        (``pip install llm-consistency[huggingface]``).

        Defaults match the ``cais/mmlu``-style schema:

        - ``question`` (``str``) — the question stem.
        - ``choices`` (``list[str]``) — option texts; labels are auto-assigned
          as ``"A"``, ``"B"``, ``"C"``, … in order.
        - ``answer``: the correct choice, read according to
          ``answer_format``:

          * ``"index"``: a 0-based position into ``choices``, as an
            ``int`` or a string of digits (``2`` or ``"2"``, as in
            HellaSwag).
          * ``"label"``: a single letter, case-insensitive (``"B"`` is
            the second choice).
          * ``"text"``: the exact text of one of the choices.
          * ``"auto"`` (the default): an ``int`` is an index.  A string
            is read in every way that applies to it: as a label if it
            is one letter, as an index if it is all ASCII digits, and as
            a choice text.  Readings that point at no choice are
            ignored.  If the remaining readings point at different
            choices, for example ``"1"`` with choices ``["1", "2", "3"]``
            (index 1, or the text of choice 0), loading fails and asks
            for an explicit ``answer_format`` rather than guessing.

        Override ``question_col`` / ``choices_col`` / ``answer_col`` for
        datasets with a different schema.  Supply ``id_col`` to use a column
        as the question ID; otherwise IDs are auto-generated as
        ``"row_0"``, ``"row_1"``, ….

        Args:
            repo_id: HuggingFace Hub dataset identifier (e.g.
                ``"cais/mmlu"``).
            split: Dataset split to load.  Defaults to ``"train"``.
            name: Subset / config name (e.g. ``"abstract_algebra"`` for
                ``cais/mmlu``).
            token: Optional HuggingFace auth token for gated/private
                datasets.
            question_col: Column containing the question stem.
            choices_col: Column containing the list of option texts.
            answer_col: Column containing the correct answer.
            id_col: Column to use as the question ID.  When ``None`` (the
                default), IDs are generated from row index.
            answer_format: How to read the answer column: ``"auto"``
                (default), ``"index"``, ``"label"`` or ``"text"``.  See
                above.
            **kwargs: Forwarded verbatim to ``datasets.load_dataset``.

        Returns:
            A new ``MCDataset`` instance.

        Raises:
            ImportError: If the ``huggingface`` extra is not installed.
            ValidationError: If a row is malformed (missing column,
                unresolvable or ambiguous answer, too many choices, etc.),
                if the resulting questions have duplicate IDs, or if
                ``answer_format`` is not a known format.
        """
        if answer_format not in get_args(AnswerFormat):
            msg = (
                f"Unknown answer_format {answer_format!r}; expected one of "
                f"{list(get_args(AnswerFormat))}"
            )
            raise ValidationError(msg)
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError:
            msg = "Install llm-consistency[huggingface] to use the Hub loader"
            raise ImportError(msg) from None

        ds: Iterable[dict[str, Any]] = load_dataset(
            repo_id, name=name, split=split, token=token, **kwargs
        )
        source = f"{repo_id}[{name}]" if name else repo_id

        questions: list[MCQuestion] = []
        for idx, row in enumerate(ds):
            questions.append(
                cls._parse_hub_row(
                    row,
                    idx,
                    source=source,
                    question_col=question_col,
                    choices_col=choices_col,
                    answer_col=answer_col,
                    id_col=id_col,
                    answer_format=answer_format,
                )
            )

        ordered = tuple(questions)
        validate_unique_ids(ordered, source)
        return cls(ordered)

    @classmethod
    def _parse_hub_row(
        cls,
        row: dict[str, Any],
        index: int,
        *,
        source: str,
        question_col: str,
        choices_col: str,
        answer_col: str,
        id_col: str | None,
        answer_format: AnswerFormat = "auto",
    ) -> MCQuestion:
        """Map one HuggingFace row to an ``MCQuestion``."""
        stem = cls._require_column(row, question_col, index, source)
        choices_raw = cls._require_column(row, choices_col, index, source)
        answer_raw = cls._require_column(row, answer_col, index, source)

        if id_col is not None:
            qid = str(cls._require_column(row, id_col, index, source))
        else:
            qid = f"row_{index}"

        if not isinstance(choices_raw, (list, tuple)):
            msg = (
                f"Column {choices_col!r} at row {index} in {source} is "
                f"{type(choices_raw).__name__}; expected a list of strings"
            )
            raise ValidationError(msg)
        choices = [str(c) for c in choices_raw]
        if not choices:
            msg = f"Row {index} in {source} has no choices"
            raise ValidationError(msg)
        if len(choices) > _MAX_MC_LABELS:
            msg = (
                f"Row {index} in {source} has {len(choices)} choices; "
                f"only up to {_MAX_MC_LABELS} (A-Z) are supported"
            )
            raise ValidationError(msg)

        correct_idx = cls._resolve_answer(
            answer_raw, choices, index, source, answer_format
        )
        options = tuple(
            MCOption(
                label=string.ascii_uppercase[i],
                text=choice,
                is_correct=(i == correct_idx),
            )
            for i, choice in enumerate(choices)
        )
        return MCQuestion(id=qid, stem=str(stem), options=options)

    @staticmethod
    def _require_column(
        row: dict[str, Any], column: str, index: int, source: str
    ) -> Any:
        """Return ``row[column]`` or raise a clear ``ValidationError``."""
        try:
            return row[column]
        except (KeyError, TypeError) as exc:
            available = sorted(row.keys()) if isinstance(row, dict) else []
            msg = (
                f"Column {column!r} not found at row {index} in {source}. "
                f"Available columns: {available}"
            )
            raise ValidationError(msg) from exc

    @staticmethod
    def _resolve_answer(
        answer: Any,
        choices: list[str],
        index: int,
        source: str,
        answer_format: AnswerFormat = "auto",
    ) -> int:
        """Resolve the answer field to a 0-based index into ``choices``.

        The rules for each ``answer_format`` are in :meth:`load_from_hub`.
        """
        where = f"at row {index} in {source}"
        # bool is an int subclass; reject it before the int branch.
        if isinstance(answer, bool):
            msg = f"Answer {where} is a bool; expected int index or str label/text"
            raise ValidationError(msg)
        if not isinstance(answer, (int, str)):
            msg = (
                f"Unsupported answer type {type(answer).__name__} {where}; "
                f"expected int or str"
            )
            raise ValidationError(msg)

        text = str(answer).strip()
        is_str = isinstance(answer, str)
        # Each applicable reading maps to the choice index it points at.
        readings: dict[str, int | None] = {}
        if answer_format in ("auto", "index") and (
            not is_str or (text.isascii() and text.isdigit())
        ):
            readings["index"] = int(text)
        if (
            answer_format in ("auto", "label")
            and is_str
            and len(text) == 1
            and text.isascii()
            and text.isalpha()
        ):
            readings["label"] = ord(text.upper()) - ord("A")
        if answer_format == "text" or (answer_format == "auto" and is_str):
            readings["text"] = next(
                (i for i, choice in enumerate(choices) if choice == text), None
            )

        valid = {
            name: i
            for name, i in readings.items()
            if i is not None and 0 <= i < len(choices)
        }
        if len(set(valid.values())) > 1:
            meanings = ", ".join(f"{name} -> choice {i}" for name, i in valid.items())
            msg = (
                f"Answer {answer!r} {where} is ambiguous ({meanings}); pass "
                f"answer_format='index', 'label' or 'text'"
            )
            raise ValidationError(msg)
        if valid:
            return next(iter(valid.values()))

        if "index" in readings:
            msg = (
                f"Answer index {readings['index']} {where} out of range for "
                f"{len(choices)} choices"
            )
        elif "label" in readings:
            msg = (
                f"Answer label {text.upper()!r} {where} out of range for "
                f"{len(choices)} choices"
            )
        elif "text" in readings:
            msg = f"Answer {answer!r} {where} does not match any choice"
        else:
            msg = f"Answer {answer!r} {where} is not a valid {answer_format}"
        raise ValidationError(msg)

    @classmethod
    def _load_json(cls, path: Path) -> tuple[MCQuestion, ...]:
        """Load from JSON format: ``{"questions": [...]}``."""
        raw_questions: list[dict[str, Any]] = load_json_questions(path)
        results: list[MCQuestion] = []
        for idx, qdict in enumerate(raw_questions):
            results.append(cls._parse_question(qdict, str(path), idx))
        return tuple(results)

    @classmethod
    def _load_jsonl(cls, path: Path) -> tuple[MCQuestion, ...]:
        """Load from JSONL format: one question per line."""
        results: list[MCQuestion] = []
        with path.open(encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    qdict: dict[str, Any] = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    msg = f"Invalid JSON at line {line_num} in {path}: {exc}"
                    raise ValidationError(msg) from exc
                results.append(cls._parse_question(qdict, str(path), line_num))
        return tuple(results)

    @classmethod
    def _load_csv(cls, path: Path) -> tuple[MCQuestion, ...]:
        """Load from CSV with option_a..option_z columns."""
        results: list[MCQuestion] = []
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                qid = row.get("id", "").strip()
                stem = row.get("stem", "").strip()
                correct_label = row.get("correct", "").strip().upper()

                if not qid:
                    msg = f"Missing required field 'id' at row {row_idx} in {path}"
                    raise ValidationError(msg)
                if not stem:
                    msg = f"Missing required field 'stem' at row {row_idx} in {path}"
                    raise ValidationError(msg)

                options: list[MCOption] = []
                for letter in string.ascii_lowercase:
                    col_name = f"option_{letter}"
                    val = row.get(col_name, "").strip()
                    if not val:
                        continue
                    label = letter.upper()
                    options.append(
                        MCOption(
                            label=label,
                            text=val,
                            is_correct=(label == correct_label),
                        )
                    )

                if not options:
                    msg = f"No options found at row {row_idx} in {path}"
                    raise ValidationError(msg)

                option_labels = {o.label for o in options}
                if correct_label not in option_labels:
                    msg = (
                        f"Invalid correct label '{correct_label}' at "
                        f"row {row_idx} in {path}. "
                        f"Available: {sorted(option_labels)}"
                    )
                    raise ValidationError(msg)

                results.append(
                    MCQuestion(
                        id=qid,
                        stem=stem,
                        options=tuple(options),
                    )
                )
        return tuple(results)

    @classmethod
    def _parse_question(
        cls,
        qdict: dict[str, Any],
        source: str,
        index: int,
    ) -> MCQuestion:
        """Parse a single MC question dict into an MCQuestion.

        Args:
            qdict: Raw question dictionary.
            source: File path for error messages.
            index: Question index/line number for error messages.

        Returns:
            An MCQuestion instance.

        Raises:
            ValidationError: On missing or invalid fields.
        """
        try:
            qid = qdict["id"]
        except KeyError:
            msg = f"Missing required field 'id' at index {index} in {source}"
            raise ValidationError(msg) from None
        try:
            stem = qdict["stem"]
        except KeyError:
            msg = f"Missing required field 'stem' at index {index} in {source}"
            raise ValidationError(msg) from None
        try:
            raw_options = qdict["options"]
        except KeyError:
            msg = f"Missing required field 'options' at index {index} in {source}"
            raise ValidationError(msg) from None

        options = tuple(
            cls._parse_option(opt, opt_idx, source, index)
            for opt_idx, opt in enumerate(raw_options)
        )
        return MCQuestion(id=str(qid), stem=str(stem), options=options)

    @staticmethod
    def _parse_option(opt: Any, opt_idx: int, source: str, index: int) -> MCOption:
        """Parse one option dict, naming the question and option on error."""
        where = f"option {opt_idx} at index {index} in {source}"
        try:
            return MCOption(
                label=str(opt["label"]),
                text=str(opt["text"]),
                is_correct=bool(opt["is_correct"]),
            )
        except KeyError as exc:
            msg = f"Missing required field {exc.args[0]!r} in {where}"
            raise ValidationError(msg) from None
        except TypeError:
            msg = (
                f"Expected an object with 'label', 'text' and 'is_correct' in "
                f"{where}, got {type(opt).__name__}"
            )
            raise ValidationError(msg) from None
