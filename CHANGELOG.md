# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-perturbation variance decomposition —
  `perturbation_impact(report)` returns
  `dict[PerturbationType, float]` mapping each perturbation type to
  its mean failure rate across the run. Identifies which
  perturbation drives the largest consistency drop. To support it,
  `ScoredResponse` grows a new optional `perturbation_type: str | None`
  field (defaulting to `None` for backward compatibility) that the
  `BatchRunner` and `StreamingRunner` now populate per variant.
  `from_dict` tolerates legacy payloads missing the field.
- Paired-model significance test — `compare_mca_paired(results_a,
  results_b, threshold)` runs McNemar's exact binomial test on
  per-question MCA pass/fail outcomes. Returns a new
  `PairedTestResult(statistic, p_value, n_discordant, method)`
  dataclass. Questions present in only one set are silently dropped;
  empty inputs or no shared IDs raise `ValidationError`. Use this
  to A/B two models on the same dataset and decide whether the MCA
  gap is statistically significant.
- Sample-size power analysis utility —
  `validate_sample_size(n, effect_size, alpha=0.05, power=0.80)`
  returns a dict with `observed_power` (achieved at `n`) and
  `recommended_n` (minimum `n` for `target_power`). One-sample
  two-sided z-test on a proportion with Cohen's h. Emits a
  `UserWarning` when `n < 200` — the typical perturbation-study
  guideline. Exported from the top-level package.
- HuggingFace Hub dataset loader — `MCDataset.load_from_hub(repo_id, *,
  split="train", name=None, token=None, question_col="question",
  choices_col="choices", answer_col="answer", id_col=None, **kwargs)`
  maps `cais/mmlu`-style schemas to `MCQuestion`s out of the box.
  Custom schemas supported via column-mapping kwargs. Answer column
  accepts `int` index, single-letter label, or choice-text match.
  Requires the new optional `huggingface` extra (`pip install
  llm-consistency[huggingface]`); raises a clear `ImportError`
  otherwise.
- `examples/` directory with six runnable end-to-end demonstrations
  (basic mock, dry-run, checkpoint/resume, custom scorer, export
  formats, model comparison) plus a bundled `datasets/sample.jsonl`.
- `CONTRIBUTING.md` covering local setup, quality gates, extension
  points (perturbation/scorer/provider), and the audit-driven roadmap.
- `CHANGELOG.md` (this file).
- HTML report exporter — `export_html(report, path, *, metadata=None,
  tau_agree=0.8)` produces a self-contained single-page HTML report
  with inline CSS and no JavaScript. CLI `--output report.html` routes
  to it automatically; `compare --format html` supported.
  ([`de32ce6`](https://github.com/mokhld/llm-consistency/commit/de32ce6))
- CSV and Markdown report exporters — `export_csv` (one row per QCR)
  and `export_markdown` (per-section tables). CLI `--output` extension
  routing: `.csv` → CSV, `.md`/`.markdown` → Markdown.
  ([`4452cd1`](https://github.com/mokhld/llm-consistency/commit/4452cd1))
- `llm-consistency run --dry-run` — validates dataset, config, provider,
  and renders one sample prompt with cost estimate, without spending
  tokens.
  ([`4452cd1`](https://github.com/mokhld/llm-consistency/commit/4452cd1))
- JSONL checkpoint/resume in `BatchRunner` — pass `checkpoint_path=` to
  persist each completed `QuestionConsistencyResult` as it's computed.
  Resumes skip already-completed questions; mismatched configs raise
  `ValidationError`; crash-truncated final lines are detected and
  skipped.
  ([`703578a`](https://github.com/mokhld/llm-consistency/commit/703578a))
- Bootstrap confidence intervals on aggregate metrics —
  `mca_with_ci`, `core_index_with_ci`,
  `agreement_gated_accuracy_with_ci`, `car_curve_with_ci` return
  `MetricResult(value, ci_lower, ci_upper, n_samples, confidence,
  method)`. Default method is **BCa** (bias-corrected accelerated);
  percentile available via `method="percentile"`. JSON export embeds
  CIs by default.
  ([`83d37f6`](https://github.com/mokhld/llm-consistency/commit/83d37f6))
- `get_scorer()` registry — `--scorer` flag now honoured in `run`.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- `.github/workflows/test.yml` — ruff + format + mypy + pytest on
  push/PR across `{3.11, 3.12, 3.13} × {ubuntu, macOS, windows}`
  (Windows on 3.12 only).
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- `CIRunner.failures` records which threshold tripped, logged at
  WARNING with actual vs target values.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))

### Fixed

- `BatchRunner` and `StreamingRunner` now catch per-variant provider
  exceptions and continue the batch; one provider failure no longer
  tears down the whole run. Failed variants are recorded as an error
  `ScoredResponse` so they participate in metrics.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- Non-`MCQuestion` items in a dataset are now counted and logged at
  WARNING instead of silently skipped.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- Empty datasets now raise `ValidationError` instead of producing an
  `EvaluationReport` over zero questions.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- `StreamingRunner` async generator cancels in-flight tasks on
  consumer early-break.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- CLI numeric range validation — `--num-variants`, `--concurrency`,
  `--seed` use `click.IntRange`; `--mca-threshold`, `--core-threshold`
  use `click.FloatRange(0.0, 1.0)`.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- Config loader raises `ValidationError` on YAML/TOML parse failure or
  non-mapping top level (previously returned `{}` silently).
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- JSON export is now atomic (tempfile + rename) with explicit UTF-8
  encoding and umask-derived permissions.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- OpenAI and Ollama providers raise explicit errors on unexpected
  response shapes instead of `IndexError`/`KeyError`.
  ([`c986eaa`](https://github.com/mokhld/llm-consistency/commit/c986eaa))
- CI green on Windows — loosened over-strict `latency_ms > 0`
  assertions to `>= 0.0` to accommodate `time.monotonic()` resolution
  on Windows.
  ([`5206d80`](https://github.com/mokhld/llm-consistency/commit/5206d80))
- OpenAI provider mypy fix — typed message payload as
  `list[ChatCompletionMessageParam]` under `TYPE_CHECKING` so the
  installed-SDK type check stays clean.
  ([`5206d80`](https://github.com/mokhld/llm-consistency/commit/5206d80))

### Documentation

- README updated with checkpoint/resume, export formats, dry-run, and
  bootstrap CI sections; pointer to `CONTRIBUTING.md` and
  `examples/`.

## [1.0] — 2026-02-22

Initial public release implementing the [CAT
framework](https://arxiv.org/abs/2512.23711) (Cavalin et al., 2025) for
LLM consistency evaluation.

### Added

- **CAT-faithful metrics** — MCA, CAR curve, CORE index, plus
  two-axis consistency: RC_correct (CAT-faithful) and RC_agree
  (answer stability).
- **Perturbation engine** — `OptionReorderPerturbation`,
  `FormatChangePerturbation`, `SeparatorChangePerturbation` with a
  registry pattern (`register_perturbation`, `get_perturbation`).
- **Provider layer** — OpenAI, Anthropic, Ollama, LiteLLM, plus
  `MockLLMProvider` for deterministic testing. Token-bucket rate
  limiter, exponential-backoff retry, USD budget enforcement with
  `BudgetExceededError`.
- **Runners** — `BatchRunner` (full pipeline), `StreamingRunner`
  (yields per-question), `CIRunner` (pass/fail exit codes against MCA
  + CORE thresholds).
- **Scoring** — `ExactMatchScorer` with cascading regex extraction
  plus `CustomScorerAdapter` for user callables.
- **Datasets** — `MCDataset.load()` auto-detects JSON, JSONL, and CSV
  formats; `OpenEndedDataset` and `CustomDataset` for in-memory
  questions.
- **CLI** — `llm-consistency run`, `compare`, `perturbations list`,
  `dataset validate`. YAML/TOML config support with CLI override.
- **Reports** — Rich console reporter with ASCII CAR curve; JSON export
  via `export_json`.
- **Packaging** — `py.typed` marker, strict mypy, ruff with `E/W/F/I/
  UP/B/SIM/N/D/C4/RUF/TC/PERF/T20/RET/PTH/ARG/PL`, optional extras
  (`openai`/`anthropic`/`ollama`/`litellm`/`embeddings`/`all`).
- **Quality** — 490 tests at 95.48% coverage.

[Unreleased]: https://github.com/mokhld/llm-consistency/compare/v1.0...HEAD
[1.0]: https://github.com/mokhld/llm-consistency/releases/tag/v1.0
