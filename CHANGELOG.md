# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-09-24

### Added

- Prompt and decoding settings: `EvaluationConfig.prompt_template`,
  `system_prompt`, `temperature`, `max_tokens` and `generation_seed`,
  the `run` flags `--prompt-template`, `--system-prompt`,
  `--temperature`, `--max-tokens` and `--generation-seed`, and the same
  keys in `run` and `compare` config files. A template has a required
  `{question}` placeholder and an optional `{labels}` placeholder, and is
  validated when the config is built. Decoding settings default to None
  and are then not sent, so the provider default applies. All five are
  part of the checkpoint hash.
- `GenerationParams` and a `generation=` keyword argument on
  `BaseLLMProvider.query()` and `_send_request()`. The runners build it
  from the config and pass it with each request. OpenAI sends
  `temperature`, `max_completion_tokens` and `seed`; Anthropic sends
  `temperature` (through `extra_body`, since anthropic 1.x has no
  temperature argument) and `max_tokens`, keeps 1024 as the default
  `max_tokens`, and ignores the seed; Ollama sends `temperature`,
  `num_predict` and `seed` in `options`; LiteLLM sends `temperature`,
  `max_tokens` and `seed`.
- `DEFAULT_PROMPT_TEMPLATE`. `render_prompt(variant, template, labels)`
  renders a variant into a template.
- Dry-run prints the decoding settings and the system prompt, and shows
  the sample prompt exactly as it will be sent.
- `--min-mca` / `min_mca` (`EvaluationConfig.min_mca`, default 1.0):
  the MCA pass target, separate from the consistency level
  `--mca-threshold`.
- `--rpm` / `rpm`: requests per minute passed to the provider's rate
  limiter.
- `pricing=CostPerToken(...)` provider argument for models missing from
  the pricing table, and a price for `claude-haiku-4-5-20251001`.
- `EmptyResponseError` for truncated or reasoning-only responses.
- `PresentedOption` and `PerturbedVariant.presented_options`, recording
  the labels each variant showed the model and the matching original
  labels.
- `BaseLLMProvider.max_budget_usd` property.
- `compare` prints a comparison table with CORE, MCA, mean RC and the
  McNemar p-value against the first model.
- `MCDataset.load_from_hub(answer_format=...)`, with numeric-string
  answers such as HellaSwag's "2" and an error for ambiguous answers.
- Dry-run warns when the estimated cost exceeds `--max-budget-usd`.

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

### Changed

- **The default prompt changed, so every reported number changes.** Each
  prompt now ends with an instruction to put the answer on the first
  line as `Answer: X`, where X is one of the labels the variant shows
  (`A, B, C, D`, or `1, 2, 3, 4` for the numbered layout). Accuracy,
  RC_correct, RC_agree, MCA and CORE are not comparable with earlier
  releases. To send the bare question and options as before, pass
  `--prompt-template "{question}"` (or `prompt_template="{question}"`).
  `render_prompt(variant)` also returns the prompt in the default
  template now.
- The runners pass `system=` and `generation=` to `provider.query()`
  only when they are set. A custom provider that overrides `query()` or
  `_send_request()` without these keyword arguments keeps working until
  a system prompt or decoding setting is configured; after that, each
  request fails with `TypeError` and is recorded as a failed variant.
- `option_reorder` and numbered-format answers are scored against the
  options as the model saw them, and `answer_distribution` is keyed by
  each option's label in the original question. Scores for
  `option_reorder` runs are not comparable with earlier releases (see
  Fixed).
- Scorers, including `CustomScorerAdapter` callables, receive an
  `MCQuestion` built from the variant's presented options instead of the
  original question. A custom scorer that looks answers up by the
  original label (for example, an external answer key) must use the
  `is_correct` flags on the options it receives instead.
- The answer extractor matches labels case-sensitively unless the whole
  output is a single character, so "Answer: b" is no longer read as B.
- `BudgetExceededError` now aborts `BatchRunner`, `StreamingRunner` and
  `CIRunner` runs instead of being recorded as failed variants. The CLI
  exits 1 with a clear message and writes no report.
- `--ci` also fails when any variant failed with a provider error.
  Console MCA status uses the same rule as CI, CORE shows "n/a" without
  a threshold, and the Mean RC rows no longer show PASS/FAIL.
- Config files reject unknown keys and list the valid ones.
- Setting `max_budget_usd` for a model with no known price raises
  `ValidationError`; pass `pricing=CostPerToken(...)` to the provider.
- `KNOWN_SCORERS` lists only implemented scorers (`exact_match`).
  `EvaluationConfig(scorer="llm_judge")` or `"semantic_similarity"`, and
  loading a report saved with either name, now raise `ValidationError`.
- Providers no longer retry `PermissionError` or HTTP 400/401/403/404.
- The rate limiter's initial burst is a tenth of a minute of requests
  rather than a full minute, and each retry takes a token. Because the
  limiter used to run at twice the configured rate, runs at the default
  `--rpm 60` now take about twice as long; raise `--rpm` for local models
  or higher API quotas.
- `option_reorder` on questions with 8 or more options now samples
  permutations directly, so the same seed gives different variants than
  earlier releases for those questions.
- `BatchRunner` evaluates up to `config.concurrency` questions at once.
  `StreamingRunner` evaluates that many ahead of the consumer and still
  yields in dataset order.
- Questions with failed variants are not written to the checkpoint, so
  a resume retries them. The checkpoint config hash covers only
  result-affecting fields (model, provider, perturbation types, scorer,
  num_variants, prompt template, system prompt, temperature, max_tokens,
  generation seed, seed), so changing concurrency, budget or thresholds
  no longer blocks a resume. The checkpoint format is now version 2;
  version 1 checkpoints from earlier releases are rejected, because
  their `option_reorder` results were scored against the wrong labels.
- Variants from a custom perturbation that does not set
  `presented_options` trigger a `UserWarning`, because the runner then
  assumes every option kept its original label.
- `core_index` requires a threshold grid that includes 0.0 and 1.0;
  metric thresholds must lie in [0, 1]; `confidence` must lie in (0, 1)
  and `n_bootstrap` must be at least 1.
- `QuestionConsistencyResult` validates its counts and rates at
  construction.
- `validate_sample_size` returns `power_at_n` (`observed_power` is kept)
  and is documented as a one-sample proportion test. Its small-sample
  warning no longer cites the CAT paper.

### Fixed

- `option_reorder` variants were scored against the original question's
  labels, so a model that always chose the correct option scored about
  0.22 RC_correct on four-option questions, and agreement was measured
  on letter positions. The numbered format template's answers ("2")
  were never extracted.
- The answer extractor misread "Answer: Definitely C" (as D), "The
  answer is clearly B." (as C), sentence-initial "A", and negated labels.
- Transient provider errors (rate limits, timeouts, connection errors,
  HTTP 408/409/429/5xx, Anthropic 529) were never retried and were
  scored as wrong answers. Each provider now retries them and honours
  `retry-after`.
- `--max-budget-usd` was never passed to the provider, so no budget was
  enforced from the CLI. Concurrent requests could also overshoot the
  cap several times over. Budget checks and reservations are now atomic;
  with a budget set, the first request runs alone and later requests
  reserve the largest cost seen so far (and at least an estimate based on
  prompt length). A request that times out or is cancelled keeps its
  reservation. The cap can still be exceeded by the in-flight requests if
  a response costs more than any before it. The runners warn when
  `EvaluationConfig.max_budget_usd` is set but the provider was created
  without a budget.
- The rate limiter ran at twice the configured rate.
- `BatchRunner` ran questions one at a time, so `--concurrency` above
  the number of variants per question had no effect.
- `option_reorder` enumerated every permutation before sampling (3.6M
  tuples, about 500 MB, for 10 options). It now samples directly when
  there are more than 7 options.
- The README's `run:` / `[run]` config format did not load, and config
  values were silently dropped. The `dataset` key is now honoured.
- `--ci -o` wrote no report. `--ci` required MCA(c) == 1.0 while the
  console showed PASS for MCA(c) >= c; the new `--min-mca` target sets
  the pass rule for both.
- A second resume after two crashes corrupted the checkpoint file.
  Duplicate records are deduplicated, and results for questions no
  longer in the dataset are dropped on resume.
- `compare_mca_paired` raised `OverflowError` past 1074 discordant
  pairs. P-values are now exact at any size.
- BCa intervals counted bootstrap ties as below the estimate, biasing
  MCA and CAR intervals downwards. Ties now count as half.
- `export_json` with CIs took about 100 s on 14,042 questions because
  the jackknife was O(n^2). The built-in CI functions now use a
  closed-form jackknife (under 2 s).
- `compare` crashed on model names containing `/` after every model had
  run, and lost all results. Reports are now written as each model
  finishes, under safe, unique file names, with run metadata.
- Dry-run counted `questions x num_variants` calls, ignoring the number
  of perturbation types.
- Refusals, truncated responses and reasoning-only responses became
  empty answers scored as wrong. Refusal text is now returned as the
  answer, and truncated or reasoning-only responses raise
  `EmptyResponseError`, recorded as failed variants.
- Dataset loaders failed on UTF-8 files with a BOM, and malformed JSON
  or a missing option field produced raw tracebacks. They now raise
  `ValidationError` with the file and position.
- Unknown providers, missing optional SDKs, and unregistered
  perturbation types (e.g. `paraphrase`) now give clean CLI errors.
- `core_index` could return values outside [0, 1] for custom threshold
  grids, and `-0.0` for empty input.
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

### Security

- CSV exports prefix cells that start with `=`, `+`, `-`, `@`, tab or
  CR with a quote, to prevent formula injection. Markdown table cells
  are escaped, so model output cannot break the table or inject HTML.
- Provider error text stored in reports and checkpoints is truncated to
  200 characters, with API keys and bearer tokens redacted.

### Documentation

- README and CONTRIBUTING corrected: budget semantics, CI rules, config
  file format, JSON metadata keys, custom perturbation registration and
  `presented_options`, the custom scorer example (it read
  `response.extracted_answer`, which is always empty),
  `validate_sample_size` and `perturbation_impact` descriptions. Removed
  links to a missing `AUDIT.md`.
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

[Unreleased]: https://github.com/mokhld/llm-consistency/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/mokhld/llm-consistency/compare/v1.0...v1.1.0
[1.0]: https://github.com/mokhld/llm-consistency/releases/tag/v1.0
