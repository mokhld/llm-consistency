# CLAUDE.md

Guidance for working in the `llm-consistency` repository, a Python implementation of the CAT framework (Cavalin et al., arXiv:2512.23711) for measuring how LLM accuracy changes when the same multiple-choice question is presented differently.

## What the package is for

- **What it does:** it loads multiple-choice datasets (JSON, JSONL or CSV, or the HuggingFace Hub via `MCDataset.load_from_hub`) and generates seeded perturbed variants of each question: `option_reorder`, `format_change` and `separator_change`. It queries an LLM provider (OpenAI, Anthropic, Ollama, LiteLLM or mock) and scores each answer by extracting the label. It then computes per-question RC_correct and RC_agree, and aggregate MCA, CAR curve, CORE and AGA, with bootstrap CIs, a McNemar paired test and power analysis. It is a library plus a `llm-consistency` CLI.
- **Value proposition (stated in the README):** measure how accuracy degrades under rephrasing, with paper-faithful metrics, seeded reproducibility and CI gating. Everything depends on the numbers being correct. Next come paid-API runs that are reliable, retried, cost-capped and resumable.
- **Users:**
  - ML researchers benchmarking robustness. They want paper fidelity and statistics.
  - ML engineers gating model or prompt changes in CI. They want exit codes they can trust and reports they can read.
- **Review and backlog:** `docs/REVIEW.md` holds the findings (A1-A23) and features (B1-B9) with statuses. `docs/FEATURES.md` holds implementation briefs an agent can pick up directly. Both are LOCAL files, gitignored because the repo is public. Never stage them. Update statuses there when you fix or build something.

## Codebase map

`src/llm_consistency/`:
- `types.py`: frozen dataclasses with `to_dict`/`from_dict`. These are `MCQuestion`, `MCOption`, `PresentedOption` (adds `original_label`), `PerturbedVariant` (with `presented_options`), `LLMResponse`, `ScoredResponse`, `QuestionConsistencyResult` (validated), `EvaluationConfig` (validated; `min_mca` is the CI pass target; also holds the prompt template, system prompt and decoding settings), and `EvaluationReport`. It also holds `GenerationParams` (not serialized) and the `KNOWN_SCORERS` constant.
- `perturbations.py`: the `BasePerturbation` ABC, the three built-ins, and a registry keyed by `PerturbationType.value`. Every built-in sets `presented_options`. The numbered format template presents labels "1".."n". `option_reorder` enumerates permutations only when there are 7 options or fewer; otherwise it samples.
- `scoring.py`: `_extract_mc_answer`, a cascading regex. It is case-sensitive unless the whole output is one character, takes the last "Answer: X", skips negated labels, and accepts numeric labels. It also holds `ExactMatchScorer`, `CustomScorerAdapter`, and `get_scorer` (only `exact_match`).
- `metrics.py`: pure functions. There are the paper metrics (`mca`, `car_curve`, `core_index` = AUCAR x norm-DTW, `agreement_gated_accuracy`), BCa/percentile bootstrap with a closed-form jackknife in the built-in `*_with_ci`, `compare_mca_paired` (exact McNemar), `validate_sample_size`, and `perturbation_impact`.
- `runners/`:
  - `_pipeline.py`: `process_question`, shared by both runners. It generates variants, queries under the semaphore, scores each response against the variant's presented options, and keys agreement by original label.
  - `_batch.py` (`BatchRunner`): a question pool bounded by `config.concurrency`, with optional checkpointing.
  - `_streaming.py` (`StreamingRunner`).
  - `_ci.py`: `CIRunner`, plus `gate_failures`, the single pass rule shared with the console.
  - `_checkpoint.py`: JSONL checkpoints. `CONFIG_HASH_FIELDS` is the result-affecting allow-list, and the file repairs a truncated last line on reopen.
  - `_metadata.py`: `RunMetadata`.
- `providers/`:
  - `_base.py`: `BaseLLMProvider.query()` reserves budget, takes a rate-limit token per attempt, retries with backoff and `retry-after`, calls `_send_request`, then settles the budget. It also defines `EmptyResponseError`.
  - Concrete providers (`_openai.py`, `_anthropic.py`, `_ollama.py`, `_litellm.py`) lazy-import their SDK and define their own retryable errors. Get them through `get_provider(name, **kwargs)`.
  - `_mock.py` overrides `query()` and bypasses budget, rate limit and retry.
  - `_cost.py` is a static pricing table, `_budget.py` does atomic reserve/settle, and `_rate_limit.py` is a token bucket.
- `datasets/`: `MCDataset` (format auto-detect, `utf-8-sig`, `load_from_hub(answer_format=...)`), `OpenEndedDataset` (loads and validates, but runners skip it), `CustomDataset` (in memory).
- `reports/`: the Rich console (`ConsoleReporter`) and exporters for JSON (atomic write, bootstrap CIs by default), CSV (formula-injection guarded), Markdown (escaped cells), HTML (escaped, no JS), and an ASCII CAR curve.
- `cli.py`: a click group with `run`, `compare`, `perturbations list` and `dataset validate`. `--config` is an eager callback that fills `ctx.default_map` through `_config_loader.run_defaults_from_config`. It accepts a `run:` section or flat keys and rejects unknown keys. Precedence is CLI flag, then config, then default.

Data flow: dataset -> `generate_variants_for_question` -> `render_prompt` -> `provider.query` -> `scorer.score` (against the presented options) -> `build_question_consistency_result` -> `EvaluationReport` -> metrics and exporters.

## Commands

- Setup: `uv sync --group dev`. The optional SDKs are NOT in the dev group: use `uv pip install -e '.[all]'`. CI installs `.[all]` on 3.11/3.12/3.13 on ubuntu and macOS, plus Windows on 3.12.
- Tests: `uv run pytest`, or `.venv/bin/python -m pytest`. `addopts` enforces `--cov-fail-under=95`, so a subset run needs `--no-cov` or it fails.
- Lint and types: `ruff check .`, `ruff format --check .`, `mypy src` (strict).
- CLI smoke test with no network: `uv run llm-consistency run -m mock-model -p mock -d examples/datasets/sample.jsonl [--dry-run]`.
- Examples: `uv run python examples/01_basic_mock.py` (all six use the mock provider).
- Release: publishing a GitHub release with tag `vX.Y.Z` runs `.github/workflows/release.yml`, which tests, builds and publishes to PyPI via trusted publishing. There is no approval gate. `workflow_dispatch` with `repository=testpypi` rehearses on TestPyPI. The version comes from the git tag (hatch-vcs). `_version.py` is generated and gitignored, and `dist/` holds stale local builds.

## Conventions and gotchas

- **Code style:** mypy strict, a large ruff rule set, Google docstrings, and `from __future__ import annotations`. Commits use Conventional Commits with no AI attribution.
- **Serialization:** new serialized fields need `from_dict` defaults, so old reports and checkpoints still load. `PerturbationType` is serialized by NAME in `to_dict` (`"OPTION_REORDER"`). `ScoredResponse.perturbation_type` stores `.value` (`"option_reorder"`).
- **Failed variants:** their `scoring_method` starts with `"error:"`. The CI gate, console, warnings and checkpoint skip logic all depend on that prefix.
- **Budget errors:** `BudgetExceededError` propagates and aborts a run. Every other provider exception becomes an error variant.
- **Budget ownership:** the provider enforces the budget. `EvaluationConfig.max_budget_usd` only takes effect when it is also passed to `get_provider` (the CLI does this; runners warn otherwise). A budget on an unpriced model raises `ValidationError` unless `pricing=` is given.
- **Checkpoint hash:** any new config field that changes results (prompt, decoding, variant sets) must be added to `CONFIG_HASH_FIELDS` in `runners/_checkpoint.py`.
- **Custom perturbations:** the runner resolves perturbations by `PerturbationType.value`, so a perturbation registered under a new name cannot be selected yet (backlog B4). Override a built-in with `register(..., force=True)`.
- **Test providers:** `MockLLMProvider` ignores the prompt, so it cannot catch scoring bugs. Use `tests/prompt_aware_providers.py` (`OracleProvider`, `PositionBiasedProvider`), which parse the rendered prompt. A correct model must score RC_correct == 1.0 under every perturbation.
- **SDK tests:** provider tests mock the SDK modules. `tests/test_providers_sdk_errors.py` exercises the real SDK error mapping and uses `importorskip`, so it is skipped in the default `.venv` and runs in CI.
- **mypy and numpy:** `pyproject.toml` skips following into numpy, because its stubs use 3.12 syntax that mypy rejects under `python_version = "3.11"` once the extras are installed.
- **Prompt contract:** `render_prompt` puts each variant in `config.prompt_template`, or `DEFAULT_PROMPT_TEMPLATE` (`runners/_pipeline.py`), which asks for `Answer: X` on the first line with the presented labels. Decoding settings default to None, meaning not sent. `process_question` passes `system=` and `generation=` to `provider.query` only when set, and `BaseLLMProvider` passes `generation=` to `_send_request` only when set, so older overrides keep working. anthropic 1.x has no `temperature` argument, so the Anthropic provider sends it in `extra_body`.

## Working Principles

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
