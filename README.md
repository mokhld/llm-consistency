# llm-consistency

**Measure how LLM accuracy degrades when users phrase the same question differently.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)]()
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)]()

llm-consistency implements the [CAT framework](https://arxiv.org/abs/2512.23711) (Cavalin et al., 2025) for evaluating LLM robustness under controlled input variations. It automates the full pipeline: load a dataset, apply deterministic perturbations (option reorder, format change, separator change), query any LLM provider, score responses, and compute paper-faithful MCA, CAR, and CORE metrics — all from a single CLI command or Python API call.

## Why This Matters

A model that gets a question right once but wrong when the options are reordered isn't reliable. llm-consistency reveals two dimensions of this problem:

- **Correctness-consistency (RC_correct):** Does the model get it right across all perturbations?
- **Answer-agreement (RC_agree):** Does the model give the *same* answer, right or wrong?

A model can be **stable-but-wrong** (high agreement, low correctness) or **unstable-but-sometimes-right** (low agreement, moderate correctness). Accuracy alone hides both failure modes.

## Key Features

- **CAT Framework Metrics** — Paper-faithful MCA, CAR curves, CORE index, plus agreement-gated accuracy (AGA)
- **Deterministic Perturbations** — Option reorder, format change, separator change with seeded reproducibility
- **Multi-Provider Support** — OpenAI, Anthropic, Ollama, LiteLLM, and a mock provider for offline testing
- **CI/CD Integration** — Pass/fail exit codes with configurable MCA and CORE thresholds
- **Plugin System** — Register custom perturbation types, scorers, and providers
- **Budget Enforcement** — Set USD spending caps with automatic cost tracking
- **Async Pipeline** — Concurrent API calls with token-bucket rate limiting and exponential backoff
- **Multiple Dataset Formats** — JSON, JSONL, CSV with automatic format detection
- **Bootstrap Confidence Intervals** — Optional statistical significance testing

## Installation

```bash
pip install llm-consistency
```

With provider extras:

```bash
# Single provider
pip install llm-consistency[openai]
pip install llm-consistency[anthropic]
pip install llm-consistency[ollama]
pip install llm-consistency[litellm]

# HuggingFace Hub dataset loader
pip install llm-consistency[huggingface]

# All providers + HuggingFace loader
pip install llm-consistency[all]
```

Development setup:

```bash
git clone https://github.com/mokhld/llm-consistency.git
cd llm-consistency
uv sync --group dev
```

## Quick Start

### 1. Prepare a Dataset

Create a JSON file with multiple-choice questions:

```json
{
  "questions": [
    {
      "id": "q1",
      "stem": "What is the capital of France?",
      "options": [
        {"label": "A", "text": "London", "is_correct": false},
        {"label": "B", "text": "Paris", "is_correct": true},
        {"label": "C", "text": "Berlin", "is_correct": false},
        {"label": "D", "text": "Madrid", "is_correct": false}
      ]
    }
  ]
}
```

### 2. Run an Evaluation

```bash
export OPENAI_API_KEY="sk-..."

llm-consistency run \
  --model gpt-5-mini \
  --provider openai \
  --dataset questions.json \
  --output report.json \
  --perturbations option_reorder \
  --perturbations format_change \
  --num-variants 3 \
  --seed 42
```

### 3. Read the Results

```
         Evaluation Summary
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric          ┃  Value ┃ Status ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ CORE            │ 0.5544 │  n/a   │
│ MCA(1.00)       │ 0.0000 │  FAIL  │
│ Mean RC Correct │ 0.7667 │        │
│ Mean RC Agree   │ 0.7667 │        │
└─────────────────┴────────┴────────┘
```

The status column uses the same rule as `--ci` (see [CI/CD Integration](#cicd-integration)). CORE shows `n/a` when no `--core-threshold` is set. The two means are informational. When any provider call failed, a `Failed variants` row shows how many.

## CLI Reference

### `llm-consistency run`

Execute an evaluation run against a single model.

```
llm-consistency run [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | *required* | LLM model identifier (e.g., `gpt-5-mini`, `claude-sonnet-4-20250514`) |
| `--provider` | `-p` | *required* | Provider name: `openai`, `anthropic`, `ollama`, `litellm`, `mock` |
| `--dataset` | `-d` | *required* | Path to dataset file (JSON, JSONL, or CSV) |
| `--config` | `-c` | none | Config file path (YAML or TOML), values merged as defaults |
| `--output` | `-o` | none | Report output path. The format follows the extension (see below) |
| `--perturbations` | | `option_reorder` | Perturbation types to apply (repeatable) |
| `--num-variants` | | `5` | Maximum variants per question per perturbation type. A type with fewer distinct variants yields fewer (a 4-option question has 23 reorderings, 7 format templates, 8 separators) |
| `--concurrency` | | `10` | Maximum concurrent API calls |
| `--seed` | | `42` | Random seed for reproducible perturbation generation |
| `--scorer` | | `exact_match` | Scoring method |
| `--mca-threshold` | | `1.0` | Consistency level c for MCA: a question passes when its RC_correct >= c |
| `--min-mca` | | `1.0` | Minimum MCA at `--mca-threshold` for a pass. `1.0` requires every question to pass |
| `--core-threshold` | | none | Minimum CORE for a pass (not checked when unset) |
| `--max-budget-usd` | | none | Spending cap in USD. The run stops with an error before a request that would exceed it (see [Budget cap](#budget-cap)) |
| `--rpm` | | `60` | Provider rate limit in requests per minute |
| `--prompt-template` | | built-in | Prompt template. `{question}` (required) is replaced by the question and its options, `{labels}` by the option labels shown. The default adds a first-line `Answer: X` instruction (see [Prompt and decoding](#prompt-and-decoding)) |
| `--system-prompt` | | none | System prompt sent with every request |
| `--temperature` | | not sent | Sampling temperature, 0.0 to 2.0. When unset the provider default applies. Use `0` for models that accept it |
| `--max-tokens` | | not sent | Maximum output tokens per response. When unset the provider default applies (1024 for Anthropic) |
| `--generation-seed` | | not sent | Provider sampling seed, separate from `--seed`. Anthropic has no seed and ignores it |
| `--ci` | | `false` | CI mode: skips the console summary, exits 0 when every check passes and 1 otherwise |
| `--dry-run` | | `false` | Validate dataset, config, provider, and render one sample prompt without spending tokens. Prints the number of provider calls the run will make and warns when the cost estimate exceeds `--max-budget-usd` |

**Output format is detected from the `--output` extension:**

| Extension | Format |
|-----------|--------|
| `.json` (default) | Full report with aggregate metrics + CIs |
| `.csv` | Per-question flat table (one row per QCR) |
| `.md` / `.markdown` | Human-readable summary with metric tables |
| `.html` / `.htm` | Self-contained single-page HTML (inline CSS, no JS) |

**Example with all perturbation types:**

```bash
llm-consistency run \
  -m gpt-5-mini -p openai \
  -d dataset.json -o report.json \
  --perturbations option_reorder \
  --perturbations format_change \
  --perturbations separator_change \
  --num-variants 5 --seed 42 \
  --max-budget-usd 1.00
```

### Budget cap

`--max-budget-usd` is enforced by the provider. Before each request it reserves an estimated cost, and if that would take total spend over the cap it raises `BudgetExceededError`. The run then stops: the command prints `Error: Budget exceeded: ...`, exits with code 1 (also under `--ci`), and writes no report, because a partial run would give misleading metrics.

The cap needs a price for the model. When `--max-budget-usd` is set and the model is not in the built-in pricing table (`providers/_cost.py`), the command refuses to start. From Python you can pass `pricing=CostPerToken(...)` to `get_provider`. Use `--dry-run` first to see the number of calls and a cost estimate.

Transient API errors (rate limits, timeouts, connection errors, HTTP 408/409/429 and 5xx) are retried with backoff, honouring `Retry-After`. A variant that still fails is recorded as a failed variant, counted in the console summary, and fails `--ci`.

### `llm-consistency compare`

Compare multiple models on the same evaluation.

```
llm-consistency compare [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | *required* | Config file with `models` list (YAML or TOML) |
| `--output` | `-o` | — | Output directory for per-model reports |
| `--format` | | `json` | Per-model file format: `json`, `csv`, `md`, or `html` |

**Config file format:**

```yaml
models:
  - model: gpt-5-mini
    provider: openai
  - model: claude-haiku-4-5-20251001
    provider: anthropic
dataset: questions.json
perturbations:
  - option_reorder
  - format_change
num_variants: 3
seed: 42
```

`models` and `dataset` are required. The other keys are `perturbations`, `num_variants`, `concurrency`, `scorer`, `seed`, `mca_threshold`, `min_mca`, `core_threshold`, `max_budget_usd`, `rpm`, `prompt_template`, `system_prompt`, `temperature`, `max_tokens` and `generation_seed`, with the same meaning and defaults as the `run` options. `max_budget_usd` applies to each model separately. Unknown keys are rejected.

Every provider is created before the first model runs, so a bad provider name, a missing SDK or an unpriced model fails before any spend. Models run one after another. Each model's summary is printed, and its report is written to `--output` with run metadata, as soon as that model finishes, so a later failure does not lose earlier results. File names come from the model name with characters other than letters, digits, `.`, `_` and `-` replaced by `_` (`openai/gpt-4o-mini` becomes `openai_gpt-4o-mini.json`); repeated names get `-2`, `-3` suffixes.

After the last model, a table compares CORE, MCA at `mca_threshold`, mean RC_correct, mean RC_agree, and the p-value of McNemar's exact test (`compare_mca_paired`) on per-question MCA pass/fail against the first model:

```
Comparison (p-value: McNemar exact test on MCA(1.00) pass/fail against gpt-5-mini)
Model                        CORE  MCA(1.00)  RC_correct  RC_agree  p-value
---------------------------------------------------------------------------
gpt-5-mini                 0.6120     0.4000      0.8133    0.8400        -
claude-haiku-4-5-20251001  0.7015     0.5333      0.8667    0.8933   0.0391
```

### `llm-consistency perturbations list`

Show all registered perturbation types.

```bash
$ llm-consistency perturbations list
Available perturbation types:
  - format_change
  - option_reorder
  - separator_change
```

### `llm-consistency dataset validate`

Validate a dataset file format and report question count.

```
llm-consistency dataset validate <PATH> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `mc` | Dataset type: `mc` or `open-ended` |

```bash
$ llm-consistency dataset validate questions.json
Valid mc dataset: 15 questions
```

## Configuration Files

Both YAML and TOML are supported. Config values serve as defaults that CLI flags override: an explicit flag beats the config file, and the config file beats the built-in default.

Put the settings under a `run` section (YAML `run:`, TOML `[run]`) as below, or at the top level of the file. Keys are the `run` option names with underscores (`num_variants`, `mca_threshold`, `min_mca`, `max_budget_usd`, `rpm`, ...), plus `dataset` for the dataset path. A key the command does not know, such as a typo, is an error that lists the valid keys.

**YAML example (`config.yaml`):**

```yaml
run:
  model: gpt-5-mini
  provider: openai
  perturbations:
    - option_reorder
    - format_change
  num_variants: 3
  seed: 42
  concurrency: 5
  mca_threshold: 0.8
  max_budget_usd: 1.00
```

**TOML example (`config.toml`):**

```toml
[run]
model = "gpt-5-mini"
provider = "openai"
perturbations = ["option_reorder", "format_change"]
num_variants = 3
seed = 42
concurrency = 5
mca_threshold = 0.8
max_budget_usd = 1.00
```

Use with: `llm-consistency run -c config.yaml -d dataset.json`

## Prompt and decoding

Each variant is sent as one user message. By default the question and its options are followed by an instruction to put the answer on the first line:

```
What is the capital of France?
A. London
B. Paris
C. Berlin
D. Madrid

Answer with the label of the correct option. The first line of your response must be "Answer: X", where X is one of A, B, C, D.
```

The label list is the one the variant shows, so the numbered `format_change` layout asks for one of `1, 2, 3, 4`. The instruction follows the fixed answer format of the CAT paper (section 4.1). The extractor reads the last `Answer: X` in a response, so a model that reasons first and states its answer at the end is still scored correctly. Use `--dry-run` to see the exact prompt, system prompt and decoding settings a run will send.

`--prompt-template` replaces the default. `{question}` is required and is replaced by the question and its options. `{labels}` is optional. Any other placeholder is an error; write a literal brace as `{{` or `}}`. To send the question and options with no instruction, as earlier releases did, pass `--prompt-template "{question}"`. A config file is the easiest place for a multi-line template:

```yaml
prompt_template: |
  {question}

  Reply with the label of the correct option only ({labels}).
system_prompt: You are taking a multiple-choice exam.
temperature: 0
max_tokens: 256
```

`--temperature`, `--max-tokens` and `--generation-seed` are sent only when set; otherwise each provider's own default applies. For repeatable results, use `--temperature 0` with models that accept it. At the default temperature (1.0 for OpenAI and Anthropic), part of the disagreement between variants is sampling noise rather than sensitivity to the perturbation, which inflates the drop in RC_agree and CORE. The default is "not sent" because some models reject the setting: the OpenAI gpt-5 family and other reasoning models accept only their default temperature; on Anthropic, Claude Opus 4.7 and later Opus models and the Claude Fable models reject any temperature, Claude Sonnet 5 accepts only the default, and Opus 4.6, Sonnet 4.6 and Haiku 4.5 accept it. Such a request fails with HTTP 400 and is recorded as a failed variant, so check a new model with a small run first. Anthropic accepts temperatures from 0 to 1.

| Setting | OpenAI | Anthropic | Ollama | LiteLLM |
|---------|--------|-----------|--------|---------|
| `temperature` | `temperature` | `temperature` | `options.temperature` | `temperature` |
| `max_tokens` | `max_completion_tokens` | `max_tokens` (1024 when unset) | `options.num_predict` | `max_tokens` |
| `generation_seed` | `seed` | not supported, ignored | `options.seed` | `seed` |

A seed makes sampling more repeatable where the provider supports it, but no provider guarantees identical outputs.

The prompt template, system prompt and decoding settings are part of the checkpoint hash, so a run cannot resume a checkpoint written with different values.

From Python, set the same fields on `EvaluationConfig`. The default template is `llm_consistency.DEFAULT_PROMPT_TEMPLATE`. The runners call `provider.query(prompt, question_id, system=..., generation=GenerationParams(...))` and leave out each argument that is not set. A custom provider that overrides `query()` or `_send_request()` should accept both keyword arguments. One that does not keeps working until a system prompt or decoding setting is configured; after that, every request fails with `TypeError`.

## Dataset Formats

### JSON

```json
{
  "questions": [
    {
      "id": "q1",
      "stem": "Question text?",
      "options": [
        {"label": "A", "text": "Option A", "is_correct": false},
        {"label": "B", "text": "Option B", "is_correct": true},
        {"label": "C", "text": "Option C", "is_correct": false},
        {"label": "D", "text": "Option D", "is_correct": false}
      ]
    }
  ]
}
```

### JSONL

One question per line:

```jsonl
{"id": "q1", "stem": "Question?", "options": [{"label": "A", "text": "Opt A", "is_correct": false}, {"label": "B", "text": "Opt B", "is_correct": true}]}
{"id": "q2", "stem": "Question 2?", "options": [{"label": "A", "text": "Opt A", "is_correct": true}, {"label": "B", "text": "Opt B", "is_correct": false}]}
```

### CSV

```csv
id,stem,option_a,option_b,option_c,option_d,correct
q1,What is 2+2?,3,4,5,6,B
```

Format is auto-detected from file extension (`.json`, `.jsonl`, `.csv`).

### HuggingFace Hub

Load MC datasets directly from the Hub (requires the `huggingface`
extra: `pip install llm-consistency[huggingface]`):

```python
from llm_consistency import MCDataset

# Defaults match the cais/mmlu schema:
#   question -> stem, choices (list[str]) -> options, answer (int) -> correct.
dataset = MCDataset.load_from_hub(
    "cais/mmlu",
    name="abstract_algebra",
    split="validation",
)

# Custom schemas via column-mapping kwargs:
dataset = MCDataset.load_from_hub(
    "my-org/mc-eval",
    question_col="prompt",
    choices_col="options",
    answer_col="label",
    id_col="qid",
)
```

The `answer` column accepts an `int` index, a single-letter label
(`"A"`/`"a"`), or a string matching one of the choice texts exactly.
Up to 26 options are supported (labels `A`–`Z`). Any extra kwargs are
forwarded to `datasets.load_dataset` (e.g. `revision="main"`,
`streaming=False`).

## Python API

### Basic Evaluation

```python
import asyncio
from llm_consistency import (
    MCDataset, EvaluationConfig, PerturbationType,
    ExactMatchScorer, BatchRunner, ConsoleReporter,
    get_provider, export_json, mca, core_index,
)

async def main():
    # Load dataset
    dataset = MCDataset.load("questions.json")

    # Configure evaluation
    config = EvaluationConfig(
        model="gpt-5-mini",
        provider="openai",
        perturbation_types=(
            PerturbationType.OPTION_REORDER,
            PerturbationType.FORMAT_CHANGE,
        ),
        num_variants=3,
        concurrency=5,
        max_budget_usd=1.00,
    )

    # Run evaluation. The provider enforces the budget, so pass it here;
    # runner.run() raises BudgetExceededError if the cap is reached.
    provider = get_provider(
        "openai",
        model="gpt-5-mini",
        max_budget_usd=config.max_budget_usd,
    )
    runner = BatchRunner()
    report = await runner.run(dataset, config, provider, ExactMatchScorer(), seed=42)

    # Display results
    ConsoleReporter().display(report, threshold=0.8)

    # Export JSON
    export_json(report, "report.json", metadata=runner.last_metadata)

    # Access metrics directly
    print(f"CORE: {core_index(report.results):.4f}")
    print(f"MCA(0.8): {mca(report.results, 0.8):.4f}")

asyncio.run(main())
```

### Resumable Long Runs

Pass `checkpoint_path` to `BatchRunner.run()` to persist each completed
`QuestionConsistencyResult` to a JSONL file as soon as it's computed. If
the run crashes (network failure, OOM, ctrl-c, host reboot), restart
with the same arguments and previously-completed questions are skipped;
the provider is not re-queried for them.

```python
report = await runner.run(
    dataset, config, provider, ExactMatchScorer(),
    seed=42,
    checkpoint_path="run-2026-05-21.jsonl",
)
```

The checkpoint header records a SHA-256 hash of the settings that change
results: `model`, `provider`, `perturbation_types`, `scorer`,
`num_variants` and the seed. Resuming with a different value for any of
them raises `ValidationError`, so results from incompatible runs can't be
mixed. Changing `concurrency`, `max_budget_usd` or the pass/fail
thresholds is allowed. Checkpoints written by releases before 1.1 are
rejected, because those releases scored `option_reorder` variants against
the wrong labels.

Questions with a failed variant (a provider error) are not written to
the checkpoint, so a resume retries them. A crash-truncated final line is
skipped on resume and removed before new records are appended; all
earlier results remain intact. The dataset itself is not hashed: results
for question IDs no longer in the dataset are dropped with a warning, but
keep the questions themselves stable between resumes.

### Alternative Report Formats

```python
from llm_consistency import export_csv, export_html, export_markdown

export_csv(report, "report.csv")           # flat per-question table
export_markdown(report, "report.md",       # human-readable summary
                metadata=runner.last_metadata)
export_html(report, "report.html",         # self-contained HTML page
            metadata=runner.last_metadata)
```

The CLI auto-detects format from the `--output` extension: `.csv` →
CSV, `.md`/`.markdown` → Markdown, `.html`/`.htm` → HTML, anything
else → JSON. The HTML output bundles its CSS inline and uses no
JavaScript or external assets — open the file directly from disk.

### Custom Perturbations

```python
from llm_consistency import (
    BasePerturbation, MCQuestion, PerturbedVariant,
    PerturbationType, register_perturbation,
)

class MyPerturbation(BasePerturbation):
    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.FORMAT_CHANGE

    def generate_variants(
        self, question: MCQuestion, *, seed: int = 0, n: int | None = None
    ) -> tuple[PerturbedVariant, ...]:
        # Your perturbation logic here
        ...

# Replace the built-in format_change generator in this Python process.
register_perturbation("format_change", MyPerturbation(), force=True)
```

Set `presented_options` on every variant: a tuple of `PresentedOption(label=..., text=..., is_correct=..., original_label=...)` giving the label the model sees for each option and that option's label in the original question. Answers are scored against these presented labels, and agreement is counted by original label. Without it, the runner assumes every option kept its original label and issues a `UserWarning`; that assumption is wrong for any perturbation that reorders or relabels options.

The runners select generators by `PerturbationType` value (`option_reorder`, `format_change`, `separator_change`), so today a custom generator is used only when it is registered under one of those values with `force=True`, as above. A generator registered under a new name, such as `"my_perturbation"`, shows up in `list_registered_perturbations()` but cannot be selected for a run yet. Registration is per process, so it does not affect separate `llm-consistency` CLI invocations.

### Custom Scorers

```python
from llm_consistency import CustomScorerAdapter, ScoredResponse, LLMResponse, MCQuestion

# Simple form: the adapter extracts the answer label and passes
# (extracted_label, correct_label) to your function.
scorer = CustomScorerAdapter(lambda extracted, correct: extracted == correct, simple=True)

# Full form: receives the raw response and the question as this variant
# presented it (labels match what the model saw), returns a ScoredResponse.
def my_scorer(response: LLMResponse, question: MCQuestion) -> ScoredResponse:
    correct = next(o for o in question.options if o.is_correct)
    is_correct = response.raw_output.strip().startswith(f"Answer: {correct.label}")
    return ScoredResponse(
        question_id=response.question_id,
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        scoring_method="my_scorer",
    )

scorer = CustomScorerAdapter(my_scorer)
```

`response.extracted_answer` is empty when a scorer is called; read `response.raw_output`.

### Offline Testing with Mock Provider

```python
from llm_consistency import get_provider

# Default: always returns "A"
mock = get_provider("mock", model="mock-model")

# Response map: specific answers per question
mock = get_provider("mock", model="mock-model",
                    responses={"q1": "B", "q2": "C", "q3": "A"})

# Cycling list: rotates through answers
mock = get_provider("mock", model="mock-model",
                    responses=["A", "B", "C", "D"])
```

## Metrics

### MCA (Minimum-Consistency Accuracy)

**MCA_cat(c)** = fraction of questions where RC_correct >= c

At a given consistency threshold *c*, what proportion of questions does the model answer correctly across *all* perturbation variants? MCA_cat(1.0) is the strictest: only questions where the model got every single variant correct.

### CAR Curve (Consistency-Accuracy Relationship)

Plots MCA_cat(c) across thresholds c = [0.0, 0.1, ..., 1.0]. Shows how accuracy degrades as you demand higher consistency. A flat curve near 1.0 means the model is both accurate and consistent.

### CORE Index (Consistency-Oriented Robustness Estimate)

**CORE = AUCAR * normalised-DTW**

A single scalar (0 to 1) combining the area under the CAR curve with how closely it tracks the ideal y=1.0 curve. Higher is better. CORE=1.0 means perfect accuracy at every consistency threshold.

### AGA (Agreement-Gated Accuracy)

**AGA(tau)** = mean accuracy among questions where RC_agree >= tau

Filters to questions where the model at least *agrees with itself*, then measures accuracy. Useful for identifying the "ambiguity zone" where the model vacillates.

### Bootstrap Confidence Intervals

Every aggregate metric ships with a `*_with_ci` sibling that returns a `MetricResult(value, ci_lower, ci_upper, n_samples, confidence, method)`. The default bootstrap is **BCa** (bias-corrected accelerated); percentile is available via `method="percentile"`.

```python
from llm_consistency import (
    mca_with_ci,
    core_index_with_ci,
    agreement_gated_accuracy_with_ci,
    car_curve_with_ci,
)

# Each call returns a MetricResult with point estimate + 95% BCa CI.
mca_ci   = mca_with_ci(results, threshold=0.8, n_bootstrap=1000, seed=42)
core_ci  = core_index_with_ci(results, n_bootstrap=1000, seed=42)
aga_ci   = agreement_gated_accuracy_with_ci(results, tau_agree=0.8, seed=42)
curve_ci = car_curve_with_ci(results, n_bootstrap=1000, seed=42)
# curve_ci is list[(threshold, MetricResult)]
```

The JSON report (`-o report.json`) embeds these CIs by default — see `aggregate.core_index_ci`, `aggregate.mca_at_threshold_ci`, and `aggregate.car_curve_ci`. The scalar fields (`core_index`, `mca_at_threshold`, `car_curve`) remain unchanged for backward compatibility.

The lower-level bootstrap primitives are exposed too: `bootstrap_ci(...)` (percentile) and `bootstrap_ci_bca(...)` (BCa), both accepting an arbitrary `statistic` callable.

### Sample-Size Power Analysis

`validate_sample_size` sizes a two-sided one-sample test of a single
proportion, such as one model's MCA at a fixed threshold against a
reference value, with Cohen's h as the effect size:

```python
from llm_consistency import validate_sample_size

result = validate_sample_size(
    n=len(dataset),
    effect_size=0.5,   # Cohen's h: 0.2 small, 0.5 medium, 0.8 large
    alpha=0.05,
    power=0.80,
)
# {
#   "n": 150.0, "effect_size": 0.5, "alpha": 0.05, "target_power": 0.80,
#   "power_at_n": 0.99, "observed_power": 0.99, "recommended_n": 32.0,
# }
```

`recommended_n` is the smallest `n` that reaches `target_power` at the
given effect size. `power_at_n` is the power at your `n` if the true
effect is exactly `effect_size`. It is computed from that assumed
effect, not observed in your data; `observed_power` is the same value
under its older name. The function does not size a comparison of two
models: the power of `compare_mca_paired` (McNemar's test) depends on
how many questions the two models disagree on, which it does not model.

Emits `UserWarning` when `n < 200`. That cut-off is this package's rule
of thumb for flagging small studies, not a figure from the CAT paper.

### Failure rate by perturbation type

`perturbation_impact` returns the failure rate of the variants of each
perturbation type:

```python
from llm_consistency import perturbation_impact

impact = perturbation_impact(report)
# {PerturbationType.OPTION_REORDER: 0.12,
#  PerturbationType.FORMAT_CHANGE: 0.41,
#  PerturbationType.SEPARATOR_CHANGE: 0.05}
```

Values are the mean failure rate (`1 - mean is_correct`) across all
variants of each type. The runner pipeline annotates each
`ScoredResponse` with its source `perturbation_type`; legacy
responses without the annotation are silently skipped.

The failure rate includes the model's base error rate. A model that is
70% accurate and never changes its answer scores about 0.30 for every
type. The unperturbed question is never asked, so the value cannot say
how much of the error a perturbation caused. It is not a variance
decomposition. Compare types against each other with that in mind.

### Paired Model Comparison

`compare_mca_paired` runs McNemar's exact binomial test on
per-question MCA pass/fail outcomes — the right test when two models
are evaluated on the same dataset:

```python
from llm_consistency import compare_mca_paired

result = compare_mca_paired(results_a, results_b, threshold=0.8)
# PairedTestResult(statistic=1.0, p_value=0.21875, n_discordant=6,
#                  method="mcnemar_exact")

if result.p_value < 0.05:
    print(f"Models differ at MCA(0.8); discordant n={result.n_discordant}")
```

Questions present in only one set are silently dropped. A small
`n_discordant` means the test is underpowered — pair this with
`validate_sample_size` to know whether your dataset can actually
distinguish the two models.

## Perturbation Types

| Type | Description | What It Tests |
|------|-------------|---------------|
| `option_reorder` | Shuffles the ordering of MC answer options | Position bias — does the model favour option A? |
| `format_change` | Changes question formatting template | Format sensitivity — does layout affect answers? |
| `separator_change` | Modifies delimiters between options | Parsing robustness — do separators matter? |

All perturbations are seeded for reproducibility. The same seed + question always produces the same variants.

## Provider Setup

### OpenAI

```bash
pip install llm-consistency[openai]
export OPENAI_API_KEY="sk-..."

llm-consistency run -m gpt-5-mini -p openai -d dataset.json
```

### Anthropic

```bash
pip install llm-consistency[anthropic]
export ANTHROPIC_API_KEY="sk-ant-..."

llm-consistency run -m claude-haiku-4-5-20251001 -p anthropic -d dataset.json
```

### Ollama (Local)

```bash
pip install llm-consistency[ollama]
# Ensure Ollama is running: ollama serve

llm-consistency run -m llama3.2 -p ollama -d dataset.json
```

### LiteLLM (Universal)

```bash
pip install llm-consistency[litellm]
# Set the appropriate API key for your backend

llm-consistency run -m gpt-5-mini -p litellm -d dataset.json
```

### Mock (Offline Testing)

No extra install needed:

```bash
llm-consistency run -m mock-model -p mock -d dataset.json
```

## CI/CD Integration

Add consistency checks to your CI pipeline:

```yaml
# GitHub Actions
- name: LLM Consistency Check
  run: |
    llm-consistency run \
      -m gpt-5-mini -p openai \
      -d tests/eval_dataset.json \
      --perturbations option_reorder \
      --perturbations format_change \
      --num-variants 3 \
      --mca-threshold 0.8 \
      --min-mca 0.95 \
      --core-threshold 0.5 \
      --max-budget-usd 2.00 \
      --output consistency-report.json \
      --ci
```

The `--ci` flag skips the console summary and exits with code **0** when every check below passes, **1** otherwise:

1. **MCA:** `mca(results, mca_threshold) >= min_mca`.
2. **CORE:** `core_index(results) >= core_threshold`, only when `--core-threshold` is set.
3. **Failed variants:** no variant failed with a provider error after retries.

Each failed check is logged to stderr, for example `MCA check failed: MCA(threshold=0.800) = 0.920, expected >= 0.950` or `Error check failed: 3 of 450 variants failed with provider errors`. The `--output` report is written before the command exits, so a failed build still has it. A budget stop (see [Budget cap](#budget-cap)) exits with code 1 and writes no report.

**Threshold semantics:** `--mca-threshold` is the consistency level c: a question passes when its RC_correct >= c. `--min-mca` is the share of questions that must pass. The defaults (`--mca-threshold 1.0 --min-mca 1.0`) require every question to be answered correctly on every variant, so a single inconsistent question fails the build. With `--mca-threshold 0.8 --min-mca 0.95`, at least 95% of questions must be correct on at least 80% of their variants. The console summary applies the same rule.

## JSON Report Format

With a `.json` path (or any extension other than `.csv`, `.md`, `.markdown`, `.html` and `.htm`), the `--output` flag produces a structured JSON report:

```json
{
  "config": {
    "model": "gpt-5-mini",
    "provider": "openai",
    "perturbation_types": ["OPTION_REORDER", "FORMAT_CHANGE"],
    "scorer": "exact_match",
    "num_variants": 3,
    "concurrency": 10,
    "max_budget_usd": null,
    "mca_threshold": 1.0,
    "min_mca": 1.0,
    "core_threshold": null,
    "ci_mode": false,
    "prompt_template": null,
    "system_prompt": null,
    "temperature": null,
    "max_tokens": null,
    "generation_seed": null
  },
  "results": [
    {
      "question_id": "q1",
      "rc_correct": 0.7667,
      "rc_agree": 0.7667,
      "total_variants": 4,
      "correct_count": 3,
      "answer_distribution": {"B": 3, "C": 1},
      "scored_responses": [...]
    }
  ],
  "total_questions": 5,
  "total_variants": 20,
  "mean_rc_correct": 0.7667,
  "mean_rc_agree": 0.7667,
  "aggregate": {
    "core_index": 0.5544,
    "mca_at_threshold": 0.6,
    "car_curve": [[0.0, 1.0], [0.1, 1.0], ...],
    "core_index_ci": {
      "value": 0.5544,
      "ci_lower": 0.4112,
      "ci_upper": 0.6831,
      "n_samples": 5,
      "confidence": 0.95,
      "method": "bca"
    },
    "mca_at_threshold_ci": {"value": 0.6, "ci_lower": 0.2, "ci_upper": 1.0, "n_samples": 5, "confidence": 0.95, "method": "bca"},
    "car_curve_ci": [[0.0, {"value": 1.0, "ci_lower": 1.0, "ci_upper": 1.0, "n_samples": 5, "confidence": 0.95, "method": "bca"}], ...]
  },
  "metadata": {
    "package_version": "0.1.0",
    "python_version": "3.12.4",
    "timestamp": "2026-02-22T10:15:03.412345+00:00",
    "config_snapshot": {"model": "gpt-5-mini", "provider": "openai", ...},
    "perturbation_seed": 42,
    "model": "gpt-5-mini",
    "provider": "openai"
  }
}
```

`metadata.timestamp` is when the run started (UTC), and `config_snapshot` is the same dictionary as `config`.

## Architecture

```
Dataset (JSON/JSONL/CSV)
    |
    v
Perturbation Engine ──> Generates N variants per question
    |                    (option_reorder, format_change, separator_change)
    v
LLM Provider Layer ──> Async queries with rate limiting, retries, budget
    |                   (OpenAI, Anthropic, Ollama, LiteLLM, Mock)
    v
Scoring Engine ──> Exact match with cascading regex extraction
    |               (or custom scorer via adapter)
    v
Metrics Engine ──> MCA, CAR curve, CORE index, AGA, bootstrap CIs
    |
    v
Reporter ──> Console (Rich tables + ASCII CAR curve) | JSON export
```

## Research Foundation

This package implements and extends the CAT framework from:

> **CAT: A Metric-Driven Framework for Analyzing the Consistency-Accuracy Relation of LLMs under Controlled Input Variations**
> Cavalin et al., arXiv:2512.23711, November 2025

The paper defines MCA, CORE, and CAR curves but leaves perturbation generation manual. llm-consistency automates the full pipeline and adds:

1. **Two-axis consistency** — RC_correct (CAT-faithful) + RC_agree (answer stability)
2. **Built-in perturbation generators** with plugin system
3. **Provider-agnostic evaluation** (any LLM backend)
4. **CI/CD mode** with pass/fail thresholds
5. **Enterprise plumbing** — retries, rate limiting, cost accounting

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for local setup, quality
gates, commit conventions, and how to add a perturbation, scorer, or
provider. Quick start:

```bash
git clone https://github.com/mokhld/llm-consistency.git
cd llm-consistency
uv sync --group dev
uv run pytest
```

Runnable end-to-end examples live in [`examples/`](examples/) — every
script uses the mock provider, so they execute without API keys or
network access:

```bash
uv run python examples/01_basic_mock.py
uv run python examples/05_export_formats.py
```

Changes are tracked in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT
