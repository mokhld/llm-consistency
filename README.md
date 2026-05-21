# llm-consistency

**Measure how LLM accuracy degrades when users phrase the same question differently.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-490%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-95.48%25-brightgreen.svg)]()
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

# All providers
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
│ CORE            │ 0.5544 │  PASS  │
│ MCA             │ 0.0000 │  FAIL  │
│ Mean RC Correct │ 0.7667 │  FAIL  │
│ Mean RC Agree   │ 0.7667 │  PASS  │
└─────────────────┴────────┴────────┘
```

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
| `--config` | `-c` | — | Config file path (YAML or TOML), values merged as defaults |
| `--output` | `-o` | — | JSON report output path |
| `--perturbations` | | `option_reorder` | Perturbation types to apply (repeatable) |
| `--num-variants` | | `5` | Number of variants to generate per question per perturbation type |
| `--concurrency` | | `10` | Maximum concurrent API calls |
| `--seed` | | `42` | Random seed for reproducible perturbation generation |
| `--scorer` | | `exact_match` | Scoring method |
| `--mca-threshold` | | `1.0` | MCA threshold for pass/fail status display |
| `--core-threshold` | | — | CORE threshold for pass/fail (disabled when unset) |
| `--max-budget-usd` | | — | Maximum spend in USD (evaluation stops when exceeded) |
| `--ci` | | `false` | CI mode: suppresses console output, exits with code 0 (pass) or 1 (fail) |
| `--dry-run` | | `false` | Validate dataset, config, provider, and render one sample prompt without spending tokens |

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

Both YAML and TOML are supported. Config values serve as defaults that CLI flags override.

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

    # Run evaluation
    provider = get_provider("openai", model="gpt-5-mini")
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
with the same arguments and previously-completed questions are skipped
— the provider is not re-queried for them.

```python
report = await runner.run(
    dataset, config, provider, ExactMatchScorer(),
    seed=42,
    checkpoint_path="run-2026-05-21.jsonl",
)
```

The checkpoint header records a SHA-256 hash of the `EvaluationConfig`
and seed. Resuming with a different config or seed raises
`ValidationError` so results from incompatible runs can't be mixed.
A crash-truncated final line is detected and skipped on resume; all
earlier results remain intact. The dataset itself is *not* hashed —
keep it stable between resumes.

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
        return PerturbationType.FORMAT_CHANGE  # or define your own

    def generate_variants(
        self, question: MCQuestion, *, seed: int = 0, n: int | None = None
    ) -> tuple[PerturbedVariant, ...]:
        # Your perturbation logic here
        ...

register_perturbation("my_perturbation", MyPerturbation())
```

### Custom Scorers

```python
from llm_consistency import CustomScorerAdapter, ScoredResponse, LLMResponse, MCQuestion

# Full-signature scorer (receives both response and question, returns ScoredResponse)
def my_scorer(response: LLMResponse, question: MCQuestion) -> ScoredResponse:
    correct = next(o for o in question.options if o.is_correct)
    is_correct = response.extracted_answer == correct.label
    return ScoredResponse(
        question_id=response.question_id,
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        scoring_method="my_scorer",
    )

scorer = CustomScorerAdapter(my_scorer)
```

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
      --core-threshold 0.5 \
      --max-budget-usd 2.00 \
      --ci
```

The `--ci` flag:
- Suppresses console output (Rich tables, CAR curves)
- Exits with code **0** if all thresholds pass
- Exits with code **1** if any threshold fails

**MCA threshold semantics:** `--mca-threshold 0.8` requires that *every question* achieves RC_correct >= 0.8 (i.e., `mca(results, 0.8) == 1.0`). This is intentionally strict — a single inconsistent question fails the check.

## JSON Report Format

The `--output` flag produces a structured JSON report:

```json
{
  "config": {
    "model": "gpt-5-mini",
    "provider": "openai",
    "perturbation_types": ["OPTION_REORDER", "FORMAT_CHANGE"],
    "num_variants": 3,
    "mca_threshold": 1.0,
    "core_threshold": null
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
    "started_at": "2026-02-22T...",
    "finished_at": "2026-02-22T...",
    "model": "gpt-5-mini",
    "provider": "openai"
  }
}
```

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

```bash
# Clone and install
git clone https://github.com/mokhld/llm-consistency.git
cd llm-consistency
uv sync --group dev

# Run tests
uv run pytest

# Type check
uv run mypy src/llm_consistency/

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

All contributions must pass:
- `pytest` (490+ tests, >= 95% coverage)
- `mypy --strict` (no errors)
- `ruff check` and `ruff format` (no violations)

## License

MIT
