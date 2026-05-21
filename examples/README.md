# Examples

Runnable end-to-end demonstrations of the llm-consistency Python API. Every
script uses the `MockLLMProvider`, so they execute without API keys,
network access, or paid tokens — clone the repo, run
`uv sync --group dev`, then:

```bash
uv run python examples/01_basic_mock.py
```

| Script | What it shows |
| --- | --- |
| [`01_basic_mock.py`](01_basic_mock.py) | Minimal end-to-end run: load dataset, configure, query mock provider, display metrics. |
| [`02_dry_run.py`](02_dry_run.py) | Programmatic dry-run — variant generation + prompt rendering + cost estimation without provider calls. Mirrors the CLI `--dry-run` flag. |
| [`03_checkpoint_resume.py`](03_checkpoint_resume.py) | Crash a run mid-flight, resume from the JSONL checkpoint, observe that completed questions are not re-queried. |
| [`04_custom_scorer.py`](04_custom_scorer.py) | `CustomScorerAdapter` wrapping a regex scorer for `\boxed{X}` LaTeX-style answers. |
| [`05_export_formats.py`](05_export_formats.py) | Export the same `EvaluationReport` to JSON, CSV, Markdown, and HTML side-by-side. |
| [`06_compare_models.py`](06_compare_models.py) | A/B two mock providers (stable vs unstable) on one dataset; print CORE, MCA, AGA deltas. |

The shared dataset is [`datasets/sample.jsonl`](datasets/sample.jsonl) — 5
general-knowledge MCQs in JSONL format. Replace it with your own
JSON/JSONL/CSV file to evaluate on a real benchmark.

## Running against a real provider

Swap `provider="mock"` and `get_provider("mock", ...)` for any of:

```python
provider = get_provider("openai", model="gpt-5-mini")   # OPENAI_API_KEY env
provider = get_provider("anthropic", model="claude-haiku-4-5-20251001")
provider = get_provider("ollama", model="llama3.2")     # ollama serve
provider = get_provider("litellm", model="gpt-5-mini")  # 100+ backends
```

Install the matching extras first — `uv sync --group dev` does *not*
pull in optional provider SDKs:

```bash
uv pip install -e '.[openai]'      # or [anthropic], [ollama], [litellm], [all]
```
