# Contributing to llm-consistency

Thanks for your interest. This document covers everything you need to
land a change: local setup, quality gates, extension points, and the
project's audit-driven roadmap.

## Local setup

```bash
git clone https://github.com/mokhld/llm-consistency.git
cd llm-consistency
uv sync --group dev
```

The `dev` group installs the test/lint/type dependencies. It does
**not** install the optional provider SDKs (`openai`, `anthropic`,
`ollama`, `litellm`). To work on those:

```bash
uv pip install -e '.[all]'   # all providers + embeddings
# or any subset:
uv pip install -e '.[openai,anthropic]'
```

CI runs against the `[all]` extras, so missing an optional dep locally
can let a type error or import bug slip past you. If you're touching
provider code, install the relevant extras.

## Quality gates

Every change must pass all four before merging:

```bash
uv run pytest                          # 591 tests, >= 95% coverage
uv run mypy --strict src/llm_consistency/
uv run ruff check src/ tests/ examples/
uv run ruff format --check src/ tests/ examples/
```

The coverage gate is enforced in `pyproject.toml`
(`--cov-fail-under=95`). New code generally needs tests — both happy
path and edge cases. See `tests/conftest.py` and the existing
`tests/test_*.py` files for fixtures and conventions.

### `mypy --strict` is mandatory

The project ships a `py.typed` marker, so downstream consumers see our
type annotations. Use the standard library types
(`list[int]`, `dict[str, str]`, `int | None`) — `from __future__ import
annotations` is enabled. If an external SDK has weak types, narrow at
the boundary rather than letting `Any` leak through.

### CI matrix

`.github/workflows/test.yml` runs ruff + format + mypy + pytest on the
full `{3.11, 3.12, 3.13} × {ubuntu, macOS, windows}` matrix (Windows is
3.12-only to keep the matrix tractable). `release.yml` only fires on
tags.

## Commit messages

Conventional Commits prefix every subject line:

| Prefix | Use for |
| --- | --- |
| `feat:` | New user-facing capability |
| `fix:` | Bug fix |
| `docs:` | Documentation-only change |
| `chore:` | Tooling, deps, CI, build |
| `test:` | Test-only change |
| `refactor:` | Internal restructuring with no behaviour change |
| `perf:` | Performance work |

Keep the subject line under ~70 chars. Detailed rationale goes in the
body. When a commit closes an `AUDIT.md` finding, mention the finding
ID (e.g. `R1`, `C3`) in the body so the audit's "what's been done"
section can be updated.

## Extension points

The library is designed to be extended without forking. The three
main extension points:

### Adding a perturbation

Subclass [`BasePerturbation`](src/llm_consistency/perturbations.py) and
register it. See `OptionReorderPerturbation`,
`FormatChangePerturbation`, and `SeparatorChangePerturbation` for
reference implementations.

```python
from llm_consistency import (
    BasePerturbation, MCQuestion, PerturbationType,
    PerturbedVariant, register_perturbation,
)

class MyPerturbation(BasePerturbation):
    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.FORMAT_CHANGE  # reuse or extend the enum

    def generate_variants(
        self, question: MCQuestion, *, seed: int = 0, n: int | None = None,
    ) -> tuple[PerturbedVariant, ...]:
        ...

register_perturbation("my_perturbation", MyPerturbation())
```

Constraints:
- Variants must be deterministic given `(question, seed, n)`.
- The variant's `is_correct` must follow the option *text*, not its
  label position.
- Tests must cover at least: identity not in output, requested `n`
  honoured, determinism across two calls with the same seed.

### Adding a scorer

Either subclass [`BaseScorer`](src/llm_consistency/scoring.py) and add
an entry to `_BUILTIN_SCORERS`, or wrap a callable via
[`CustomScorerAdapter`](src/llm_consistency/scoring.py).

Built-in scorers are exposed through the
`llm-consistency run --scorer <name>` flag via `get_scorer()`. If you
add a built-in, also add the name to the `KNOWN_SCORERS` set in
[`types.py`](src/llm_consistency/types.py).

For one-off scoring logic (regex extractors, model-specific
post-processing), `CustomScorerAdapter` is the right tool — see
[`examples/04_custom_scorer.py`](examples/04_custom_scorer.py).

### Adding a provider

Providers live in [`src/llm_consistency/providers/`](src/llm_consistency/providers/).
Each is a thin subclass of `BaseLLMProvider` whose `_send_request`
method maps the provider's SDK response into a `_RawResponse`.

To add a new provider `foo`:

1. Create `src/llm_consistency/providers/_foo.py` with a
   `FooProvider(BaseLLMProvider)` class. See
   [`_openai.py`](src/llm_consistency/providers/_openai.py) for the
   shape — handle `ImportError` on the SDK import with a clear
   "install llm-consistency[foo]" message.
2. Register the provider in two dicts in
   [`providers/__init__.py`](src/llm_consistency/providers/__init__.py)
   (`_PROVIDER_REGISTRY` and `_PROVIDER_CLASS_NAMES`).
3. Add an optional extra in `pyproject.toml`:
   `foo = ["foo-sdk>=1.0"]`, then update the `all` group.
4. Write tests under `tests/test_providers_foo.py` using `AsyncMock` —
   real network calls are not allowed in the suite.
5. If the provider has public pricing, add entries to the static
   pricing table in
   [`providers/_cost.py`](src/llm_consistency/providers/_cost.py).

## Reporting bugs and proposing changes

File a GitHub issue for bugs or enhancement proposals. Include a
minimal reproducer for bugs and the motivating use case for
enhancements.

## Running the examples

Every example in [`examples/`](examples/) is a runnable end-to-end
demonstration using the mock provider. Run them from the repo root:

```bash
uv run python examples/01_basic_mock.py
uv run python examples/02_dry_run.py
uv run python examples/03_checkpoint_resume.py
uv run python examples/04_custom_scorer.py
uv run python examples/05_export_formats.py
uv run python examples/06_compare_models.py
```

If you change a public API, run the relevant examples to verify the
demonstrated UX still works. They're the canonical "how does this
look from the outside" smoke test.

## Reporting bugs and security issues

Bugs: open a GitHub issue with a minimal reproduction.

Security issues: please email the maintainer privately rather than
filing a public issue.

## License

By contributing, you agree your contributions are licensed under the
MIT License (see [`LICENSE`](LICENSE)).
