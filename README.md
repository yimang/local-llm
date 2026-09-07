# local-llm

Completely vibe-coded experiments with local large language models.

## Card Oracle

[Trading-card projects](card-oracle/README.md), starting with
[query segmentation](card-oracle/query-segmentation/README.md).

## Development setup

This is a non-packaged project using Astral's Python toolchain:

- [uv](https://docs.astral.sh/uv/) manages Python, the virtual environment,
  dependencies, and the lockfile.
- [Ruff](https://docs.astral.sh/ruff/) provides linting, import sorting, and
  formatting.
- [ty](https://docs.astral.sh/ty/) provides static type checking.

```bash
# Install the pinned Python version and reproduce the locked environment.
uv python install
uv sync --locked

# Run Python tools inside the managed environment.
uv run python --version
uv run mlx_lm.benchmark --help

# Run all static checks.
uv run ruff format --check .
uv run ruff check .
uv run ty check

# Apply safe formatting and lint fixes while developing.
uv run ruff format .
uv run ruff check --fix .
```

Use `uv add <package>` and `uv remove <package>` when changing dependencies.
Use `uv add --dev <package>` for development-only tools. Commit both
`pyproject.toml` and `uv.lock`; do not install project dependencies with `pip`
or edit `.venv` directly.

## First project: MacBook M5 Pro benchmarking

The first project will benchmark local LLM performance on a MacBook M5 Pro. The goal is to compare models and runtimes using practical measurements such as:

- Time to first token
- Prompt processing speed
- Token generation speed
- Memory usage
- Power and thermal behavior
- Output quality for real-world tasks

Expect quick experiments, reproducible results, and plenty of iteration.
