# Data Science

A commercial / decision-science data-science workspace — revenue management, pricing, forecasting,
customer analytics, experimentation, optimization, and risk, across industries. Built for
**gigabyte-scale tabular, text, and geo data** analyzed locally in Jupyter, with the durable logic in
the importable **`core`** package so every change is a clean diff on GitHub.

**Stack:** Polars + DuckDB · pydantic-settings + `.env` · jupytext + nbstripout · uv · ruff + mypy + pytest.

## Quickstart

```bash
# 1. Install uv once:  https://docs.astral.sh/uv/getting-started/installation/
uv sync                       # create .venv and install everything
Copy-Item .env.example .env   # then fill in real secrets  (bash: cp .env.example .env)
uv run jupyter lab            # analyze in the browser
```

Confirm the package imports: `uv run python -c "import core"`.

## How it works

- **Explore in notebooks, keep logic in `core/`.** Notebooks under `notebooks/` are where you run and
  see analysis and plots; anything reusable moves into the `core` package and is imported back
  (`from core.db.query import load_orders`). uv installs `core` in editable mode, so imports work the
  same in notebooks, scripts, and tests.
- **Scan, don't load.** Lazy Polars / DuckDB over Parquet for GB-scale data; choose dtypes deliberately.
- **Secrets in `.env`** (typed via `core.config`), non-secret config in `conf/*.yaml`.
- Notebooks are paired to `.py` (jupytext) and stripped of outputs (nbstripout) — commit the `.py`.

## Layout

```
conf/      versioned, non-secret config (config.yaml, databases.yaml, logging.yaml)
data/      datasets (gitignored): raw/ interim/ processed/ external/
notebooks/ jupytext-paired exploration — commit the .py
sql/       .sql files by domain; parameterized
core/      the package: config + io/ db/ api/ features/ analytics/ viz/ utils/
tests/     pytest
```

## Common commands

```bash
uv run jupyter lab        # notebooks in the browser
uv add <package>          # add a dependency
uv run pytest             # tests
uv run ruff check .       # lint   (uv run ruff format .  to format)
uv run mypy core          # type-check
```

Working conventions (what to do / not do, naming, where code goes) live in
[CLAUDE.md](CLAUDE.md).
