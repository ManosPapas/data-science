# CLAUDE.md

Always-on guidance for Claude Code in this repository. Read it before doing anything here.

## What this project is

A **commercial / decision-science** data-science workspace. The work is making business decisions
from data across domains like revenue management, pricing, forecasting, customer analytics,
experimentation & causal inference, optimization, and risk/fraud — the techniques generalize across
industries (banking, retail, energy, telecom, transport, etc.).

- **Data scale:** gigabytes, held in memory on a local machine.
- **Data types:** tabular (the backbone), text, and geo. **Not** images, and no exotic/extreme formats.
- **Where work happens:** exploration in **Jupyter in the browser**; durable logic in importable `.py`
  modules so every change is a clean, reviewable diff on GitHub.

## Goal

Reproducible, memory-efficient, reviewable analysis that drives commercial decisions. Notebooks are for
*seeing* and showing a result; modules are for *keeping*. If something is worth reusing or trusting
tomorrow, it lives in the package with a test — not in a notebook cell.

## Tech stack (the defaults — don't substitute without being asked)

| Concern | Tool |
|---|---|
| DataFrames / compute | **Polars** (lazy/streaming) + **DuckDB** (SQL over files, out-of-core) |
| Config & secrets | **pydantic-settings** + `.env` (secrets) + `conf/*.yaml` (non-secret) |
| Notebooks | **jupytext** (pair to `.py`) + **nbstripout** (strip outputs) |
| Dependencies | **uv** (+ committed lockfile) |
| Quality | **ruff** (lint+format), **mypy** (strict), **pytest**, **pre-commit** |
| Last-mile / interop | pandas only where a library needs it (scikit-learn, statsmodels, plotting) |

## How I write code here (style)

- Type-hinted, with short docstrings that say *why*. `snake_case` functions/modules, `PascalCase`
  classes, `UPPER_SNAKE` constants. Functions read as verbs (`load_orders`, `compute_elasticity`).
- **Absolute imports from the package** (`from <package>.db.query import load_orders`). Never
  `from x import *`, never `sys.path` hacks, never `%run`.
- ruff-formatted: double quotes, line length 100. Keep comments sparse and meaningful.
- Match the surrounding code's idioms and density.
- Notebooks read like a narrative (load → inspect → analyze → visualize) and delegate every non-trivial
  step to a named, tested function in the package.

## Do

- **Scan, don't load.** Start from lazy Polars (`pl.scan_*`) or DuckDB over Parquet; select columns and
  filter rows early; stream when it won't fit; choose dtypes deliberately (`Int32`/`Float32`/`Categorical`).
- **Prefer Parquet** over CSV for anything that persists.
- **Parameterize all SQL** — bind values (`:param`), never string-format them into the query.
- **One typed loader per data source** (load → pin schema → return), reused everywhere.
- **Secrets through settings**: read via a typed `Settings`/`get_settings()`, sourced from `.env`.
- **Promote** reusable notebook code into `src/<package>/` and import it back.
- Cache expensive pulls to Parquet rather than re-querying.

## Don't

- Don't load multi-GB CSVs straight into pandas.
- Don't put secrets in YAML, code, or notebooks; don't commit `.env` or notebook outputs.
- Don't build SQL by string-substituting runtime values.
- Don't use `import *`, `sys.path` edits, or `warnings.filterwarnings('ignore')` — fix warnings instead.
- Don't pull in image/CV or deep-learning stacks; this project is tabular/text/geo.
- Don't commit `data/` — treat it as a rebuildable cache.

## Project structure (target)

```
conf/        versioned, non-secret config (YAML): config.yaml, databases.yaml, logging.yaml
data/        gitignored datasets: raw/ interim/ processed/ external/
notebooks/   jupytext-paired exploration — commit the .py, outputs stripped
sql/         .sql files grouped by domain; parameterized
src/<package>/
  config.py     typed settings from .env + conf/*.yaml (single get_settings entry point)
  io/           readers, writers, parquet cache (Polars/DuckDB)
  db/           pooled engines, parameterized typed query loaders
  api/          HTTP clients (retries, auth, pagination)
  features/     reusable feature engineering (e.g. temporal)
  analytics/    stats, effect sizes, CIs, experiment/A-B analysis
  viz/          consistent plot defaults
  utils/        logging, memory profiling
tests/       pytest, mirrors src/<package>/
```

Organize `src/` by **capability and pipeline stage**, not by business entity. Add capability
subpackages (e.g. `pricing/`, `forecasting/`, `churn/`) inside this structure; they reuse the shared
`io`/`db`/`features`/`analytics` layers rather than re-implementing loaders.

## Tooling & commands

```bash
uv sync                 # create .venv and install from the lockfile
uv add <pkg>            # add a dependency
uv run pytest           # tests
uv run ruff check .     # lint   (uv run ruff format . to format)
uv run mypy src         # types
uv run jupyter lab      # analyze in the browser
```

**Dependency supply-chain policy:** pin via the lockfile and keep a release-age gate in `pyproject.toml`
(`[tool.uv] exclude-newer = "7d"`, the org default of 7 days) so freshly-published — possibly
compromised — releases aren't pulled in. Requires uv ≥ 0.9.17 for the relative-duration form.

## How to work with me

- Do exactly what's asked, then stop. Don't gold-plate or expand scope; don't append a menu of extra
  proposals. If you have an idea beyond the ask, mention it in one line — don't act on it.
- Be concise and direct.
- Confirm before destructive or hard-to-reverse actions (deleting/overwriting files, anything outward-facing).
