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
| Config & secrets | **pydantic-settings**; secrets in `.env` / gitignored `conf/*.local.yaml`; non-secret in `conf/*.yaml` |
| Notebooks | **jupytext** (pair to `.py`) + **nbstripout** (strip outputs) |
| Dependencies | **uv** (+ committed lockfile) |
| Quality | **ruff** (lint+format), **mypy** (strict), **pytest**, **pre-commit** |
| Last-mile / interop | pandas only where a library needs it (scikit-learn, statsmodels, plotting) |

## How I write code here (style)

- Type-hinted, with short docstrings that say *why*. `snake_case` functions/modules, `PascalCase`
  classes, `UPPER_SNAKE` constants. Functions read as verbs (`load_orders`, `compute_elasticity`).
- **Absolute imports from the package** (`from core.db.query import load_orders`). No `sys.path`
  hacks, no `%run`. The one sanctioned star-import is `from core.prelude import *` in notebooks
  (curated via `__all__`); library modules never use `import *`.
- ruff-formatted: double quotes, line length 100. Keep comments sparse and meaningful.
- Match the surrounding code's idioms and density.
- Notebooks read like a narrative (load → inspect → analyze → visualize) and delegate every non-trivial
  step to a named, tested function in the package.

## Standardized patterns (plan first, don't sprawl)

Prefer standardized, reusable units over ad-hoc functions — agree the shape before writing many of
them. Every reusable unit is a function with a **fixed input format → a predictable output**, and
cross-cutting boilerplate is owned by a decorator or shared base, not repeated.

- **Visualization.** Every chart is a `@chart`-decorated function `f(ax, ...)` that only *draws*; the
  decorator (`core/viz/base.py`) owns figure/axes creation, theme, title, saving, and returning the
  `Axes` (charts compose into grids via `base.grid`). Charts take prepared data — arrays
  (`y_true, y_score`) or a `pl.DataFrame` + columns — and live in
  `core/viz/{eda,model,cluster,explain,timeseries}.py`.
- **Compute ≠ present.** Heavy computation (fitting for elbow/silhouette, SHAP, PDP) lives in
  `analytics`/`modeling`; the chart receives the result. Cheap, plot-specific math may stay in the chart.
- **Imports.** Notebooks import the toolkit in one line — `from core.prelude import *`. Library modules
  keep explicit per-module imports; Python caches modules in `sys.modules`, so that costs nothing at
  runtime and keeps each module independent and type-checkable.

## Do

- **Scan, don't load.** Start from lazy Polars (`pl.scan_*`) or DuckDB over Parquet; select columns and
  filter rows early; stream when it won't fit; choose dtypes deliberately (`Int32`/`Float32`/`Categorical`).
- **Prefer Parquet** over CSV for anything that persists.
- **Parameterize all SQL** — bind values (`:param`), never string-format them into the query.
- **One typed loader per data source** (load → pin schema → return), reused everywhere.
- **Secrets never in git**: app env in `.env`, connection secrets in gitignored `conf/*.local.yaml`;
  read through `core.config` (`get_settings` / `get_connection` / `get_api`). One typed loader per source.
- **Promote** reusable notebook code into `core/` and import it back.
- Cache expensive pulls to Parquet rather than re-querying.

## Don't

- Don't load multi-GB CSVs straight into pandas.
- Don't put secrets in YAML, code, or notebooks; don't commit `.env` or notebook outputs.
- Don't build SQL by string-substituting runtime values.
- Don't use `import *` in library code (the notebook prelude is the one exception), `sys.path` edits,
  or `warnings.filterwarnings('ignore')` in library code — fix warnings instead. (The notebook
  prelude suppresses warnings for the interactive layer; library modules and tests must not.)
- Don't pull in image/CV or deep-learning stacks; this project is tabular/text/geo.
- Don't commit `data/` — treat it as a rebuildable cache.

## Project structure (target)

```
conf/        versioned, non-secret config (YAML): config.yaml, databases.yaml, logging.yaml
data/        gitignored datasets: raw/ interim/ processed/ external/
notebooks/   jupytext-paired exploration — commit the .py, outputs stripped
sql/         .sql files grouped by domain; parameterized
scripts/     one-off / dev scripts (e.g. sample-data generation)
core/        the package (flat layout at the repo root — no src/ wrapper)
  config.py     typed settings from .env + conf/*.yaml (single get_settings entry point)
  prelude.py    one-line notebook toolkit: from core.prelude import *
  METHODS.md    reference for every stats/ML function: when, how, what it offers statistically
  io/           readers, writers, parquet cache, typed source catalog (Polars/DuckDB)
  db/           pooled engines, parameterized typed query loaders
  api/          HTTP + GraphQL clients (auth, retries, REST & cursor pagination)
  features/     stateless transforms: clean, transform, temporal, period (MoM/QoQ/YoY), text, geo, validate
  analytics/    stats (summaries/distributions/MLE fits/effects/tests/MI/power), regression
                (OLS assumption checks: VIF/Breusch-Pagan/Durbin-Watson), experiment (A/B +
                Bayesian A/B, SRM, CUPED, mSPRT), causal (DiD/PSM/IPW/IV/ITT-TOT/subgroups/uplift)
  modeling/     registry (make_model), train (fit/predict/cross-val/partial_fit), tune, ensemble,
                imbalance, evaluate + compare (+ curves/permutation importance/RFECV), persist,
                split, preprocess, segment (clustering/PCA/t-SNE), anomaly, monitor (PSI/KS/label drift)
  decision/     contextual bandits (epsilon-greedy, Thompson, UCB, LinUCB) + optimization (LP, assignment)
  forecasting/  classical (arima/sarimax, ets) + ml-reduction forecasters (+ prediction intervals)
                + stationarity/seasonality diagnostics + rolling-origin backtest
  pricing/      demand elasticity (log-log) + price/revenue optimization (capability subpackage)
  kpi/          business KPIs: financial (revenue/economy), behaviour (GA/marketing), profit (cost-sensitive)
  viz/          base.py = @chart decorator + theme + grid; static charts by group:
                eda, model, cluster, explain, timeseries; plus interactive (Plotly,
                @interactive_chart): interactive
  utils/        memory profiling, structured logging, HTML report
tests/       pytest, mirrors core/
```

Organize `core/` by **capability and pipeline stage**, not by business entity. Add capability
subpackages (e.g. `pricing/`, `forecasting/`, `churn/`) inside this structure; they reuse the shared
`io`/`db`/`features`/`analytics` layers rather than re-implementing loaders.

## Tooling & commands

```bash
uv sync                 # create .venv and install from the lockfile
uv add <pkg>            # add a dependency
uv run pytest           # tests
uv run ruff check .     # lint   (uv run ruff format . to format)
uv run mypy core        # types
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
