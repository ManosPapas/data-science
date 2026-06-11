# Data Science

A **commercial / decision-science** workspace: turning data into business decisions across revenue
management, pricing, forecasting, customer analytics, experimentation & causal inference,
optimization, and risk/fraud — techniques that generalize across industries (banking, retail,
energy, telecom, transport, …).

Built for **gigabyte-scale tabular, text, and geo data** analyzed locally in **Jupyter**, with all
durable logic in the importable **`core`** package so every change is a clean, reviewable diff on
GitHub. Notebooks are for *seeing* a result; `core` is for *keeping* it.

**Stack:** Polars + DuckDB · pydantic-settings + `.env` · jupytext + nbstripout · uv · ruff + mypy(strict) + pytest.

---

## What it offers

`core/` is a batteries-included toolkit, organized by capability and pipeline stage. Everything is
type-hinted, tested, and importable in one line from notebooks (`from core.prelude import *`).

| Area | Module | What you get |
|---|---|---|
| **Read data** | `io` | Lazy/eager readers (Parquet, CSV, JSON, NDJSON, Excel), DuckDB SQL over files, a `@cached` Parquet loader cache |
| | `db` | Pooled SQLAlchemy engines from named connections; **parameterized** SQL loaders (`read_sql`/`write_sql`) |
| | `api` | httpx clients (auth + retries), REST pagination, and GraphQL (queries + Relay cursor pagination) |
| **Prepare** | `features.clean` | dtype inference/fix, fill missing, dedupe, winsorize, text cleanup, downcast for memory |
| | `features.transform` | reshape, aggregate, join (with cardinality checks), bin, rank, sample, share-of-total |
| | `features.temporal` | calendar parts, holidays, lags, rolling windows, cyclical encodes |
| | `features.period` | rolling windows (7/28/30/60/90/120d…) + period-over-period (WoW/MoM/QoQ/YoY) |
| | `features.validate` / `text` / `geo` | schema checks · normalize/TF-IDF · haversine/bounding-box |
| **Analyze** | `analytics.stats` | summaries, distributions, hypothesis tests, effect sizes, mutual information, power |
| | `analytics.experiment` | A/B analysis for means and conversions (lift, CI, verdict) |
| | `analytics.causal` | difference-in-differences, uplift, propensity scores + matching |
| **Model** | `modeling` | 40+ models (`registry`), leakage-free `split`/`preprocess`, `train`, `tune`, `compare` (leaderboard + paired tests), `evaluate`, `ensemble`, `imbalance`, `segment` (clustering/PCA), `anomaly`, versioned `persist` |
| **Forecast** | `forecasting` | one interface over baselines, ARIMA/SARIMAX, ETS, and ML-reduction; rolling-origin backtest |
| **Decide** | `decision` | contextual bandits (ε-greedy, Thompson, UCB1, LinUCB) and optimization (LP, assignment) |
| **Price** | `pricing` | demand elasticity estimation (log-log) + revenue/profit-maximizing price optimization |
| **Measure** | `kpi` | ~30 financial KPIs, ~28 behaviour KPIs, and cost-sensitive `profit` curves |
| **Visualize** | `viz` | a `@chart` decorator + theme + grid; static charts for EDA, models, clustering, explainability, time series — plus **interactive Plotly** charts (`viz.interactive`) |
| **Utilities** | `utils` | memory profiling, structured logging, HTML reports |

See the runnable [example notebooks](#example-notebooks) for these in action end-to-end.

---

## Install

Requires **Python 3.13+**. Two supported paths:

### Option A — `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) gives a committed lockfile and a supply-chain release-age gate.

```bash
uv sync --all-extras          # create .venv and install everything (deps + extras + dev tools)
uv run python scripts/make_sample_data.py   # build the sample datasets
uv run jupyter lab            # analyze in the browser
```

### Option B — plain `venv` + `pip`

Use this if `uv` can't run in your environment (e.g. a locked-down Windows machine where AppLocker
blocks `uv.exe`). It installs the same things; you just run tools via `python -m …`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1                      # bash/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[geo,explain,excel,gam,imbalance,interactive]"
python -m pip install jupyterlab jupytext ipykernel ruff mypy pytest pre-commit nbstripout types-PyYAML
python -m ipykernel install --user --name data-science --display-name "Python (data-science)"
python scripts\make_sample_data.py              # build the sample datasets
python -m jupyterlab                            # analyze in the browser
```

> On a locked-down machine the pip-generated `.exe` shims (`jupyter`, `mypy`, `pytest`) may be
> blocked, while the venv's `python.exe` is not — so always invoke tools as `python -m <tool>`
> (e.g. `python -m jupyterlab`, `python -m pytest`). See [Good to know](#good-to-know).

Confirm it worked: `python -c "import core; print('ok')"`.

---

## Quickstart

1. **Install** (above) and **generate the sample data**: `python scripts/make_sample_data.py`
   → writes `data/raw/{transactions.csv, customers.parquet, daily_sales.parquet}` (seeded, so it's
   reproducible; `data/` is a gitignored, rebuildable cache).
2. **Open JupyterLab**, pick the **`Python (data-science)`** kernel, and run the example notebooks
   **in order** (notebook 01 caches a clean frame that 02 reads).
3. In any notebook, `from core.prelude import *` gives you the whole toolkit.

---

## Example notebooks

Five end-to-end notebooks under `notebooks/`, each a `load → inspect → analyze → visualize`
narrative that delegates every non-trivial step to a tested `core` function:

| Notebook | Demonstrates |
|---|---|
| `01_cleaning_and_eda` | Profiling a messy export → fixing dtypes, filling gaps, de-duping, winsorizing, downcasting, validating → distribution & relationship EDA |
| `02_features_kpis_periods` | Feature engineering + a customer join → financial & behaviour KPIs → MoM/QoQ/YoY and per-segment group analysis |
| `03_churn_modeling` | Stratified split → leakage-free pipelines → multi-model leaderboard + paired test → held-out evaluation → **profit-based threshold** → explainability → versioned save |
| `04_segmentation_and_experiments` | k-means (elbow/silhouette) + PCA, anomaly detection, an A/B test with power analysis, and a causal cross-check |
| `05_forecasting` | Decomposition/ACF/PACF → time features → forecaster bake-off on a holdout → rolling-origin backtest |

---

## How it works

- **Explore in notebooks, keep logic in `core/`.** Notebooks are where you run and see analysis;
  anything reusable is promoted into the package and imported back. `core` is installed editable, so
  imports work the same in notebooks, scripts, and tests — no `sys.path` hacks.
- **Notebooks are jupytext `.py` files** (percent format). The repo commits the readable `.py`
  (clean diffs, no outputs); the `.ipynb` is a local, gitignored pairing. Edit either and saving
  syncs the other. To get a runnable `.ipynb`: in Lab right-click a `.py` → **Open With → Jupytext
  Notebook**, or generate them with `python -m jupytext --to notebook notebooks/*.py`.
- **Scan, don't load.** Start from lazy Polars (`scan_*`) or DuckDB over Parquet; select columns and
  filter rows early; choose dtypes deliberately (`Int32`/`Float32`/`Categorical`).
- **Config & secrets** are typed via `core.config`: app env in `.env`, non-secret config in
  `conf/*.yaml`, connection/API secrets in gitignored `conf/*.local.yaml` (deep-merged on top).
  Read through `get_settings()` / `get_connection(name)` / `get_api(name)`.

---

## Layout

```
conf/        versioned, non-secret config (config.yaml, databases.yaml, apis.yaml, logging.yaml)
data/        datasets (gitignored, rebuildable): raw/ interim/ processed/ external/
notebooks/   jupytext-paired exploration — commit the .py, outputs stripped
scripts/     one-off / dev scripts (e.g. make_sample_data.py)
sql/         .sql files grouped by workspace/domain; parameterized
core/        the package: config + io/ db/ api/ features/ analytics/ modeling/
             decision/ forecasting/ kpi/ viz/ utils/   (prelude.py = one-line notebook toolkit)
tests/       pytest, mirrors core/
models/      versioned model store (gitignored)
```

---

## Common commands

| Task | `uv` | plain venv |
|---|---|---|
| Notebooks | `uv run jupyter lab` | `python -m jupyterlab` |
| Build sample data | `uv run python scripts/make_sample_data.py` | `python scripts/make_sample_data.py` |
| Tests | `uv run pytest` | `python -m pytest` |
| Lint / format | `uv run ruff check .` / `ruff format .` | `python -m ruff check .` / `python -m ruff format .` |
| Type-check | `uv run mypy core` | `python -m mypy core` |
| Add a dependency | `uv add <pkg>` | `python -m pip install <pkg>` (then add to `pyproject.toml`) |

(`make sync` / `make check` etc. wrap the `uv` forms — see the `Makefile`.)

---

## Testing & the full pipeline

```powershell
.venv\Scripts\Activate.ps1

python -m pytest                   # fast unit tests (notebooks skipped)
python -m pytest --run-notebooks   # unit tests + every example notebook run end-to-end
make pipeline                      # the whole gate at once (see below)
```

**`make pipeline`** is the one command that runs everything CI runs — ruff lint, format check,
mypy, the unit suite, and all six example notebooks executed top-to-bottom (the real integration
test). Without `make`, run the steps directly:

```powershell
python -m ruff check . ; python -m ruff format --check . ; python -m mypy core
python scripts\make_sample_data.py            # notebooks need the sample data
python -m pytest --run-notebooks              # unit + notebook integration tests
```

Notebook execution needs the sample data and the `interactive` extra (Plotly). **CI**
(`.github/workflows/ci.yml`) runs the same on every push/PR as two jobs: **checks** (ruff + mypy +
unit tests) and **notebooks** (builds data, runs all notebooks).

## Good to know

- **Run tools via `python -m` on locked-down machines.** AppLocker (common on managed Windows)
  blocks third-party `.exe`s from user directories — that includes `uv.exe` and the pip-generated
  `jupyter`/`mypy`/`pytest` shims, but **not** the venv's signed `python.exe`. `python -m <tool>`
  always works. To get real `uv` back, have IT allow-list it or install it into an allowed path.
- **Pick the right Jupyter kernel.** A fresh `.ipynb` may open on the system `python3` kernel, which
  doesn't have `core` (→ `ModuleNotFoundError: No module named 'core'`). Switch to
  **`Python (data-science)`** (top-right kernel name → Change Kernel) and save.
- **`data/` and `models/` are rebuildable caches** (gitignored). Re-run `make_sample_data.py` (or
  your own loaders) to repopulate; never commit data.
- **Never commit secrets or notebook outputs.** Secrets live in `.env` / `conf/*.local.yaml`;
  nbstripout (via pre-commit) strips outputs from committed notebooks.
- **Parameterize SQL** — pass values as bound params (`:name`), never string-format them in.
- **Optional extras**: `geo` (geopandas/shapely), `explain` (SHAP), `excel`, `gam` (pygam),
  `imbalance` (imbalanced-learn), `interactive` (Plotly). Install only what you need.
- **Supply-chain gate**: `[tool.uv] exclude-newer = "7d"` in `pyproject.toml` makes `uv` ignore
  releases younger than 7 days, so a freshly-compromised package version can't be pulled in. (pip
  doesn't enforce this — another reason to prefer `uv` once it's available.)

---

Working conventions — coding style, what to do / not do, where code goes — live in [CLAUDE.md](CLAUDE.md).
