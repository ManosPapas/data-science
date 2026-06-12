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
| **Read data** | `io` | Lazy/eager readers (Parquet, CSV, JSON, NDJSON, Excel), DuckDB SQL over files, a `@cached` Parquet loader cache, and a typed source `catalog` (pinned schemas) |
| | `db` | Pooled SQLAlchemy engines from named connections; **parameterized** SQL loaders (`read_sql`/`write_sql`) |
| | `api` | httpx clients (auth + retries), REST pagination, and GraphQL (queries + Relay cursor pagination) |
| **Prepare** | `features.clean` | dtype inference/fix, fill missing, dedupe, winsorize, text cleanup, downcast for memory |
| | `features.transform` | reshape, aggregate, join (with cardinality checks), bin, rank, sample, share-of-total |
| | `features.temporal` | calendar parts, holidays, lags, rolling windows, cyclical encodes |
| | `features.period` | rolling windows (7/28/30/60/90/120d…) + period-over-period (WoW/MoM/QoQ/YoY) |
| | `features.validate` / `text` / `geo` | schema checks · normalize/TF-IDF · haversine/bounding-box |
| **Analyze** | `analytics.stats` | summaries, distributions, hypothesis tests, effect sizes, mutual information, power |
| | `analytics.experiment` | A/B analysis for means and conversions (lift, CI, verdict), SRM check, CUPED variance reduction, always-valid mSPRT (safe peeking) |
| | `analytics.causal` | difference-in-differences, uplift, propensity scores + matching |
| | `analytics.curves` | numerical derivatives, local maxima/minima, inflection points, convexity, response curves |
| | `analytics.drivers` | root-cause decomposition: segment contributions, price-volume-mix bridge, revenue-leakage detection |
| | `analytics.risk` | VaR, expected shortfall (CVaR), drawdown, downside deviation, Sharpe/Sortino, target probabilities |
| | `analytics.graph` / `basket` | network analytics (centrality, PageRank, components, paths, max-flow) · market basket (frequent itemsets, association rules) |
| **Model** | `modeling` | 40+ models (`registry`), leakage-free `split`/`preprocess`, `train`, `tune`, `compare` (leaderboard + paired tests), `evaluate` (+ ranking metrics), `ensemble`, `imbalance`, `segment` (clustering/PCA), `anomaly`, drift `monitor` (PSI/KS + EWMA early warning), versioned `persist` |
| | `modeling.survival` | Kaplan-Meier retention curves, Cox hazard ratios, restricted mean survival (censoring-correct churn) |
| | `modeling.recommend` | item-item collaborative filtering + popularity baseline |
| | `modeling.checks` / `interpret` | monotonicity & business-logic validation, perturbation robustness · counterfactual explanations, conformal prediction intervals, confidence scores |
| **Forecast** | `forecasting` | one interface over baselines, ARIMA/SARIMAX, ETS, and ML-reduction; rolling-origin backtest; hierarchical reconciliation (`hierarchy`) |
| **Decide** | `decision` | contextual bandits (ε-greedy, Thompson, UCB1, LinUCB) · optimization (LP + shadow prices, MILP/knapsack, nonlinear, assignment, portfolio, Pareto fronts, stochastic/robust) · Monte Carlo `simulate` (correlated inputs, stress tests, paths) · `inventory` (newsvendor/EOQ/safety stock) · `capacity` (Erlang C) · `game` (Nash, best-response dynamics) |
| **Price** | `pricing` | elasticity with CIs, cross-price & segment elasticity, drift monitoring · demand curves, willingness-to-pay, Van Westendorp · market equilibrium, censored-demand unconstraining, saturation, HHI · optimal/markup prices, marginal economics, dynamic-pricing DP |
| **Measure** | `kpi` | ~30 financial KPIs, ~28 behaviour KPIs, and cost-sensitive `profit` curves |
| **Visualize** | `viz` | a `@chart` decorator + theme + grid; static charts for EDA, models, clustering, explainability, time series — plus **interactive Plotly** charts (`viz.interactive`) |
| **Utilities** | `utils` | memory profiling, structured logging, HTML reports |

See the runnable [example notebooks](#example-notebooks) for these in action end-to-end.

---

## Install

Requires **Python 3.13+**. Pick one — both install *everything* (deps + all extras + dev tools):

### With `uv` — one command

[`uv`](https://docs.astral.sh/uv/) gives a committed lockfile and a 7-day supply-chain gate.

```bash
uv sync --all-extras                         # .venv + all deps, extras, and dev tools
uv run python scripts/make_sample_data.py    # build the sample datasets
```

### Without `uv` — one-command bootstrap

No `uv` (e.g. a locked-down Windows box where AppLocker blocks `uv.exe`)? One script does the whole
setup — creates `.venv`, installs the package + all extras + dev tools, registers the
**`Python (data-science)`** Jupyter kernel, and builds the sample data. It uses `python -m`
throughout, so it's AppLocker-safe. (Note: this path installs the *latest* matching versions with
pip — only the `uv` path gets the lockfile and the 7-day release-age gate.)

```powershell
python scripts/bootstrap.py        # run with any Python 3.13+
```

<details><summary>…or run the same steps by hand</summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1                      # bash/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[geo,explain,excel,gam,imbalance,interactive]"
python -m pip install jupyterlab jupytext ipykernel ruff mypy pytest pre-commit nbstripout types-PyYAML
python -m ipykernel install --user --name data-science --display-name "Python (data-science)"
python scripts\make_sample_data.py
```
</details>

Confirm it worked: `python -c "import core; print('ok')"`, then **[run the notebooks](#quickstart)**.

---

## Quickstart

1. **Install** (above) and **generate the sample data**: `python scripts/make_sample_data.py`
   → writes `data/raw/{transactions.csv, customers.parquet, daily_sales.parquet}` (seeded and
   reproducible; `data/` is a gitignored, rebuildable cache).
2. **Launch JupyterLab in your browser** — activate the venv, then start it with `python -m`:
   ```powershell
   .venv\Scripts\Activate.ps1
   python -m jupyterlab        # opens http://localhost:8888/lab; the left sidebar is the file browser
   ```
   Use `python -m jupyterlab`, **not** `jupyter lab` (see [Good to know](#good-to-know)). `Ctrl+C`
   twice in the terminal stops the server. On a fresh clone the paired `.ipynb` files don't exist
   yet — create them with `python -m jupytext --to notebook notebooks/*.py`, or right-click a `.py`
   → *Open With → Jupytext Notebook*.
3. Open `notebooks/01_cleaning_and_eda.ipynb`, pick the **`Python (data-science)`** kernel, **Run
   All**, and work through them **in order** (01 caches a clean frame that 02 reads). In any cell,
   `from core.prelude import *` gives you the whole toolkit.

---

## Example notebooks

Twenty-one end-to-end notebooks under `notebooks/`, each a `load → inspect → analyze → visualize`
narrative that delegates every non-trivial step to a tested `core` function:

| Notebook | Demonstrates |
|---|---|
| `01_cleaning_and_eda` | Profiling a messy export → fixing dtypes, filling gaps, de-duping, winsorizing, downcasting, validating → distribution & relationship EDA |
| `02_features_kpis_periods` | Feature engineering + a customer join → financial & behaviour KPIs → MoM/QoQ/YoY and per-segment group analysis |
| `03_churn_modeling` | Stratified split → leakage-free pipelines → multi-model leaderboard + paired test → held-out evaluation → **profit-based threshold** → explainability → versioned save |
| `04_segmentation_and_experiments` | k-means (elbow/silhouette) + PCA, anomaly detection, an A/B test with power analysis, and a causal cross-check |
| `05_forecasting` | Decomposition/ACF/PACF → time features → forecaster bake-off on a holdout → rolling-origin backtest |
| `06_scale_lazy_duckdb` | The headline GB-scale path: lazy Polars scan-with-pushdown + DuckDB SQL straight over Parquet/CSV (out-of-core) |
| `07_statistical_inference` | Distributions & MLE fits, hypothesis tests, effect sizes, CIs & bootstrap, power and sample-size design |
| `08_regression_inference` | OLS/GLM for *inference*, fixed vs mixed effects, and the assumption checks (VIF, Breusch-Pagan, Durbin-Watson) |
| `09_causal_inference` | DiD, propensity matching/IPW, IV, RDD, synthetic control, uplift models + Qini |
| `10_bayesian_methods` | Conjugate posteriors, hierarchical shrinkage, MCMC, Bayesian A/B decisions |
| `11_timeseries_diagnostics` | Stationarity (ADF/KPSS), ACF/PACF, seasonal decomposition, trend tests, change points |
| `12_decision_and_pricing` | Elasticity → optimal price, profit-based churn threshold, scenarios & tornado, expected utility, LP + assignment, bandit shoot-out |
| `13_demand_elasticity_wtp` | Elasticity CIs & bootstrap, cross-price (substitutes/complements), segment & rolling elasticity, drift test, nonlinearity, decomposition, logit WTP, Van Westendorp |
| `14_pricing_optimization_curves` | Marginal revenue/profit, curve calculus (extrema/convexity), linear-demand closed form, dynamic-pricing DP, revenue leakage, price/volume/mix bridge |
| `15_market_supply_demand_ops` | Equilibrium & supply shocks, market balance, censored-demand unconstraining, saturation S-curve, HHI, newsvendor/EOQ/safety stock, Erlang-C staffing |
| `16_monte_carlo_risk` | Correlated-input Monte Carlo, P10/P50/P90 & target probabilities, simulation tornado, stress tests, path fans, VaR/CVaR/drawdown, EWMA early warning |
| `17_optimization_game_theory` | LP shadow prices, MILP/knapsack, nonlinear allocation, mean-variance portfolio, Pareto frontier, robust optimization, max-flow/MST, Nash equilibria & price-war dynamics |
| `18_hierarchical_forecasting` | Per-node ETS forecasts → coherence gaps → bottom-up/top-down/OLS reconciliation scored on a holdout |
| `19_recommendation_basket_graph` | Association rules (confidence vs lift), item-item recommender vs popularity with ranking metrics, co-purchase network (PageRank, communities) |
| `20_churn_survival_analysis` | Censoring done right: Kaplan-Meier vs naive churn, retention by segment, Cox hazard ratios, RMST → CLV |
| `21_model_governance_explainability` | Business-rule gates, monotonicity checks (catching a confounded model), perturbation robustness, counterfactual offers, conformal intervals, confidence routing |

Notebooks **01** and **04** also render a couple of charts with interactive **Plotly**
(`viz.interactive`) instead of matplotlib — a deliberate mix, to show both options without
duplicating any chart. Start a new analysis by copying **`notebooks/_template.py`** (the
load → inspect → analyze → visualize skeleton).

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
  `config/*.yaml`, connection/API secrets in gitignored `config/*.local.yaml` (deep-merged on top).
  Read through `get_settings()` / `get_connection(name)` / `get_api(name)`.

---

## Layout

```
config/      versioned, non-secret config (config.yaml, databases.yaml, apis.yaml, logging.yaml)
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
| Notebooks (browser) | `uv run python -m jupyterlab` | `python -m jupyterlab` |
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

**`make pipeline`** is the one command that runs the whole local gate — ruff lint, format check,
mypy, the unit suite, and every example notebook executed top-to-bottom (the real integration
test). Without `make`, run the steps directly:

```powershell
python -m ruff check . ; python -m ruff format --check . ; python -m mypy core
python -m pytest -q --cov=core --cov-fail-under=55   # unit tests + coverage floor
python scripts\make_sample_data.py                   # notebooks need the sample data
python -m pytest --run-notebooks -m notebook -q      # every notebook end-to-end
```

Notebook execution needs the sample data and the `interactive` extra (Plotly). There's no online CI
by design — run `make pipeline` (or the commands above) locally after a change to confirm nothing
broke.

## Good to know

- **Run tools via `python -m` on locked-down machines.** AppLocker (common on managed Windows)
  blocks third-party `.exe`s from user directories — including the venv's pip-generated
  `jupyter`/`mypy`/`pytest` shims, but **not** the venv's signed `python.exe`. So launch Lab with
  `python -m jupyterlab` (even `uv run jupyter lab` invokes the blocked shim), and run tools as
  `python -m <tool>` (or `uv run …`). `uv` itself works once it lives in an allowed path — e.g.
  `pip install uv` into the Program Files Python so `uv.exe` lands in its (allowed) `Scripts\`.
- **PowerShell execution policy:** if `.venv\Scripts\Activate.ps1` is blocked, run once
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-activate.
- **Pick the right Jupyter kernel.** A fresh `.ipynb` may open on the system `python3` kernel, which
  doesn't have `core` (→ `ModuleNotFoundError: No module named 'core'`). Switch to
  **`Python (data-science)`** (top-right kernel name → Change Kernel) and save.
- **`data/` and `models/` are rebuildable caches** (gitignored). Re-run `make_sample_data.py` (or
  your own loaders) to repopulate; never commit data.
- **Never commit secrets or notebook outputs.** Secrets live in `.env` / `config/*.local.yaml`;
  nbstripout (via pre-commit) strips outputs from committed notebooks.
- **Parameterize SQL** — pass values as bound params (`:name`), never string-format them in.
- **Optional extras**: `geo` (geopandas/shapely), `explain` (SHAP), `excel`, `gam` (pygam),
  `imbalance` (imbalanced-learn), `interactive` (Plotly). Install only what you need.
- **Supply-chain gate**: `[tool.uv] exclude-newer = "7d"` in `pyproject.toml` makes `uv` ignore
  releases younger than 7 days, so a freshly-compromised package version can't be pulled in. (pip
  doesn't enforce this — another reason to prefer `uv` once it's available.)

---

Working conventions — coding style, what to do / not do, where code goes — live in [CLAUDE.md](CLAUDE.md).
