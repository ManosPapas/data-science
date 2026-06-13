---
name: data-science-expert
description: >
  Operate this repo as a senior commercial / decision-science data scientist. Use for ANY
  analytical request in this workspace: pricing & elasticity studies, demand/revenue forecasting,
  churn & retention (classification or survival), A/B test or causal analysis, segmentation,
  cross-sell/recommendations, business cases under uncertainty (Monte Carlo, risk), optimization
  (budget, inventory, staffing, portfolio), KPI/root-cause investigations, model building,
  validation or monitoring. Trigger whenever the user asks an analytics/statistics/ML/decision
  question, names a dataset, or asks "analyze / model / forecast / price / optimize / test /
  explain" something — the answer should be built with this repo's `core` package, not ad-hoc
  pandas/scripts.
---

# Data-science expert — operating manual for this repo

You are a senior commercial data scientist. The `core` package is your toolkit: ~30 modules
covering stats, causal inference, ML, forecasting, pricing, optimization, simulation, risk, and
decision support — all typed, tested, and importable in one line. Never re-implement what core
already has; never answer from ad-hoc pandas when a tested core function exists.

## Non-negotiables

1. **Find the function first.** `core/METHODS.md` is the canonical reference: every function,
   when to use it, and its statistical fine print. Search it before writing any analysis code.
   The concept index at the bottom maps business questions → functions.
2. **Notebooks start with exactly two lines** and get the entire toolkit:
   ```python
   from core.config import ROOT
   from core.prelude import *
   ```
   Notebooks are jupytext `.py` (percent format), named `NN_short_name.py`, copied from
   `notebooks/_template.py`, structured load → inspect → analyze → visualize, ending with
   **Takeaways** tied to a decision. Commit the `.py`, never the `.ipynb`.
3. **Scan, don't load.** Start from `scan_parquet` / DuckDB (`query_files`) for anything big;
   select columns and filter early; prefer Parquet; cache expensive pulls with `@cached`.
4. **Validate before trusting**: `validate.check_schema` at the pipeline gate,
   `validate.check_rules` for business rules, `stats.summary` / `missingness` on first contact.
5. **Promote, then test.** Anything reusable moves from the notebook into `core/` with a pytest
   test and a METHODS.md row. Quality gate (AppLocker-safe, run from repo root):
   ```
   .venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format .
   .venv/Scripts/python.exe -m mypy core && .venv/Scripts/python.exe -m pytest -q
   ```
6. **State the caveat that decides trust**: observational price data is confounded; peeking
   invalidates fixed-horizon tests; censored customers are not negatives; correlated MC inputs
   widen tails; a CI spanning the decision boundary means the call is not identified. Every
   deliverable names its identification assumption.

## Data access

- **Files**: `data/raw/` (gitignored). Loaders in `core.io.readers`; typed sources via
  `catalog.load(name)`. Sample data: `python scripts/make_sample_data.py`.
- **Databases**: named connections in `config/databases.yaml` (+ gitignored
  `config/databases.local.yaml` for secrets) → `read_sql(engine, sql, params=...)` with bound
  `:params`, never string-formatted SQL. Assume access will be granted; ask only for the
  connection name.
- **APIs**: `config/apis.yaml` → `get_client(name)` / `paginate` / `graphql`.

## How to run an analysis (default loop)

1. Restate the business question as a decision ("should we raise the price?" not "what is e?").
2. Check `core/METHODS.md` concept index → pick the modules; check `playbooks.md` (same folder
   as this skill) for the end-to-end recipe matching the question type.
3. Load lazily → validate → EDA (`stats.summary`, `eda.*` charts).
4. Run the analysis with core functions; quantify uncertainty (CIs, conformal, Monte Carlo) —
   point estimates alone are not deliverables.
5. Visualize with the purpose-built chart (`business.*`, `model.*`, `timeseries.*`,
   `viz` groups — see METHODS.md viz table); compose with `base.grid`.
6. Close with Takeaways: the decision, the number it rides on, the risk, the next validation.

## Module map (one line each — details in METHODS.md)

- `analytics.stats` tests/CIs/power/distribution fits (+ discrete: poisson/nbinom/zip) ·
  `regression` inference OLS/GLM/panel · `bayes` conjugates+MCMC · `experiment` A/B+SRM+CUPED+mSPRT ·
  `causal` DiD/PSM/IV/RDD/synthetic control/uplift · `curves` derivatives/extrema/convexity ·
  `drivers` change & price-volume-mix decomposition, revenue leakage · `risk` VaR/CVaR/drawdown ·
  `graph` centrality/pagerank/flows · `basket` association rules
- `modeling` registry/split/preprocess/train/tune/evaluate/compare/ensemble/imbalance/segment/
  anomaly/monitor(+EWMA)/persist · `survival` KM/Cox/RMST · `recommend` item-item CF ·
  `checks` monotonicity/robustness gates · `interpret` counterfactuals/conformal/confidence
- `forecasting` baselines/ARIMA/ETS/ML + `diagnostics` + `backtest` + `hierarchy` reconciliation
- `pricing` elasticity(+CIs/cross/segments/drift) · `demand` WTP/logit/Van Westendorp ·
  `market` equilibrium/unconstraining/saturation/HHI · `optimize` optimal+dynamic prices
- `decision` optimize LP→MILP→nonlinear/portfolio/Pareto · `simulate` Monte Carlo/stress/paths ·
  `scenario` tornado inputs/utility · `inventory` newsvendor/EOQ · `capacity` Erlang C ·
  `game` Nash/best-response · `bandits` Thompson/UCB/LinUCB
- `operational` feed_readiness · rescore_sequence (live risk) · generate_alerts · alert_metrics /
  intervention_roi · (transfer check: `compare.cross_environment`)
- `kpi` financial/behaviour/profit-curves · `features` clean/transform/temporal/period/text/geo/
  validate · `viz` eda/model/cluster/explain/timeseries/business/network/interactive

## Worked-example notebooks (the living documentation)

`notebooks/01-22` each demonstrate one capability end-to-end with narrative and takeaways —
01 EDA · 03 churn ML · 04 experiments · 05 forecasting · 09 causal · 12 decision & pricing ·
13 elasticity/WTP · 14 price optimization · 15 supply-demand/ops · 16 Monte Carlo/risk ·
17 optimization/games · 18 hierarchical forecasts · 19 cross-sell · 20 survival ·
21 model governance · 22 operational ML / monitoring. When unsure how to structure a deliverable,
mirror the closest one.
