# Playbooks — end-to-end recipes by question type

Each playbook: the business question → the core functions in order → the deliverable. Mirror the
named notebook for narrative structure. All assume the two-line bootstrap and validated data.

## 1. Pricing / elasticity study (nb 13, 14)
"What should we charge?" → `pricing.elasticity.fit_demand_ci` (CI must exclude -1 before any
directional call) → `segment_elasticity` + `cross_price_elasticity` (substitutes move together) →
`nonlinear_elasticity_check` → `pricing.optimize.optimal_price`/`markup_price` (+ `dynamic_prices`
for fixed stock & deadline) → sanity: `marginal_profit` sign at current price → chart
`business.price_curves(optimum=)`. Caveat to state: observational prices are confounded — prefer
randomized/IV variation (`causal.iv_effect`). Deliverable: recommended price + revenue delta + CI.

## 2. Churn / retention (nb 03 classification, nb 20 timing)
*Who* churns: `split.train_test_split(stratify=)` → `registry.make_model` + `train`/`tune` →
`evaluate.classification_metrics` + `compare.leaderboard` → **money threshold** via
`kpi.profit.profit_threshold` (never F1) → `checks.expected_directions` before shipping.
*When* they churn: `survival.kaplan_meier`/`cox_ph`/`restricted_mean_survival` → CLV via
`financial.clv`; censored customers are NOT negatives. Intervention: `interpret.counterfactual`
on actionable levers only. Deliverable: targeting list + per-customer offer budget ceiling.

## 3. Forecasting (nb 05 single series, nb 18 hierarchy)
`diagnostics.stationarity_report` + `dominant_period` first → baselines ALWAYS
(`make_forecaster("naive"/"seasonal_naive")`) → `arima`/`ets`/`ml` → `backtest.rolling_origin`
picks the winner (never in-sample fit) → `predict_interval` (a point forecast is not a
deliverable) → multiple levels? forecast every node, `hierarchy.coherence_error`, then
`hierarchy.reconcile(method="ols")`. Sold-out periods? unconstrain first
(`pricing.market.unconstrain_demand`). Deliverable: forecast + interval + backtest MAE vs naive.

## 4. Experiment analysis (nb 04, 07)
Design first: `stats.sample_size_*` / `power`. Then ALWAYS `experiment.srm_check` before reading
anything → `analyze_means`/`analyze_conversions` (or `bayes_conversions` for ship/no-ship
decision quantities) → variance reduction via `cuped_adjust` if pre-metric exists → peeking?
`msprt_means` only. Heterogeneity: `causal.subgroup_effects` (re-test surprises). Deliverable:
verdict + lift CI + the decision.

## 5. Observational impact / no experiment possible (nb 09)
Pick identification you can defend: known event + control group → `difference_in_differences`
(plot parallel pre-trends); observed confounders → `propensity_scores` + `match_on_propensity` /
`ipw_ate`; threshold rule → `regression_discontinuity`; one treated market →
`synthetic_control` (demand small pre_rmse, placebo-test donors); instrument →
`iv_effect`. Draw the DAG (`conceptual.dag`) to choose the adjustment set. Deliverable: effect +
the assumption it rides on, stated in one sentence.

## 6. Business case under uncertainty (nb 16)
Model value as a function → `simulate.monte_carlo` with distributions on uncertain inputs
(correlate the ones that move together — independence understates tails) → read
`summary(targets=)`, P10/P50/P90, `prob_above(target)` → `result.drivers()` ranks what to
de-risk → `simulate.stress_test` for survival under named shocks → `risk.risk_summary`
(VaR/CVaR) → chart `business.outcome_distribution` + `business.tornado`. Deliverable: the
distribution, not a number; funding case = P10.

## 7. Allocation / optimization (nb 17)
Linear & divisible → `optimize.linear_program` + `shadow_prices` (the "what's one more unit of
budget worth" answer); all-or-nothing → `integer_program`/`knapsack`; diminishing returns →
`nonlinear_program` (check `curves.convexity` first); risky options → `portfolio_weights`;
several objectives → `pareto_front` + chart; uncertain future → `scenario_optimize`
(criterion="worst" when the bad future is unaffordable). Stock: `inventory.newsvendor`/`eoq`.
Staffing: `capacity.required_servers`. Competitor reacts: `game.best_response_dynamics`.

## 8. Cross-sell / recommendations (nb 19)
`basket.association_rules` — rank by LIFT, not confidence → `recommend.ItemItemRecommender` vs
`popularity_baseline`, judged by `evaluate.ranking_metrics` on held-out purchases →
`analytics.graph.pagerank`/`connected_components` on lift-filtered edges for structure →
`network.network` to show it. Deliverable: rules table + uplift over popularity.

## 9. Segmentation (nb 04)
Scale features (`preprocess.make_scaler`) → `segment.elbow_scores` + `silhouette_scores` pick k
→ `make_clusterer("kmeans")` (or `gaussian_mixture` for soft, `dbscan` for shapes) → profile
segments with `stats.group_summary` → name them in business terms → visualize
`cluster.cluster_scatter` on PCA coords. t-SNE is for eyes only — never compute on its coords.

## 10. Root cause / "why did the KPI move?" (nb 14 §5-6)
`drivers.change_decomposition(current, baseline, value=, by=)` per candidate dimension →
revenue specifically: `drivers.price_volume_mix` (price vs volume vs mix) →
`drivers.revenue_leakage` for expected-vs-realized gaps → chart `business.waterfall` →
`stats.simpsons_check` before trusting any pooled trend. Decomposition locates, causal tools
explain — say which one you did.

## 11. Model governance / pre-deployment gate (nb 21)
`validate.check_rules` on the data → `checks.expected_directions` (business-sign table) +
`prediction_bounds` + `perturbation_stability` on the model → `interpret.conformal_intervals`
for honest ranges → `interpret.confidence_score` for human-review routing → after deploy:
`monitor.drift_report` + `monitor.ewma_alerts` (chart `business.control_chart`).

## 12. First contact with any new dataset (nb 01)
`stats.summary` → `missingness` (+ `missingness_dependence` before imputing) → `cardinality` →
`outlier_bounds`/`clean.winsorize` → dtype downcasts → `validate.check_schema` pinned as the
pipeline gate → distributions: `best_distribution`/`best_discrete` + `eda.fit_overlay`;
counts: `dispersion_check` before any Poisson assumption.

## 13. Operationalize a model against a live process (nb 22)
The model→live bridge for checkpointed processes (deliveries, claims, loans, ops). Gate the feed
first: `operational.feed_readiness(expected=)` + `entities_missing` — a model needing a low-coverage
signal can't score live yet. Re-score as state arrives: `operational.rescore_sequence(model,
snapshots, feature_columns=)` → the per-entity risk trajectory. Turn risk into action:
`operational.generate_alerts(score=, bands=, lead_time=, min_lead=)` — the lead-time gate separates
"act" from "too late". Prove it: `operational.alert_metrics` (detection/precision/lead) +
`intervention_roi` (explicit cost model, never a buried placeholder). Before rollout, the honesty
check: `compare.cross_environment(make_model, environments, scoring=)` — strong diagonal, weak
off-diagonal = overfit to a regime, re-train/calibrate per environment.
