# Statistical & ML methods reference

Every statistics / machine-learning function in `core`: **when** to reach for it, **how** to call
it, and **what it offers statistically** — including the assumptions and failure modes that decide
whether the number can be trusted. Notebooks get everything in one line via
`from core.prelude import *`; the tables below are grouped by module (`stats.welch_t_test(...)`,
`evaluate.rfecv_scores(...)`, ...).

House conventions:

- **Compute ≠ present.** Heavy fitting lives in `analytics` / `modeling` / `forecasting`; the
  `viz` charts only draw what those return.
- **Hypothesis tests** return a `TestResult(statistic, p_value)`. The p-value is
  P(data at least this extreme | H0 true) — it is *not* P(H0). Significance defaults to
  `alpha = 0.05`; a 95% CI excluding 0 and p < 0.05 tell the same story.
- **Fit on train only.** Anything stateful (imputers, scalers, encoders, models) fits on the
  training split and transforms the rest — that is what `modeling.preprocess` pipelines enforce.
- Randomized routines take `seed` (default 42) so results are reproducible.

---

## analytics.stats — EDA, tests, effect sizes, power

### Describe & profile

| Function | Use when | What it tells you |
|---|---|---|
| `summary(df)` | First contact with any frame | Per column: dtype, nulls, distinct count, mean/std/min/quartiles/max — the screening pass for scale, skew, and junk |
| `cardinality(df)` | Choosing categoricals / spotting IDs | Distinct count and % of rows; ~100% unique = identifier, low % = categorical candidate |
| `missingness(df)` | Before imputing or dropping | Null count/% per column, most-missing first |
| `missingness_dependence(df, col)` | Deciding *how* to handle missing values | MCAR-vs-MAR triage: tests every other column between rows where `col` is null vs not (Welch t / chi-square). Small p = missingness depends on that column (MAR) → dropping rows or mean-imputing biases; impute conditionally (`preprocess.make_imputer("knn"/"iterative")`) and/or flag (`clean.add_missing_indicators`) |
| `describe_distribution(x)` | Shape check on one sample | Mean, std, skew (asymmetry), kurtosis (tail weight), p05–p95 — skew/kurtosis ≫ 0 argue for transforms or non-parametric tests |
| `correlation(df)` / `spearman(df)` | Linear vs monotonic association scan | Pearson r (linear, outlier-sensitive) vs Spearman rank ρ (monotonic, robust); both in [-1, 1], neither implies causation |
| `correlation_test(a, b, method=)` | One pair, with evidence | r (or ρ) plus a p-value for H0: no association |
| `mutual_information(df, target, task=)` | Non-linear feature relevance | MI ≥ 0 in nats between each numeric feature and the target; catches dependence correlation misses (task = `regression`/`classification`) |
| `pct_change(current, previous)` | Quick relative change | (cur − prev)/prev, `None` on zero base |

### Distribution fitting & outliers

| Function | Use when | What it tells you |
|---|---|---|
| `normality_test(x, method=)` | Before t-tests/ANOVA, on residuals | H0: sample is normal. `shapiro` (best power < ~5k rows), `dagostino` (skew+kurtosis based, large n), `ks` (Lilliefors-corrected KS — plain KS would be anticonservative with estimated mean/std). Small p → go non-parametric or transform |
| `fit_distribution(x, dist)` | You need a parametric model of a metric (revenue, delays, demand) | Maximum-likelihood fit of any scipy distribution; returns `params` (shape..., loc, scale), log-likelihood, AIC, and a KS goodness-of-fit p. MLE = the parameter values that make the observed data most probable |
| `best_distribution(x, candidates=)` | Which family fits best? | MLE-fits each candidate, ranked by AIC (lower = better fit after complexity penalty); check the winner's `ks_p` too — best-of-bad is still bad |
| `outlier_bounds(x, method=, factor=)` | Flagging univariate outliers | Cut-offs via IQR fences (robust, default 1.5×IQR) or z-score (mean ± k·std — itself distorted by the outliers). Treat with `clean.winsorize`, transforms, or removal only for genuine errors |

### Hypothesis tests (two or more samples)

| Function | Use when | What it tells you |
|---|---|---|
| `welch_t_test(a, b)` | Compare two means | Welch's t — does not assume equal variances (safer default than Student's t). Assumes rough normality of the *means* (CLT covers you at n ≳ 30/arm) |
| `mann_whitney(a, b)` | Two samples, skewed/ordinal/outliers | Non-parametric rank test of H0: same distribution; trades a little power for robustness |
| `anova(*groups)` | 3+ group means | One-way ANOVA F-test of "all means equal"; assumes normality + similar variances. A small p says *some* group differs, not which |
| `kruskal(*groups)` | 3+ groups, assumptions broken | Rank-based ANOVA analogue |
| `chi_square(a, b)` | Two categorical variables | Test of independence on the contingency table; expected counts ≥ 5 per cell to be trustworthy |
| `proportions_test(successes, totals)` | Two conversion rates | Two-proportion z-test (the classical A/B significance test) |
| `compare_groups(df, value, group)` | One-call group comparison | Auto-picks the right test (checks normality; 2 levels → Welch/Mann-Whitney, 3+ → ANOVA/Kruskal) and reports effect size with the p-value |
| `group_summary(df, value, group)` | Table for the deck | Per-group n, mean, std, and 95% CI half-width (±1.96·SE) |

### Effect sizes — "significant" is not "large"

| Function | Use when | What it tells you |
|---|---|---|
| `cohens_d(a, b)` | Standardized mean difference | Difference in pooled-SD units; ~0.2 small / 0.5 medium / 0.8 large. Comparable across metrics, unlike raw differences |
| `hedges_g(a, b)` | Same, small samples | Cohen's d with the small-sample bias correction |
| `cliffs_delta(a, b)` | Ordinal / non-normal data | P(a > b) − P(a < b) ∈ [-1, 1]; no distributional assumptions at all |
| `eta_squared(groups)` | After ANOVA | Share of total variance explained by group membership (0.01/0.06/0.14 ≈ small/medium/large) |

### Confidence intervals, sample size & power

| Function | Use when | What it tells you |
|---|---|---|
| `mean_confidence_interval(x, confidence=)` | Uncertainty on one mean | t-distribution CI; "95%" = the long-run coverage of the *procedure*. Width shrinks with √n (CLT) |
| `sample_size_mean(effect_size, power=, alpha=)` | Planning a means experiment | Per-arm n to detect a given Cohen's d at the target power (default 80%) and alpha. Run *before* the experiment — peeking instead is what `experiment.msprt_means` is for |
| `sample_size_proportion(p1, p2, ...)` | Planning a conversion experiment | Per-arm n to detect p1 → p2 |
| `power(effect_size, n, alpha=)` | Sanity-check an existing design | P(detect the effect if it is real) = 1 − β; an underpowered test mostly produces false "inconclusive"s and exaggerated significant effects |

---

## analytics.regression — OLS assumption diagnostics

Predictions survive mild violations; *inference* (coefficients, CIs, p-values) does not.

| Function | Use when | What it tells you |
|---|---|---|
| `vif(df)` | Coefficients unstable / signs flip | Variance inflation factor = 1/(1−R²) of each feature on the rest. 1 = independent, > 5 worrying, > 10 unstable — drop/combine features or move to Ridge/Lasso/PCA. (Quick screen: `clean.drop_highly_correlated`) |
| `breusch_pagan(residuals, features)` | Funnel-shaped residual plot | H0: constant error variance. Small p = heteroscedasticity → OLS standard errors are wrong; use robust (HC) errors or transform y |
| `durbin_watson(residuals)` | Time-ordered data | First-order residual autocorrelation: ~2 none, < 1.5 positive, > 2.5 negative. Autocorrelated residuals make naive SEs overstate evidence — consider lags or time-series models |
| `linear_assumptions(features, residuals)` | One-stop check after fitting | normality p, Breusch-Pagan p, Durbin-Watson, max VIF (+ which feature). Linearity itself is visual: `viz.model.residuals` should be a flat cloud |

Residual normality: `stats.normality_test(y_true - y_pred)` + `viz.eda.qq`.

---

## analytics.experiment — A/B testing

| Function | Use when | What it tells you |
|---|---|---|
| `analyze_means(control, treatment)` | Continuous metric per user (revenue, minutes) | Welch t-test + lift + CI + verdict (`win`/`loss`/`inconclusive`). One look at the planned n — not valid under repeated peeking |
| `analyze_conversions(c_conv, c_n, t_conv, t_n)` | Binary outcome | Two-proportion z-test, same result shape |
| `bayes_conversions(c_conv, c_n, t_conv, t_n, prior=)` | You want decision quantities, not a p-value | Beta-Binomial posterior: `prob_treatment_better` = P(T > C \| data), `expected_loss` = expected rate given up if you ship T and it's actually worse (ship when below tolerance), credible interval = "effect in this range with 95% probability" (the intuitive reading a CI doesn't license). `prior` encodes history when data are thin |
| `bayes_means(control, treatment)` | Same, continuous metric | Normal posterior of each mean (valid from ~dozens of obs/arm via CLT) |
| `srm_check(counts, expected=)` | **Always, before reading any metric** | Sample-ratio-mismatch chi-square: do arm sizes match the intended split? p < 0.001 = assignment is broken — fix before trusting anything |
| `cuped_adjust(metric, covariate, theta=)` | Pre-experiment covariate available | CUPED variance reduction: residualizes the metric on the covariate — same mean (unbiased effect), lower variance → same power with fewer users. Compute theta once on both arms pooled |
| `msprt_means(control, treatment, tau=)` | You must peek as data arrives | Always-valid p-value (mixture SPRT): stays correct under continuous monitoring; stop the moment it crosses alpha. A fixed-horizon t-test peeked at repeatedly inflates false positives badly |

Design checklist (PDF §7/§9): randomize units, define the primary metric up front, size with
`stats.sample_size_*`, check `srm_check`, then read effect + interval, not just the verdict.

---

## analytics.causal — effects without (or beyond) randomization

Correlation ≠ causation: confounders and reverse causality. Each tool buys identification with a
different assumption — pick the one you can defend.

| Function | Use when | What it tells you / assumes |
|---|---|---|
| `uplift(treatment_outcome, control_outcome)` | Randomized data | Plain ATE = mean(T) − mean(C); only unbiased because randomization balanced everything else |
| `difference_in_differences(c_before, c_after, t_before, t_after)` | Policy hit one group at a known time | (ΔT) − (ΔC): differences out *time-invariant* unobserved confounders. Assumes parallel trends — plot pre-period trends to defend it |
| `propensity_scores(x, treatment)` | Observational data, observed confounders | P(treated \| x) via logistic regression — the balancing score for matching/weighting |
| `match_on_propensity(scores, treatment, caliper=)` | Comparable-units comparison | Nearest-neighbour control per treated unit within `caliper`; estimates ATT on matched pairs. Handles *observed* confounding only |
| `ipw_ate(outcome, treatment, propensity)` | Keep all rows instead of matching | Inverse-propensity weighting (normalized/Hájek, scores clipped to [0.01, 0.99]): reweights groups to the same covariate mix. Sensitive to propensity misspecification |
| `itt_tot(assigned, treated, outcome)` | Experiment with non-compliance | `itt` = effect of *assignment* (what shipping delivers), `compliance` = uptake moved, `tot` = itt/compliance — Wald/IV estimate on compliers (LATE) |
| `iv_effect(outcome, treatment, instrument)` | Treatment self-selected, instrument available | cov(z,y)/cov(z,t) = 2SLS with one instrument. Needs relevance (z moves t — raises if ~0) and the *untestable* exclusion restriction (z affects y only through t) |
| `subgroup_effects(df, outcome=, treatment=, segment=)` | Who benefits most? (HTE) | Per-segment uplift + Welch p. Many slices = multiplied false positives — treat surprising subgroups as hypotheses to re-test |

---

## modeling.split — honest train/test separation

Leakage rule (PDF §5): split **first**, fit everything (scalers, imputers, encoders, resamplers)
inside the training side only.

| Function | Use when | What it guards statistically |
|---|---|---|
| `train_test_split(df, test_size=, stratify=)` | Default split | `stratify=` keeps class shares equal across splits — essential for imbalanced targets |
| `train_val_test_split(df, ...)` | Tuning + final estimate | Validation picks hyper-parameters; test is touched once, at the end |
| `group_split(df, group)` | Repeated entities (customer, store) | Whole entity stays on one side — otherwise the model memorizes entities and the test score lies |
| `time_split(df, time_col)` | Anything temporal | Train on past, test on future; random splits leak the future (lookahead bias) |
| `make_cv(strategy, n_splits=)` | Building a CV splitter | `kfold` / `stratified` / `group` / `repeated` (stabler estimate) / `timeseries` (expanding window). Pass the same splitter to every model you compare |

---

## modeling.preprocess — impute, scale, encode (fit on train)

| Function | Use when | What it offers statistically |
|---|---|---|
| `make_imputer(strategy)` | Filling numeric gaps | `mean`/`median` for MCAR gaps (median is outlier-robust); `knn` (neighbour mean) and `iterative` (MICE-style — each column regressed on the rest, in rounds) impute *conditionally* for MAR. Diagnose first with `stats.missingness_dependence`; keep flags via `clean.add_missing_indicators` when missingness is informative (MNAR) |
| `make_scaler(strategy)` | Features on different scales | `standard` z-scores (linear/SVM/PCA assume centred, comparable scales); `minmax` 0-1 normalization (distance-based models, NNs); `robust` median/IQR (outliers would otherwise set the scale). Trees don't care |
| `make_encoder(strategy)` | Categoricals → numbers | `onehot`: no fake order, column-per-level (explodes with cardinality); `ordinal`: compact integer codes but *imposes an order* — alphabetical by default, so supply real orderings yourself; `target`: mean-target per level in one column for high cardinality, cross-fitted internally to limit target leakage (needs `y` at fit) |
| `make_preprocessor(numeric=, categorical=, ...)` | The standard pipeline | ColumnTransformer wiring the above; drop into `train.fit(model, x, y, preprocessor=...)` so CV refits it per fold (no leakage) |

Stateless polars-side alternatives: `transform.frequency_encode`, `transform.group_rare`,
`temporal.cyclical_encode`.

---

## modeling.registry — one factory for every estimator

| Function | Use when | Notes |
|---|---|---|
| `make_model(name, task=, **params)` | Constructing any model | `task` = `regression`/`classification`; `**params` passes straight to the estimator, so every hyper-parameter is available |
| `available_models(task)` / `register(...)` | Discovering / extending the menu | xgboost / lightgbm / pygam load lazily |

Statistical map of the menu:

- **Linear family** — `linear`, `logistic` (linear in log-odds; log-loss not MSE), `ridge` (L2:
  shrinks coefficients smoothly, treats multicollinearity), `lasso` (L1: zeroes coefficients =
  embedded feature selection), `elasticnet` (L1+L2: sparsity with correlated-group stability),
  `huber`/`quantile` (robust / conditional-quantile loss), `poisson` (counts),
  `bayesian_ridge` (posterior over coefficients). Regularization is the lever on the
  bias-variance trade-off: penalty up → variance down, bias up.
- **Margin / distance / probabilistic** — `svc`/`svr` (kernel margins), `knn` (local averaging —
  scale features; suffers the curse of dimensionality first), `gaussian_nb`/`bernoulli_nb`
  (Bayes' rule with conditional independence).
- **Trees & ensembles** — `tree` (greedy impurity-minimizing splits; interpretable, prone to
  overfit — prune via `max_depth`, `min_samples_leaf`), `random_forest`/`extra_trees`/`bagging`
  (bagging: average independent bootstrap trees → variance ↓),
  `gradient_boosting`/`hist_gradient_boosting`/`adaboost`/`xgboost`/`lightgbm` (boosting:
  sequential error-correction → bias ↓; control overfit with `learning_rate`, depth/leaves,
  subsampling, L1/L2 on leaves, early stopping).
- **Other** — `mlp` (in-stack neural net), `gam` (smooth additive effects), `sgd` (online linear;
  pairs with `train.partial_fit`).

---

## modeling.train / tune — fitting, CV, search

| Function | Use when | What it offers statistically |
|---|---|---|
| `train.fit(model, x, y, preprocessor=)` | Plain fit | Wraps preprocessing + model into one Pipeline → CV revalidates the *whole* chain |
| `train.cross_validate(model, x, y, cv=)` | Generalization estimate | k-fold scores: mean = expected out-of-sample performance, std = its stability. Use instead of one lucky split |
| `train.cross_val_predict(model, x, y, method=)` | Honest predictions for curves/plots | Out-of-fold predictions — every row predicted by a model that never saw it; feed ROC/calibration/threshold tools |
| `train.predict` / `predict_proba` / `score_frame` | Scoring | Probabilities feed ranking metrics (AUC), calibration, and threshold tuning |
| `train.partial_fit(model, x, y, classes=)` | Streaming / out-of-core | Incremental learning for SGD/NB/MLP-style models |
| `tune.grid_search(model, grid, x, y, cv=)` | Few parameters, exhaustive | Best CV combination; beware: the best *reported* CV score is optimistically biased — confirm on held-out test |
| `tune.random_search(model, dists, x, y, n_iter=)` | Many/continuous parameters | Samples the space; usually finds near-optima far cheaper than grids |

---

## modeling.evaluate — metrics & model diagnostics

### Metrics

| Function | Use when | Statistical reading |
|---|---|---|
| `regression_metrics(y_true, y_pred)` | Any regressor | RMSE (quadratic loss — punishes big misses; same units as y), MAE (linear loss, robust), median-AE (outlier-immune), MAPE (% terms; explodes near zero actuals), RMSLE (relative errors, log-scale), R²/explained variance (share of variance explained; R² can be negative out-of-sample = worse than predicting the mean) |
| `classification_metrics(y_true, y_pred, y_score=)` | Any classifier | Accuracy (misleading under imbalance), balanced accuracy (mean per-class recall), precision (TP/(TP+FP) — cost of acting), recall (TP/(TP+FN) — cost of missing), F1 (harmonic mean), MCC & kappa (chance-corrected, imbalance-robust). With scores: ROC-AUC (P(random + ranked above random −); misleading under heavy imbalance), average precision (PR-AUC — prefer it there), log-loss & Brier (probability *quality*, not just ranking) |
| `report(y_true, y_pred)` | Multi-class detail | Per-class precision/recall/F1/support — finds the class the headline number hides |
| `pinball_loss(y_true, y_pred, alpha=)` | Quantile models / intervals | Proper loss for the α-quantile (asymmetric penalty) |

Money-units alternative: `kpi.profit` (below). Choose the metric by error costs, not habit.

### Diagnostics compute (pairs with `viz.model` / `viz.explain`)

| Function | Use when | Statistical reading |
|---|---|---|
| `permutation_importance(model, x, y, scoring=)` | Model-agnostic importance | Score drop when one feature is shuffled, on *held-out* data. Unbiased toward high-cardinality features (impurity importances aren't); correlated features share credit — read clusters together |
| `learning_curve_scores(model, x, y, cv=)` | Under- or over-fitting? More data? | Train vs validation score vs training size — the bias-variance diagnostic: both converge low = high bias (add capacity/features); persistent gap = high variance (regularize, simplify, more data). Total error = bias² + variance + noise |
| `validation_curve_scores(model, x, y, param_name=, param_range=)` | Tuning one complexity knob | Validation peaks at the best generalizing complexity; past it train keeps rising while validation falls = overfitting onset |
| `rfecv_scores(model, x, y, cv=)` | How many features earn their keep | Recursive elimination by `coef_`/`feature_importances_` with CV at each size: curve for the chart + `selected` list. Fewer features → variance ↓, interpretability ↑, possible bias ↑ |

Other importance routes: coefficients on standardized features (linear), `mutual_information`
(model-free), SHAP charts (local + global, `viz.explain`).

---

## modeling.compare — is model A actually better than B?

| Function | Use when | Statistical reading |
|---|---|---|
| `fold_scores(model, x, y, cv=, scoring=)` | Inputs for a paired test | Per-fold scores; use the *same* splitter for all models |
| `leaderboard(models, x, y, cv=, scoring=)` | Ranking candidates | Mean ± std per metric on identical folds — overlapping spreads = no real difference (see `viz.model.model_comparison`) |
| `paired_test(scores_a, scores_b, method=)` | The winner looks close | Paired t (or Wilcoxon) on per-fold differences. Folds overlap, so p-values are optimistic — a guide, not a verdict |

---

## modeling.ensemble / imbalance

| Function | Use when | Statistical reading |
|---|---|---|
| `ensemble.make_voting(estimators, voting=)` | Diverse, similar-strength models | Averaging cuts variance where members err independently; `soft` uses probabilities |
| `ensemble.make_stacking(estimators, final_estimator=)` | Squeeze more than voting | Meta-model learns optimal combination weights on out-of-fold predictions (CV inside guards leakage) |
| `imbalance.class_weights(y)` | First lever for rare positives | Balanced weights {class: weight} → reweights the loss; no synthetic data, works everywhere `class_weight=` exists |
| `imbalance.make_resampler(strategy)` | Weights aren't enough | `smote` (synthetic minority interpolation), `random_over`/`random_under`, `smote_tomek`/`smoteenn` (+boundary cleaning). Changes the training prior → probabilities come out miscalibrated; check `viz.model.calibration` |
| `imbalance.imbalanced_pipeline(model, resampler=, preprocessor=)` | Resampling + CV | Resamples *inside training folds only* — resampling before the split would leak synthetic copies into validation |
| `imbalance.tune_threshold(y_true, y_score, metric=)` | After training | 0.5 is arbitrary; picks the score cut-off maximizing F1/precision/recall. For money-optimal: `kpi.profit_threshold` |

Evaluation under imbalance: prefer PR-AUC, MCC, per-class recall — accuracy and ROC-AUC flatter.

---

## modeling.segment — clustering & dimensionality reduction

| Function | Use when | Statistical reading |
|---|---|---|
| `make_clusterer(name, **params)` | Customer/store segmentation | `kmeans` (spherical, needs k, scale features), `minibatch_kmeans` (big n), `dbscan` (density-based: arbitrary shapes, finds noise, no k — but ε is touchy), `agglomerative` (hierarchy → `viz.cluster.dendrogram`), `gaussian_mixture` (soft probabilistic assignment, elliptical clusters) |
| `elbow_scores(x, k_values)` | Choosing k, pass 1 | Inertia (within-cluster SS) vs k — read the bend (`viz.cluster.elbow`) |
| `silhouette_scores(x, k_values)` | Choosing k, pass 2 | Mean silhouette ∈ [-1, 1]: cohesion vs separation per k; ≳ 0.5 = real structure |
| `pca(x, n_components=)` | Linear compression, decorrelation | Orthogonal directions of max variance (global structure preserved); `explained_variance_ratio` says how much information k components keep. Standardize first |
| `tsne(x, perplexity=)` | *Visualizing* high-dim structure | Non-linear, preserves local neighbourhoods: tight groups are meaningful; inter-cluster distances, axes, and densities are **not**. Never cluster/measure on the coords; `perplexity` (5-50) sets neighbourhood size. Curse-of-dimensionality escape hatch for the eye, not the model |

---

## modeling.anomaly — unsupervised outlier scoring

| Function | Use when | Statistical reading |
|---|---|---|
| `make_detector(name, **params)` | Fraud, data quality, extreme-imbalance fallback | `isolation_forest` (anomalies isolate in few random splits; scales well; set `contamination` = expected outlier share), `local_outlier_factor` (local-density ratio — finds outliers *relative to their neighbourhood*), `one_class_svm` (boundary around normal data) |
| `anomaly_labels(detector, x)` | Apply | Fit-predict: 1 = inlier, −1 = outlier. When imbalance is so extreme classification fails, model "normal" and flag deviations (PDF §2) |

Univariate first pass: `stats.outlier_bounds`; treatment: `clean.winsorize`.

---

## modeling.monitor — drift after deployment

| Function | Use when | Statistical reading |
|---|---|---|
| `psi(expected, actual, bins=)` | Covariate/score drift, scalar | Population stability index over baseline-quantile bins: < 0.1 stable, 0.1–0.2 drifting, > 0.2 drifted (act). Magnitude-style measure — no p-value, insensitive to n |
| `ks_drift(expected, actual)` | Same, with a test | Two-sample Kolmogorov-Smirnov: max ECDF gap + p-value. At very large n it flags trivial differences — pair with PSI for "big enough to matter" |
| `label_drift(expected, actual)` | Class-mix shift (labels or predicted classes) | Chi-square on label proportions (prior shift); unseen-in-baseline classes get a floor share so a new class registers as drift. PSI/KS cover numeric columns only |
| `drift_report(baseline, current, columns=)` | Routine monitoring sweep | Per-column PSI + KS, sorted. Run it on features **and the model's scores** — score drift is the early warning. Concept drift (X→y changing, PDF §5) needs labels: re-evaluate metrics when they arrive; until then drift here is the proxy |

---

## modeling.persist — model store (workflow, not statistics)

`save_model(model, name, metadata=)` / `load_model(name, version=)` / `model_versions(name)` /
`list_models()` — versioned joblib store under `models/<name>/v<N>/` with a metadata sidecar
(timestamp + metrics/params you pass). Reproducibility: persist the seed and metrics with the
artifact.

---

## decision.bandits — learn *while* deciding

Explore-exploit alternatives to a fixed A/B: keep choosing, keep learning. All expose
`select()` → arm and `update(arm, reward)`.

| Policy | Use when | Statistical reading |
|---|---|---|
| `EpsilonGreedy(n_arms, epsilon=)` | Simple baseline | Exploit the best-known arm, explore at rate ε; ε is a fixed regret tax |
| `ThompsonSampling(n_arms)` | Binary rewards (conversion) | Beta-Bernoulli posterior per arm, sample → pick max: exploration self-tunes to uncertainty (Bayesian; near-optimal regret in practice) |
| `UCB1(n_arms)` | Deterministic optimism | Mean + √(2·ln t / n) bonus = upper confidence bound; under-tried arms get the benefit of the doubt, regret grows O(log t) |
| `LinUCB(n_arms, n_features, alpha=)` | Context matters (user/segment features) | Per-arm linear reward model with a confidence-ellipsoid bonus — personalizes the choice |

---

## decision.optimize — turn estimates into actions

| Function | Use when | What it does |
|---|---|---|
| `linear_program(cost, a_ub=, b_ub=, ..., maximize=)` | Budget/capacity allocation | LP via scipy linprog; the standard "maximize linear objective under linear constraints" |
| `assign(cost_matrix, maximize=)` | One-to-one matching (tasks↔people, slots↔ads) | Hungarian algorithm — exact optimal assignment |

---

## forecasting — models, diagnostics, backtesting

### forecasting.diagnostics — check the series first

| Function | Use when | Statistical reading |
|---|---|---|
| `adf_test(y)` | Stationarity check | Augmented Dickey-Fuller, H0: unit root. p < 0.05 → stationary. Stationarity (stable mean/variance/autocovariance) is what AR/MA machinery assumes; modelling a non-stationary series invites spurious relationships |
| `kpss_test(y, regression=)` | The mirror test | H0: stationary (level `c` / trend `ct`). p clipped to the [0.01, 0.1] table range — read extremes as bounds |
| `stationarity_report(y)` | One verdict | ADF + KPSS grid: both agree stationary → model it; both non-stationary → difference; ADF-only → difference-stationary (difference); KPSS-only → trend-stationary (detrend). Differencing a trend-stationary series (or vice versa) leaves the problem in place |
| `ljung_box(residuals, lags=)` | After fitting | H0: residuals are white noise up to `lags`. Small p = structure missed → raise AR/MA order, add the seasonal term or features. Set lags ≥ one season |
| `dominant_period(y)` | What's the seasonality? | Periodogram peak (linearly detrended) → cycle length, e.g. 7 on daily data. Confirm visually (`viz.timeseries.acf` / `seasonal_subseries`) before wiring into a model |

### forecasting.models — one interface: `fit(y)` → `predict(h)` / `predict_interval(h)`

| Forecaster (`make_forecaster(name)`) | Use when | Statistical reading |
|---|---|---|
| `naive` / `seasonal_naive` / `mean` | **Always, as the benchmark** | Last value / last season / global mean. A model that can't beat these on a backtest isn't a model |
| `arima` / `sarimax` (`order=`, `seasonal_order=`) | Autocorrelated, (difference-)stationary series | AR (lags of y) + I (differencing) + MA (lags of errors) + seasonal terms + exogenous regressors. Pick orders from `viz.timeseries.acf`/`pacf`; intervals come from the state-space model |
| `ets` / `holt_winters` (`trend=`, `seasonal=`, `seasonal_periods=`) | Trend + stable seasonality, business series | Exponential smoothing: recency-weighted level/trend/season (the same decomposition idea Prophet popularizes; pair with `temporal.add_holiday_flags` for holidays) |
| `ml` (`estimator=`, `lags=`) | Non-linear dynamics, any sklearn regressor | Reduction to supervised learning on lag features, recursive multi-step. Intervals use a *time-ordered holdout* residual σ (in-sample residuals of forests/boosters are dishonestly small), widened by √horizon |

`predict_interval(h, alpha=)` → (lower, upper): a range for each future value; width growing with
horizon is the honest compounding of uncertainty.

### forecasting.backtest — honest error estimates

| Function | Use when | Statistical reading |
|---|---|---|
| `rolling_origin(make, y, initial=, horizon=, step=)` | Model selection for forecasts | Expanding-window backtest: train on past, score the next `horizon`, roll forward. The time-series replacement for random CV (which would leak the future); slice errors by step-ahead `h` |
| `mae` / `rmse` | Error in units of y | MAE linear (robust), RMSE quadratic (punishes large misses) |
| `mape` / `smape` | Comparable across series | Percentage errors; MAPE explodes near zero actuals and penalizes over-forecasting more — sMAPE bounds both sides |

---

## pricing — elasticity & price optimization

| Function | Use when | Statistical reading |
|---|---|---|
| `elasticity.fit_demand(price, quantity)` | Estimate demand response | OLS on ln(q) = a + e·ln(p): constant-elasticity model, slope = elasticity. **Observational price-demand data is confounded** (prices were set in response to demand) — prefer experimental/IV variation (`causal.iv_effect`) when you can |
| `elasticity.price_elasticity(price, quantity)` | Just the number | e < −1 elastic (price ↑ → revenue ↓), −1 < e < 0 inelastic (price ↑ → revenue ↑) |
| `elasticity.predict_demand(intercept, elasticity, price)` | Scenario lines | Quantity under the fitted model |
| `optimize.revenue_at` / `profit_at(intercept, elasticity, price, unit_cost=)` | Money curves | Revenue/profit at candidate prices under the model |
| `optimize.optimal_price(intercept, elasticity, candidates, unit_cost=)` | Pick the price | Grid-search over realistic candidates (robust to any demand shape) |
| `optimize.markup_price(elasticity, unit_cost)` | Closed form | c·e/(e+1) — the textbook constant-elasticity optimum; requires elastic demand (e < −1) |

---

## kpi.profit — predictions → money

| Function | Use when | Statistical reading |
|---|---|---|
| `expected_value(y_true, y_pred, costs=)` | Value a classifier in currency | Confusion matrix × per-cell value `{tp, fp, tn, fn}` (e.g. fraud: missed fraud −500, investigation −10). The decision-theoretic metric when error costs are asymmetric — which commercially they always are |
| `profit_curve(y_true, y_score, costs=)` | See value vs threshold | Expected value across all cut-offs (chart-ready) |
| `profit_threshold(y_true, y_score, costs=)` | Deploy setting | The threshold maximizing expected value — the cost-aware upgrade over `imbalance.tune_threshold`'s F1 |

(`kpi.financial` / `kpi.behaviour` are deterministic business arithmetic — growth, margins, LTV,
funnel rates — not statistical estimators; see their docstrings.)

---

## features — statistically relevant transforms

Stateless polars functions (`f(df, ...) -> df`); anything that must learn from train only lives in
`modeling.preprocess` instead.

| Function | Use when | Statistical reading |
|---|---|---|
| `clean.fill_missing(df, strategy=)` | Simple gap-filling | constant/forward/backward/mean/median/mode. Mean/median shrink variance and distort correlations — fine for sparse MCAR gaps, wrong for MAR (check `stats.missingness_dependence`) |
| `clean.add_missing_indicators(df)` | Missingness might be signal | Boolean `<col>_missing` flags added *before* imputing, so the model keeps the information (MNAR-friendly) |
| `clean.winsorize(df, columns, lower=, upper=)` | Outlier treatment | Clips to quantiles: keeps rows, caps leverage of extremes (vs removal: only for genuine errors; vs log: for multiplicative skew) |
| `clean.drop_constant(df)` | Hygiene | Zero-variance columns carry no information and break some estimators |
| `clean.drop_highly_correlated(df, threshold=)` | Quick collinearity pruning | Drops later columns correlated ≥ threshold with an earlier one; the principled diagnosis is `regression.vif` |
| `transform.log1p(df, columns)` | Right-skewed positives (revenue, counts) | Variance-stabilizing; turns multiplicative effects additive, reins in heteroscedasticity |
| `transform.discretize(df, column, breaks=/quantiles=)` | Non-linearity for linear models, reporting bands | Binning trades resolution for robustness/interpretability |
| `transform.frequency_encode(df, columns, normalize=)` | High-cardinality categoricals, leakage-light | Category → its count/share: one numeric column, no target involved (target encoding is in `preprocess.make_encoder("target")` because it must fit on train) |
| `transform.group_rare(df, column, min_share=)` | Long-tail categories | Pools levels too thin to estimate into `other` — stabler estimates, smaller encodings |
| `transform.add_interactions(df, pairs)` | Effect of A depends on B | Product features (e.g. days-before-departure × route-demand). Linear models can't invent interactions — you supply them; trees find their own |
| `transform.sample` / `stratified_sample(df, by=, fraction=)` | Downsampling for prototyping | Stratified keeps group proportions, so estimates stay representative (selection-bias guard) |
| `temporal.add_lags(df, column, lags=, by=)` | Forecasting features | y(t−k) as predictors — temporal dependence without full ARIMA; sort by time first, lag within groups via `by` (leakage guard) |
| `temporal.add_rolling(df, column, windows=, stat=)` | Smoothed history features | Rolling mean/std/min/max — local level and volatility |
| `temporal.cyclical_encode(df, column, period=)` | Hour/weekday/month features | sin/cos pair so December sits next to January — distance-respecting periodic encoding |
| `validate.check_schema(df, ...)` | Pipeline entry gate | Required columns / non-null / unique / ranges — fail fast before bad data reaches a model |

---

## viz — what each statistical chart is *for*

All `@chart` functions: pass prepared data, get an `Axes` (multi-panel ones return a `Figure`).

| Chart | Read it for |
|---|---|
| `eda.histogram` / `eda.ecdf` | Shape, modes, skew; ECDF = quantiles without binning artifacts |
| `eda.qq(sample)` | Normality: points on the line = normal; S-curve = heavy/light tails. Run on residuals |
| `eda.ks(a, b)` | Two ECDFs + KS statistic — distribution gap (also: classifier class separation) |
| `eda.boxplot_by` / `eda.scatter` / `eda.pairplot` | Group spreads/outliers; pairwise relationships |
| `eda.correlation_heatmap` / `eda.crosstab_heatmap` | Collinearity clusters; categorical association |
| `eda.missingness_bar` | Null share per column |
| `model.roc` | Ranking quality across all thresholds (AUC) — flattering under imbalance |
| `model.precision_recall` | Same, focused on the positive class — the imbalance-honest curve (AP) |
| `model.threshold_curve` | Precision/recall/F1 vs cut-off → choose the operating point |
| `model.confusion` | Where errors land (normalize='true' for per-class rates) |
| `model.calibration` | Are probabilities honest? (predicted 0.8 ⇒ ~80% positive). Resampling/boosting usually miscalibrate |
| `model.gains_curve` / `model.lift_curve` | Targeting value: % positives captured per % contacted; lift over base rate by decile |
| `model.score_distribution` | Class separation of scores |
| `model.predicted_vs_actual` / `model.residuals` | Bias and structure; residuals should be a flat cloud |
| `model.scale_location` | Heteroscedasticity visually (test: `regression.breusch_pagan`) |
| `model.residuals_vs_leverage` | Influential points (Cook's distance sizing) |
| `model.error_by_feature` / `model.regression_calibration` | Where the model is biased; banded mean accuracy |
| `model.learning_curve` / `model.validation_curve` | Bias-variance read (compute: `evaluate.learning_curve_scores` / `validation_curve_scores`) |
| `model.feature_selection_curve` | Score vs #features, peak marked (compute: `evaluate.rfecv_scores`) |
| `model.model_comparison` | Per-fold score boxes — overlap = no real difference |
| `cluster.explained_variance` | PCA components worth keeping |
| `cluster.elbow` / `cluster.silhouette` / `cluster.silhouette_plot` | Choosing k; per-sample cohesion |
| `cluster.cluster_scatter` / `cluster.dendrogram` | Segments in 2-D (PCA/t-SNE coords); merge hierarchy |
| `explain.feature_importance` / `explain.permutation_importance` | Global drivers (impurity/coefficients vs shuffle-based) |
| `explain.partial_dependence` | Average effect shape of a feature (ICE: per-row heterogeneity) |
| `explain.shap_summary` / `shap_bar` / `shap_dependence` / `shap_waterfall` | Additive per-prediction attributions: global beeswarm/bar, feature interaction, single-prediction breakdown |
| `timeseries.rolling_stats` | Mean/variance stability — visual stationarity check |
| `timeseries.acf` / `timeseries.pacf` | Autocorrelation structure → ARIMA orders (q from ACF, p from PACF); seasonality spikes |
| `timeseries.lag_plot` / `timeseries.seasonal_subseries` | Lag dependence; seasonal profile consistency |
| `timeseries.seasonal_decomposition` | Trend / seasonal / residual split |
| `timeseries.forecast` / `timeseries.forecast_residuals` | Forecast vs actual with interval band; residual whiteness over time (test: `diagnostics.ljung_box`) |
| `conceptual.*` | Teaching sketches (bias-variance, etc.) — no data statistics |

---

## Concept → function index

| Concept | Where it lives |
|---|---|
| Bias-variance trade-off | `evaluate.learning_curve_scores` / `validation_curve_scores` + `viz.model.learning_curve`; regularized models in `registry`; bagging vs boosting (ensembles) |
| Linear-regression assumptions | `regression.linear_assumptions` (+ `vif`, `breusch_pagan`, `durbin_watson`), `stats.normality_test`, `viz.eda.qq`, `viz.model.residuals`/`scale_location` |
| Multicollinearity | `regression.vif`; `clean.drop_highly_correlated`; Ridge/Lasso/PCA |
| p-value vs confidence interval | `stats.TestResult`, `stats.mean_confidence_interval`, `experiment.ExperimentResult.confidence_interval` |
| CLT | Why `mean_confidence_interval`, t-tests, and `bayes_means` work at modest n |
| Likelihood & MLE | `stats.fit_distribution` / `best_distribution` |
| Bayesian vs frequentist | `experiment.bayes_conversions` / `bayes_means` vs `analyze_*`; `ThompsonSampling` |
| Decision trees & overfitting | `registry.make_model("tree", max_depth=, min_samples_leaf=)`; ensembles |
| Bagging vs boosting | `random_forest`/`bagging` (variance ↓) vs `gradient_boosting`/`xgboost`/`lightgbm` (bias ↓); `ensemble.make_voting`/`make_stacking` |
| Regularization (L1/L2/elastic net) | `ridge` / `lasso` / `elasticnet` in the registry |
| Cross-validation & leakage | `split.make_cv`, `train.cross_validate`, fit-on-train `preprocess`, `imbalance.imbalanced_pipeline`, `group_split`/`time_split` |
| Class imbalance | `imbalance.*`, anomaly detection fallback, PR-AUC/MCC in `evaluate`, `kpi.profit_threshold` |
| Evaluation beyond accuracy | `evaluate.classification_metrics`, `viz.model.roc`/`precision_recall`/`calibration`, `kpi.profit` |
| Curse of dimensionality | `segment.pca` (+ `rfecv_scores`, `mutual_information` for selection); `segment.tsne` for the eyes |
| Missing data (MCAR/MAR/MNAR) | `stats.missingness` / `missingness_dependence`, `clean.add_missing_indicators`, `clean.fill_missing`, `preprocess.make_imputer("knn"/"iterative")` |
| Outliers | `stats.outlier_bounds`, `clean.winsorize`, `anomaly.make_detector` |
| Normalize vs standardize | `preprocess.make_scaler("minmax"/"standard"/"robust")` |
| Encoding (one-hot/label/target/frequency/rare) | `preprocess.make_encoder`, `transform.frequency_encode`, `transform.group_rare` |
| Interaction features | `transform.add_interactions` |
| Feature importance | `evaluate.permutation_importance`, `viz.explain.*` (SHAP/PDP), model attributes |
| Feature selection | `evaluate.rfecv_scores`, `stats.mutual_information`, lasso |
| PCA vs t-SNE | `segment.pca` (model-ready, global) vs `segment.tsne` (visual, local) |
| Stationarity (ADF/KPSS) | `diagnostics.stationarity_report` |
| ARIMA/SARIMA & smoothing | `make_forecaster("arima"/"sarimax"/"ets")` |
| Lag features & seasonality | `temporal.add_lags`/`add_rolling`, `diagnostics.dominant_period`, `viz.timeseries.acf`/`seasonal_decomposition` |
| Drift (covariate/label/concept) | `monitor.psi`/`ks_drift` (covariate), `monitor.label_drift` (prior), score-drift via `drift_report` + delayed-label re-evaluation (concept) |
| A/B design, SRM, power, peeking | `stats.sample_size_*`/`power`, `experiment.srm_check`, `experiment.msprt_means`, `experiment.cuped_adjust` |
| Correlation vs causation; confounders | `analytics.causal` (matching, IPW, DiD, IV); stratify via `subgroup_effects` |
| ITT vs TOT | `causal.itt_tot` |
| Instrumental variables | `causal.iv_effect` |
| DiD vs PSM | `causal.difference_in_differences` (unobserved time-invariant confounders) vs `propensity_scores` + `match_on_propensity` (observed ones) |
| Heterogeneous treatment effects | `causal.subgroup_effects`; uplift by segment |
| Selection bias | randomize; `transform.stratified_sample`; matching/weighting in `causal` |

Out of scope by design: deep learning (CNNs/RNNs/transformers — this workspace is tabular/text/geo;
`mlp` and `sgd` in the registry are the in-stack neural options) and tools requiring extra
dependencies (Prophet → covered by `ets`/`sarimax` + holiday flags; causal forests → start with
`subgroup_effects`; UMAP → `tsne`).
