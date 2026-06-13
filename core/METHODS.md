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
  `alpha = 0.05`; a 95% CI excluding 0 and p < 0.05 tell the same story. `alpha` *is* your
  Type I (false-positive) rate; power = 1 − β is the chance of catching a real effect, so
  Type II (missing it) is what `stats.sample_size_*` keeps low at design time.
- **Why n matters twice**: the law of large numbers says sample averages converge on the truth;
  the CLT says their error becomes approximately normal at a √n rate — that is what licenses
  t-tests and `mean_confidence_interval`, and what `bootstrap_ci` reproduces by simulation when
  no formula exists.
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
| `simpsons_check(df, x=, y=, group=)` | A headline trend smells like composition | Simpson's-paradox detector: pooled slope vs per-group slopes; `reversal=True` = the aggregate association flips within groups. Which margin to report is causal: condition on a confounder, not on a mediator |
| `pct_change(current, previous)` | Quick relative change | (cur − prev)/prev, `None` on zero base |

### Distribution fitting & outliers

| Function | Use when | What it tells you |
|---|---|---|
| `normality_test(x, method=)` | Before t-tests/ANOVA, on residuals | H0: sample is normal. `shapiro` (best power < ~5k rows), `dagostino` (skew+kurtosis based, large n), `ks` (Lilliefors-corrected KS — plain KS would be anticonservative with estimated mean/std). Small p → go non-parametric or transform |
| `fit_distribution(x, dist)` | You need a parametric model of a metric (revenue, delays, demand) | Maximum-likelihood fit of any scipy distribution; returns `params` (shape..., loc, scale), log-likelihood, AIC, and a KS goodness-of-fit p. MLE = the parameter values that make the observed data most probable |
| `best_distribution(x, candidates=)` | Which family fits best? | MLE-fits each candidate, ranked by AIC (lower = better fit after complexity penalty); incompatible candidates are skipped. Defaults span the commercial staples (norm, lognorm, expon, gamma, weibull_min, pareto, t) and *any* scipy continuous name can be added ('beta', 'genextreme', ...). Check the winner's `ks_p` too — best-of-bad is still bad |
| `fit_discrete(x, dist, trials=)` | Count data (orders, claims, tickets) | MLE for the discrete families scipy can't `.fit`: `poisson` (variance = mean), `geometric` (trials until first success, support ≥ 1), `nbinom` (overdispersed counts — usually the right one), `binom` (successes out of known `trials`), `zip` (zero-inflated Poisson — never-buyers mixed with buyers; estimates the structural-zero share directly). Params plug into `scipy.stats.<dist>(**params)`; eyeball with `viz.eda.fit_overlay` |
| `best_discrete(x, candidates=)` | Which count model? | Ranks the discrete fits by AIC; geometric drops out automatically when zeros are present. Poisson vs nbinom vs zip decides whether forecasts carry the real variance and the real zero mass |
| `dispersion_check(x)` | Before any Poisson assumption | Variance/mean ratio + the Poisson dispersion test ((n−1)s²/x̄ ~ chi²). Overdispersed (ratio > 1, small p) → negative binomial, or intervals come out too narrow and stock-outs too frequent |
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
| `permutation_test(a, b)` | Small n / ugly distributions, but you want to compare *means* | Shuffles group labels to build the exact null of mean(a) − mean(b) — assumption-free significance (`mann_whitney` tests ranks, a different question) |
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
| `proportion_confidence_interval(s, n)` | Error bar on a rate | Wilson score interval: stays inside [0, 1] and behaves at small n / extreme rates, where the Wald p ± z·SE interval collapses or spills out |
| `bootstrap_ci(x, statistic=)` | CI for a median / ratio / quantile — no clean formula | Resamples with replacement and reads the CI off the empirical distribution ('BCa' corrects bias and skew) — the CLT by simulation, for any statistic |
| `bayes_rule(prior, tpr, fpr)` | Updating a base rate with a test / alert signal | Posterior P(H \| signal). The base-rate-neglect guard: a 99%-sensitive, 5%-false-positive test on a 1% prior gives only ~17% |
| `sample_size_mean(effect_size, power=, alpha=)` | Planning a means experiment | Per-arm n to detect a given Cohen's d at the target power (default 80%) and alpha. Run *before* the experiment — peeking instead is what `experiment.msprt_means` is for |
| `sample_size_proportion(p1, p2, ...)` | Planning a conversion experiment | Per-arm n to detect p1 → p2 |
| `power(effect_size, n, alpha=)` | Sanity-check an existing design | P(detect the effect if it is real) = 1 − β; an underpowered test mostly produces false "inconclusive"s and exaggerated significant effects |

### Information theory

| Function | Use when | What it tells you |
|---|---|---|
| `entropy(labels)` | How unpredictable is a categorical? | Shannon entropy in bits: 0 = constant, log2(k) = uniform over k — the currency that information gain, MI, and tree splits trade in |
| `kl_divergence(p, q)` | How far is distribution Q from P? | Extra bits paid modelling P with Q; asymmetric, 0 iff identical. Relatives: `monitor.psi` is a symmetrized binned KL; classifier cross-entropy (`log_loss`) = H(P) + KL(P‖Q) |
| `information_gain(df, feature, target)` | Which categorical feature splits the target best? | H(target) − expected conditional entropy — exactly what entropy-based decision trees maximize per split; the categorical sibling of `mutual_information` |

---

## analytics.regression — regression for *inference* (fits + assumption checks)

`modeling` predicts; this module explains. Fitters return a `FitSummary`: a coefficient table
(term, coef, std_err, statistic, p_value, ci_low/ci_high) plus model stats (R², AIC, n).

### Fitting

| Function | Use when | What it tells you |
|---|---|---|
| `ols_fit(df, y=, x=)` | Read effect sizes off a continuous outcome | Each coef = expected Δy per unit of that feature, others held fixed, with SE/t/p/CI. Run `linear_assumptions` on the residuals before quoting the p-values; add `transform.add_interactions` columns to test moderation |
| `glm_fit(df, y=, x=, family=)` | y isn't normal: counts, 0/1, skewed amounts | The right likelihood + link: 'poisson' (log link — exp(coef) is a rate multiplier), 'binomial' (logit — coefs are log-odds), 'gamma' (log link, positive skew), 'gaussian' (= OLS). Wald z inference; compare specs by AIC |
| `fixed_effects(df, y=, x=, entity=)` | Panel data; entities differ in level (stores, customers) | Within-entity demeaning absorbs *every* time-invariant entity confounder — slopes come from within-entity variation only. Dof-corrected SEs; `r_squared` is the within-R²; entity-constant features drop out |
| `mixed_effects(df, y=, x=, group=)` | Group structure; groups should share strength | Random-intercept model (partial pooling): small groups borrow from the rest — the regression cousin of `bayes.hierarchical_rates`. `group_variance` = between-group intercept variance; Wald z inference, wants ≳ 8-10 groups |

Fixed vs random/mixed: fixed effects assume nothing about the entities and kill all level
confounding, but can't estimate entity-constant features; mixed models assume intercepts ~ normal
and buy efficiency, shrinkage, and group-level inference. Confounding worry → fixed; many small
groups → mixed.

### Assumption diagnostics

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

## analytics.bayes — Bayesian building blocks

The grammar everywhere in this repo's `bayes_*` tooling: **prior** (belief before data) ×
**likelihood** (how probable the data are under each parameter value) → **posterior** (belief
after). Conjugate pairs make the update closed-form; MCMC samples it when nothing is closed-form.

| Function | Use when | What it tells you |
|---|---|---|
| `beta_posterior(successes, trials, prior=)` | One rate, with honest uncertainty | Conjugate Beta-Binomial update: Beta(a, b) + s/n → Beta(a+s, b+n−s). The interval is *credible* — "the rate is in here with 95% probability". The prior is explicit: Beta(1, 1) = uniform; larger a+b = stronger belief (mean a/(a+b)) for thin data |
| `gamma_posterior(events, exposure, prior=)` | One *rate* per unit exposure (arrivals, defects, claims) | Conjugate Gamma-Poisson update: Gamma(a, b) + k events over exposure t → Gamma(a+k, b+t), mean (a+k)/(b+t). The Poisson sibling of `beta_posterior` for when "successes out of trials" doesn't fit; prior reads as a pseudo-events over b pseudo-exposure |
| `hierarchical_rates(successes, trials, labels=)` | Many small rates (per store / SKU / campaign) | Hierarchical model via empirical Bayes: fits one shared Beta prior, then partially pools — small-n groups shrink hard toward the global mean, big-n barely move. Rank league tables on `shrunk_rate`, not raw rates |
| `mcmc_sample(log_density, start, step=)` | No conjugate form for your posterior | Random-walk Metropolis: feed any log-density (log-prior + log-likelihood), get posterior draws + acceptance rate. Summarize with means / `np.quantile` credible intervals; tune `step` to ~20-40% acceptance, eyeball the trace. For big models use a dedicated PPL |

Likelihood itself: `stats.fit_distribution` is the maximum-likelihood half of the machinery; the
A/B decision wrappers are `experiment.bayes_conversions` / `bayes_means`; the bandit that *acts*
on posteriors is `ThompsonSampling`.

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
| `regression_discontinuity(running, outcome, cutoff=, bandwidth=)` | A threshold rule assigns treatment (score cutoffs, spend tiers, exam marks) | Sharp RDD: the fitted jump at the cutoff is the *local* causal effect — units just either side are comparable. Vary `bandwidth` to check stability; invalid if units manipulate the running variable (bunching) |
| `synthetic_control(treated_pre, donors_pre, treated_post, donors_post)` | One treated unit (market, region), no natural control | Non-negative sum-to-one donor weights fitted to track the pre-period; the post-period gap to that synthetic twin is the effect — an explicit counterfactual. Demand a small `pre_rmse`; placebo-test by rerunning on donors |
| `subgroup_effects(df, outcome=, treatment=, segment=)` | Who benefits most? (HTE) | Per-segment uplift + Welch p. Many slices = multiplied false positives — treat surprising subgroups as hypotheses to re-test |
| `TLearner(model).fit(x, treatment, outcome)` | Target *persuadables*, not likely converters | Uplift model: one outcome model per arm, predicted uplift = the difference. Rank customers by it for campaign targeting — incrementality beats propensity (sure things convert anyway) |
| `qini_points` / `qini_auc(outcome, treatment, scores)` | Evaluate an uplift *ranking* | Qini curve = incremental successes vs share targeted; the area over the random-targeting line is the uplift world's AUC. Accuracy/AUC on outcomes say nothing about persuasion |

---

## analytics.curves — derivatives & turning points on any curve

The calculus layer under pricing/optimization: works on sampled `(x, y)` curves (uneven spacing
ok) or callables. Numerical derivatives amplify noise — `smooth_series` first, and distrust
extrema found on raw noisy data.

| Function | Use when | What it tells you |
|---|---|---|
| `slope(x, y)` / `curvature(x, y)` | Rate of change / acceleration along a curve | First/second derivative by central differences. Slope crossing 0 = a turning point; curvature < 0 = concave (diminishing returns) |
| `point_elasticity(x, y)` | %-for-% sensitivity at each x | d ln(y)/d ln(x) along the curve — the local elasticity read for any response, not just demand |
| `local_extrema(x, y)` | Find optima on a sampled curve | Interior maxima/minima, parabolic-refined. A "best" value at the grid edge means the real optimum may be outside the grid — widen it |
| `inflection_points(x, y)` | Where acceleration flips | Sign changes of the second derivative: peak growth on an S-curve, onset of diminishing returns on a response curve |
| `convexity(x, y)` | Trust an interior optimum? | convex/concave/mixed verdict from curvature signs. Concave objective → hill-climbing and interior optima are trustworthy; mixed → multiple local optima, grid-search first |
| `marginal_effect(fn, base, name)` / `gradient(fn, base)` | Local what-if rate on a business model | Numerical ∂fn/∂input at the base point (all inputs at once via `gradient`) — "one more euro of X is worth this". Local: re-evaluate at different bases |
| `response_curve(fn, base, name, values)` | The full curve behind a sensitivity | Sweep one input, get output + local slope — shows *where* returns diminish, not just that they do (the two-point version is `scenario.sensitivity`) |

---

## analytics.drivers — diagnostic decomposition (root cause, in exactly-summing parts)

Decomposition is accounting, not causation: it says *where* a change sits; `analytics.causal`
says what made it happen. All outputs sum exactly to the headline change — no residual.

| Function | Use when | What it tells you |
|---|---|---|
| `change_decomposition(current, baseline, value=, by=)` | "Why is the metric down?" — first pass | Per-segment contribution to the total change (entries/exits counted in full), sorted by impact. The root-cause shortlist |
| `price_volume_mix(current, baseline, price=, volume=, by=)` | Revenue bridge for the deck | ΔRevenue split into price effect (charging differently), volume effect (market grew/shrank), mix effect (share shifted toward cheap/expensive segments) — separates "pricing worked" from "mix flattered us" |
| `revenue_leakage(df, expected=, actual=, by=)` | Entitled vs realized revenue | Leakage = expected − actual, ranked by group: unapproved discounting, billing gaps, fee waivers. Persistent positive leakage in one rep/segment is the audit signal |

---

## analytics.risk — risk & uncertainty measures on outcome samples

Consumes outcome draws (Monte Carlo from `decision.simulate`, bootstrap replicates, historical
P&L). Convention: outcomes are better-is-bigger, so risk lives in the low quantiles.

| Function | Use when | What it tells you |
|---|---|---|
| `value_at_risk(outcomes, alpha=)` | "How bad, with 95% confidence?" | The alpha-quantile outcome — a *threshold*, silent about how bad the tail beyond it gets |
| `expected_shortfall(outcomes, alpha=)` | What a bad year actually looks like | Mean of the worst-alpha tail (CVaR). Coherent (sub-additive) — aggregates across a portfolio sensibly, unlike VaR |
| `downside_deviation(outcomes, target=)` | Spread that only counts the bad side | RMS shortfall below target — plain std punishes upside surprises too; the Sortino denominator |
| `max_drawdown(series)` | Path risk of a cumulative series | Worst peak-to-trough fraction — two paths with the same endpoint can differ wildly in survivability |
| `probability_below/above(outcomes, threshold)` | Probability of failure / of hitting the target | Tail mass each side of the line — the number a commitment can be made on |
| `sharpe_ratio` / `sortino_ratio(returns, ...)` | Risk-adjusted comparison | Mean excess return per unit of (downside) volatility; `periods_per_year` annualizes by √t |
| `risk_summary(outcomes, targets=)` | The one-slide risk read | Mean/std, P5-P95, VaR/CVaR, P(≥ target) — P50 is the plan, P10 the funding case |

---

## analytics.graph — network analytics on edge lists

Polars edge-list in (`source`, `target`[, `weight`]), scipy.sparse.csgraph underneath. Commercial
graphs: co-purchase (from `analytics.basket`), referrals, logistics, money movement.

| Function | Use when | What it tells you |
|---|---|---|
| `degree_centrality(edges)` | First-pass importance | Degree, weighted degree, normalized centrality — hub products, super-referrers. Local measure only |
| `pagerank(edges, damping=)` | Importance through the structure | Random-surfer score (sums to 1): high when *important* nodes point at you, not just many |
| `connected_components(edges)` | Segmentation by reachability | Component label + size per node — customer communities, product islands, fraud rings |
| `shortest_paths(edges, origin=)` | Distance/cost from one node | Dijkstra distances (weights = costs, lower = closer; invert similarities before routing) |
| `minimum_spanning_tree_edges(edges)` | Cheapest network connecting everything | The minimal backbone — network design's lower bound; extra edges buy redundancy, not reach |
| `max_flow(edges, origin=, sink=)` | Throughput ceiling of a network | Max-flow value + per-edge flows; the saturated cut is the binding bottleneck. Capacities are rounded to ints (`scale=` for fractional) |

---

## analytics.basket — market basket analysis

| Function | Use when | What it tells you |
|---|---|---|
| `frequent_itemsets(df, transaction=, item=, min_support=)` | What sells together (sets of 1-3) | Support = share of transactions containing the whole set; apriori pruning means rare items can't form frequent sets (raise `min_support` if the pair join blows up) |
| `association_rules(df, transaction=, item=, min_support=, min_confidence=)` | Cross-sell rules, ranked | Confidence = P(consequent \| antecedent) — the hit rate if you recommend on the rule; **lift** > 1 separates real affinity from "both are just popular"; leverage is the same gap in absolute share |

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
  `huber`/`quantile` (robust / conditional-quantile loss), `poisson`/`gamma`/`tweedie` (GLM
  losses for counts / positive skew / zero-inflated continuous — inference twins in
  `regression.glm_fit`), `bayesian_ridge` (posterior over coefficients). Regularization is the
  lever on the bias-variance trade-off: penalty up → variance down, bias up.
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

## modeling.survival — *when* churn/failure happens (censoring done right)

Classification throws away timing and mishandles customers who haven't churned *yet* (censored,
not negative). `events`: 1 = churned/failed at `duration`, 0 = still active at last sight.

| Function | Use when | Statistical reading |
|---|---|---|
| `kaplan_meier(durations, events)` | Retention curve from censored data | Non-parametric S(t) = P(survive past t) with Greenwood CIs — every account contributes exposure up to its last sighting, which is exactly what naive churn rates get wrong |
| `survival_at(..., times)` / `median_survival(...)` | Retention at a tenure / typical lifetime | Step-read of S(t); median = first crossing of 0.5 (NaN = most of the base outlives the window — itself informative) |
| `restricted_mean_survival(..., horizon=)` | CLV-ready retention time | Area under KM to the horizon = expected retained periods per customer; multiply by margin/period. Always estimable under censoring (the unrestricted mean isn't) |
| `cox_ph(df, duration=, event=, x=)` | Which factors drive churn hazard | Semi-parametric regression: `hazard_ratio` = exp(coef) multiplies the churn hazard at *every* tenure (the proportional-hazards assumption — check by fitting early/late tenure splits). No baseline-hazard shape assumed |

---

## modeling.recommend — item-item collaborative filtering

| Function | Use when | Statistical reading |
|---|---|---|
| `ItemItemRecommender().fit(df, user=, item=, rating=)` | "Customers who bought X also bought…" | Cosine similarity over the user dimension; implicit feedback (no rating) scores interactions 1. Item-item is the production classic: item neighbourhoods are stabler than user tastes and every pick has a "because you bought X" explanation |
| `.recommend(user, k=)` / `.similar_items(item, k=)` | Personal top-k / the bundle shelf | Similarity-weighted sum over the user's history (seen items excluded); cold users/items need a fallback — `popularity_baseline` |
| `popularity_baseline(df, item=, k=)` | The bar to clear | Most-interacted items; a recommender that can't beat popularity on `evaluate.ranking_metrics` isn't earning its complexity |
| `evaluate.ranking_metrics(relevance, scores, k=)` | Score a *ranking*, not a label | NDCG@k (graded, position-discounted), precision/recall@k, MRR — classification metrics ignore order; these score what the shelf actually serves |

---

## modeling.monitor — drift after deployment

| Function | Use when | Statistical reading |
|---|---|---|
| `psi(expected, actual, bins=)` | Covariate/score drift, scalar | Population stability index over baseline-quantile bins: < 0.1 stable, 0.1–0.2 drifting, > 0.2 drifted (act). Magnitude-style measure — no p-value, insensitive to n |
| `ks_drift(expected, actual)` | Same, with a test | Two-sample Kolmogorov-Smirnov: max ECDF gap + p-value. At very large n it flags trivial differences — pair with PSI for "big enough to matter" |
| `label_drift(expected, actual)` | Class-mix shift (labels or predicted classes) | Chi-square on label proportions (prior shift); unseen-in-baseline classes get a floor share so a new class registers as drift. PSI/KS cover numeric columns only |
| `drift_report(baseline, current, columns=)` | Routine monitoring sweep | Per-column PSI + KS, sorted. Run it on features **and the model's scores** — score drift is the early warning. Concept drift (X→y changing, PDF §5) needs labels: re-evaluate metrics when they arrive; until then drift here is the proxy |
| `control_limits(baseline, sigmas=)` | Watch a KPI/metric stream | Shewhart mean ± k·sigma band from an in-control baseline; points outside are signals (~0.3% false alarms at 3 sigma). Compute limits on a period you *trust* — recomputing on drifting data hides the drift |
| `ewma_alerts(values, baseline, lam=)` | Early warning on slow drifts | EWMA control chart: exponentially weighted average vs widening-to-asymptote limits — catches small persistent shifts a Shewhart band misses. Persistent alerts → fire `drift_report` |

---

## modeling.checks — model behaviour validation (business logic, not test scores)

A model can score well and be unshippable: demand rising with price, risk falling with exposure.
Probe the fitted model like a domain reviewer would; data-side rules → `validate.check_rules`.

| Function | Use when | Statistical reading |
|---|---|---|
| `monotonicity(model, x, feature=, direction=)` | Sign-check one relationship | Sweeps the feature per row, flags wrong-direction prediction moves (`violation_rate`, `worst_gap`). Violations in flexible models usually mean confounded training data — fix with monotone constraints (lightgbm/xgboost) or features, don't ship and hope |
| `expected_directions(model, x, {feature: direction})` | Model governance as a table | All sign expectations re-checked per retrain; alert on regressions |
| `prediction_bounds(model, x, lower=, upper=)` | Outputs in the plausible range? | Share of predictions outside business bounds (negative price, conversion > 1) — extrapolation or leakage symptoms; understand before clipping |
| `perturbation_stability(model, x, scale=)` | Robustness to measurement noise | Prediction movement under noise-sized input jitter; big `p95_abs_change` = over-sharp boundaries that will thrash downstream decisions — regularize/ensemble |
| `validate.check_rules(df, {name: pl.Expr})` (features) | Business-rule gate on data | Named boolean predicates that must hold per row (nulls count as violations); `raise_on_error=True` makes it a pipeline gate |

---

## modeling.interpret — counterfactuals, conformal intervals, confidence

| Function | Use when | Statistical reading |
|---|---|---|
| `counterfactual(model, row, candidates=, target=)` | "What would change the outcome?" | Greedy minimal-change search over *actionable* levers only (never immutables — a counterfactual on tenure is an explanation, not an action). Holding other features fixed can describe an impossible customer — sanity-check the winner. SHAP says which features mattered; this says what to *do* |
| `conformal_intervals(model, x_cal, y_cal, x_new, alpha=)` | Honest intervals for any regressor | Split-conformal: intervals cover the truth with P ≥ 1-alpha on exchangeable data, no normality or model trust needed. Calibration data must be *held out from training*; the guarantee is marginal, breaks under drift — re-calibrate periodically |
| `confidence_score(probabilities, method=)` | Triage / human-review routing | Per-row confidence in [0,1]: `margin` (top-1 − top-2) or `entropy` (distribution concentration). Only as honest as the probabilities — check `viz.model.calibration` first |

---

## modeling.persist — model store (workflow, not statistics)

`save_model(model, name, metadata=)` / `load_model(name, version=)` / `model_versions(name)` /
`list_models()` — versioned joblib store under `models/<name>/v<N>/` with a metadata sidecar
(timestamp + metrics/params you pass). Reproducibility: persist the seed and metrics with the
artifact.

### modeling.compare.cross_environment — does the model generalize across regimes?

| Function | Use when | Statistical reading |
|---|---|---|
| `cross_environment(make_model, environments, scoring=)` | Will it transfer to a new segment/period/region? | Trains a fresh model on each environment, scores it on every other → long (train, test, score) matrix. The diagonal is in-domain, off-diagonal is transfer; a model strong at home and weak away is overfit to a regime — read the row spread, not one number, before rollout |

---

## operational — operational ML / live monitoring

The layer between a fitted model and a live process (events, milestones, custody checkpoints).
Generic over any checkpointed operational process (deliveries, claims, loans, clinical pathways);
exposed as the ``operational`` namespace.

| Function | Use when | What it tells you |
|---|---|---|
| `feed_readiness(df, entity=, milestone=, expected=, timestamp=)` | Before scoring against a live feed | Per-milestone coverage (share of entities with it) + recency; `expected` surfaces milestones that never arrive (coverage 0). The go/no-go gate: a model needing a 5%-coverage signal can't score live yet |
| `entities_missing(df, entity=, milestone=, required=)` | Triage incomplete entities | One row per entity short of a required milestone, with the `missing` list — chase the feed or hold the score before running on incomplete custody |
| `rescore_sequence(model, snapshots, feature_columns=, id_column=)` | Risk evolves as state arrives | Scores one fitted model on a sequence of progressively richer per-checkpoint snapshots → the risk *trajectory* per entity (the rising-toward-deadline ones are the ones to act on). Snapshots must be leak-free |
| `generate_alerts(df, score=, bands=, lead_time=, min_lead=)` | Turn risk into an action | Band ladder maps score → action by severity; the lead-time gate downgrades anything past the point of no return (`too_late`) — the distinction between "act" and "too late to act" that makes an alert operational |
| `alert_metrics(alerted, event, lead_time=)` | Backtest an alerting system | Detection rate (recall — the costly miss), precision (false-alarm cost), and mean lead time on detected events (an alert too late to act on isn't a catch) |
| `intervention_roi(events_detected=, value_per_prevented=, prevention_rate=, interventions=, cost_per_intervention=)` | Prove it paid | Prevented losses minus intervention cost → benefit/cost/net/ROI; every input is a number you must justify (the cost model stays explicit, not a buried placeholder) |

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

Results follow scipy conventions: with `maximize=True` the objective is negated internally —
read the optimum off `-result.fun`.

| Function | Use when | What it does |
|---|---|---|
| `linear_program(cost, a_ub=, b_ub=, ..., maximize=)` | Budget/capacity allocation | LP via scipy linprog; the standard "maximize linear objective under linear constraints" |
| `shadow_prices(result, names_ub=, maximize=)` | What is one more unit of capacity worth? | Duals of a solved LP: the marginal objective value of relaxing each constraint — the rational ceiling on paying for extra resource, 0 = not binding. The principled opportunity-cost number |
| `integer_program(cost, ..., integrality=, maximize=)` | Whole-unit / yes-no decisions | Mixed-integer LP (scipy milp). LP-then-round is *not* a substitute: rounding can break constraints and land far from the optimum |
| `knapsack(values, weights, capacity)` | Project/campaign selection under a budget | Exact 0/1 knapsack via MILP — greedy value-per-weight can be arbitrarily bad at integer scale |
| `nonlinear_program(objective, x0, bounds=, constraints=, maximize=)` | Curved objectives (diminishing returns, saturation) | scipy minimize wrapper. Local solver: check `analytics.curves.convexity` or multi-start before trusting one run |
| `assign(cost_matrix, maximize=)` | One-to-one matching (tasks↔people, slots↔ads) | Hungarian algorithm — exact optimal assignment |
| `portfolio_weights(expected_returns, covariance, risk_aversion=)` | Spread a budget over risky options | Mean-variance optimum (full investment, no shorting, concentration cap). The covariance is what diversifies — and μ/Σ are *estimates*: stress them before betting the budget |
| `pareto_front(points, maximize=)` | Several objectives, no honest single score | Mask of non-dominated options — the genuine trade-off menu (margin vs volume vs risk) without arbitrary weights |
| `scenario_optimize(value_fn, x0, scenarios, criterion=)` | One decision against many futures | Stochastic optimization: maximize the mean (`'mean'`) or the worst case (`'worst'`, robust max-min) over sampled scenarios — deciding under uncertainty, not per-scenario hindsight |

---

## decision.simulate — Monte Carlo: propagate uncertainty through a business model

Point estimates hide the tail — a plan built on means can still lose money 30% of the time.
Risk measures over the resulting samples live in `analytics.risk`.

| Function | Use when | What it tells you |
|---|---|---|
| `monte_carlo(value_fn, inputs, n=, correlation=)` | Any business case with uncertain inputs | Samples inputs (scipy distributions / custom samplers / constants; optionally rank-correlated via Gaussian copula — costs and volumes rarely move independently, and independence understates tail risk), pushes them through `value_fn`, returns the outcome *distribution* |
| `SimulationResult.summary(targets=)` / `.p10/.p50/.p90` / `.prob_above` | The deck numbers | P50 = plan, P10 = funding case, P(≥ target) = the commitment you can defend |
| `SimulationResult.drivers()` | Which uncertainty drives the spread | \|Spearman\| of each input vs the outcome across draws — the simulation-native tornado; de-risk (research, hedge, contract) the top driver first |
| `stress_test(value_fn, base, stresses)` | "Do we survive *this*?" | Named adverse shocks + the all-at-once combined row (correlations go to 1 in a crisis). Complements `monte_carlo`: probability vs survivability |
| `simulate_paths(start=, drift=, volatility=, periods=)` | Demand/revenue trajectories | Compounding (or additive) random-walk paths; `path_percentiles` turns them into the chart-ready uncertainty fan |

---

## decision.inventory — newsvendor, EOQ, safety stock

| Function | Use when | What it tells you |
|---|---|---|
| `newsvendor(price=, cost=, salvage=, demand_mean=/demand_samples=)` | One-shot stocking under uncertain demand | Optimal stock covers demand with P = critical fractile (p−c)/(p−s), *not* mean demand: fat margins justify deliberate overstock, thin ones deliberate stock-outs. Feed forecast Monte Carlo draws as `demand_samples` |
| `eoq(demand=, order_cost=, holding_cost=)` | Steady-demand order sizing | √(2DS/H); the optimum is famously flat (±20% on Q costs ~2%) — get holding cost roughly right and move on |
| `safety_stock(...)` / `reorder_point(...)` | Service through lead time | z·√(LT·σ_d² + μ_d²·σ_LT²) buffer + expected lead-time demand. Service is convex in cost: 95→99% costs far more than 90→95% — set levels per item value |
| `simulate_inventory_policy(demand, reorder_at=, order_quantity=)` | Validate the closed forms | Replays an (R, Q) policy against real/simulated demand; negative stock = stock-out depth |

---

## decision.capacity — Erlang C staffing & queueing economics

| Function | Use when | What it tells you |
|---|---|---|
| `erlang_c(arrival_rate=, service_rate=, servers=)` | Wait/queue metrics at a staffing level | M/M/c steady state: P(wait), average wait, queue length, utilization. The core economics: waiting explodes *nonlinearly* as utilization → 1, so "run at 95%" is a queueing disaster, not efficiency |
| `QueueMetrics.service_level(t)` | The SLA number | P(wait ≤ t) — "80% answered in 20s" |
| `required_servers(..., target_wait_probability=/target_service_level=, answer_within=)` | Staff to an SLA | Smallest c meeting the target, with full metrics attached — capacity-utilization optimization is the gap between this and what you run today |

---

## decision.game — game theory: the other side moves too

Optimization assumes the environment holds still; competitors don't. An equilibrium is a
*prediction of where dynamics settle*, not a recommendation.

| Function | Use when | What it tells you |
|---|---|---|
| `pure_nash(payoff_row, payoff_col)` | Discrete strategy games (discount vs hold) | Cells where neither side gains by deviating alone. The pricing prisoner's dilemma in one check: "both discount" can be the unique equilibrium even though "both hold" pays more |
| `mixed_nash_2x2(payoff_row, payoff_col)` | No pure equilibrium (matching-pennies structure) | The interior mixing probabilities — each side mixes to make the *other* indifferent, so your equilibrium mix comes from their payoffs |
| `iterated_dominance(payoff_row, payoff_col)` | Prune the strategy space | Strategies a rational player never uses, eliminated iteratively; one surviving cell = dominance-solvable game |
| `best_response_dynamics(responses, start, damping=)` | Competitive/price-war response simulation | Iterates everyone's best reply (reaction functions from each player's profit model) to its resting point — a continuous-strategy Nash equilibrium; seed `start` with your contemplated move to simulate the match-and-settle path |

---

## decision.scenario — valuing decisions under uncertainty

| Function | Use when | What it tells you |
|---|---|---|
| `expected_utility(outcomes, probs, risk_aversion=)` | Comparing gambles when risk appetite matters | Probability-weighted CARA utility: `risk_aversion=0` is plain expected value; a > 0 makes downside hurt more than equal upside helps, so volatile options score below their mean |
| `certainty_equivalent(outcomes, probs, risk_aversion=)` | Price a gamble in money | u⁻¹(EU): the sure amount worth exactly the gamble. EV − CE = the risk premium (what insurance or a guaranteed contract is rationally worth) |
| `scenario_table(value_fn, base, scenarios)` | Stress a decision under coherent futures | Re-values `value_fn(**inputs)` per named scenario (downturn, optimistic, ...) with `vs_base` swings — scenario analysis; inputs move *together* |
| `sensitivity(value_fn, base, ranges)` | Which assumption actually drives the answer? | One-at-a-time low→high swings ranked by absolute swing (tornado-ready): refine the biggest lever first. Misses input interactions by design — probe those with scenarios |

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
| `trend_test(y)` | Is the series drifting, and how fast? | Mann-Kendall test (non-parametric, outlier-proof: small p = monotonic trend) + Sen's slope = robust per-step rate. Deseasonalize first when seasonality is strong |
| `change_points(y, min_size=)` | Did the level break — and when? | Mean-shift detection by binary segmentation with a BIC-style penalty (stable series → `[]`). Line the indices up with deploys, price moves, campaigns, outages |

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

### forecasting.hierarchy — coherent forecasts across levels

| Function | Use when | Statistical reading |
|---|---|---|
| `reconcile(forecasts, hierarchy, method=)` | Total/region/product forecasts disagree | Makes every level add up. `ols` (default) projects all base forecasts onto the coherent subspace — pools information across levels and usually *improves* accuracy; `bottom_up` trusts the leaves; `top_down` splits the total by `proportions` (stable aggregate, leaf accuracy only as good as the split) |
| `coherence_error(forecasts, hierarchy)` | Pre-reconciliation diagnostic | Mean \|parent − Σchildren\| per node — a large gap at one node deserves a look before projection hides the disagreement |
| `summing_matrix(hierarchy)` | Build your own reconciler | The S matrix mapping leaf values to every node, plus node/leaf order |

---

## pricing — demand & elasticity, WTP, market analysis, price optimization

**Observational price-demand data is confounded** (prices were set in response to demand) —
prefer experimental/IV variation (`causal.iv_effect`) before pricing off any fit below.

### pricing.elasticity — estimation, uncertainty, segments, dynamics

| Function | Use when | Statistical reading |
|---|---|---|
| `fit_demand(price, quantity)` | Estimate demand response | OLS on ln(q) = a + e·ln(p): constant-elasticity model, slope = elasticity |
| `price_elasticity(price, quantity)` | Just the number | e < −1 elastic (price ↑ → revenue ↓), −1 < e < 0 inelastic (price ↑ → revenue ↑) |
| `predict_demand(intercept, elasticity, price)` | Scenario lines | Quantity under the fitted model |
| `fit_demand_ci(price, quantity, confidence=)` | The number *with its uncertainty* | Elasticity ± SE and t-CI. A CI spanning −1 means the data can't tell elastic from inelastic — the raise-vs-cut call is not identified; get more price variation |
| `bootstrap_elasticity(price, quantity)` | Small n / ugly residuals | Percentile-bootstrap CI, no normality assumption; agreement with `fit_demand_ci` is itself a robustness check |
| `cross_price_elasticity(quantity, own_price, {name: price})` | Substitutes & complements | One multivariate log-log fit: cross-e > 0 substitute, < 0 complement. Joint estimation matters — competitor prices move with yours, and a univariate fit absorbs their effect into your own slope |
| `segment_elasticity(df, price=, quantity=, segment=)` | Who is price-sensitive? | Per-segment fits with CIs → differentiated pricing: protect margin where \|e\| is small, compete where it's large |
| `rolling_elasticity(df, ..., window=)` | Dynamic elasticity through time | Re-fit over a rolling window; drifting/widening bands = the constant-elasticity assumption breaking |
| `elasticity_drift(df, ..., split=)` | Has price sensitivity moved? | Baseline-vs-recent z-test on the slopes; a drifted elasticity quietly invalidates the optimal price — re-optimize when it fires |
| `nonlinear_elasticity_check(price, quantity)` | Is one elasticity number wrong? | Adds (ln p)²: significant curvature + lower AIC = elasticity varies with price level; read `local_elasticity` at the prices you actually charge |
| `aggregate_elasticity` / `elasticity_decomposition(before, after)` | Portfolio elasticity & why it moved | Weighted mean; shift-share split into within-segment change vs mix shift (sums exactly) — a pure-mix move needs portfolio action, not price action |

### pricing.demand — demand curves, purchase probability, willingness-to-pay

| Function | Use when | Statistical reading |
|---|---|---|
| `fit_linear_demand(price, quantity)` | Demand with a choke price | q = a + b·p: elasticity varies along the line (`elasticity_at`), demand hits 0 at `choke_price` — predictions beyond it are extrapolation |
| `fit_logit_demand(price, purchased)` | Purchase probability from buy/no-buy offers | P(buy\|p) = sigmoid(a + b·p); implies WTP ~ Logistic(−a/b, −1/b), so `wtp_median`/`wtp_quantile` come closed-form. A positive slope = confounding (price proxies quality/segment), not a Veblen good |
| `willingness_to_pay(price, purchased)` | The WTP table | Quantile → price that share won't exceed: median = mass-market price, upper quantiles = the premium tier's room |
| `van_westendorp(too_cheap, cheap, expensive, too_expensive)` | Survey-based price range (no transactions yet) | Price sensitivity meter: optimal price point, indifference price, acceptable range from curve crossings. Stated preference — calibrate against transactions when you have them |
| `demand_schedule(quantity_fn, prices)` | Chart/optimize any demand model | (price, quantity, revenue) table — input for `analytics.curves` turning-point analysis |

### pricing.market — supply & demand, censored demand, saturation, structure

| Function | Use when | Statistical reading |
|---|---|---|
| `equilibrium(demand_fn, supply_fn, price_low=, price_high=)` / `linear_equilibrium(...)` | Where the market clears | Root of excess demand (Brent) / the closed form for linear curves; below the price the market is short, above it oversupplied |
| `supply_demand_gap(demand, supply)` | Market balance per period | Gap (excess demand), served, unmet, shortage/surplus regime — persistent one-sided gaps are the mismatch signal |
| `unconstrain_demand(sales, capacity)` | True demand when sales were capped | Censored-normal MLE: sold-out periods are right-censored (demand ≥ capacity), so raw averages understate demand exactly where it matters. `spill` = unmet demand/period, `spill_rate` = share never served — the classic revenue-management unconstraining step |
| `saturation_fit(t, y)` | How much market is left? | Logistic S-curve: `capacity` = market potential, `time_to_share(0.9)` = near-saturation date. Weakly identified before the inflection — treat early-stage capacity estimates as speculative |
| `market_share(df, value=, by=)` / `hhi(shares)` | Structure & concentration | Shares + cumulative; HHI on the 0-10,000 scale (< 1,500 competitive, > 2,500 concentrated) |

### pricing.optimize — optimal prices, marginal economics, dynamic pricing

| Function | Use when | Statistical reading |
|---|---|---|
| `revenue_at` / `profit_at(intercept, elasticity, price, unit_cost=)` | Money curves | Revenue/profit at candidate prices under the model |
| `optimal_price(intercept, elasticity, candidates, unit_cost=)` | Pick the price | Grid-search over realistic candidates (robust to any demand shape) |
| `markup_price(elasticity, unit_cost)` | Closed form, constant elasticity | c·e/(e+1); requires elastic demand (e < −1) |
| `optimal_price_linear(intercept, slope, unit_cost=)` | Closed form, linear demand | Midway between unit cost and the choke price — the parabola's vertex |
| `marginal_revenue(elasticity, price)` / `marginal_profit(..., unit_cost=)` | Direction-of-adjustment signal | MR = p(1 + 1/e) < p always; MR = 0 at e = −1 (the revenue peak). Marginal profit = 0 exactly at `markup_price`: its *sign* says raise vs cut even when the level is rough |
| `dynamic_prices(demand_rate, capacity=, periods=, candidates=)` | Fixed stock, finite horizon (revenue management) | Backward-induction DP over (period, remaining): Poisson sales around `demand_rate(price, t)`. The solved policy shows both classic forces — prices fall as the deadline nears and rise when stock runs scarce |

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
| `eda.fit_overlay` | Histogram + fitted distribution (pdf or pmf) — the eyeball check on `fit_distribution` / `fit_discrete`; a good AIC with a visibly wrong tail is what it catches |
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
| `model.decision_boundary` | Predicted regions over a 2-feature plane with the data on top: KNN neighbourhoods, tree tiles, logistic lines, SVM curves; k-means regions (no target) and regression surfaces too. `soft=True` shades P(class) — the honest view near the boundary. With >2 features it's a slice at the median row |
| `model.tree_diagram` | The fitted tree's actual split rules drawn (sklearn plot_tree) — interpretability you can hand a domain expert; unwraps `train.fit` pipelines, pass `forest.estimators_[0]` for one ensemble member |
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
| `timeseries.survival_curve` | Kaplan-Meier step curve(s) with CI bands, `{label: frame}` overlays segments — retention compared the censoring-correct way |
| `business.tornado` | `scenario.sensitivity` as the tornado: biggest lever on top, bars spanning low→high outcome, base line as the pivot |
| `business.waterfall` | The finance bridge: named contributions as floating bars (`change_decomposition`, summed `price_volume_mix`), gains/losses coloured apart, closing total |
| `business.fan` | Shaded quantile bands + center line — one chart for `path_percentiles` fans, rolling-elasticity CIs, conformal intervals |
| `business.control_chart` | `monitor.ewma_alerts` drawn: raw points, EWMA vs widening limits, alerts highlighted — the early-warning picture |
| `business.pareto_frontier` | Options scattered, the efficient set traced and labelled — the trade-off menu from `optimize.pareto_front` |
| `business.outcome_distribution` | Monte Carlo outcomes annotated with P10/P50/P90 and target lines — the deck version of a histogram |
| `business.price_curves` | Revenue/profit (any response) vs price with the recommended optimum marked |
| `business.van_westendorp` | The four survey curves with the crossing points (optimal / range bounds) marked |
| `business.price_policy` | Dynamic-pricing policy heatmap (period × remaining stock → price): markdown toward the deadline, scarcity premium toward the stock-out corner |
| `network.network` | Edge list as a force-directed graph (no networkx): node size = degree, edge width = weight; hubs central, communities clustered. Qualitative — read exact structure off `analytics.graph` tables; filter big edge lists first |
| `conceptual.dag(edges)` | Causal DAG from (cause, effect) pairs — pick the adjustment set: block backdoor paths (confounders), don't condition on colliders or mediators |
| `conceptual.gini_vs_entropy` / `conceptual.bias_variance` | Teaching sketches (impurity criteria, error decomposition) — functions of a parameter, not data |

---

## Concept → function index

| Concept | Where it lives |
|---|---|
| Bias-variance trade-off | `evaluate.learning_curve_scores` / `validation_curve_scores` + `viz.model.learning_curve`; regularized models in `registry`; bagging vs boosting (ensembles) |
| Linear & logistic regression | Inference: `regression.ols_fit` / `glm_fit(family="binomial")`; prediction: `registry.make_model("linear"/"logistic")` |
| Generalized linear models | `regression.glm_fit` (poisson / binomial / gamma / gaussian); registry `poisson`/`gamma`/`tweedie` for the prediction side |
| Fixed / random / mixed effects | `regression.fixed_effects` (within-entity, kills level confounding) vs `regression.mixed_effects` (random intercepts, partial pooling) |
| Linear-regression assumptions / residual analysis | `regression.linear_assumptions` (+ `vif`, `breusch_pagan`, `durbin_watson`), `stats.normality_test`, `viz.eda.qq`, `viz.model.residuals`/`scale_location` |
| Multicollinearity | `regression.vif`; `clean.drop_highly_correlated`; Ridge/Lasso/PCA |
| p-value vs confidence interval | `stats.TestResult`, `stats.mean_confidence_interval` / `proportion_confidence_interval` / `bootstrap_ci`, `experiment.ExperimentResult.confidence_interval` |
| CLT & law of large numbers | Why `mean_confidence_interval`, t-tests, and `bayes_means` work at modest n; `bootstrap_ci` is the same idea by simulation |
| Type I vs Type II error | `alpha` in every test = the false-positive rate you accept; power = 1 − β via `stats.power` / `sample_size_*`; `experiment.msprt_means` keeps Type I controlled under peeking |
| Bayes' theorem | `stats.bayes_rule` — the arithmetic behind all of `analytics.bayes` and `experiment.bayes_*` |
| Bootstrapping | `stats.bootstrap_ci` (BCa resampling CI for any statistic) |
| Permutation testing | `stats.permutation_test` (shuffle-based, assumption-free significance) |
| Likelihood & MLE | `stats.fit_distribution` / `best_distribution` (continuous); `fit_discrete` / `best_discrete` (counts); visual check via `viz.eda.fit_overlay` |
| Count distributions & overdispersion (Poisson / geometric / negative binomial / binomial / zero-inflated) | `stats.fit_discrete` / `best_discrete` / `dispersion_check`; regression side via `glm_fit(family="poisson")` and registry `poisson`/`tweedie` |
| Decision boundaries / class separation (KNN, trees, SVM, k-means) | `viz.model.decision_boundary` (hard regions or `soft=True` probability shading); the tree's rules via `viz.model.tree_diagram` |
| Bayesian vs frequentist | `experiment.bayes_conversions` / `bayes_means` vs `analyze_*`; `analytics.bayes`; `ThompsonSampling` |
| Prior / posterior / conjugate priors / credible intervals | `bayes.beta_posterior` (Beta-Binomial) and `bayes.gamma_posterior` (Gamma-Poisson, for rates); credible intervals also in `experiment.bayes_*` |
| MCMC | `bayes.mcmc_sample` (random-walk Metropolis) |
| Hierarchical models | `bayes.hierarchical_rates` (empirical-Bayes shrinkage); `regression.mixed_effects` (random intercepts) |
| Decision trees & overfitting | `registry.make_model("tree", max_depth=, min_samples_leaf=)`; ensembles |
| Bagging vs boosting | `random_forest`/`bagging` (variance ↓) vs `gradient_boosting`/`xgboost`/`lightgbm` (bias ↓); `ensemble.make_voting`/`make_stacking` |
| Regularization (L1/L2/elastic net) | `ridge` / `lasso` / `elasticnet` in the registry |
| Cross-validation & leakage | `split.make_cv`, `train.cross_validate`, fit-on-train `preprocess`, `imbalance.imbalanced_pipeline`, `group_split`/`time_split` |
| Class imbalance | `imbalance.*`, anomaly detection fallback, PR-AUC/MCC in `evaluate`, `kpi.profit_threshold` |
| Evaluation beyond accuracy | `evaluate.classification_metrics`, `viz.model.roc`/`precision_recall`/`calibration`, `kpi.profit` |
| Curse of dimensionality | `segment.pca` (+ `rfecv_scores`, `mutual_information` for selection); `segment.tsne` for the eyes |
| Feature engineering | The `features` package: `temporal` lags/rolling/calendar/cyclical, encodings + interactions in `transform`, text/geo helpers — promoted from notebooks, tested |
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
| Trend analysis | `diagnostics.trend_test` (Mann-Kendall + Sen's slope); `viz.timeseries.rolling_stats` |
| Change-point detection | `diagnostics.change_points` |
| Forecast error metrics | `backtest.mae` / `rmse` / `mape` / `smape`, computed on `rolling_origin` backtests |
| Drift (covariate/label/concept) | `monitor.psi`/`ks_drift` (covariate), `monitor.label_drift` (prior), score-drift via `drift_report` + delayed-label re-evaluation (concept) |
| Data / feed readiness before scoring live | `operational.feed_readiness` / `entities_missing` |
| Sequential / live risk re-scoring at checkpoints | `operational.rescore_sequence` |
| Alerting (risk + lead time → action) | `operational.generate_alerts` |
| Alert backtest & intervention ROI | `operational.alert_metrics` / `intervention_roi` |
| Cross-environment / does-it-generalize check | `modeling.compare.cross_environment` |
| A/B design, SRM, power, peeking | `stats.sample_size_*`/`power`, `experiment.srm_check`, `experiment.msprt_means`, `experiment.cuped_adjust` |
| Correlation vs causation; confounders | `analytics.causal` (matching, IPW, DiD, IV); stratify via `subgroup_effects`; draw the graph with `viz.conceptual.dag` |
| RCTs / A/B as gold standard | `analytics.experiment` end-to-end: `stats.sample_size_*` → `srm_check` → `analyze_*` / `bayes_*`; the causal tools exist for when you *can't* randomize |
| Average treatment effect (ATE) | `causal.uplift` (randomized), `causal.ipw_ate` (weighted), matched ATT via `match_on_propensity`; LATE via `itt_tot` / `iv_effect` |
| ITT vs TOT | `causal.itt_tot` |
| Instrumental variables | `causal.iv_effect` |
| DiD vs PSM | `causal.difference_in_differences` (unobserved time-invariant confounders) vs `propensity_scores` + `match_on_propensity` (observed ones) |
| Regression discontinuity (RDD) | `causal.regression_discontinuity` — threshold rules as local experiments |
| Synthetic control & counterfactual analysis | `causal.synthetic_control` builds the explicit "what would have happened" series; DiD and `experiment` lifts are counterfactual differences |
| Causal graphs (DAGs) | `viz.conceptual.dag` — sketch the graph, choose what to adjust for |
| Heterogeneous treatment effects | `causal.subgroup_effects`; uplift by segment |
| Uplift modeling & incrementality | `causal.TLearner` + `qini_points` / `qini_auc` (individual-level); campaign incrementality via `experiment.analyze_*`, DiD, or `synthetic_control` |
| Simpson's paradox | `stats.simpsons_check` — pooled vs within-group slopes, reversal flag |
| Selection / sampling bias | randomize; `transform.stratified_sample`; matching/weighting in `causal`; check sample-vs-population representativeness with `monitor.drift_report` |
| Survivorship bias | A design trap, not a function: build cohorts from *entry* (`features.period`, `viz.timeseries.cohort_heatmap`) so churned/failed units stay in the denominator |
| Measurement error | Noisy regressors attenuate effects toward 0 — gate inputs with `validate.check_schema`, and use an instrument (`causal.iv_effect`) when the mismeasured variable is the treatment |
| Expected value & cost-benefit | `kpi.profit.expected_value` / `profit_curve` / `profit_threshold`; `kpi.financial.roi` |
| Expected utility & risk attitude | `scenario.expected_utility` / `certainty_equivalent` (risk premium = EV − CE) |
| Sensitivity & scenario analysis | `scenario.sensitivity` (tornado), `scenario.scenario_table` (coherent what-ifs) |
| Customer lifetime value | `kpi.financial.clv` (+ churn / retention / NRR helpers); retention time from censored data → `survival.restricted_mean_survival` |
| Price elasticity & revenue optimization | `pricing.elasticity.fit_demand` / `fit_demand_ci` + `pricing.optimize.optimal_price` / `markup_price` / `optimal_price_linear` |
| Cross-price elasticity (substitutes/complements) | `pricing.elasticity.cross_price_elasticity` |
| Segment-level / dynamic elasticity & drift | `pricing.elasticity.segment_elasticity` / `rolling_elasticity` / `elasticity_drift` / `nonlinear_elasticity_check` / `elasticity_decomposition` |
| Willingness to pay / purchase probability | `pricing.demand.fit_logit_demand` / `willingness_to_pay` / `van_westendorp` |
| Demand curves & schedules | `pricing.demand.fit_linear_demand` / `demand_schedule`; constant-elasticity in `pricing.elasticity` |
| Dynamic pricing / revenue management | `pricing.optimize.dynamic_prices` (finite-horizon DP); censored-demand unconstraining → `pricing.market.unconstrain_demand` |
| Marginal revenue / marginal profit / marginal effects | `pricing.optimize.marginal_revenue` / `marginal_profit`; generic → `analytics.curves.marginal_effect` / `gradient` |
| Supply-demand equilibrium & market balance | `pricing.market.equilibrium` / `linear_equilibrium` / `supply_demand_gap` |
| Saturation & market potential | `pricing.market.saturation_fit` (logistic growth) |
| Market share & concentration | `pricing.market.market_share` / `hhi` |
| Derivatives, turning points, convexity | `analytics.curves`: `slope` / `curvature` / `local_extrema` / `inflection_points` / `convexity` / `response_curve` |
| Root-cause / driver decomposition | `analytics.drivers.change_decomposition`; revenue bridge → `price_volume_mix`; regression/SHAP for model-based drivers |
| Revenue leakage detection | `analytics.drivers.revenue_leakage` |
| Monte Carlo simulation & what-if | `decision.simulate.monte_carlo` (+ correlated inputs) / `simulate_paths`; coherent named futures → `scenario.scenario_table` |
| Stress testing | `decision.simulate.stress_test` (named shocks + combined worst case) |
| Value-at-risk / expected shortfall / drawdown | `analytics.risk.value_at_risk` / `expected_shortfall` / `max_drawdown` / `risk_summary` |
| Probability of hitting a target | `analytics.risk.probability_above` / `SimulationResult.prob_above` |
| Risk-adjusted performance | `analytics.risk.sharpe_ratio` / `sortino_ratio` / `downside_deviation` |
| Integer / mixed-integer optimization | `decision.optimize.integer_program` / `knapsack` |
| Nonlinear & stochastic/robust optimization | `decision.optimize.nonlinear_program` / `scenario_optimize` (mean or worst-case) |
| Portfolio optimization | `decision.optimize.portfolio_weights` (mean-variance) |
| Multi-objective trade-offs | `decision.optimize.pareto_front` |
| Shadow prices / opportunity cost | `decision.optimize.shadow_prices` on a solved LP |
| Inventory optimization | `decision.inventory.newsvendor` / `eoq` / `safety_stock` / `reorder_point` |
| Capacity & queueing (Erlang C) | `decision.capacity.erlang_c` / `required_servers` |
| Game theory / competitive response | `decision.game.pure_nash` / `mixed_nash_2x2` / `iterated_dominance` / `best_response_dynamics` |
| Hierarchical forecast reconciliation | `forecasting.hierarchy.reconcile` (ols / bottom-up / top-down) + `coherence_error` |
| Survival analysis (churn timing, censoring) | `modeling.survival.kaplan_meier` / `cox_ph` / `median_survival` / `restricted_mean_survival` |
| Recommendation systems | `modeling.recommend.ItemItemRecommender` (+ `popularity_baseline`); evaluate with `evaluate.ranking_metrics` |
| Ranking quality (NDCG/MRR) | `evaluate.ranking_metrics` |
| Market basket / association rules | `analytics.basket.frequent_itemsets` / `association_rules` |
| Network & graph analytics | `analytics.graph`: `degree_centrality` / `pagerank` / `connected_components` / `shortest_paths` / `minimum_spanning_tree_edges` / `max_flow` |
| Monotonicity & business-rule validation | `modeling.checks.monotonicity` / `expected_directions` / `prediction_bounds`; data rules → `features.validate.check_rules` |
| Model robustness / prediction consistency | `modeling.checks.perturbation_stability` |
| Counterfactual explanations | `modeling.interpret.counterfactual` (actionable levers only) |
| Uncertainty quantification for predictions | `modeling.interpret.conformal_intervals` (distribution-free); forecast intervals → `predict_interval`; bootstrap → `stats.bootstrap_ci` |
| Confidence scoring / triage routing | `modeling.interpret.confidence_score` |
| Early warning / control charts | `monitor.control_limits` / `ewma_alerts` (+ `drift_report` for the follow-up) |
| Entropy / cross-entropy / KL / information gain | `stats.entropy`; cross-entropy = `log_loss` in `evaluate.classification_metrics`; `stats.kl_divergence` (+ `monitor.psi`); `stats.information_gain` / `mutual_information` |

Out of scope by design: deep learning (CNNs/RNNs/transformers — this workspace is tabular/text/geo;
`mlp` and `sgd` in the registry are the in-stack neural options) and tools requiring extra
dependencies (Prophet → covered by `ets`/`sarimax` + holiday flags; causal forests → start with
`subgroup_effects` / `TLearner`; UMAP → `tsne`; LIME → the SHAP charts cover local explanations;
PyMC/Stan → `bayes.mcmc_sample` for small problems; ruptures → `diagnostics.change_points`;
networkx → `analytics.graph` for analytics, `viz.conceptual.dag` for DAG sketches; lifelines →
the statsmodels-backed `modeling.survival`; mlxtend → `analytics.basket`; hierarchicalforecast →
`forecasting.hierarchy`; MAPIE → `interpret.conformal_intervals`; dice-ml →
`interpret.counterfactual`; cvxpy/OR-tools/gurobi → the scipy-backed `decision.optimize`).
Full reinforcement learning and agent-based modeling are also out: `decision.bandits` is the
in-stack online learner, `pricing.optimize.dynamic_prices` the planned sequential decision, and
`decision.simulate` + `decision.game.best_response_dynamics` cover the simulation questions ABM
usually answers ("digital twin" asks, at this scale, are those two plus a good value function).
