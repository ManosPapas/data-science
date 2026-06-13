# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 · Churn modeling — supervised, end-to-end
#
# Predict customer churn: split → preprocess (no leakage) → compare several models on identical
# CV folds → evaluate the winner on held-out test → choose an operating threshold by *profit* →
# explain → persist. Imbalanced target (~17% churn), so we use ROC-AUC / PR and threshold tuning.

# %%
from sklearn.inspection import permutation_importance as sk_permutation
from sklearn.pipeline import Pipeline

from core.config import ROOT
from core.prelude import *

set_theme()

customers = read_parquet(ROOT / "data" / "raw" / "customers.parquet")
target = "churned"
numeric = [
    "age",
    "tenure_months",
    "num_products",
    "sessions_30d",
    "support_tickets",
    "monthly_spend",
    "satisfaction",
]
categorical = ["region", "segment", "plan"]
features = [*numeric, *categorical]
print(f"churn rate: {customers[target].mean():.1%}   n={customers.height}")

# %% [markdown]
# ## 1. Signal check
# Mutual information ranks how much each feature says about churn (drop nulls for the estimator).

# %%
stats.mutual_information(
    customers.select([*numeric, target]).drop_nulls(), target, task="classification"
)

# %%
fig, axes = base.grid(2)
eda.count_bar(customers, "churned", ax=axes[0], title="Churn (0 = stay, 1 = churn)")
eda.boxplot_by(customers, "tenure_months", "churned", ax=axes[1], title="Tenure by churn")

# %% [markdown]
# ## 2. Split + preprocessing
# Stratified split keeps the churn rate steady; the preprocessor is fit *inside* each pipeline so
# CV has no leakage.

# %%
train_df, test_df = split.train_test_split(customers, test_size=0.2, stratify=target, seed=42)
x_train, y_train = train_df.select(features), train_df[target]
x_test, y_test = test_df.select(features), test_df[target]
print(f"train={train_df.height}  test={test_df.height}")
print("balanced class weights:", imbalance.class_weights(y_train.to_numpy()))

# %% [markdown]
# ## 3. Compare models on identical folds
# Each model is a `preprocessor → estimator` pipeline so the leaderboard is leakage-free and fair.

# %%
pre = dict(numeric=numeric, categorical=categorical)
models = {
    "logistic": Pipeline(
        [
            ("pre", preprocess.make_preprocessor(**pre)),
            ("clf", registry.make_model("logistic", task="classification", max_iter=1000)),
        ]
    ),
    "random_forest": Pipeline(
        [
            ("pre", preprocess.make_preprocessor(**pre)),
            (
                "clf",
                registry.make_model(
                    "random_forest", task="classification", n_estimators=200, random_state=42
                ),
            ),
        ]
    ),
    "gradient_boosting": Pipeline(
        [
            ("pre", preprocess.make_preprocessor(**pre)),
            (
                "clf",
                registry.make_model("gradient_boosting", task="classification", random_state=42),
            ),
        ]
    ),
    "xgboost": Pipeline(
        [
            ("pre", preprocess.make_preprocessor(**pre)),
            (
                "clf",
                registry.make_model(
                    "xgboost",
                    task="classification",
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                ),
            ),
        ]
    ),
    "lightgbm": Pipeline(
        [
            ("pre", preprocess.make_preprocessor(**pre)),
            (
                "clf",
                registry.make_model(
                    "lightgbm", task="classification", n_estimators=200, random_state=42, verbose=-1
                ),
            ),
        ]
    ),
}
list(models)

# %%
# `n_jobs=-1` fits folds across all cores; leave at 1 for fast models or reproducible serial runs.
cv = split.make_cv("stratified", n_splits=5)
board = compare.leaderboard(
    models, x_train, y_train, cv=cv, scoring=["roc_auc", "average_precision", "f1"], n_jobs=-1
)
board

# %%
# Per-fold ROC-AUC (same folds) → a fair box plot and a paired significance test on the top two.
fold = {
    name: compare.fold_scores(pipe, x_train, y_train, cv=cv, scoring="roc_auc")
    for name, pipe in models.items()
}
ranked = sorted(fold, key=lambda k: fold[k].mean(), reverse=True)
print("top two:", ranked[:2], "->", compare.paired_test(fold[ranked[0]], fold[ranked[1]]))

fig, axes = base.grid(1, ncols=1)
model.model_comparison(fold, ax=axes[0], title="Per-fold ROC-AUC by model")

# %% [markdown]
# ### Imbalance strategies — does reweighting the loss change anything?
# `class_weight="balanced"` makes minority mistakes cost more. The ranking metric (ROC-AUC)
# barely moves — reweighting shifts the *operating point*, which is exactly what default-threshold
# F1 shows. (The resampling route — SMOTE via `imbalance.make_resampler` +
# `imbalance.imbalanced_pipeline`, which resamples inside training folds only — is the next lever;
# it changes the training prior, so recheck calibration before reading its outputs as
# probabilities.)

# %%
weighted = Pipeline(
    [
        ("pre", preprocess.make_preprocessor(**pre)),
        (
            "clf",
            registry.make_model(
                "logistic", task="classification", max_iter=1000, class_weight="balanced"
            ),
        ),
    ]
)
for name, pipe in {"plain": models["logistic"], "class_weight": weighted}.items():
    auc = compare.fold_scores(pipe, x_train, y_train, cv=cv, scoring="roc_auc")
    f1 = compare.fold_scores(pipe, x_train, y_train, cv=cv, scoring="f1")
    print(f"{name:12s} ROC-AUC {auc.mean():.3f} ± {auc.std():.3f}   F1@0.5 {f1.mean():.3f}")

# %% [markdown]
# ### Hyper-parameter search + the validation curve
# Randomized search samples the parameter space (cheaper than a grid at equal quality); the
# validation curve shows *why* the chosen depth wins — past it, train keeps improving while
# validation slips: overfitting's signature.

# %%
search = tune.random_search(
    models["xgboost"],
    {
        "clf__max_depth": [2, 3, 4, 6],
        "clf__learning_rate": [0.03, 0.1, 0.3],
        "clf__n_estimators": [100, 200, 400],
    },
    x_train,
    y_train,
    n_iter=8,
    cv=3,
    scoring="roc_auc",
    seed=42,
)
print(f"best CV ROC-AUC {search.best_score_:.3f} with {search.best_params_}")

# %%
depths, train_curve, val_curve = evaluate.validation_curve_scores(
    models["xgboost"],
    x_train,
    y_train,
    param_name="clf__max_depth",
    param_range=[1, 2, 3, 4, 6, 8],
    cv=3,
    scoring="roc_auc",
)
fig, axes = base.grid(1, ncols=1)
model.validation_curve(
    depths, train_curve, val_curve, ax=axes[0], title="ROC-AUC vs tree depth (xgboost)"
)

# %% [markdown]
# ### Ensembling the leaderboard
# A soft-voting blend pays off when members are diverse *and* comparably strong — equal-weight
# averaging with weaker members drags the blend below the best single model, which is exactly
# what the fold scores adjudicate here. `ensemble.make_stacking` is the next step up: it *learns*
# the blend weights on out-of-fold predictions instead of assuming them equal.

# %%
blend = ensemble.make_voting(
    [
        ("logistic", models["logistic"]),
        ("xgboost", models["xgboost"]),
        ("lgbm", models["lightgbm"]),
    ],
    voting="soft",
)
blend_auc = compare.fold_scores(blend, x_train, y_train, cv=cv, scoring="roc_auc")
print(f"voting blend ROC-AUC {blend_auc.mean():.3f} ± {blend_auc.std():.3f}")
print(f"best single  ROC-AUC {fold[ranked[0]].mean():.3f} ± {fold[ranked[0]].std():.3f}")

# %% [markdown]
# ## 4. Fit the winner, evaluate on held-out test
# First an honest preview that spends no test data: out-of-fold predictions — every training row
# scored by a fold-model that never saw it.

# %%
best_name = board["model"][0]
best = models[best_name]
oof_score = train.cross_val_predict(best, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
oof_auc = evaluate.classification_metrics(
    y_train.to_numpy(), (oof_score >= 0.5).astype(int), y_score=oof_score
)["roc_auc"]
print(f"out-of-fold ROC-AUC on train: {oof_auc:.3f}  (the held-out test should land nearby)")

# %%
fitted = train.fit(best, x_train, y_train)
y_score = train.predict_proba(fitted, x_test)[:, 1]
y_pred = train.predict(fitted, x_test)
print("winner:", best_name)

# %%
# Confusion-derived rates by name: sensitivity (recall, churners caught), specificity (stayers
# cleared), NPV (trust in a "won't churn" call), beside precision/F1/MCC. Which to optimize is a
# business call — false-alarm-averse teams watch specificity; can't-miss-churners teams watch
# sensitivity.
report = evaluate.classification_metrics(y_test, y_pred, y_score=y_score)
print({k: round(v, 3) for k, v in report.items()})
print(
    f"sensitivity {report['sensitivity']:.0%} (churners caught) · "
    f"specificity {report['specificity']:.0%} (stayers cleared) · NPV {report['npv']:.0%}"
)

# %%
evaluate.report(y_test, y_pred)

# %%
fig, axes = base.grid(6, ncols=3)
model.roc(y_test, y_score, ax=axes[0])
model.precision_recall(y_test, y_score, ax=axes[1])
model.confusion(y_test, y_pred, ax=axes[2])
model.calibration(y_test, y_score, ax=axes[3])
model.gains_curve(y_test, y_score, ax=axes[4])
model.lift_curve(y_test, y_score, ax=axes[5])

# %% [markdown]
# ## 5. Operating point — tune the threshold by F1 and by profit
# A missed churner (FN) costs a lost customer; a retention offer to a happy customer (FP) is cheap.

# %%
print(f"F1-optimal threshold: {imbalance.tune_threshold(y_test, y_score, metric='f1'):.2f}")
costs = {"tp": 80.0, "fp": -20.0, "tn": 0.0, "fn": -200.0}
threshold, value = profit.profit_threshold(y_test, y_score, costs=costs)
print(f"profit-optimal threshold: {threshold:.2f}  expected value: {value:,.0f}")

# %% [markdown]
# ## 6. Explain
# Permutation importance is model-agnostic (works for any winner); native importances if available.

# %%
x_test_pd = x_test.to_pandas()
x_test_pd[numeric] = x_test_pd[numeric].astype("float64")  # PDP/perm need float, not int
perm = sk_permutation(
    fitted, x_test_pd, y_test.to_numpy(), n_repeats=5, random_state=42, scoring="roc_auc"
)
fig, axes = base.grid(1, ncols=1)
explain.permutation_importance(
    features, perm.importances_mean, perm.importances_std, ax=axes[0], top=12
)

# %%
clf = fitted.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    names = list(fitted.named_steps["pre"].get_feature_names_out())
    fig, axes = base.grid(1, ncols=1)
    explain.feature_importance(names, clf.feature_importances_, ax=axes[0], top=12)

# %%
explain.partial_dependence(fitted, x_test_pd, ["tenure_months", "monthly_spend"])

# %% [markdown]
# ### SHAP (needs the `explain` extra)
# Per-customer, per-feature contributions on the boosted challenger — TreeExplainer wants a tree
# model, so we explain the xgboost pipeline rather than the linear winner.

# %%
import shap

xgb_fitted = train.fit(models["xgboost"], x_train, y_train)
pre_fitted = xgb_fitted.named_steps["pre"]
shap_x = pd.DataFrame(
    pre_fitted.transform(x_test_pd), columns=list(pre_fitted.get_feature_names_out())
)
shap_values = shap.TreeExplainer(xgb_fitted.named_steps["clf"]).shap_values(shap_x)
explain.shap_summary(shap_values, shap_x)

# %%
explain.shap_bar(shap_values, shap_x)

# %% [markdown]
# ## 7. Persist (versioned model store)

# %%
saved = persist.save_model(
    fitted, "churn", metadata={"model": best_name, "roc_auc": float(board["roc_auc_mean"][0])}
)
print(f"saved -> {saved}   versions: {persist.model_versions('churn')}")

# %%
# The round trip a scoring job does: load newest version, batch-score fresh rows.
reloaded = persist.load_model("churn")
train.score_frame(reloaded, test_df.select(features).head(5)).select(
    ["age", "tenure_months", "plan", "prediction"]
)

# %% [markdown]
# **Takeaways:** a fair, leakage-free model comparison (with imbalance handled two ways and the
# blend checked against the best single model); a tuned depth justified by its validation curve;
# an out-of-fold preview that matched the held-out test; a profit-driven threshold; model-agnostic
# explanations; and a versioned artifact that reloads straight into batch scoring.
