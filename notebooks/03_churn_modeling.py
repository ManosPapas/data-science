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
cv = split.make_cv("stratified", n_splits=5)
board = compare.leaderboard(
    models, x_train, y_train, cv=cv, scoring=["roc_auc", "average_precision", "f1"]
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
# ## 4. Fit the winner, evaluate on held-out test

# %%
best_name = board["model"][0]
best = models[best_name]
fitted = train.fit(best, x_train, y_train)
y_score = train.predict_proba(fitted, x_test)[:, 1]
y_pred = train.predict(fitted, x_test)
print("winner:", best_name)

# %%
evaluate.classification_metrics(y_test, y_pred, y_score=y_score)

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
x_test_pd[numeric] = x_test_pd[numeric].astype("float64")  # PDP/perm want float, not int
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
fig, axes = base.grid(2)
explain.partial_dependence(fitted, x_test_pd, ["tenure_months"], ax=axes[0])
explain.partial_dependence(fitted, x_test_pd, ["monthly_spend"], ax=axes[1])

# %% [markdown]
# ## 7. Persist (versioned model store)

# %%
saved = persist.save_model(
    fitted, "churn", metadata={"model": best_name, "roc_auc": float(board["roc_auc_mean"][0])}
)
print(f"saved -> {saved}   versions: {persist.model_versions('churn')}")

# %% [markdown]
# **Takeaways:** a fair, leakage-free model comparison; the winner evaluated on held-out data with
# the full curve suite; a profit-driven threshold; model-agnostic explanations; and a versioned,
# reloadable artifact.
