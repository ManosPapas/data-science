"""Tests for the modeling registry + train/evaluate harness."""

from __future__ import annotations

import numpy as np
import polars as pl

from core.modeling import (
    anomaly,
    compare,
    evaluate,
    persist,
    preprocess,
    registry,
    segment,
    split,
    train,
)


def test_make_model_returns_estimator() -> None:
    model = registry.make_model("ridge", task="regression", alpha=0.5)
    assert hasattr(model, "fit")


def test_available_models_nonempty() -> None:
    assert "xgboost" in registry.available_models("regression")


def test_glm_regressors_construct() -> None:
    for name in ("gamma", "tweedie"):
        assert hasattr(registry.make_model(name, task="regression"), "fit")


def test_fit_and_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    model = train.fit(registry.make_model("linear", task="regression"), x, y)
    predictions = train.predict(model, x)
    assert predictions.shape == (60,)


def test_regression_metrics() -> None:
    metrics = evaluate.regression_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert metrics["rmse"] >= 0.0


def test_classification_metrics() -> None:
    metrics = evaluate.classification_metrics(
        [0, 1, 1, 0], [0, 1, 0, 0], y_score=[0.2, 0.8, 0.4, 0.1]
    )
    assert metrics["accuracy"] == 0.75  # 3 of 4 correct


def test_train_test_split_sizes() -> None:
    df = pl.DataFrame({"x": list(range(100)), "y": [0, 1] * 50})
    train_df, test_df = split.train_test_split(df, test_size=0.2, stratify="y", seed=0)
    assert train_df.height == 80
    assert test_df.height == 20


def test_make_preprocessor_fit_transform() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]}).to_pandas()
    pre = preprocess.make_preprocessor(numeric=["a"], categorical=["b"])
    transformed = pre.fit_transform(frame)
    assert transformed.shape[0] == 3


def test_leaderboard_ranks_models(rng: np.random.Generator) -> None:
    x = pl.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pl.Series("y", (x["a"] > 0).cast(int))
    models = {
        "logistic": registry.make_model("logistic", task="classification", max_iter=200),
        "tree": registry.make_model("tree", task="classification", random_state=0),
    }
    board = compare.leaderboard(models, x, y, cv=3, scoring="accuracy")
    assert board.height == 2
    assert "accuracy_mean" in board.columns
    means = board["accuracy_mean"].to_list()
    assert means == sorted(means, reverse=True)


def test_segment_clusterer_and_elbow(rng: np.random.Generator) -> None:
    x = rng.normal(size=(60, 3))
    clusterer = segment.make_clusterer("kmeans", n_clusters=3, n_init=10, random_state=0)
    labels = clusterer.fit_predict(x)
    assert len(set(labels.tolist())) == 3
    _, inertias = segment.elbow_scores(x, [2, 3, 4])
    assert len(inertias) == 3


def test_anomaly_labels(rng: np.random.Generator) -> None:
    x = rng.normal(size=(100, 2))
    detector = anomaly.make_detector("isolation_forest", contamination=0.05, random_state=0)
    labels = anomaly.anomaly_labels(detector, x)
    assert set(labels.tolist()) <= {-1, 1}
    assert (labels == -1).sum() > 0  # contamination=0.05 actually flags outliers


def test_persist_roundtrip(tmp_path: object, monkeypatch: object) -> None:
    from sklearn.linear_model import LinearRegression

    monkeypatch.setattr(persist, "MODELS_DIR", tmp_path)  # type: ignore[attr-defined]
    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    saved = persist.save_model(model, "demo", metadata={"k": 1})
    assert saved.exists()
    assert persist.model_versions("demo") == [1]
    assert hasattr(persist.load_model("demo"), "predict")


def test_permutation_importance_ranks_the_real_signal(rng: np.random.Generator) -> None:
    x = pl.DataFrame({"signal": rng.normal(size=300), "noise": rng.normal(size=300)})
    y = pl.Series("y", x["signal"].to_numpy() * 2.0 + rng.normal(scale=0.1, size=300))
    model = train.fit(registry.make_model("linear", task="regression"), x, y)
    importance = evaluate.permutation_importance(model, x, y, n_repeats=5, scoring="r2", seed=0)
    assert importance["feature"][0] == "signal"
    assert importance["importance_mean"][0] > importance["importance_mean"][1]


def test_learning_and_validation_curve_scores(rng: np.random.Generator) -> None:
    x = rng.normal(size=(120, 3))
    y = (x[:, 0] > 0).astype(int)
    model = registry.make_model("logistic", task="classification", max_iter=200)
    sizes, train_scores, val_scores = evaluate.learning_curve_scores(model, x, y, cv=3)
    assert sizes.shape[0] == train_scores.shape[0] == val_scores.shape[0] == 5
    assert train_scores.shape[1] == 3  # one column per cv fold
    values, train_curve, val_curve = evaluate.validation_curve_scores(
        model, x, y, param_name="C", param_range=[0.01, 1.0, 100.0], cv=3
    )
    assert values.shape == (3,)
    assert train_curve.shape == val_curve.shape == (3, 3)


def test_rfecv_keeps_the_informative_features(rng: np.random.Generator) -> None:
    x = pl.DataFrame(
        {
            "s1": rng.normal(size=250),
            "s2": rng.normal(size=250),
            "n1": rng.normal(size=250),
            "n2": rng.normal(size=250),
        }
    )
    y = pl.Series("y", x["s1"].to_numpy() + x["s2"].to_numpy() + rng.normal(scale=0.2, size=250))
    result = evaluate.rfecv_scores(
        registry.make_model("linear", task="regression"), x, y, cv=3, scoring="r2"
    )
    assert {"s1", "s2"} <= set(result.selected)
    assert result.n_features.shape == result.mean_scores.shape == result.std_scores.shape


def test_tsne_embeds_to_2d(rng: np.random.Generator) -> None:
    coords = segment.tsne(rng.normal(size=(60, 4)), perplexity=10.0, seed=0)
    assert coords.shape == (60, 2)
    assert np.isfinite(coords).all()


def test_preprocessor_strategy_options() -> None:
    frame = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": ["x", "y", "x", None]}).to_pandas()
    pre = preprocess.make_preprocessor(
        numeric=["a"], categorical=["b"], imputer="knn", scaler="minmax", encoder="ordinal"
    )
    transformed = pre.fit_transform(frame)
    assert transformed.shape == (4, 2)  # min-max scaled 'a' + ordinal-coded 'b'
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_make_imputer_strategies_construct() -> None:
    for strategy in ("mean", "median", "knn", "iterative"):
        assert hasattr(preprocess.make_imputer(strategy), "fit")


def test_target_encoder_consumes_y() -> None:
    frame = pl.DataFrame({"c": ["a", "b"] * 20}).to_pandas()
    y = np.array([1.0, 0.0] * 20)
    pre = preprocess.make_preprocessor(categorical=["c"], encoder="target")
    transformed = pre.fit_transform(frame, y)
    assert transformed.shape == (40, 1)  # one numeric column, no one-hot explosion
