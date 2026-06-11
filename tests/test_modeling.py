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
    assert means == sorted(means, reverse=True)  # ranked best-first


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
