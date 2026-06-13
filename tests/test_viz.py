"""Smoke tests: every chart builds without error and returns the expected object."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.viz import base, cluster, conceptual, eda, explain, interactive, model, timeseries


def test_eda_charts_return_axes(rng: np.random.Generator) -> None:
    df = pl.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "g": ["x", "y"] * 25})
    assert isinstance(eda.histogram(df["a"].to_numpy()), Axes)
    assert isinstance(eda.scatter(df, "a", "b"), Axes)
    assert isinstance(eda.boxplot_by(df, "a", "g"), Axes)
    assert isinstance(eda.count_bar(df, "g"), Axes)
    assert isinstance(eda.correlation_heatmap(df), Axes)


def test_model_charts_return_axes(rng: np.random.Generator) -> None:
    y_true = (rng.uniform(size=100) > 0.5).astype(int)
    y_score = rng.uniform(size=100)
    assert isinstance(model.roc(y_true, y_score), Axes)
    assert isinstance(model.confusion(y_true, (y_score > 0.5).astype(int)), Axes)
    assert isinstance(model.calibration(y_true, y_score), Axes)


def test_cluster_and_conceptual_charts() -> None:
    assert isinstance(cluster.elbow([2, 3, 4], [10.0, 6.0, 4.0]), Axes)
    assert isinstance(conceptual.gini_vs_entropy(), Axes)
    edges = [("season", "price"), ("season", "demand"), ("price", "demand")]
    assert isinstance(conceptual.dag(edges), Axes)


def test_chart_decorator_options(rng: np.random.Generator) -> None:
    _, axes = base.grid(2)
    assert len(axes) == 2
    ax = eda.histogram(rng.normal(size=30), ax=axes[0], title="T", grid=True)
    assert ax.get_title() == "T"


def test_timeseries_and_multipanel(rng: np.random.Generator) -> None:
    series = rng.normal(size=60)
    assert isinstance(timeseries.acf(series, lags=10), Axes)
    fig = eda.pairplot(pl.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)}))
    assert isinstance(fig, Figure)


def test_partial_dependence_returns_figure(rng: np.random.Generator) -> None:
    from sklearn.linear_model import LogisticRegression

    x = rng.normal(size=(60, 2))
    y = (x[:, 0] > 0).astype(int)
    fig = explain.partial_dependence(LogisticRegression().fit(x, y), x, [0])
    assert isinstance(fig, Figure)


def test_chart_signature_shows_injected_kwargs() -> None:
    import inspect

    params = inspect.signature(eda.histogram).parameters
    assert "values" in params  # the draw function's own data parameter
    for injected in ("ax", "title", "grid", "save"):
        assert injected in params  # Jupyter help must show the decorator's keywords


def test_interactive_charts_return_plotly_figure() -> None:
    pytest.importorskip("plotly")  # optional 'interactive' extra — skip, don't fail, without it
    df = pl.DataFrame({"x": [1, 2, 3], "y": [3.0, 1.0, 2.0]})
    fig = interactive.line(df, "x", "y", title="t")
    assert type(fig).__module__.startswith("plotly")
    assert fig.layout.xaxis.showgrid in (False, None)


def test_business_charts_return_axes(rng: np.random.Generator) -> None:
    from core.viz import business

    tornado_table = pl.DataFrame(
        {
            "input": ["volume", "cost"],
            "low_value": [80.0, 95.0],
            "high_value": [120.0, 104.0],
            "swing": [40.0, 9.0],
        }
    )
    assert isinstance(business.tornado(tornado_table, base=100.0), Axes)

    bridge = pl.DataFrame({"effect": ["price", "volume", "mix"], "value": [300.0, -120.0, 80.0]})
    assert isinstance(business.waterfall(bridge, label="effect", value="value", start=1000.0), Axes)

    bands = pl.DataFrame(
        {"period": [0, 1, 2], "p10": [9.0, 8.5, 8.0], "p50": [10.0] * 3, "p90": [11.0, 11.5, 12.0]}
    )
    assert isinstance(business.fan(bands, x="period", bands=[("p10", "p90")], line="p50"), Axes)

    ewma = pl.DataFrame(
        {
            "t": [0, 1, 2, 3],
            "value": [1.0, 1.1, 2.0, 2.2],
            "ewma": [1.0, 1.02, 1.2, 1.4],
            "lower": [0.8] * 4,
            "upper": [1.3] * 4,
            "alert": [False, False, False, True],
        }
    )
    assert isinstance(business.control_chart(ewma), Axes)

    frontier = pl.DataFrame(
        {
            "config": ["a", "b", "c"],
            "margin": [10.0, 8.0, 9.0],
            "volume": [100.0, 300.0, 90.0],
            "efficient": [True, True, False],
        }
    )
    assert isinstance(
        business.pareto_frontier(
            frontier, x="margin", y="volume", efficient="efficient", label="config"
        ),
        Axes,
    )

    assert isinstance(
        business.outcome_distribution(rng.normal(100.0, 10.0, 500), targets=[90.0]), Axes
    )

    schedule = pl.DataFrame(
        {"price": [10.0, 20.0, 30.0], "revenue": [900.0, 1600.0, 2100.0], "profit": [1.0, 2.0, 1.5]}
    )
    assert isinstance(business.price_curves(schedule, optimum=20.0), Axes)

    vw_curves = pl.DataFrame(
        {
            "price": [50.0, 100.0, 150.0],
            "too_cheap": [0.9, 0.4, 0.1],
            "cheap": [0.95, 0.6, 0.2],
            "expensive": [0.1, 0.5, 0.9],
            "too_expensive": [0.05, 0.3, 0.8],
        }
    )
    assert isinstance(business.van_westendorp(vw_curves, points={"optimal": 100.0}), Axes)

    policy = pl.DataFrame(
        {
            "period": [0, 0, 1, 1],
            "remaining": [1, 2, 1, 2],
            "price": [120.0, 100.0, 110.0, 90.0],
        }
    )
    assert isinstance(business.price_policy(policy), Axes)


def test_network_chart(rng: np.random.Generator) -> None:
    from core.viz import network as network_viz

    edges = pl.DataFrame(
        {"source": ["a", "a", "b"], "target": ["b", "c", "c"], "w": [1.0, 2.0, 1.0]}
    )
    ax = network_viz.network(edges, weight="w", seed=1)
    assert isinstance(ax, Axes)
    with pytest.raises(ValueError, match="legibly"):
        big = pl.DataFrame(
            {"source": [f"n{i}" for i in range(600)], "target": [f"n{i + 1}" for i in range(600)]}
        )
        network_viz.network(big)


def test_survival_curve_chart(rng: np.random.Generator) -> None:
    from core.modeling import survival

    durations = rng.exponential(10.0, 300)
    events = (durations < 15.0).astype(float)
    durations = np.minimum(durations, 15.0)
    km = survival.kaplan_meier(durations, events)
    assert isinstance(timeseries.survival_curve(km), Axes)
    assert isinstance(timeseries.survival_curve({"a": km, "b": km}), Axes)


def test_fit_overlay_chart(rng: np.random.Generator) -> None:
    continuous = eda.fit_overlay(rng.normal(10.0, 2.0, 400), dist="norm", params=(10.0, 2.0))
    assert isinstance(continuous, Axes)
    discrete = eda.fit_overlay(
        rng.poisson(3.0, 400).astype(float), dist="poisson", params={"mu": 3.0}
    )
    assert isinstance(discrete, Axes)


def test_decision_boundary_and_tree(rng: np.random.Generator) -> None:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    points = pl.DataFrame({"f1": rng.normal(0, 1, 120), "f2": rng.normal(0, 1, 120)})
    labelled = points.with_columns(
        ((pl.col("f1") + pl.col("f2")) > 0).cast(pl.Int32).alias("label")
    )
    features = labelled.select("f1", "f2").to_pandas()
    target = labelled["label"].to_numpy()

    knn = KNeighborsClassifier(5).fit(features, target)
    assert isinstance(
        model.decision_boundary(knn, labelled, x="f1", y="f2", target="label", resolution=40),
        Axes,
    )

    logit = LogisticRegression().fit(features, target)
    assert isinstance(
        model.decision_boundary(
            logit, labelled, x="f1", y="f2", target="label", soft=True, resolution=30
        ),
        Axes,
    )

    km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(features)
    assert isinstance(model.decision_boundary(km, points, x="f1", y="f2", resolution=30), Axes)

    tree = DecisionTreeClassifier(max_depth=3).fit(features, target)
    assert isinstance(
        model.tree_diagram(tree, feature_names=["f1", "f2"], class_names=["stay", "churn"]), Axes
    )


def test_fit_overlay_zip(rng: np.random.Generator) -> None:
    counts = np.where(rng.random(500) < 0.3, 0, rng.poisson(5.0, 500)).astype(float)
    assert isinstance(eda.fit_overlay(counts, dist="zip", params={"pi": 0.3, "mu": 5.0}), Axes)
