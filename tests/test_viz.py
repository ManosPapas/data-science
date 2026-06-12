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
