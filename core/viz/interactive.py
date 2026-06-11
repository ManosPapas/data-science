"""Interactive charts (Plotly) — the interactive counterpart to the static matplotlib charts.

Mirrors ``viz.base``: ``@interactive_chart`` is the Plotly analogue of ``@chart``. Each function
just *builds* a Plotly figure from a prepared Polars frame + columns; the decorator owns the
cross-cutting concerns — shared high-contrast theme, an optional ``title``, an optional ``grid``
(off by default), and an optional ``save`` (``.html``, or a static image if the optional ``kaleido``
package is present) — then returns the figure. Plotly is optional (the ``interactive`` extra) and
imported lazily, so this module loads even without it — only *calling* a chart needs it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import polars as pl

_TEMPLATE = "simple_white"  # clean, gridless, high-contrast base
_COLORWAY = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
]


def _write(fig: Any, path: str) -> None:
    """Save to ``.html`` (self-contained) or, for image extensions, via the optional ``kaleido``."""
    if path.lower().endswith((".html", ".htm")):
        fig.write_html(path)
    else:
        fig.write_image(path)


def interactive_chart(draw: Callable[..., Any]) -> Callable[..., Any]:
    """Plotly analogue of ``@chart``: the function builds the figure, this wrapper finishes it.

    Adds ``title`` / ``grid`` / ``save`` keywords, applies the shared theme (high-contrast colorway,
    readable left-aligned title, no gridlines unless ``grid=True``), then returns the Plotly figure.
    """

    @functools.wraps(draw)
    def wrapper(
        *args: Any,
        title: str | None = None,
        grid: bool = False,
        save: str | None = None,
        **kwargs: Any,
    ) -> Any:
        fig = draw(*args, **kwargs)
        layout: dict[str, Any] = {
            "colorway": _COLORWAY,
            "title_font_size": 16,
            "title_x": 0.01,
            "font": {"size": 12},
            "margin": {"t": 60, "r": 25, "b": 50, "l": 60},
        }
        if title is not None:
            layout["title_text"] = title
        fig.update_layout(**layout)
        fig.update_xaxes(showgrid=grid, gridcolor="#e9e9e9")
        fig.update_yaxes(showgrid=grid, gridcolor="#e9e9e9")
        if save is not None:
            _write(fig, save)
        return fig

    return wrapper


@interactive_chart
def line(df: pl.DataFrame, x: str, y: str | Sequence[str], *, color: str | None = None) -> Any:
    """Interactive line chart (trends, multiple series)."""
    import plotly.express as px

    return px.line(df.to_pandas(), x=x, y=y, color=color, template=_TEMPLATE)


@interactive_chart
def scatter(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    size: str | None = None,
    hover: Sequence[str] | None = None,
) -> Any:
    """Interactive scatter plot with hover detail."""
    import plotly.express as px

    return px.scatter(
        df.to_pandas(),
        x=x,
        y=y,
        color=color,
        size=size,
        hover_data=list(hover or []),
        template=_TEMPLATE,
    )


@interactive_chart
def bar(
    df: pl.DataFrame, x: str, y: str, *, color: str | None = None, barmode: str = "group"
) -> Any:
    """Interactive bar chart."""
    import plotly.express as px

    return px.bar(df.to_pandas(), x=x, y=y, color=color, barmode=barmode, template=_TEMPLATE)


@interactive_chart
def histogram(
    df: pl.DataFrame, x: str, *, color: str | None = None, nbins: int | None = None
) -> Any:
    """Interactive histogram."""
    import plotly.express as px

    return px.histogram(df.to_pandas(), x=x, color=color, nbins=nbins, template=_TEMPLATE)


@interactive_chart
def box(df: pl.DataFrame, y: str, *, x: str | None = None, color: str | None = None) -> Any:
    """Interactive box plot (optionally split by the categorical ``x``)."""
    import plotly.express as px

    return px.box(df.to_pandas(), x=x, y=y, color=color, template=_TEMPLATE)


@interactive_chart
def time_series(
    df: pl.DataFrame, date_col: str, value: str | Sequence[str], *, color: str | None = None
) -> Any:
    """Interactive time series with a range slider for zooming the date axis."""
    import plotly.express as px

    fig = px.line(df.to_pandas(), x=date_col, y=value, color=color, template=_TEMPLATE)
    fig.update_xaxes(rangeslider_visible=True)
    return fig


@interactive_chart
def correlation_heatmap(df: pl.DataFrame) -> Any:
    """Interactive correlation heatmap of the numeric columns."""
    import plotly.express as px

    corr = df.to_pandas().corr(numeric_only=True)
    return px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        template=_TEMPLATE,
    )
