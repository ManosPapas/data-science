"""Charting foundation: the ``@chart`` decorator, the theme, and a ``grid`` helper.

Every chart in ``core/viz`` is a function whose first parameter is a matplotlib ``Axes`` and which
only *draws*. ``@chart`` supplies the ``Axes`` (creating a themed figure when one isn't passed),
applies the title, optionally saves, and returns the ``Axes`` so charts compose into grids.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import cast

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_FIGSIZE = (10.0, 6.0)
PALETTE = "colorblind"  # accessible, high-contrast qualitative palette

DrawFn = Callable[..., None]


def set_theme(style: str = "white", context: str = "notebook", palette: str = PALETTE) -> None:
    """Apply a clean, high-contrast theme: no gridlines, despined, bold padded titles.

    Call once per notebook (the prelude also applies it on import). Charts are gridless by
    default — pass ``grid=True`` to a chart to overlay a light grid.
    """
    sns.set_theme(style=style, context=context, palette=palette)
    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 12.0,
            "axes.titlecolor": "#1a1a1a",
            "axes.labelsize": 11,
            "axes.labelcolor": "#1a1a1a",
            "axes.labelweight": "medium",
            "text.color": "#1a1a1a",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
        }
    )


def chart(
    title: str | None = None, figsize: tuple[float, float] = DEFAULT_FIGSIZE
) -> Callable[[DrawFn], Callable[..., Axes]]:
    """Wrap a drawing function ``f(ax, ...)``, adding ``ax`` / ``title`` / ``grid`` / ``save``.

    Makes a themed figure when ``ax`` is omitted, applies the title, keeps the chart gridless
    unless ``grid=True``, tightens the layout, optionally saves, and returns the ``Axes``.
    """

    def decorator(draw: DrawFn) -> Callable[..., Axes]:
        @functools.wraps(draw)
        def wrapper(
            *args: object,
            ax: Axes | None = None,
            title: str | None = title,
            grid: bool = False,
            save: str | None = None,
            **kwargs: object,
        ) -> Axes:
            created = ax is None
            if ax is None:
                ax = plt.subplots(figsize=figsize)[1]
            draw(ax, *args, **kwargs)
            if title:
                ax.set_title(title)
            if grid:
                ax.grid(visible=True, alpha=0.3, linewidth=0.6)
            else:
                ax.grid(visible=False)
            if created:
                cast(Figure, ax.figure).tight_layout()
            if save:
                cast(Figure, ax.figure).savefig(save, bbox_inches="tight", dpi=150)
            return ax

        return wrapper

    return decorator


def grid(
    n: int, ncols: int = 2, figsize: tuple[float, float] | None = None
) -> tuple[Figure, list[Axes]]:
    """Create a figure with exactly ``n`` axes (in ``ncols`` columns) for composing charts.

    Returns the figure and a flat list of ``n`` axes (extra cells are removed)::

        fig, axes = grid(3)
        roc(y, p, ax=axes[0]); pr(y, p, ax=axes[1]); calibration(y, p, ax=axes[2])
    """
    nrows = (n + ncols - 1) // ncols
    size = figsize or (6.0 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size, squeeze=False)
    flat = [cast(Axes, item) for item in axes.flatten()]
    for extra in flat[n:]:
        fig.delaxes(extra)
    return fig, flat[:n]
