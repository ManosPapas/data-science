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

DrawFn = Callable[..., None]


def set_theme(style: str = "whitegrid", context: str = "notebook") -> None:
    """Apply a clean, readable default theme. Call once per notebook/session."""
    sns.set_theme(style=style, context=context)
    plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE
    plt.rcParams["axes.titleweight"] = "bold"


def chart(
    title: str | None = None, figsize: tuple[float, float] = DEFAULT_FIGSIZE
) -> Callable[[DrawFn], Callable[..., Axes]]:
    """Turn a drawing function ``f(ax, ...)`` into ``f(..., *, ax=None, title=None, save=None)``.

    The wrapped chart makes a themed figure when ``ax`` is omitted, sets the title, optionally
    saves to ``save``, and always returns the ``Axes``.
    """

    def decorator(draw: DrawFn) -> Callable[..., Axes]:
        @functools.wraps(draw)
        def wrapper(
            *args: object,
            ax: Axes | None = None,
            title: str | None = title,
            save: str | None = None,
            **kwargs: object,
        ) -> Axes:
            if ax is None:
                ax = cast(Axes, plt.subplots(figsize=figsize)[1])
            draw(ax, *args, **kwargs)
            if title:
                ax.set_title(title)
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
