"""Charting foundation: the ``@chart`` decorator, the theme, and a ``grid`` helper.

Every chart in ``core/viz`` is a function whose first parameter is a matplotlib ``Axes`` and which
only *draws*. ``@chart`` supplies the ``Axes`` (creating a themed figure when one isn't passed),
applies the title, optionally saves, and returns the ``Axes`` so charts compose into grids.
"""

from __future__ import annotations

import functools
import inspect
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
    """Apply a clean, high-contrast theme: despined, bold padded titles, no gridlines by default.

    Call once per notebook (the prelude also applies it on import). The default ``style="white"``
    is gridless — pass ``grid=True`` to a chart for a light per-chart grid, or use
    ``style="whitegrid"`` to turn gridlines on globally.
    """
    sns.set_theme(style=style, context=context, palette=palette)
    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
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
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
        }
    )


def _public_signature(draw: DrawFn, default_title: str | None) -> inspect.Signature:
    """The signature Jupyter shows: the draw params minus ``ax``, plus the injected keywords."""
    own = list(inspect.signature(draw).parameters.values())[1:]  # drop the wrapper-supplied ax
    keyword = inspect.Parameter.KEYWORD_ONLY
    injected = [
        inspect.Parameter("ax", keyword, default=None),
        inspect.Parameter("title", keyword, default=default_title),
        inspect.Parameter("grid", keyword, default=False),
        inspect.Parameter("save", keyword, default=None),
    ]
    return inspect.Signature(own + injected)


def chart(
    title: str | None = None, figsize: tuple[float, float] = DEFAULT_FIGSIZE
) -> Callable[[DrawFn], Callable[..., Axes]]:
    """Wrap a drawing function ``f(ax, ...)``, adding ``ax`` / ``title`` / ``grid`` / ``save``.

    Makes a themed figure when ``ax`` is omitted, applies the title (``title=""`` clears a
    chart-set one), overlays a light grid when ``grid=True`` (never touches the grid otherwise,
    so the theme/caller's choice stands), tightens the layout, optionally saves, and returns
    the ``Axes``.
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
            if title is not None:
                ax.set_title(title)
            if grid:
                ax.grid(visible=True, alpha=0.3, linewidth=0.6)
            if created:
                cast(Figure, ax.figure).tight_layout()
            if save is not None:
                cast(Figure, ax.figure).savefig(save, bbox_inches="tight", dpi=150)
            return ax

        # functools.wraps copies the draw signature, which hides the injected keywords (and shows
        # ``ax`` as required) in Jupyter help — publish the real calling convention instead.
        setattr(wrapper, "__signature__", _public_signature(draw, title))  # noqa: B010
        return wrapper

    return decorator


def grid(
    n: int, ncols: int = 2, figsize: tuple[float, float] | None = None
) -> tuple[Figure, list[Axes]]:
    """Create a figure with exactly ``n`` axes (in ``ncols`` columns) for composing charts.

    Uses constrained layout so composed titles and tick labels never overlap. Returns the figure
    and a flat list of ``n`` axes (extra cells are removed)::

        fig, axes = grid(3)
        roc(y, p, ax=axes[0]); pr(y, p, ax=axes[1]); calibration(y, p, ax=axes[2])
    """
    nrows = (n + ncols - 1) // ncols
    size = figsize or (6.0 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=size, squeeze=False, layout="constrained"
    )
    flat = [cast(Axes, item) for item in axes.flatten()]
    for extra in flat[n:]:
        fig.delaxes(extra)
    return fig, flat[:n]
