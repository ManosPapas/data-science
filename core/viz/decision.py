"""Decision-science charts — tornado, waterfall, fans, control charts, frontiers, pricing.

The deck-ready visuals for the ``decision`` / ``pricing`` / ``analytics.drivers`` toolkits. Same
contract as every chart group: prepared data in (the frames those modules return), an ``Axes``
out; compute stays in the owning module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from core.viz.base import chart


@chart(title="Sensitivity tornado")
def tornado(ax: Axes, table: pl.DataFrame, *, base: float | None = None) -> None:
    """Tornado chart for ``scenario.sensitivity`` output — the biggest lever lands on top.

    Each bar spans the outcome from the input's low to its high value; pass ``base`` (the value
    at the base case) to draw the anchor line the bars pivot around.
    """
    ordered = table.sort(pl.col("swing").abs())  # ascending: barh stacks bottom-up
    names = ordered["input"].to_list()
    low = ordered["low_value"].to_numpy()
    high = ordered["high_value"].to_numpy()
    left = np.minimum(low, high)
    width = np.abs(high - low)
    ax.barh(names, width, left=left, color=sns.color_palette()[0], alpha=0.85)
    if base is not None:
        ax.axvline(base, color="#444444", linewidth=1.2, linestyle="--", label="base")
        ax.legend()
    ax.set(xlabel="outcome", ylabel="")


@chart(title="Bridge")
def waterfall(
    ax: Axes,
    df: pl.DataFrame,
    *,
    label: str,
    value: str,
    start: float = 0.0,
    total_label: str = "total",
) -> None:
    """Waterfall of named contributions — the finance bridge slide.

    Feed it ``drivers.change_decomposition`` (label/change) or a summed
    ``drivers.price_volume_mix`` (effect names/values); ``start`` anchors the running level
    (e.g. baseline revenue). Gains and losses are coloured apart and a closing total bar lands
    at ``start + sum(values)``.
    """
    palette = sns.color_palette()
    labels = df[label].to_list()
    values = df[value].to_numpy()
    running = start
    for index, (name, amount) in enumerate(zip(labels, values, strict=True)):
        color = palette[2] if amount >= 0 else palette[3]
        ax.bar(str(name), amount, bottom=running, color=color, alpha=0.9)
        next_level = running + float(amount)
        ax.plot(
            [index - 0.4, index + 1.4 if index < len(labels) - 1 else index + 0.4],
            [next_level, next_level],
            color="#888888",
            linewidth=0.8,
        )
        running = next_level
    ax.bar(total_label, running - start, bottom=start, color=palette[0], alpha=0.9)
    ax.axhline(start, color="#444444", linewidth=1.0)
    ax.set(ylabel="value")
    ax.tick_params(axis="x", rotation=30)


@chart(title="Uncertainty fan")
def fan(
    ax: Axes,
    df: pl.DataFrame,
    *,
    x: str,
    bands: Sequence[tuple[str, str]],
    line: str | None = None,
) -> None:
    """Shaded quantile band(s) plus an optional center line — the uncertainty fan.

    One chart for every band-shaped output: ``simulate.path_percentiles`` (``bands=[("p10",
    "p90")]``, ``line="p50"``), rolling-elasticity CIs, and conformal intervals. Order bands
    outermost first when nesting several.
    """
    color = sns.color_palette()[0]
    xs = df[x].to_numpy()
    for index, (low, high) in enumerate(bands):
        ax.fill_between(
            xs,
            df[low].to_numpy(),
            df[high].to_numpy(),
            alpha=0.15 + 0.1 * index,
            color=color,
            label=f"{low}-{high}",
        )
    if line is not None:
        ax.plot(xs, df[line].to_numpy(), color=color, label=line)
    ax.legend()
    ax.set(xlabel=x)


@chart(title="Control chart")
def control_chart(ax: Axes, df: pl.DataFrame, *, x: str = "t") -> None:
    """EWMA control chart from ``monitor.ewma_alerts`` output — alerts highlighted.

    Raw observations sit in the background; the EWMA line against its widening limits is the
    signal, and red markers are the points where the metric has provably walked away.
    """
    palette = sns.color_palette()
    xs = df[x].to_numpy()
    ax.scatter(xs, df["value"].to_numpy(), s=14, color="#bbbbbb", label="observed")
    ax.plot(xs, df["ewma"].to_numpy(), color=palette[0], label="EWMA")
    ax.plot(xs, df["lower"].to_numpy(), color="#444444", linewidth=1.0, linestyle="--")
    ax.plot(
        xs, df["upper"].to_numpy(), color="#444444", linewidth=1.0, linestyle="--", label="limits"
    )
    alerts = df.filter(pl.col("alert"))
    if alerts.height:
        ax.scatter(
            alerts[x].to_numpy(),
            alerts["ewma"].to_numpy(),
            color=palette[3],
            s=40,
            zorder=3,
            label="alert",
        )
    ax.legend()
    ax.set(xlabel=x, ylabel="metric")


@chart(title="Efficient frontier")
def pareto_frontier(
    ax: Axes,
    df: pl.DataFrame,
    *,
    x: str,
    y: str,
    efficient: str,
    label: str | None = None,
) -> None:
    """Scatter of options with the Pareto-efficient set highlighted and traced.

    ``efficient`` is the boolean column from ``optimize.pareto_front``; ``label`` optionally
    annotates the efficient points (config names) so the trade-off menu reads off the chart.
    """
    palette = sns.color_palette()
    dominated = df.filter(~pl.col(efficient))
    front = df.filter(pl.col(efficient)).sort(x)
    ax.scatter(dominated[x].to_numpy(), dominated[y].to_numpy(), color="#bbbbbb", label="dominated")
    ax.plot(
        front[x].to_numpy(),
        front[y].to_numpy(),
        marker="o",
        color=palette[0],
        label="efficient frontier",
    )
    if label is not None:
        for row in front.iter_rows(named=True):
            ax.annotate(
                str(row[label]),
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
    ax.legend()
    ax.set(xlabel=x, ylabel=y)


@chart(title="Outcome distribution")
def outcome_distribution(
    ax: Axes,
    samples: ArrayLike,
    *,
    percentiles: Sequence[float] = (10, 50, 90),
    targets: Sequence[float] | None = None,
    bins: int = 50,
) -> None:
    """Monte Carlo outcomes with P10/P50/P90 markers and optional target lines — the deck slide.

    The annotated version of a plain histogram: the percentile lines are the plan/funding-case
    numbers, the dashed target lines show where commitments sit in the distribution.
    """
    values = np.asarray(samples, dtype=float)
    palette = sns.color_palette()
    sns.histplot(x=values, bins=bins, stat="density", color=palette[0], alpha=0.45, ax=ax)
    for q in percentiles:
        level = float(np.percentile(values, q))
        ax.axvline(level, color=palette[0], linewidth=1.4)
        ax.annotate(
            f"P{int(q)}\n{level:,.0f}",
            (level, ax.get_ylim()[1] * 0.92),
            ha="center",
            fontsize=9,
            color="#1a1a1a",
        )
    for target in targets or []:
        ax.axvline(float(target), color=palette[3], linewidth=1.4, linestyle="--")
        ax.annotate(
            f"target {target:,.0f}",
            (float(target), ax.get_ylim()[1] * 0.05),
            ha="center",
            fontsize=9,
            color=palette[3],
        )
    ax.set(xlabel="outcome", ylabel="density")


@chart(title="Price response")
def price_curves(
    ax: Axes,
    df: pl.DataFrame,
    *,
    price: str = "price",
    curves: Sequence[str] = ("revenue", "profit"),
    optimum: float | None = None,
) -> None:
    """Revenue/profit (or any response) against price, with the chosen optimum marked.

    Feed it ``pricing.demand.demand_schedule`` output or a frame built from
    ``optimize.revenue_at`` / ``profit_at``; ``optimum`` draws the recommended price.
    """
    xs = df[price].to_numpy()
    for column in curves:
        ax.plot(xs, df[column].to_numpy(), label=column)
    if optimum is not None:
        ax.axvline(optimum, color="#444444", linewidth=1.2, linestyle="--")
        ax.annotate(
            f"optimum {optimum:,.0f}",
            (optimum, ax.get_ylim()[1] * 0.95),
            ha="left",
            fontsize=9,
            xytext=(6, 0),
            textcoords="offset points",
        )
    ax.legend()
    ax.set(xlabel=price, ylabel="value")


@chart(title="Van Westendorp price sensitivity")
def van_westendorp(
    ax: Axes, curves: pl.DataFrame, *, points: Mapping[str, float] | None = None
) -> None:
    """The four Van Westendorp survey curves with the classic crossing points marked.

    Takes the ``curves`` frame from ``pricing.demand.van_westendorp`` and (optionally) a mapping
    of point labels to prices, e.g. ``{"optimal": vw.optimal_price, "lower": ..., "upper": ...}``.
    """
    xs = curves["price"].to_numpy()
    for column in ("too_cheap", "cheap", "expensive", "too_expensive"):
        ax.plot(xs, curves[column].to_numpy(), label=column.replace("_", " "))
    for name, level in (points or {}).items():
        ax.axvline(float(level), color="#444444", linewidth=1.0, linestyle="--")
        ax.annotate(
            f"{name} {level:,.0f}",
            (float(level), 1.02),
            ha="center",
            fontsize=9,
            annotation_clip=False,
        )
    ax.set(xlabel="price", ylabel="share of respondents", ylim=(0, 1.05))
    ax.legend()


@chart(title="Dynamic pricing policy")
def price_policy(
    ax: Axes,
    policy: pl.DataFrame,
    *,
    period: str = "period",
    remaining: str = "remaining",
    price: str = "price",
) -> None:
    """Heatmap of the solved dynamic-pricing policy: price per (period, remaining stock).

    Takes ``DynamicPricingPolicy.policy_frame()``. Both revenue-management forces should be
    visible at a glance: colours cool toward the deadline (markdown) and warm toward the
    stock-out corner (scarcity premium).
    """
    wide = policy.pivot(on=period, index=remaining, values=price).sort(remaining)
    levels = wide[remaining].to_numpy()
    matrix = wide.drop(remaining).to_numpy()
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.figure.colorbar(image, ax=ax, label=price)
    ax.set(xlabel=period, ylabel=remaining)
    step = max(1, levels.size // 8)
    ax.set_yticks(np.arange(0, levels.size, step), [f"{v:g}" for v in levels[::step]])
