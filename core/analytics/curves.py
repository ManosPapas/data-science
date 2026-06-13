"""Numerical curve analysis — derivatives, turning points, curvature, response curves.

The calculus layer under pricing and optimization: any sampled curve (price→revenue,
spend→conversions) or callable model can be differentiated, scanned for optima and inflections,
and classified convex/concave. Numerical derivatives amplify noise — smooth a noisy series first
(:func:`smooth_series`) and treat extrema found on raw data with suspicion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


def _as_curve(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length")
    if xs.size < 3:
        raise ValueError("need at least 3 points to differentiate")
    if np.any(np.diff(xs) <= 0):
        raise ValueError("x must be strictly increasing — sort the curve first")
    return xs, ys


def smooth_series(y: ArrayLike, *, window: int = 5) -> NDArray[np.float64]:
    """Centered moving average (edges use the partial window) — pre-step for noisy derivatives."""
    values = np.asarray(y, dtype=float)
    if window < 1:
        raise ValueError("window must be at least 1")
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.asarray(smoothed[: values.size], dtype=float)


def slope(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """First derivative dy/dx along a sampled curve (central differences, uneven spacing ok).

    The rate-of-change read: marginal revenue from a revenue curve, marginal response from a
    spend curve. Where it crosses zero, the curve turns (:func:`local_extrema`).
    """
    xs, ys = _as_curve(x, y)
    return np.asarray(np.gradient(ys, xs), dtype=float)


def curvature(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Second derivative d²y/dx² — acceleration. Negative = concave (diminishing returns)."""
    xs, ys = _as_curve(x, y)
    return np.asarray(np.gradient(np.gradient(ys, xs), xs), dtype=float)


def point_elasticity(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Local elasticity d ln(y) / d ln(x) along the curve — the %-for-% sensitivity at each x."""
    xs, ys = _as_curve(x, y)
    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("elasticity needs strictly positive x and y")
    return np.asarray(np.gradient(np.log(ys), np.log(xs)), dtype=float)


def local_extrema(x: ArrayLike, y: ArrayLike) -> pl.DataFrame:
    """Interior maxima/minima of a sampled curve, refined by parabolic interpolation.

    Returns (x, y, kind) per turning point. Boundary points are *not* extrema here — if the best
    value sits on the edge of your grid, the real optimum may lie outside it: widen the grid.
    Noise creates fake extrema; smooth first.
    """
    xs, ys = _as_curve(x, y)
    rows = []
    for i in range(1, xs.size - 1):
        if (ys[i] > ys[i - 1] and ys[i] > ys[i + 1]) or (ys[i] < ys[i - 1] and ys[i] < ys[i + 1]):
            kind = "maximum" if ys[i] > ys[i - 1] else "minimum"
            # Parabola through the three points; vertex refines the grid-snapped location.
            denominator = ys[i - 1] - 2.0 * ys[i] + ys[i + 1]
            if denominator != 0:
                offset = 0.5 * (ys[i - 1] - ys[i + 1]) / denominator
                offset = float(np.clip(offset, -1.0, 1.0))
            else:
                offset = 0.0
            step = (xs[i + 1] - xs[i - 1]) / 2.0
            x_ref = float(xs[i] + offset * step)
            y_ref = float(ys[i] - 0.25 * (ys[i - 1] - ys[i + 1]) * offset)
            rows.append({"x": x_ref, "y": y_ref, "kind": kind})
    return pl.DataFrame(rows, schema={"x": pl.Float64, "y": pl.Float64, "kind": pl.String})


def inflection_points(x: ArrayLike, y: ArrayLike) -> pl.DataFrame:
    """Where curvature changes sign — acceleration flips to deceleration (or back).

    On an adoption curve the inflection is peak growth (saturation begins); on a response curve
    it separates the accelerating spend region from diminishing returns.
    """
    xs, ys = _as_curve(x, y)
    second = curvature(xs, ys)
    rows = []
    for i in range(second.size - 1):
        if second[i] == 0 or second[i] * second[i + 1] >= 0:
            continue
        frac = second[i] / (second[i] - second[i + 1])
        rows.append(
            {
                "x": float(xs[i] + frac * (xs[i + 1] - xs[i])),
                "direction": "convex→concave" if second[i] > 0 else "concave→convex",
            }
        )
    return pl.DataFrame(rows, schema={"x": pl.Float64, "direction": pl.String})


@dataclass(frozen=True)
class ConvexityReport:
    """Share of the curve that is convex vs concave, and the overall verdict."""

    convex_share: float
    concave_share: float
    verdict: str  # "convex" | "concave" | "mixed"


def convexity(x: ArrayLike, y: ArrayLike, *, tolerance: float = 0.9) -> ConvexityReport:
    """Classify a curve convex / concave / mixed from the sign of its second derivative.

    Concave objectives (diminishing returns) make hill-climbing and interior optima trustworthy;
    mixed curvature means multiple local optima are possible — grid-search before trusting a
    solver. ``tolerance`` is the share of points one sign needs for a clean verdict.
    """
    second = curvature(x, y)
    meaningful = second[np.abs(second) > 1e-12]
    if meaningful.size == 0:
        return ConvexityReport(0.0, 0.0, "linear")
    convex_share = float((meaningful > 0).mean())
    concave_share = float((meaningful < 0).mean())
    if convex_share >= tolerance:
        verdict = "convex"
    elif concave_share >= tolerance:
        verdict = "concave"
    else:
        verdict = "mixed"
    return ConvexityReport(convex_share, concave_share, verdict)


def marginal_effect(
    fn: Callable[..., float], base: Mapping[str, Any], name: str, *, step: float | None = None
) -> float:
    """Numerical ∂fn/∂name at the base point (central difference) — the local what-if rate.

    "One more euro of ``name`` is worth this many units of output, holding the rest fixed."
    Local by construction: re-evaluate at materially different base points before extrapolating.
    """
    value = float(base[name])
    h = step if step is not None else max(abs(value), 1.0) * 1e-5
    high = float(fn(**{**base, name: value + h}))
    low = float(fn(**{**base, name: value - h}))
    return (high - low) / (2.0 * h)


def gradient(
    fn: Callable[..., float], base: Mapping[str, Any], *, names: list[str] | None = None
) -> dict[str, float]:
    """Numerical gradient of a scalar business function at a point — all marginal effects at once.

    The steepest-ascent direction: which lever, moved a little, buys the most output. Pair with
    constraints via ``decision.optimize`` before acting on it.
    """
    if names is not None:
        keys = names
    else:
        # bool is a subclass of int — exclude flags, which have no meaningful derivative.
        keys = [
            k for k, v in base.items() if isinstance(v, int | float) and not isinstance(v, bool)
        ]
    return {name: marginal_effect(fn, base, name) for name in keys}


def response_curve(
    fn: Callable[..., float], base: Mapping[str, Any], name: str, values: ArrayLike
) -> pl.DataFrame:
    """Sweep one input through ``values`` (others at base): output and its local slope.

    The full curve behind ``decision.scenario.sensitivity``'s two-point swing — shows *where*
    returns diminish, not just that they do. Feed the result to :func:`local_extrema` /
    :func:`inflection_points` for turning-point detection.
    """
    grid = np.asarray(values, dtype=float)
    outputs = np.array([float(fn(**{**base, name: float(v)})) for v in grid])
    return pl.DataFrame({name: grid, "output": outputs, "slope": np.gradient(outputs, grid)})
