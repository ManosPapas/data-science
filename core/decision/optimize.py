"""Optimization / OR — linear, integer, and nonlinear programs; portfolios; trade-off fronts.

Turn estimates into actions under constraints. ``linear_program``/``integer_program`` cover
allocation and selection; ``nonlinear_program`` handles curved objectives (diminishing returns);
``scenario_optimize`` optimizes under uncertainty; ``pareto_front`` exposes the trade-offs no
single objective can settle. Results follow scipy conventions: for ``maximize=True`` the
objective is negated internally, so read the optimum off ``-result.fun``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


def linear_program(
    cost: ArrayLike,
    *,
    a_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    a_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Any = None,
    maximize: bool = False,
) -> Any:
    """Solve a linear program (scipy.optimize.linprog). ``maximize=True`` flips the objective."""
    from scipy.optimize import linprog

    c = np.asarray(cost, dtype=float)
    return linprog(
        c=-c if maximize else c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds
    )


def shadow_prices(
    result: Any,
    *,
    names_ub: Sequence[str] | None = None,
    names_eq: Sequence[str] | None = None,
    maximize: bool = False,
) -> pl.DataFrame:
    """Marginal value of each constraint from a solved :func:`linear_program` — opportunity cost.

    A shadow price of 1.4 on a capacity row means one more unit of that resource is worth 1.4 in
    the objective — the rational ceiling on what to pay for extra capacity, and 0 means the
    constraint isn't binding (slack > 0). Pass the same ``maximize`` you solved with so signs
    read in objective units.
    """
    # shadow price = d(objective)/d(b); linprog marginals are for the minimized problem, so a
    # maximize solve flips the sign
    sign = -1.0 if maximize else 1.0
    rows = []
    ub = getattr(result, "ineqlin", None)
    if ub is not None and len(ub.marginals):
        labels = names_ub or [f"ub_{i}" for i in range(len(ub.marginals))]
        for label, marginal, slack in zip(labels, ub.marginals, result.slack, strict=True):
            rows.append(
                {
                    "constraint": label,
                    "kind": "ub",
                    "shadow_price": sign * float(marginal),
                    "slack": float(slack),
                }
            )
    eq = getattr(result, "eqlin", None)
    if eq is not None and len(eq.marginals):
        labels = names_eq or [f"eq_{i}" for i in range(len(eq.marginals))]
        for label, marginal in zip(labels, eq.marginals, strict=True):
            rows.append(
                {
                    "constraint": label,
                    "kind": "eq",
                    "shadow_price": sign * float(marginal),
                    "slack": 0.0,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "constraint": pl.String,
            "kind": pl.String,
            "shadow_price": pl.Float64,
            "slack": pl.Float64,
        },
    )


def integer_program(
    cost: ArrayLike,
    *,
    a_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    a_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    integrality: ArrayLike | int = 1,
    maximize: bool = False,
) -> Any:
    """Mixed-integer linear program (scipy.optimize.milp) — yes/no and whole-unit decisions.

    ``integrality`` per variable: 0 = continuous, 1 = integer (scalar broadcasts). Bounds default
    to ``(0, inf)`` like ``linear_program``. Use it where fractional answers are meaningless:
    open/close a site, buy whole machines, pick projects. LP + rounding is *not* a substitute —
    rounding can break constraints and land far from the true optimum.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    c = np.asarray(cost, dtype=float)
    constraints = []
    if a_ub is not None:
        constraints.append(LinearConstraint(np.asarray(a_ub, dtype=float), ub=b_ub))
    if a_eq is not None:
        b = np.asarray(b_eq, dtype=float)
        constraints.append(LinearConstraint(np.asarray(a_eq, dtype=float), lb=b, ub=b))
    if bounds is None:
        variable_bounds = Bounds(lb=np.zeros(c.size), ub=np.full(c.size, np.inf))
    else:
        lb = np.array([lo if lo is not None else -np.inf for lo, _ in bounds], dtype=float)
        ub = np.array([hi if hi is not None else np.inf for _, hi in bounds], dtype=float)
        variable_bounds = Bounds(lb=lb, ub=ub)
    return milp(
        c=-c if maximize else c,
        constraints=constraints,
        bounds=variable_bounds,
        integrality=np.broadcast_to(np.asarray(integrality), c.shape),
    )


@dataclass(frozen=True)
class KnapsackResult:
    """Chosen item indices with the value packed and capacity used."""

    chosen: list[int]
    total_value: float
    total_weight: float


def knapsack(values: ArrayLike, weights: ArrayLike, capacity: float) -> KnapsackResult:
    """Pick the value-maximizing subset under a budget/capacity — exact, via integer programming.

    The canonical project-portfolio / campaign-selection shape. Greedy value-per-weight picking
    is the LP intuition but can be arbitrarily bad at integer scale; this solves it exactly.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    result = integer_program(
        v,
        a_ub=w.reshape(1, -1),
        b_ub=[capacity],
        bounds=[(0.0, 1.0)] * v.size,
        integrality=1,
        maximize=True,
    )
    if not result.success:
        raise ValueError(f"knapsack solve failed: {result.message}")
    chosen = [int(i) for i in np.flatnonzero(np.round(result.x) == 1)]
    return KnapsackResult(
        chosen=chosen,
        total_value=float(v[chosen].sum()),
        total_weight=float(w[chosen].sum()),
    )


def nonlinear_program(
    objective: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    *,
    bounds: Any = None,
    constraints: Any = (),
    maximize: bool = False,
    method: str | None = None,
) -> Any:
    """Smooth nonlinear optimization (scipy.optimize.minimize) — curved objectives, real life.

    Diminishing returns, saturation effects, and interaction terms all bend the objective; this
    handles them with bounds and (in)equality constraints. Local solver: from a bad start it
    finds a local optimum — check curvature first (``analytics.curves.convexity``) or multi-start.
    """
    from scipy.optimize import minimize

    def fun(x: NDArray[np.float64]) -> float:
        value = float(objective(np.asarray(x, dtype=float)))
        return -value if maximize else value

    return minimize(
        fun, np.asarray(x0, dtype=float), bounds=bounds, constraints=constraints, method=method
    )


def assign(
    cost_matrix: ArrayLike, *, maximize: bool = False
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Optimal one-to-one assignment (Hungarian algorithm). Returns (row_indices, col_indices)."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(np.asarray(cost_matrix, dtype=float), maximize=maximize)
    return np.asarray(rows), np.asarray(cols)


@dataclass(frozen=True)
class PortfolioResult:
    """Mean-variance optimal weights with the portfolio's expected return and volatility."""

    weights: NDArray[np.float64]
    expected_return: float
    volatility: float


def portfolio_weights(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    *,
    risk_aversion: float = 2.0,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """Mean-variance allocation: maximize ``μ'w - (a/2)·w'Σw`` with full investment, no shorting.

    Works for any "spread a budget over risky options" problem — channels, products, projects —
    not just securities: the covariance term is what diversifies (correlated bets don't).
    ``risk_aversion`` trades return against variance; ``max_weight`` caps concentration. Garbage
    in, garbage out: μ and Σ are *estimates* — shrink or stress them before betting the budget.
    """
    from scipy.optimize import minimize

    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n = mu.size
    if sigma.shape != (n, n):
        raise ValueError("covariance must be square and match expected_returns")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if max_weight * n < 1.0:
        raise ValueError("max_weight too small to invest the whole budget")

    def negative_utility(w: NDArray[np.float64]) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ sigma @ w))

    result = minimize(
        negative_utility,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
    )
    if not result.success:
        raise ValueError(f"portfolio optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    return PortfolioResult(
        weights=weights,
        expected_return=float(mu @ weights),
        volatility=float(np.sqrt(weights @ sigma @ weights)),
    )


def pareto_front(points: ArrayLike, *, maximize: bool | Sequence[bool] = True) -> NDArray[np.bool_]:
    """Mask of non-dominated options across several objectives — the efficient frontier.

    An option is dominated when another is at least as good on every objective and better on one;
    nothing rational picks it. What remains is the genuine trade-off menu (e.g. margin vs volume
    vs risk) — multi-objective optimization without arbitrary objective weights. Pair with
    ``viz`` to put the frontier in front of the decision-maker.
    """
    matrix = np.asarray(points, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("points must be 2-D (options x objectives)")
    directions = (
        np.full(matrix.shape[1], maximize)
        if isinstance(maximize, bool)
        else np.asarray(maximize, dtype=bool)
    )
    if directions.size != matrix.shape[1]:
        raise ValueError("maximize must be a bool or one flag per objective")
    oriented = np.where(directions, matrix, -matrix)
    better_equal = (oriented[None, :, :] >= oriented[:, None, :]).all(axis=2)
    strictly_better = (oriented[None, :, :] > oriented[:, None, :]).any(axis=2)
    dominated = np.any(better_equal & strictly_better, axis=1)
    return np.asarray(~dominated)


def scenario_optimize(
    value_fn: Callable[..., float],
    x0: ArrayLike,
    scenarios: Sequence[Mapping[str, Any]],
    *,
    bounds: Any = None,
    maximize: bool = True,
    criterion: str = "mean",
) -> Any:
    """Optimize a decision against many sampled futures — stochastic / robust optimization.

    ``value_fn(x, **scenario)`` values decision ``x`` in one future; scenarios come from
    ``decision.simulate`` draws or hand-built cases. ``criterion='mean'`` maximizes the expected
    value; ``'worst'`` maximizes the worst case (max-min — the robust answer when you can't
    afford the bad future). One decision is optimized for *all* futures: this is deciding under
    uncertainty, not per-scenario hindsight.
    """
    if criterion not in ("mean", "worst"):
        raise ValueError("criterion must be 'mean' or 'worst'")
    if not scenarios:
        raise ValueError("need at least one scenario")

    def objective(x: NDArray[np.float64]) -> float:
        values = np.array([float(value_fn(x, **scenario)) for scenario in scenarios])
        return float(values.mean()) if criterion == "mean" else float(values.min())

    return nonlinear_program(objective, x0, bounds=bounds, maximize=maximize)
