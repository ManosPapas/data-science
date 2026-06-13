"""Monte Carlo simulation — propagate input uncertainty through a business model.

Point estimates hide the tail: a plan built on means can still lose money 30% of the time.
``monte_carlo`` samples the inputs (scipy distributions, custom samplers, constants — optionally
rank-correlated via a Gaussian copula), pushes them through your value function, and returns the
*distribution* of outcomes: P10/P50/P90, probability of hitting targets, and which input
uncertainty actually drives the spread. Risk measures over the result live in ``analytics.risk``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SimulationResult:
    """Sampled outcomes plus the input draws that produced them."""

    samples: NDArray[np.float64]
    input_samples: dict[str, NDArray[np.float64]]

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def std(self) -> float:
        return float(self.samples.std(ddof=1))

    def percentile(self, q: float) -> float:
        """The q-th percentile outcome (q in 0-100)."""
        return float(np.percentile(self.samples, q))

    @property
    def p10(self) -> float:
        return self.percentile(10)

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p90(self) -> float:
        return self.percentile(90)

    def prob_above(self, target: float) -> float:
        """P(outcome ≥ target) — the probability of achieving the goal."""
        return float(np.mean(self.samples >= target))

    def prob_below(self, target: float) -> float:
        """P(outcome < target) — shortfall probability."""
        return float(np.mean(self.samples < target))

    def summary(
        self,
        *,
        targets: ArrayLike | None = None,
        quantiles: Sequence[float] = (10, 50, 90),
    ) -> pl.DataFrame:
        """Headline table: mean, std, chosen percentiles, and P(≥ target) for each target.

        ``quantiles`` are percentiles (0-100); a scalar ``targets`` is accepted. For the fuller
        VaR/CVaR view of the same samples use ``analytics.risk.risk_summary``.
        """
        rows: list[dict[str, Any]] = [
            {"metric": "mean", "value": self.mean},
            {"metric": "std", "value": self.std},
        ]
        for q in quantiles:
            rows.append({"metric": f"p{round(q)}", "value": self.percentile(float(q))})
        if targets is not None:
            for target in np.atleast_1d(np.asarray(targets, dtype=float)):
                value = self.prob_above(float(target))
                rows.append({"metric": f"prob ≥ {target:g}", "value": value})
        return pl.DataFrame(rows)

    def drivers(self) -> pl.DataFrame:
        """Rank inputs by |Spearman rho| with the outcome — which uncertainty moves the result.

        The simulation-native tornado: reduce uncertainty (research, hedge, contract) on the top
        driver first. Correlation here is association across draws, not causation — but inside
        the simulation the inputs *are* the only causes.
        """
        from scipy.stats import spearmanr

        rows = []
        for name, draws in self.input_samples.items():
            if np.unique(draws).size < 2:
                continue  # constants carry no uncertainty
            rho = float(spearmanr(draws, self.samples).statistic)
            rows.append({"input": name, "spearman": rho})
        frame = pl.DataFrame(rows, schema={"input": pl.String, "spearman": pl.Float64})
        return frame.sort(pl.col("spearman").abs(), descending=True)


def _sample_inputs(
    inputs: Mapping[str, Any],
    n: int,
    rng: np.random.Generator,
    correlation: Mapping[tuple[str, str], float] | None,
) -> dict[str, NDArray[np.float64]]:
    from scipy.stats import norm

    distributions = {k: v for k, v in inputs.items() if hasattr(v, "rvs")}
    samples: dict[str, NDArray[np.float64]] = {}

    if correlation:
        names = list(distributions)
        index = {name: i for i, name in enumerate(names)}
        matrix = np.eye(len(names))
        for (a, b), rho in correlation.items():
            if a not in index or b not in index:
                raise ValueError(
                    f"correlation pair ({a!r}, {b!r}) must reference distribution inputs"
                )
            if not -1.0 < rho < 1.0:
                raise ValueError("correlations must be strictly between -1 and 1")
            matrix[index[a], index[b]] = matrix[index[b], index[a]] = rho
        try:
            cholesky = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("correlation matrix is not positive definite") from error
        z = rng.standard_normal((n, len(names))) @ cholesky.T
        uniforms = norm.cdf(z)
        for name in names:
            dist = distributions[name]
            if not hasattr(dist, "ppf"):
                raise ValueError(f"correlated input {name!r} needs a distribution with .ppf")
            samples[name] = np.asarray(dist.ppf(uniforms[:, index[name]]), dtype=float)
    else:
        for name, dist in distributions.items():
            samples[name] = np.asarray(dist.rvs(size=n, random_state=rng), dtype=float)

    for name, spec in inputs.items():
        if name in samples:
            continue
        if callable(spec):
            drawn = np.asarray(spec(rng, n), dtype=float)
            if drawn.shape != (n,):
                raise ValueError(f"sampler {name!r} must return an array of shape ({n},)")
            samples[name] = drawn
        else:
            samples[name] = np.full(n, float(spec))
    return samples


def monte_carlo(
    value_fn: Callable[..., ArrayLike],
    inputs: Mapping[str, Any],
    *,
    n: int = 10_000,
    seed: int = 42,
    correlation: Mapping[tuple[str, str], float] | None = None,
    vectorized: bool = True,
) -> SimulationResult:
    """Simulate ``value_fn(**inputs)`` under input uncertainty; returns the outcome distribution.

    Each input is a frozen scipy distribution (``stats.norm(100, 10)``), a sampler
    ``f(rng, n) -> array``, or a constant. ``correlation`` couples distribution inputs by rank
    (Gaussian copula) — costs and volumes rarely move independently, and ignoring that understates
    tail risk. ``value_fn`` receives full sample arrays; set ``vectorized=False`` if it only
    handles scalars (slower).
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)
    draws = _sample_inputs(inputs, n, rng, correlation)
    if vectorized:
        outcomes = np.asarray(value_fn(**draws), dtype=float)
        if outcomes.shape != (n,):
            raise ValueError("value_fn did not return one outcome per draw — set vectorized=False")
    else:
        outcomes = np.array(
            [value_fn(**{name: arr[i] for name, arr in draws.items()}) for i in range(n)],
            dtype=float,
        )
    return SimulationResult(samples=outcomes, input_samples=draws)


def stress_test(
    value_fn: Callable[..., float],
    base: Mapping[str, Any],
    stresses: Mapping[str, Mapping[str, Any]],
    *,
    include_combined: bool = True,
) -> pl.DataFrame:
    """Re-value the model under named adverse shocks, plus all shocks hitting at once.

    Stress testing complements :func:`monte_carlo`: the simulation says how likely bad outcomes
    are, the stress test says whether you *survive* specific ones (the regulator's question).
    The combined row is the joint worst case — correlations go to 1 in a crisis.
    """
    base_value = float(value_fn(**base))
    rows = [{"scenario": "base", "value": base_value, "vs_base": 0.0}]
    for name, overrides in stresses.items():
        value = float(value_fn(**{**base, **overrides}))
        rows.append({"scenario": name, "value": value, "vs_base": value - base_value})
    if include_combined and stresses:
        merged: dict[str, Any] = dict(base)
        for overrides in stresses.values():
            merged.update(overrides)
        value = float(value_fn(**merged))
        rows.append({"scenario": "combined", "value": value, "vs_base": value - base_value})
    return pl.DataFrame(rows)


def simulate_paths(
    *,
    start: float,
    drift: float,
    volatility: float,
    periods: int,
    n: int = 1000,
    seed: int = 42,
    model: str = "multiplicative",
) -> NDArray[np.float64]:
    """Random-walk trajectories for a level (demand, revenue, users): (n, periods+1) incl. start.

    ``multiplicative``: each period grows by ``Normal(drift, volatility)`` compounded — the
    natural model for revenue/demand (% growth, no negative levels for plausible inputs).
    ``additive``: level changes by the draw in absolute units. Summarize with
    :func:`path_percentiles`; the fan you get is forecast-uncertainty growth made visible.
    """
    if periods < 1 or n < 1:
        raise ValueError("periods and n must be at least 1")
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, volatility, size=(n, periods))
    if model == "multiplicative":
        factors = np.cumprod(1.0 + shocks, axis=1)
        paths = start * np.hstack([np.ones((n, 1)), factors])
    elif model == "additive":
        paths = start + np.hstack([np.zeros((n, 1)), np.cumsum(shocks, axis=1)])
    else:
        raise ValueError("model must be 'multiplicative' or 'additive'")
    return np.asarray(paths, dtype=float)


def path_percentiles(
    paths: ArrayLike, *, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> pl.DataFrame:
    """Per-period quantile bands of simulated paths — the chart-ready fan (period, p10, p50, …)."""
    matrix = np.asarray(paths, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("paths must be 2-D (n_paths, n_periods)")
    bands = np.quantile(matrix, quantiles, axis=0)
    data: dict[str, Any] = {"period": np.arange(matrix.shape[1])}
    for q, band in zip(quantiles, bands, strict=True):
        data[f"p{round(q * 100)}"] = band
    return pl.DataFrame(data)
