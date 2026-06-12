"""Price elasticity of demand — estimation, uncertainty, segments, dynamics, decomposition.

Fit ``ln(quantity) = intercept + elasticity * ln(price)``; the slope is the price elasticity
(typically negative — demand falls as price rises). Feed the fitted ``(intercept, elasticity)`` to
``core.pricing.optimize`` to find a revenue- or profit-maximizing price.

Observational price-demand data is confounded (prices were set in response to demand) — prefer
randomized or instrumented price variation (``analytics.causal.iv_effect``) before trusting any
elasticity here for a pricing decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


def _validate_loglog(
    price: ArrayLike, quantity: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(price, dtype=float)
    q = np.asarray(quantity, dtype=float)
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("log-log demand requires strictly positive price and quantity")
    if np.unique(p).size < 2:
        raise ValueError("need at least two distinct prices to identify elasticity")
    return p, q


def _ols(
    design: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """OLS for a full-rank design; returns (coef, std_err, residuals, dof)."""
    n, k = design.shape
    if n <= k:
        raise ValueError(f"need more than {k} observations, got {n}")
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < k:
        raise ValueError("design matrix is rank-deficient (collinear or constant regressors)")
    residuals = y - design @ coef
    dof = n - k
    sigma2 = float(residuals @ residuals) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return (
        np.asarray(coef, dtype=float),
        np.asarray(np.sqrt(np.diag(cov)), dtype=float),
        np.asarray(residuals, dtype=float),
        int(dof),
    )


def fit_demand(price: ArrayLike, quantity: ArrayLike) -> tuple[float, float]:
    """Fit constant-elasticity demand by OLS on logs; return ``(intercept, elasticity)``.

    ``price`` and ``quantity`` must be strictly positive (the fit is on their logs) — filter out
    zero-price/zero-sales rows first.
    """
    p, q = _validate_loglog(price, quantity)
    elasticity, intercept = np.polyfit(np.log(p), np.log(q), 1)
    return float(intercept), float(elasticity)


def price_elasticity(price: ArrayLike, quantity: ArrayLike) -> float:
    """Point price elasticity of demand (log-log slope); < -1 is elastic, -1..0 inelastic."""
    return fit_demand(price, quantity)[1]


def predict_demand(intercept: float, elasticity: float, price: ArrayLike) -> NDArray[np.float64]:
    """Predicted quantity at ``price`` under the fitted constant-elasticity model."""
    p = np.asarray(price, dtype=float)
    return np.asarray(np.exp(intercept) * p**elasticity, dtype=float)


@dataclass(frozen=True)
class ElasticityFit:
    """A log-log demand fit with inference — the elasticity plus its sampling uncertainty."""

    intercept: float
    elasticity: float
    std_err: float
    ci_low: float
    ci_high: float
    r_squared: float
    n: int


def fit_demand_ci(
    price: ArrayLike, quantity: ArrayLike, *, confidence: float = 0.95
) -> ElasticityFit:
    """Log-log demand fit with a standard error and t-based CI on the elasticity.

    A CI spanning -1 means the data cannot tell elastic from inelastic — the revenue-direction
    call (raise vs cut price) is not identified; collect more price variation first.
    """
    from scipy import stats as sps

    p, q = _validate_loglog(price, quantity)
    x, y = np.log(p), np.log(q)
    design = np.column_stack([np.ones(x.size), x])
    coef, std_err, residuals, dof = _ols(design, y)
    t_crit = float(sps.t.ppf(1.0 - (1.0 - confidence) / 2.0, dof))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(residuals @ residuals) / tss if tss > 0 else float("nan")
    return ElasticityFit(
        intercept=float(coef[0]),
        elasticity=float(coef[1]),
        std_err=float(std_err[1]),
        ci_low=float(coef[1] - t_crit * std_err[1]),
        ci_high=float(coef[1] + t_crit * std_err[1]),
        r_squared=r_squared,
        n=x.size,
    )


def bootstrap_elasticity(
    price: ArrayLike,
    quantity: ArrayLike,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the elasticity — no normality assumption on the residuals.

    Prefer this over :func:`fit_demand_ci` when n is small or residuals are heavy-tailed; the two
    agreeing is itself a robustness check.
    """
    p, q = _validate_loglog(price, quantity)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    log_p, log_q = np.log(p), np.log(q)
    for b in range(n_boot):
        idx = rng.integers(0, p.size, p.size)
        if np.unique(log_p[idx]).size < 2:  # degenerate resample — redraw is biased, just skip
            estimates[b] = np.nan
            continue
        estimates[b] = np.polyfit(log_p[idx], log_q[idx], 1)[0]
    alpha = 1.0 - confidence
    low, high = np.nanquantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def cross_price_elasticity(
    quantity: ArrayLike,
    own_price: ArrayLike,
    cross_prices: Mapping[str, ArrayLike],
    *,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Own- and cross-price elasticities from one multivariate log-log regression.

    Cross elasticity > 0 = substitute (their price up → your demand up), < 0 = complement.
    Estimating own and cross effects *jointly* matters: competitor prices move with yours, and a
    univariate fit would push their effect into your own-price slope (omitted-variable bias).
    """
    from scipy import stats as sps

    q = np.asarray(quantity, dtype=float)
    p_own = np.asarray(own_price, dtype=float)
    if np.any(q <= 0) or np.any(p_own <= 0):
        raise ValueError("log-log demand requires strictly positive price and quantity")
    names = list(cross_prices)
    columns = [np.log(p_own)]
    for name in names:
        p_cross = np.asarray(cross_prices[name], dtype=float)
        if np.any(p_cross <= 0):
            raise ValueError(f"cross price {name!r} must be strictly positive")
        columns.append(np.log(p_cross))
    design = np.column_stack([np.ones(q.size), *columns])
    coef, std_err, _, dof = _ols(design, np.log(q))
    t_crit = float(sps.t.ppf(1.0 - (1.0 - confidence) / 2.0, dof))

    rows = []
    for i, term in enumerate(["own", *names]):
        estimate, se = float(coef[i + 1]), float(std_err[i + 1])
        t_stat = estimate / se
        p_value = 2.0 * float(sps.t.sf(abs(t_stat), dof))
        if term == "own":
            relationship = "own"
        elif estimate > 0:
            relationship = "substitute"
        else:
            relationship = "complement"
        rows.append(
            {
                "term": term,
                "elasticity": estimate,
                "std_err": se,
                "p_value": p_value,
                "ci_low": estimate - t_crit * se,
                "ci_high": estimate + t_crit * se,
                "relationship": relationship,
            }
        )
    return pl.DataFrame(rows)


def segment_elasticity(
    df: pl.DataFrame,
    *,
    price: str,
    quantity: str,
    segment: str,
    confidence: float = 0.95,
    min_rows: int = 8,
) -> pl.DataFrame:
    """Per-segment elasticity with CIs — who is price-sensitive and who isn't.

    Segments with fewer than ``min_rows`` rows, non-positive values, or no price variation are
    skipped (you can't estimate a slope there). Differentiated pricing follows directly: protect
    margin where |e| is small, compete on price where it is large.
    """
    rows = []
    for (label,), part in df.partition_by(segment, as_dict=True).items():
        if part.height < min_rows:
            continue
        try:
            fit = fit_demand_ci(
                part[price].to_numpy(), part[quantity].to_numpy(), confidence=confidence
            )
        except ValueError:
            continue
        rows.append(
            {
                "segment": label,
                "n": fit.n,
                "elasticity": fit.elasticity,
                "std_err": fit.std_err,
                "ci_low": fit.ci_low,
                "ci_high": fit.ci_high,
                "intercept": fit.intercept,
            }
        )
    if not rows:
        raise ValueError("no segment had enough rows and price variation to fit")
    return pl.DataFrame(rows).sort("elasticity")


def rolling_elasticity(
    df: pl.DataFrame,
    *,
    price: str,
    quantity: str,
    time: str,
    window: int = 60,
    step: int = 1,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Elasticity re-fit over a rolling window — dynamic elasticity through time.

    Each row is the fit on the last ``window`` observations ending at ``time``. Widening or
    drifting bands mean the constant-elasticity assumption is breaking (season, competition,
    mix); test formally with :func:`elasticity_drift`.
    """
    data = df.sort(time)
    p = data[price].to_numpy()
    q = data[quantity].to_numpy()
    stamps = data[time]
    if window < 8:
        raise ValueError("window must be at least 8 observations")
    if p.size < window:
        raise ValueError(f"need at least window={window} rows, got {p.size}")
    rows = []
    for end in range(window, p.size + 1, step):
        try:
            fit = fit_demand_ci(p[end - window : end], q[end - window : end], confidence=confidence)
        except ValueError:
            continue
        rows.append(
            {
                time: stamps[end - 1],
                "elasticity": fit.elasticity,
                "ci_low": fit.ci_low,
                "ci_high": fit.ci_high,
                "n": fit.n,
            }
        )
    if not rows:
        raise ValueError("no window had enough price variation to fit")
    return pl.DataFrame(rows)


@dataclass(frozen=True)
class ElasticityDrift:
    """Baseline-vs-recent elasticity comparison (two-sample z on the slopes)."""

    baseline: ElasticityFit
    recent: ElasticityFit
    difference: float
    z: float
    p_value: float
    drifted: bool


def elasticity_drift(
    df: pl.DataFrame,
    *,
    price: str,
    quantity: str,
    time: str,
    split: float = 0.5,
    alpha: float = 0.05,
) -> ElasticityDrift:
    """Has price sensitivity moved? Fit the first ``split`` share vs the rest and z-test the gap.

    A drifted elasticity quietly invalidates the optimal price — re-optimize when this fires.
    Pair with :func:`rolling_elasticity` to *see* when the move happened.
    """
    from scipy import stats as sps

    if not 0.1 <= split <= 0.9:
        raise ValueError("split must be in [0.1, 0.9]")
    data = df.sort(time)
    cut = int(data.height * split)
    baseline = fit_demand_ci(data[price][:cut].to_numpy(), data[quantity][:cut].to_numpy())
    recent = fit_demand_ci(data[price][cut:].to_numpy(), data[quantity][cut:].to_numpy())
    difference = recent.elasticity - baseline.elasticity
    z = difference / float(np.hypot(baseline.std_err, recent.std_err))
    p_value = 2.0 * float(sps.norm.sf(abs(z)))
    return ElasticityDrift(baseline, recent, difference, z, p_value, p_value < alpha)


@dataclass(frozen=True)
class CurvatureCheck:
    """Quadratic-in-log-price fit: does elasticity itself vary with the price level?"""

    slope: float
    curvature: float
    p_value: float
    aic_linear: float
    aic_quadratic: float
    nonlinear: bool

    def local_elasticity(self, price: ArrayLike) -> NDArray[np.float64]:
        """Elasticity at each price under the quadratic model: slope + 2·curvature·ln(price)."""
        p = np.asarray(price, dtype=float)
        return np.asarray(self.slope + 2.0 * self.curvature * np.log(p), dtype=float)


def nonlinear_elasticity_check(
    price: ArrayLike, quantity: ArrayLike, *, alpha: float = 0.05
) -> CurvatureCheck:
    """Test the constant-elasticity assumption by adding a ``(ln price)²`` term.

    A significant curvature term (small ``p_value``, lower ``aic_quadratic``) means one elasticity
    number is wrong across the price range — price response is locally steeper or flatter; read
    ``local_elasticity`` at the prices you actually charge.
    """
    from scipy import stats as sps

    p, q = _validate_loglog(price, quantity)
    x, y = np.log(p), np.log(q)
    n = x.size
    linear = np.column_stack([np.ones(n), x])
    quadratic = np.column_stack([np.ones(n), x, x**2])
    _, _, resid_lin, _ = _ols(linear, y)
    coef, std_err, resid_quad, dof = _ols(quadratic, y)
    t_stat = float(coef[2]) / float(std_err[2])
    p_value = 2.0 * float(sps.t.sf(abs(t_stat), dof))
    aic_linear = n * float(np.log(resid_lin @ resid_lin / n)) + 2 * 2
    aic_quadratic = n * float(np.log(resid_quad @ resid_quad / n)) + 2 * 3
    return CurvatureCheck(
        slope=float(coef[1]),
        curvature=float(coef[2]),
        p_value=p_value,
        aic_linear=aic_linear,
        aic_quadratic=aic_quadratic,
        nonlinear=p_value < alpha and aic_quadratic < aic_linear,
    )


def aggregate_elasticity(elasticities: ArrayLike, weights: ArrayLike) -> float:
    """Portfolio elasticity as the weighted mean of segment elasticities (weights = value share)."""
    e = np.asarray(elasticities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and sum to a positive total")
    return float(np.average(e, weights=w))


def elasticity_decomposition(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    segment: str = "segment",
    elasticity: str = "elasticity",
    weight: str = "weight",
) -> pl.DataFrame:
    """Why did aggregate elasticity move? Split the change into within-segment vs mix per segment.

    Shift-share with midpoint weights — ``within`` (segments became more/less price-sensitive) and
    ``mix`` (value shifted toward more/less sensitive segments) sum *exactly* to the aggregate
    change. A pure-mix move needs portfolio action, not price action. Segments present on one side
    only contribute through ``mix``.
    """
    norm_before = before.with_columns((pl.col(weight) / pl.col(weight).sum()).alias("_w0")).select(
        pl.col(segment), pl.col(elasticity).alias("_e0"), "_w0"
    )
    norm_after = after.with_columns((pl.col(weight) / pl.col(weight).sum()).alias("_w1")).select(
        pl.col(segment), pl.col(elasticity).alias("_e1"), "_w1"
    )
    joined = norm_before.join(norm_after, on=segment, how="full", coalesce=True).with_columns(
        pl.col("_w0").fill_null(0.0),
        pl.col("_w1").fill_null(0.0),
        # a segment absent on one side has no "own" elasticity there: carry the other side's
        # value so its whole contribution lands in the mix term.
        pl.col("_e0").fill_null(pl.col("_e1")),
        pl.col("_e1").fill_null(pl.col("_e0")),
    )
    result = (
        joined.with_columns(
            (((pl.col("_w0") + pl.col("_w1")) / 2) * (pl.col("_e1") - pl.col("_e0"))).alias(
                "within"
            ),
            (((pl.col("_e0") + pl.col("_e1")) / 2) * (pl.col("_w1") - pl.col("_w0"))).alias("mix"),
        )
        .with_columns((pl.col("within") + pl.col("mix")).alias("total"))
        .rename(
            {
                "_e0": "elasticity_before",
                "_e1": "elasticity_after",
                "_w0": "weight_before",
                "_w1": "weight_after",
            }
        )
    )
    return result.sort(pl.col("total").abs(), descending=True)
