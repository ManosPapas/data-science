"""Demand curves beyond log-log — linear demand, purchase probability, willingness-to-pay.

Constant elasticity (``pricing.elasticity``) assumes demand reacts proportionally at every price
level. The models here let the response vary: a linear curve with a choke price, a logit purchase
model that turns individual buy/no-buy decisions into a willingness-to-pay distribution, and the
survey-based Van Westendorp price sensitivity meter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LinearDemand:
    """Linear demand ``q = intercept + slope·price`` — elasticity varies along the line."""

    intercept: float
    slope: float
    r_squared: float
    n: int

    @property
    def choke_price(self) -> float:
        """The price where demand hits zero — predictions beyond it are extrapolation."""
        return -self.intercept / self.slope

    def predict(self, price: ArrayLike) -> NDArray[np.float64]:
        """Predicted quantity (floored at 0 past the choke price)."""
        p = np.asarray(price, dtype=float)
        return np.asarray(np.maximum(self.intercept + self.slope * p, 0.0), dtype=float)

    def elasticity_at(self, price: ArrayLike) -> NDArray[np.float64]:
        """Point elasticity slope·p/q — rises in magnitude toward the choke price."""
        p = np.asarray(price, dtype=float)
        q = self.intercept + self.slope * p
        return np.asarray(self.slope * p / q, dtype=float)


def fit_linear_demand(price: ArrayLike, quantity: ArrayLike) -> LinearDemand:
    """OLS linear demand curve. Expect ``slope < 0``; a positive slope means confounded data."""
    p = np.asarray(price, dtype=float)
    q = np.asarray(quantity, dtype=float)
    if np.unique(p).size < 2:
        raise ValueError("need at least two distinct prices to identify a demand slope")
    slope, intercept = np.polyfit(p, q, 1)
    fitted = intercept + slope * p
    tss = float(np.sum((q - q.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((q - fitted) ** 2)) / tss if tss > 0 else float("nan")
    return LinearDemand(float(intercept), float(slope), r_squared, p.size)


@dataclass(frozen=True)
class LogitDemand:
    """Purchase-probability model ``P(buy|price) = sigmoid(intercept + slope·price)``.

    Under this model individual willingness-to-pay is Logistic(median=-intercept/slope,
    scale=-1/slope), so WTP quantiles come in closed form.
    """

    intercept: float
    slope: float
    std_err_slope: float
    n: int

    def predict(self, price: ArrayLike) -> NDArray[np.float64]:
        """Purchase probability at each price."""
        p = np.asarray(price, dtype=float)
        return np.asarray(1.0 / (1.0 + np.exp(-(self.intercept + self.slope * p))), dtype=float)

    def demand_at(self, price: ArrayLike, *, market_size: float) -> NDArray[np.float64]:
        """Expected buyers at each price = market_size · P(buy)."""
        return np.asarray(market_size * self.predict(price), dtype=float)

    @property
    def wtp_median(self) -> float:
        """Median willingness-to-pay — the price where purchase probability crosses 50%."""
        return -self.intercept / self.slope

    def wtp_quantile(self, q: float | ArrayLike) -> NDArray[np.float64]:
        """WTP quantile(s): the price only the top ``1-q`` share of customers would pay."""
        qs = np.asarray(q, dtype=float)
        scale = -1.0 / self.slope
        return np.asarray(self.wtp_median + scale * np.log(qs / (1.0 - qs)), dtype=float)


def fit_logit_demand(price: ArrayLike, purchased: ArrayLike) -> LogitDemand:
    """Fit purchase probability vs price by logistic regression (statsmodels GLM, no penalty).

    ``purchased`` is 0/1 per offer shown. A *positive* slope means higher price → more purchases:
    almost always confounding (price proxies quality/segment), not a Veblen good — fix the data.
    """
    import statsmodels.api as sm

    p = np.asarray(price, dtype=float)
    y = np.asarray(purchased, dtype=float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("purchased must be binary 0/1")
    if np.unique(p).size < 2:
        raise ValueError("need at least two distinct prices to identify price response")
    if np.unique(y).size < 2:
        raise ValueError("need both purchases and non-purchases to fit")
    design = sm.add_constant(p)
    result = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    return LogitDemand(
        intercept=float(result.params[0]),
        slope=float(result.params[1]),
        std_err_slope=float(result.bse[1]),
        n=p.size,
    )


def willingness_to_pay(
    price: ArrayLike,
    purchased: ArrayLike,
    *,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> pl.DataFrame:
    """WTP distribution table from buy/no-buy data — quantile → the price that share won't exceed.

    Read it as a pricing menu: the median is the mass-market price; upper quantiles are what a
    premium tier can charge into.
    """
    model = fit_logit_demand(price, purchased)
    if model.slope >= 0:
        raise ValueError("fitted slope is non-negative — WTP is undefined; check for confounding")
    wtp = model.wtp_quantile(np.asarray(quantiles, dtype=float))
    return pl.DataFrame({"quantile": list(quantiles), "wtp": wtp})


@dataclass(frozen=True)
class VanWestendorp:
    """Van Westendorp price sensitivity meter — survey-based acceptable price range.

    ``optimal_price`` (too-cheap/too-expensive crossing) minimizes resistance at both ends;
    ``acceptable`` range is [point of marginal cheapness, point of marginal expensiveness].
    """

    optimal_price: float
    indifference_price: float
    lower_bound: float
    upper_bound: float
    curves: pl.DataFrame

    @property
    def acceptable_range(self) -> tuple[float, float]:
        return self.lower_bound, self.upper_bound


def _crossing(grid: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """First crossing of two curves on a shared grid, linearly interpolated."""
    diff = a - b
    signs = np.sign(diff)
    for i in range(signs.size - 1):
        if signs[i] == 0:
            return float(grid[i])
        if signs[i] * signs[i + 1] < 0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            return float(grid[i] + frac * (grid[i + 1] - grid[i]))
    raise ValueError("curves do not cross — check the survey columns are in the right order")


def van_westendorp(
    too_cheap: ArrayLike, cheap: ArrayLike, expensive: ArrayLike, too_expensive: ArrayLike
) -> VanWestendorp:
    """Price sensitivity meter from the four survey prices per respondent.

    Respondents answer: at what price is it *too cheap* (quality doubt), *cheap* (a bargain),
    *expensive* (getting dear), *too expensive* (out of the question). Crossings of the cumulative
    curves give the classic points. Stated-preference data — calibrate against transactions
    (:func:`willingness_to_pay`) when you have them.
    """
    tc = np.asarray(too_cheap, dtype=float)
    ch = np.asarray(cheap, dtype=float)
    ex = np.asarray(expensive, dtype=float)
    te = np.asarray(too_expensive, dtype=float)
    grid = np.unique(np.concatenate([tc, ch, ex, te]))
    share_too_cheap = (tc[None, :] >= grid[:, None]).mean(axis=1)
    share_cheap = (ch[None, :] >= grid[:, None]).mean(axis=1)
    share_expensive = (ex[None, :] <= grid[:, None]).mean(axis=1)
    share_too_expensive = (te[None, :] <= grid[:, None]).mean(axis=1)
    curves = pl.DataFrame(
        {
            "price": grid,
            "too_cheap": share_too_cheap,
            "cheap": share_cheap,
            "expensive": share_expensive,
            "too_expensive": share_too_expensive,
        }
    )
    return VanWestendorp(
        optimal_price=_crossing(grid, share_too_cheap, share_too_expensive),
        indifference_price=_crossing(grid, share_cheap, share_expensive),
        lower_bound=_crossing(grid, share_too_cheap, share_expensive),
        upper_bound=_crossing(grid, share_cheap, share_too_expensive),
        curves=curves,
    )


def demand_schedule(
    quantity_fn: Callable[[NDArray[np.float64]], ArrayLike], prices: ArrayLike
) -> pl.DataFrame:
    """Tabulate any demand model over candidate prices: price, quantity, revenue.

    Works with every model here (``model.predict``) and with ``elasticity.predict_demand`` via a
    lambda — chart-ready, and the input for ``analytics.curves`` turning-point analysis.
    """
    p = np.asarray(prices, dtype=float)
    q = np.asarray(quantity_fn(p), dtype=float)
    return pl.DataFrame({"price": p, "quantity": q, "revenue": p * q})
