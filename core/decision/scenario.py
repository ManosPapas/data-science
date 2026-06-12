"""Decision analysis — expected utility, scenario tables, one-at-a-time sensitivity (tornado).

Models produce estimates; decisions need them folded with probabilities, risk attitude, and
what-ifs. ``kpi.profit`` prices a classifier's errors; this module prices *decisions*: what is the
gamble worth (``expected_utility`` / ``certainty_equivalent``), what happens under coherent
futures (``scenario_table``), and which assumption actually drives the answer (``sensitivity``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike


def expected_utility(
    outcomes: ArrayLike, probabilities: ArrayLike, *, risk_aversion: float = 0.0
) -> float:
    """Probability-weighted utility of a gamble; ``risk_aversion=0`` reduces to expected value.

    Uses exponential (CARA) utility u(x) = (1 - e^(-a·x))/a: with a > 0 the downside hurts more
    than equal upside helps, so volatile options score below their expected value — expected
    *value* alone ignores risk appetite. ``a`` is per outcome-unit, so keep units consistent
    (e.g. k€) and compare options on the same scale, or convert back to money with
    :func:`certainty_equivalent`. Probabilities are normalized to sum to 1.
    """
    x = np.asarray(outcomes, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    if risk_aversion == 0.0:
        return float(np.sum(p * x))
    utilities = (1.0 - np.exp(-risk_aversion * x)) / risk_aversion
    return float(np.sum(p * utilities))


def certainty_equivalent(
    outcomes: ArrayLike, probabilities: ArrayLike, *, risk_aversion: float = 0.0
) -> float:
    """The sure amount worth exactly the gamble: u⁻¹(expected utility), in outcome units.

    Expected value minus the certainty equivalent is the risk premium — what it is rational to
    pay to avoid the gamble (the price of insurance, the discount for a guaranteed contract).
    Equals the expected value when ``risk_aversion`` is 0.
    """
    eu = expected_utility(outcomes, probabilities, risk_aversion=risk_aversion)
    if risk_aversion == 0.0:
        return eu
    return float(-np.log(1.0 - risk_aversion * eu) / risk_aversion)


def scenario_table(
    value_fn: Callable[..., float],
    base: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> pl.DataFrame:
    """Value a decision under named what-if input sets (base / optimistic / downturn / ...).

    Each scenario overrides part of ``base`` and re-evaluates ``value_fn(**inputs)``; ``vs_base``
    is the swing. Scenarios should be coherent futures (several inputs moving together — a
    downturn moves volume *and* price); for "which single input matters most" use
    :func:`sensitivity` instead.
    """
    base_value = float(value_fn(**base))
    rows = [{"scenario": "base", "value": base_value, "vs_base": 0.0}]
    for name, overrides in scenarios.items():
        value = float(value_fn(**{**base, **overrides}))
        rows.append({"scenario": name, "value": value, "vs_base": value - base_value})
    return pl.DataFrame(rows)


def sensitivity(
    value_fn: Callable[..., float],
    base: Mapping[str, Any],
    ranges: Mapping[str, tuple[Any, Any]],
) -> pl.DataFrame:
    """One-at-a-time sensitivity: swing each input low→high, others at base (tornado chart data).

    ``swing`` = value(high) - value(low); the largest |swing| marks the assumption the decision
    actually hinges on — spend your data-collection budget there first. One-at-a-time by design,
    so it misses interactions between inputs; probe those with :func:`scenario_table`.
    """
    rows = []
    for name, (low, high) in ranges.items():
        low_value = float(value_fn(**{**base, name: low}))
        high_value = float(value_fn(**{**base, name: high}))
        rows.append(
            {
                "input": name,
                "low_value": low_value,
                "high_value": high_value,
                "swing": high_value - low_value,
            }
        )
    return pl.DataFrame(rows).sort(pl.col("swing").abs(), descending=True)
