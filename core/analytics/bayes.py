"""Bayesian building blocks — conjugate updates, hierarchical shrinkage, a generic MCMC sampler.

The grammar: prior (belief before data) times likelihood (how probable the data are under each
parameter value) → posterior (belief after). Conjugate pairs (Beta-Binomial here) make that update
closed-form; MCMC samples the posterior when no closed form exists. The A/B-decision wrappers
built on these ideas live in ``analytics.experiment`` (``bayes_conversions`` / ``bayes_means``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Posterior:
    """A posterior summary: point estimate, credible interval, and the Beta parameters."""

    mean: float
    lower: float
    upper: float
    alpha: float
    beta: float


def beta_posterior(
    successes: int,
    trials: int,
    *,
    prior: tuple[float, float] = (1.0, 1.0),
    confidence: float = 0.95,
) -> Posterior:
    """Conjugate Beta-Binomial update for one rate (conversion, churn, defect share).

    Beta(a, b) prior + s successes in n trials → Beta(a+s, b+n-s) posterior — conjugacy means no
    simulation needed. The interval is *credible*: "the rate is in this range with 95%
    probability", the plain-language reading a frequentist CI doesn't license. Beta(1, 1) is
    uniform; a stronger prior (larger a+b, mean a/(a+b)) encodes history when n is small.
    """
    from scipy.stats import beta as beta_dist

    a = prior[0] + successes
    b = prior[1] + trials - successes
    lower, upper = beta_dist.ppf([(1 - confidence) / 2, (1 + confidence) / 2], a, b)
    return Posterior(a / (a + b), float(lower), float(upper), float(a), float(b))


def hierarchical_rates(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> tuple[pl.DataFrame, tuple[float, float]]:
    """Partial pooling for many small rates (per store / SKU / campaign) — empirical Bayes.

    Fits one shared Beta prior to the observed rates (method of moments), then updates each group
    with its own data: small-n groups shrink hard toward the global mean, large-n groups barely
    move. Cures the league-table pathology where tiny groups top and bottom the ranking by luck —
    rank on ``shrunk_rate``. Returns (per-group frame with credible intervals, fitted prior (a, b)).
    """
    from scipy.stats import beta as beta_dist

    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    rates = s / n
    mu = float(rates.mean())
    var = float(rates.var(ddof=1)) if rates.size > 1 else 0.0
    cap = mu * (1.0 - mu)
    # Beta method of moments: a + b = mu(1-mu)/var - 1; degenerate spreads get a strong prior.
    strength = max(cap / var - 1.0, 1.0) if 0.0 < var < cap else 1000.0
    a0, b0 = mu * strength, (1.0 - mu) * strength
    post_a = a0 + s
    post_b = b0 + (n - s)
    frame = pl.DataFrame(
        {
            "group": list(labels) if labels is not None else [str(i) for i in range(s.size)],
            "successes": s.astype(int),
            "trials": n.astype(int),
            "rate": rates,
            "shrunk_rate": post_a / (post_a + post_b),
            "lower": np.asarray(beta_dist.ppf((1 - confidence) / 2, post_a, post_b), dtype=float),
            "upper": np.asarray(beta_dist.ppf((1 + confidence) / 2, post_a, post_b), dtype=float),
        }
    )
    return frame, (float(a0), float(b0))


def mcmc_sample(
    log_density: Callable[[NDArray[np.float64]], float],
    start: ArrayLike,
    *,
    n_samples: int = 5000,
    burn_in: int = 1000,
    step: float = 0.5,
    seed: int = 42,
) -> tuple[NDArray[np.float64], float]:
    """Random-walk Metropolis sampler — posterior draws when no conjugate form exists.

    Pass any log-density (log-prior + log-likelihood; additive constants may be dropped) and a
    start point; get (samples, acceptance_rate) back. Summarize the samples like any posterior:
    means, ``np.quantile`` credible intervals. Tune ``step`` toward ~20-40% acceptance and eyeball
    the trace for mixing before trusting; for big multi-parameter models reach for a dedicated
    PPL — this is the dependency-free workhorse.
    """
    rng = np.random.default_rng(seed)
    current = np.atleast_1d(np.asarray(start, dtype=float)).copy()
    current_lp = float(log_density(current))
    samples = np.empty((n_samples, current.size), dtype=float)
    accepted = 0
    for i in range(n_samples + burn_in):
        proposal = current + rng.normal(0.0, step, current.size)
        proposal_lp = float(log_density(proposal))
        if np.log(rng.random()) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            accepted += 1
        if i >= burn_in:
            samples[i - burn_in] = current
    return samples, accepted / (n_samples + burn_in)
