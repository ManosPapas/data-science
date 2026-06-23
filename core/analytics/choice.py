"""Discrete choice & preference measurement — conjoint, MaxDiff, and the designs that feed them.

This is the *stated-preference* counterpart to ``pricing.demand`` (which reads preference off
transactions). The estimation engine is McFadden's conditional (multinomial) logit over choice
sets — :func:`fit_conditional_logit` — fitted by maximising the within-set choice likelihood; its
globally-concave log-likelihood means a clean Newton/quasi-Newton fit and exact information-matrix
standard errors. Everything else composes on top:

- **Choice-based conjoint (CBC)** — :func:`choice_based_conjoint`: dummy-codes attribute levels,
  fits the logit, and returns part-worth utilities, attribute importances, a share-of-preference
  simulator, and (with a price attribute) willingness-to-pay.
- **Metric (ratings) conjoint** — :func:`metric_conjoint`: the OLS read when respondents *rate*
  rather than *choose*; coefficients are the part-worths.
- **MaxDiff (best-worst scaling)** — :func:`maxdiff_counts` (the assumption-free counting score)
  and :func:`maxdiff_logit` (the exploded best-worst logit, on the same engine).
- **Experimental structures** — :func:`full_factorial`, :func:`orthogonal_design` (D-efficient
  fractional design via coordinate exchange), :func:`choice_design`, :func:`maxdiff_design`.

Aggregate models here estimate population-mean part-worths. Heterogeneity is real: read it via
``segment`` on respondents, per-respondent ``maxdiff_counts``, or a hierarchical-Bayes extension
(``analytics.bayes``) when individual-level utilities matter.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

# --- Conditional (multinomial) logit — the estimation engine --------------------------------


def _group_index(values: NDArray[Any]) -> tuple[NDArray[np.intp], int]:
    """Contiguous 0..G-1 codes for arbitrary choice-set labels, plus the group count."""
    _, inverse = np.unique(values, return_inverse=True)
    groups = np.asarray(inverse, dtype=np.intp).reshape(-1)
    return groups, (int(groups.max()) + 1 if groups.size else 0)


def _softmax_by_group(
    v: NDArray[np.float64], groups: NDArray[np.intp], n_sets: int
) -> NDArray[np.float64]:
    """Within-set choice probabilities: exp(v) normalised per set (max-shifted for stability)."""
    gmax = np.full(n_sets, -np.inf)
    np.maximum.at(gmax, groups, v)
    ev = np.exp(v - gmax[groups])
    denom = np.bincount(groups, weights=ev, minlength=n_sets)
    return np.asarray(ev / denom[groups], dtype=float)


@dataclass(frozen=True)
class ChoiceModel:
    """A fitted conditional/multinomial logit over choice sets, read for inference.

    ``coefficients`` carries the part-worth scale utilities (term / coef / std_err / statistic /
    p_value / ci_low / ci_high) — utilities are identified only up to scale, so read *differences*
    and signs, not absolute levels. ``mcfadden_r2`` is the pseudo-R² 1 - LL/LL₀ (0.2-0.4 is a good
    logit fit, not the 0.7+ you'd want from OLS).
    """

    coefficients: pl.DataFrame
    log_likelihood: float
    ll_null: float
    mcfadden_r2: float
    mcfadden_r2_adj: float
    aic: float
    n_choices: int
    feature_names: list[str]
    beta: NDArray[np.float64]

    def utilities(self, df: pl.DataFrame) -> NDArray[np.float64]:
        """Deterministic utility xβ for each row of a long design (same feature columns)."""
        x = df.select(self.feature_names).to_numpy().astype(float)
        return np.asarray(x @ self.beta, dtype=float)

    def predict_shares(self, df: pl.DataFrame, *, choice_set: str) -> pl.DataFrame:
        """Append a ``share`` column: the logit choice probability within each ``choice_set``."""
        groups, n_sets = _group_index(df[choice_set].to_numpy())
        share = _softmax_by_group(self.utilities(df), groups, n_sets)
        return df.with_columns(pl.Series("share", share))


def _fit_clogit(
    groups: NDArray[np.intp],
    chosen: NDArray[np.float64],
    x: NDArray[np.float64],
    names: Sequence[str],
) -> ChoiceModel:
    """Maximum-likelihood conditional logit given group codes, 0/1 choices, and a feature matrix."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    k = x.shape[1]
    n_sets = int(groups.max()) + 1 if groups.size else 0
    chosen_per_set = np.bincount(groups, weights=chosen, minlength=n_sets)
    if n_sets == 0 or not np.allclose(chosen_per_set, 1.0):
        raise ValueError("each choice set must contain exactly one chosen alternative")
    set_sizes = np.bincount(groups, minlength=n_sets).astype(float)

    def neg_ll(beta: NDArray[np.float64]) -> float:
        v = x @ beta
        gmax = np.full(n_sets, -np.inf)
        np.maximum.at(gmax, groups, v)
        lse = gmax + np.log(np.bincount(groups, weights=np.exp(v - gmax[groups]), minlength=n_sets))
        return float(lse.sum() - np.dot(chosen, v))

    def grad(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        p = _softmax_by_group(x @ beta, groups, n_sets)
        return np.asarray(x.T @ (p - chosen), dtype=float)

    result = minimize(neg_ll, np.zeros(k), jac=grad, method="BFGS")
    beta = np.asarray(result.x, dtype=float)
    ll = -float(result.fun)
    ll_null = -float(np.log(set_sizes).sum())

    # Observed = Fisher information for the logit: sum_rows p*xx' - sum_sets (sum p*x)(sum p*x)'.
    p = _softmax_by_group(x @ beta, groups, n_sets)
    px = p[:, None] * x
    group_mean = np.zeros((n_sets, k))
    np.add.at(group_mean, groups, px)
    info = x.T @ px - group_mean.T @ group_mean
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:  # singular: collinear or set-invariant features
        raise ValueError(
            "information matrix is singular — features are collinear or constant within sets"
        ) from exc
    std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err > 0)
    p_value = 2.0 * np.asarray(norm.sf(np.abs(z)), dtype=float)
    half = float(norm.ppf(0.975)) * std_err
    coefficients = pl.DataFrame(
        {
            "term": list(names),
            "coef": beta,
            "std_err": std_err,
            "statistic": z,
            "p_value": p_value,
            "ci_low": beta - half,
            "ci_high": beta + half,
        }
    )
    mcfadden = 1.0 - ll / ll_null if ll_null != 0 else float("nan")
    mcfadden_adj = 1.0 - (ll - k) / ll_null if ll_null != 0 else float("nan")
    return ChoiceModel(
        coefficients=coefficients,
        log_likelihood=ll,
        ll_null=ll_null,
        mcfadden_r2=mcfadden,
        mcfadden_r2_adj=mcfadden_adj,
        aic=2.0 * k - 2.0 * ll,
        n_choices=n_sets,
        feature_names=list(names),
        beta=beta,
    )


def fit_conditional_logit(
    df: pl.DataFrame, *, choice: str, choice_set: str, features: Sequence[str]
) -> ChoiceModel:
    """McFadden's conditional logit on a long choice table — one row per alternative-in-set.

    ``choice`` is the 0/1 chosen flag (exactly one 1 per ``choice_set``); ``features`` are the
    numeric, alternative-specific attributes (dummy-code categoricals first via
    :func:`choice_based_conjoint`, which calls this for you). Coefficients are utilities on the
    logit scale — a one-unit feature change multiplies a set's odds of being chosen by exp(coef).
    Conditional logit assumes IIA (independence of irrelevant alternatives): a near-duplicate
    alternative should not steal share disproportionately — violations want nested/mixed logit.
    """
    frame = df.select([choice_set, choice, *features]).drop_nulls()
    chosen = frame[choice].cast(pl.Float64).to_numpy()
    if not np.isin(chosen, (0.0, 1.0)).all():
        raise ValueError("choice must be a binary 0/1 chosen flag")
    groups, _ = _group_index(frame[choice_set].to_numpy())
    x = frame.select(features).to_numpy().astype(float)
    return _fit_clogit(groups, chosen, x, list(features))


# --- Attribute encoding & conjoint readouts -------------------------------------------------


def encode_attributes(
    df: pl.DataFrame, attributes: Sequence[str], *, reference: Mapping[str, Any] | None = None
) -> tuple[pl.DataFrame, dict[str, list[str]], list[str]]:
    """Dummy-code categorical attributes (reference level dropped) for a part-worth design.

    Returns the 0/1 design frame, a ``{attribute: [levels]}`` map with the reference level first,
    and the ordered list of generated ``"attribute=level"`` term names. The dropped reference level
    is the baseline every part-worth is measured against (its utility is pinned to 0).
    """
    reference = reference or {}
    columns: dict[str, pl.Series] = {}
    level_maps: dict[str, list[str]] = {}
    terms: list[str] = []
    for attr in attributes:
        levels = df[attr].unique().sort().to_list()
        if len(levels) < 2:
            raise ValueError(
                f"attribute {attr!r} needs at least two levels to estimate a part-worth"
            )
        ref = reference.get(attr, levels[0])
        if ref not in levels:
            raise ValueError(f"reference level {ref!r} not found in attribute {attr!r}")
        ordered = [ref, *[lvl for lvl in levels if lvl != ref]]
        level_maps[attr] = ordered
        for lvl in ordered[1:]:
            term = f"{attr}={lvl}"
            terms.append(term)
            columns[term] = (df[attr] == lvl).cast(pl.Int8).alias(term)
    return pl.DataFrame(columns), level_maps, terms


def _part_worths(coefficients: pl.DataFrame, level_maps: Mapping[str, list[str]]) -> pl.DataFrame:
    """Turn fitted ``attribute=level`` coefficients into a tidy part-worth table (reference = 0)."""
    lookup = {row["term"]: row for row in coefficients.iter_rows(named=True)}
    rows: list[dict[str, Any]] = []
    for attr, levels in level_maps.items():
        for index, lvl in enumerate(levels):
            if index == 0:  # reference level: pinned baseline
                rows.append(
                    {
                        "attribute": attr,
                        "level": str(lvl),
                        "utility": 0.0,
                        "std_err": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                    }
                )
            else:
                coef = lookup[f"{attr}={lvl}"]
                rows.append(
                    {
                        "attribute": attr,
                        "level": str(lvl),
                        "utility": float(coef["coef"]),
                        "std_err": float(coef["std_err"]),
                        "ci_low": float(coef["ci_low"]),
                        "ci_high": float(coef["ci_high"]),
                    }
                )
    return pl.DataFrame(rows)


def attribute_importance(part_worths: pl.DataFrame) -> pl.DataFrame:
    """Relative attribute importance: each attribute's part-worth range as a share of the total.

    Importance = (max - min part-worth) for the attribute, divided by the summed ranges across all
    attributes — the standard conjoint read of *which attribute moves choice most*. It depends on
    the levels you tested: a wider price range mechanically inflates price's importance, so compare
    importances only across studies with comparable level spans.
    """
    ranges = part_worths.group_by("attribute").agg(
        (pl.col("utility").max() - pl.col("utility").min()).alias("range")
    )
    total = float(ranges["range"].sum())
    if total <= 0:
        raise ValueError("part-worths have zero spread — nothing to rank")
    return ranges.with_columns(
        (pl.col("range") / total).alias("importance"),
        (pl.col("range") / total * 100.0).alias("importance_pct"),
    ).sort("importance", descending=True)


@dataclass(frozen=True)
class Conjoint:
    """A fitted conjoint study: part-worths, importances, and a share/WTP simulator.

    ``part_worths`` (attribute / level / utility + SE + CI) and ``importance`` are the headline
    deliverables. Use :meth:`simulate` to predict choice shares for candidate product line-ups and
    :meth:`willingness_to_pay` (when a numeric ``price`` attribute was fitted) to read part-worths
    in money. ``model`` is the underlying :class:`ChoiceModel` for CBC and ``None`` for metric
    conjoint (which carries ``intercept`` / ``r_squared`` instead).
    """

    part_worths: pl.DataFrame
    importance: pl.DataFrame
    attributes: list[str]
    level_maps: dict[str, list[str]]
    price: str | None
    price_coef: float | None
    model: ChoiceModel | None
    intercept: float | None = None
    r_squared: float | None = None

    def _utility(self, profiles: pl.DataFrame) -> NDArray[np.float64]:
        lookup = {
            (row["attribute"], row["level"]): row["utility"]
            for row in self.part_worths.iter_rows(named=True)
        }
        base = self.intercept or 0.0
        util = np.full(profiles.height, base, dtype=float)
        for attr in self.attributes:
            values = profiles[attr].to_list()
            for value in values:
                if (attr, str(value)) not in lookup:
                    raise ValueError(
                        f"level {value!r} of {attr!r} was not in the fitted design — "
                        "cannot simulate an unfitted level"
                    )
            util += np.array([lookup[(attr, str(v))] for v in values], dtype=float)
        if self.price is not None and self.price_coef is not None:
            util += self.price_coef * profiles[self.price].to_numpy().astype(float)
        return util

    def simulate(
        self, profiles: pl.DataFrame, *, label: str | None = None, scale: float = 1.0
    ) -> pl.DataFrame:
        """Share of preference for a candidate line-up via the logit rule (utilities → shares).

        ``profiles`` is one row per product (attribute columns, plus the price column if fitted).
        Shares are softmax(``scale`` · total utility) and sum to 1 — a first-choice simulation with
        no outside option. ``scale`` > 1 sharpens toward the top product (useful for tuning against
        a holdout); ``label`` names the rows in the output, else they're indexed.
        """
        util = self._utility(profiles)
        shares = np.exp(scale * (util - util.max()))
        shares = shares / shares.sum()
        names = profiles[label].to_list() if label else list(range(profiles.height))
        return pl.DataFrame({"product": names, "utility": util, "share": shares}).sort(
            "share", descending=True
        )

    def willingness_to_pay(self) -> pl.DataFrame:
        """Each level's part-worth re-expressed in price units: utility ÷ (-price coefficient).

        The money a respondent would trade to move from the reference level to this one. Needs a
        numeric ``price`` attribute with a negative coefficient (higher price → lower utility); a
        non-negative price coefficient means WTP is undefined — check for confounding.
        """
        if self.price is None or self.price_coef is None:
            raise ValueError("willingness_to_pay needs a numeric price attribute in the fit")
        if self.price_coef >= 0:
            raise ValueError("price coefficient is non-negative — WTP is undefined; check the data")
        return self.part_worths.with_columns(
            (pl.col("utility") / (-self.price_coef)).alias("wtp")
        ).select("attribute", "level", "wtp")


def choice_based_conjoint(
    df: pl.DataFrame,
    *,
    choice: str,
    choice_set: str,
    attributes: Sequence[str],
    price: str | None = None,
    reference: Mapping[str, Any] | None = None,
) -> Conjoint:
    """Choice-based conjoint end-to-end: encode levels → fit conditional logit → part-worths.

    ``df`` is the long choice table (one row per alternative shown), ``choice`` the 0/1 chosen flag,
    ``choice_set`` the task identifier. ``attributes`` are dummy-coded; pass ``price`` separately to
    keep it a continuous feature (so :meth:`Conjoint.willingness_to_pay` works). The returned
    :class:`Conjoint` holds the utilities, importances, and the share simulator.
    """
    design, level_maps, terms = encode_attributes(df, attributes, reference=reference)
    feature_cols = list(terms)
    frame = df.select([choice_set, choice]).hstack(design)
    if price is not None:
        frame = frame.with_columns(df[price])
        feature_cols.append(price)
    model = fit_conditional_logit(
        frame, choice=choice, choice_set=choice_set, features=feature_cols
    )
    part_worths = _part_worths(model.coefficients, level_maps)
    price_coef = None
    if price is not None:
        price_coef = float(model.coefficients.filter(pl.col("term") == price)["coef"][0])
    return Conjoint(
        part_worths=part_worths,
        importance=attribute_importance(part_worths),
        attributes=list(attributes),
        level_maps=level_maps,
        price=price,
        price_coef=price_coef,
        model=model,
    )


def metric_conjoint(
    df: pl.DataFrame,
    *,
    rating: str,
    attributes: Sequence[str],
    price: str | None = None,
    reference: Mapping[str, Any] | None = None,
) -> Conjoint:
    """Ratings-based (metric) conjoint via OLS — for *rated* profiles, not chosen ones.

    Each profile carries a numeric ``rating`` (preference / purchase-intent score); the dummy-coded
    level coefficients are the part-worths and the intercept is the baseline rating at every
    attribute's reference level (and, when a numeric ``price`` is fitted, at price = 0). Simpler and
    lower-variance than CBC when ratings are available, but rating scales are noisier and less
    behaviourally grounded than real choices.
    """
    from core.analytics import regression

    design, level_maps, terms = encode_attributes(df, attributes, reference=reference)
    feature_cols = list(terms)
    frame = df.select([rating]).hstack(design)
    if price is not None:
        frame = frame.with_columns(df[price])
        feature_cols.append(price)
    fit = regression.ols_fit(frame, y=rating, x=feature_cols)
    lookup = {row["term"]: row for row in fit.coefficients.iter_rows(named=True)}
    part_worths = _part_worths(fit.coefficients, level_maps)
    price_coef = float(lookup[price]["coef"]) if price is not None else None
    return Conjoint(
        part_worths=part_worths,
        importance=attribute_importance(part_worths),
        attributes=list(attributes),
        level_maps=level_maps,
        price=price,
        price_coef=price_coef,
        model=None,
        intercept=float(lookup["intercept"]["coef"]),
        r_squared=fit.r_squared,
    )


# --- MaxDiff (best-worst scaling) -----------------------------------------------------------


def maxdiff_counts(
    df: pl.DataFrame, *, item_col: str, best_col: str, worst_col: str
) -> pl.DataFrame:
    """Counting analysis for MaxDiff — the assumption-free best-minus-worst score per item.

    ``df`` is long (one row per item shown in a task) with 0/1 ``best_col`` / ``worst_col`` flags.
    ``score`` = (times best - times worst) / times shown, in [-1, 1]: +1 always best, -1 always
    worst, 0 neutral. The score pools across all tasks, so no set identifier is needed; filter to
    one respondent first for individual-level scores. :func:`maxdiff_logit` puts the same data on
    an interval utility scale.
    """
    aggregated = (
        df.group_by(item_col)
        .agg(
            pl.len().alias("n_shown"),
            pl.col(best_col).sum().alias("n_best"),
            pl.col(worst_col).sum().alias("n_worst"),
        )
        .with_columns(((pl.col("n_best") - pl.col("n_worst")) / pl.col("n_shown")).alias("score"))
        .rename({item_col: "item"})
        .sort("score", descending=True)
    )
    return aggregated.select("item", "n_shown", "n_best", "n_worst", "score")


def maxdiff_logit(
    df: pl.DataFrame,
    *,
    set_col: str,
    item_col: str,
    best_col: str,
    worst_col: str,
    reference: Any | None = None,
) -> pl.DataFrame:
    """MaxDiff utilities via the exploded best-worst logit (interval scale, reference item = 0).

    Models each best pick as a max-utility choice among the shown items and each worst pick as a
    *min*-utility choice (a logit on negated utilities), then fits one conditional logit over the
    pooled pseudo-choices. Returns item / utility (+ SE / CI) and a softmax ``share`` (preference
    probability). The interval scale supports ratio-style statements the counting score can't.
    """
    items = sorted(df[item_col].unique().to_list(), key=str)
    if len(items) < 2:
        raise ValueError("need at least two items for MaxDiff")
    ref = items[0] if reference is None else reference
    if ref not in items:
        raise ValueError(f"reference item {ref!r} not found")
    non_ref = [it for it in items if it != ref]
    term_names = [f"item={it}" for it in non_ref]

    syn_set: list[str] = []
    chosen: list[float] = []
    columns: dict[str, list[float]] = {term: [] for term in term_names}
    for part in df.partition_by(set_col):
        set_id = str(part[set_col][0])
        shown = part[item_col].to_list()
        best_rows = part.filter(pl.col(best_col) == 1)[item_col].to_list()
        worst_rows = part.filter(pl.col(worst_col) == 1)[item_col].to_list()
        if len(best_rows) != 1 or len(worst_rows) != 1:
            raise ValueError(f"set {set_id} must have exactly one best and one worst item")
        best_item, worst_item = best_rows[0], worst_rows[0]
        if best_item == worst_item:
            raise ValueError(f"set {set_id}: the same item cannot be both best and worst")
        for sign, label, picked in ((1.0, "b", best_item), (-1.0, "w", worst_item)):
            for item in shown:
                syn_set.append(f"{set_id}_{label}")
                chosen.append(1.0 if item == picked else 0.0)
                for term, term_item in zip(term_names, non_ref, strict=True):
                    columns[term].append(sign if item == term_item else 0.0)

    design = pl.DataFrame({"set": syn_set, "chosen": chosen, **columns})
    model = fit_conditional_logit(design, choice="chosen", choice_set="set", features=term_names)
    coefs = {row["term"]: row for row in model.coefficients.iter_rows(named=True)}
    rows: list[dict[str, Any]] = [
        {"item": str(ref), "utility": 0.0, "std_err": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    ]
    for item, term in zip(non_ref, term_names, strict=True):
        coef = coefs[term]
        rows.append(
            {
                "item": str(item),
                "utility": float(coef["coef"]),
                "std_err": float(coef["std_err"]),
                "ci_low": float(coef["ci_low"]),
                "ci_high": float(coef["ci_high"]),
            }
        )
    table = pl.DataFrame(rows)
    util = table["utility"].to_numpy()
    shares = np.exp(util - util.max())
    return table.with_columns(pl.Series("share", shares / shares.sum())).sort(
        "utility", descending=True
    )


# --- Experimental structures (design generation) --------------------------------------------


def full_factorial(levels: Mapping[str, Sequence[Any]]) -> pl.DataFrame:
    """Every attribute-level combination — the full-factorial profile space.

    ``levels`` maps each attribute to its allowed values; the result has one column per attribute
    and one row per profile (∏ level counts rows). The exhaustive design is estimable but explodes
    combinatorially — pass it to :func:`orthogonal_design` to pick an efficient fraction.
    """
    names = list(levels)
    combos = list(itertools.product(*(list(levels[name]) for name in names)))
    return pl.DataFrame({name: [combo[i] for combo in combos] for i, name in enumerate(names)})


def _design_matrix(
    profiles: pl.DataFrame, levels: Mapping[str, Sequence[Any]]
) -> NDArray[np.float64]:
    """Intercept + reference-dropped dummy columns, coded against the *full* intended level set.

    Encoding the fixed level set (not just the levels present in ``profiles``) is what makes designs
    comparable: a design missing a level gets that all-zero dummy column, so it reads as singular
    rather than masquerading as a smaller, falsely-efficient model.
    """
    columns: list[NDArray[np.float64]] = [np.ones((profiles.height, 1))]
    for attr, values in levels.items():
        ordered = pl.Series(list(values)).unique().sort().to_list()
        for lvl in ordered[1:]:  # drop the first (reference) level
            indicator = (profiles[attr] == lvl).cast(pl.Int8).to_numpy().astype(float)
            columns.append(indicator.reshape(-1, 1))
    return np.column_stack(columns)


def d_efficiency(profiles: pl.DataFrame, levels: Mapping[str, Sequence[Any]]) -> float:
    """D-efficiency score of a design: (det(Xᵀ·X / n))^(1/p) for the dummy-coded matrix X.

    Higher = more orthogonal and balanced = tighter, less-correlated part-worth estimates; 0 means
    a singular (rank-deficient) design that can't identify every effect — e.g. a missing level.
    Compare designs only for the *same* model (same ``levels``); ``levels`` is the full intended
    attribute → values spec, so coverage of every level is required to score above 0.
    """
    x = _design_matrix(profiles, levels)
    n, p = x.shape
    sign, logdet = np.linalg.slogdet(x.T @ x / n)
    return float(np.exp(logdet / p)) if sign > 0 else 0.0


def orthogonal_design(
    levels: Mapping[str, Sequence[Any]],
    *,
    n_profiles: int | None = None,
    seed: int = 42,
    restarts: int = 10,
) -> pl.DataFrame:
    """A compact D-efficient fractional design — the profiles to actually field, not all of them.

    Searches the full factorial with a Fedorov coordinate-exchange algorithm (``restarts`` random
    starts, ``seed`` for reproducibility) to maximise :func:`d_efficiency`. ``n_profiles`` defaults
    to the number of estimable parameters (the identifiability floor); raise it to add precision and
    let you test interactions. Returns the chosen profiles; check their :func:`d_efficiency`.
    """
    full = full_factorial(levels)
    candidates = _design_matrix(full, levels)
    n_candidates, p = candidates.shape
    if n_candidates > 50_000:
        raise ValueError(
            f"full factorial has {n_candidates} profiles — reduce levels before searching"
        )
    size = p if n_profiles is None else n_profiles
    size = min(max(size, p), n_candidates)
    rng = np.random.default_rng(seed)

    def logdet(rows: NDArray[np.intp]) -> float:
        sign, value = np.linalg.slogdet(candidates[rows].T @ candidates[rows])
        return float(value) if sign > 0 else -np.inf

    best_rows: NDArray[np.intp] | None = None
    best_score = -np.inf
    for _ in range(restarts):
        rows: NDArray[np.intp] = np.asarray(
            rng.choice(n_candidates, size=size, replace=False), dtype=np.intp
        )
        improved = True
        while improved:
            improved = False
            current = logdet(rows)
            for position in range(size):
                original = int(rows[position])
                for candidate in range(n_candidates):
                    if candidate in rows:
                        continue
                    rows[position] = candidate
                    score = logdet(rows)
                    if score > current + 1e-9:
                        current, original, improved = score, candidate, True
                    else:
                        rows[position] = original
                rows[position] = original
        score = logdet(rows)
        if best_rows is None or score > best_score:
            best_score, best_rows = score, rows.copy()

    if best_rows is None or not np.isfinite(best_score):
        raise ValueError(
            "no full-rank (estimable) design found — increase n_profiles or restarts, "
            "or check the level spec"
        )
    selected = full[np.sort(best_rows).tolist()]
    assert isinstance(selected, pl.DataFrame)
    return selected


def choice_design(
    profiles: pl.DataFrame,
    *,
    n_sets: int,
    alternatives: int,
    seed: int = 42,
    include_none: bool = False,
) -> pl.DataFrame:
    """Assemble profiles into choice tasks for a CBC survey (no responses yet).

    Draws ``alternatives`` distinct profiles per task for ``n_sets`` tasks (seeded). Output is the
    long format :func:`fit_conditional_logit` expects once a ``choice`` flag is collected:
    ``choice_set`` / ``alternative`` / the attribute columns. ``include_none`` appends a no-choice
    option per set (a ``none`` indicator column = the outside-option constant) so share simulation
    can include "buys nothing".
    """
    if alternatives < 2:
        raise ValueError("a choice task needs at least two alternatives")
    if alternatives > profiles.height:
        raise ValueError("not enough distinct profiles for that many alternatives")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for set_id in range(n_sets):
        picks = rng.choice(profiles.height, size=alternatives, replace=False)
        for alt, index in enumerate(picks):
            rows.append(
                {
                    "choice_set": set_id,
                    "alternative": alt,
                    **profiles.row(int(index), named=True),
                    "none": 0,
                }
            )
        if include_none:
            empty = {name: None for name in profiles.columns}
            rows.append({"choice_set": set_id, "alternative": alternatives, **empty, "none": 1})
    return pl.DataFrame(rows)


def maxdiff_design(
    items: Sequence[Any], *, n_sets: int, items_per_set: int, seed: int = 42
) -> pl.DataFrame:
    """A balanced MaxDiff design — assign items to tasks so each appears about equally often.

    Chunks repeated random permutations of the item pool into tasks of ``items_per_set`` distinct
    items (seeded), which keeps exposure even without solving a full BIBD. Returns long ``set`` /
    ``item`` rows; collect a best and a worst pick per set, then read with :func:`maxdiff_logit`.
    """
    pool_items = list(items)
    if items_per_set < 2 or items_per_set > len(pool_items):
        raise ValueError("items_per_set must be between 2 and the number of items")
    rng = np.random.default_rng(seed)
    queue: list[int] = []
    rows: list[dict[str, Any]] = []
    for set_id in range(n_sets):
        chosen: list[int] = []
        while len(chosen) < items_per_set:
            if not queue:
                queue.extend(int(i) for i in rng.permutation(len(pool_items)))
            candidate = queue.pop(0)
            if candidate not in chosen:
                chosen.append(candidate)
        for index in chosen:
            rows.append({"set": set_id, "item": pool_items[index]})
    return pl.DataFrame(rows)


__all__ = [
    "ChoiceModel",
    "Conjoint",
    "attribute_importance",
    "choice_based_conjoint",
    "choice_design",
    "d_efficiency",
    "encode_attributes",
    "fit_conditional_logit",
    "full_factorial",
    "maxdiff_counts",
    "maxdiff_design",
    "maxdiff_logit",
    "metric_conjoint",
    "orthogonal_design",
]
